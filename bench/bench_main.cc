#ifndef ADMC_HAVE_GBENCH
#define ADMC_HAVE_GBENCH 0
#endif

#if ADMC_HAVE_GBENCH
#include "benchmark/benchmark.h"
#endif

#include "bench_csv_schema.hpp"
#include "contact_gen.hpp"
#include "joints.hpp"
#include "metrics.hpp"
#include "metrics_micro.hpp"
#include "scenes.hpp"
#include "solver_baseline_vec.hpp"
#include "solver_scalar_cached.hpp"
#include "solver_scalar_soa.hpp"
#include "solver_scalar_soa_vectorized.hpp"
#include "solver_scalar_soa_mt.hpp"
#include "solver_scalar_soa_simd.hpp"
#include "soa_pack.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <ctime>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
struct BenchmarkResult {
  std::string scene;
  std::string solver;
  int iterations = 0;
  int steps = 0;
  double dt = 0.0;
  std::size_t bodies = 0;
  std::size_t contacts = 0;
  std::size_t joints = 0;
  int tile_size = 0;
  double ms_per_step = 0.0;
  double drift_max = 0.0;
  double penetration_linf = 0.0;
  double energy_drift = 0.0;
  double cone_consistency = 0.0;
  double joint_Linf = 0.0;
  bool simd = false;
  int threads = 1;
  bool has_soa_timings = false;
  SoaTimingBreakdown soa_timings;
  std::string soa_debug_summary;
  std::string commit_sha;
};

std::string default_results_csv_path() {
  using clock = std::chrono::system_clock;
  const auto now = clock::now();
  const std::time_t tt = clock::to_time_t(now);
  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &tt);
#else
  localtime_r(&tt, &tm);
#endif
  std::ostringstream oss;
  oss << "results/" << std::put_time(&tm, "%Y%m%d") << "/results.csv";
  return oss.str();
}

std::vector<BenchmarkResult> g_results;
std::string g_results_csv_path = default_results_csv_path();

using Clock = std::chrono::steady_clock;
using DurationMs = std::chrono::duration<double, std::milli>;

double elapsed_ms(const Clock::time_point& begin, const Clock::time_point& end) {
  return DurationMs(end - begin).count();
}

struct MicrobenchResult {
  std::string kernel;
  std::string variant;
  int lane = 1;
  int threads = 1;
  std::size_t rows = 0;
  double ns_per_row = 0.0;
};

std::vector<MicrobenchResult> g_micro_results;
std::string g_micro_csv_path = "results/microbench.csv";

void record_result(const BenchmarkResult& result) {
  g_results.push_back(result);
}

void record_micro_result(const MicrobenchResult& result) {
  g_micro_results.push_back(result);
}

void write_results_csv();
void write_microbench_csv();
void print_results_table();

void configure_baseline_params(const std::string& scene_name,
                               BaselineParams& params) {
  if (scene_name == "two_spheres") {
    params.beta = 0.0;
    params.slop = 0.0;
    params.restitution = 1.0;
  }
}

void configure_solver_params(const std::string& scene_name,
                             SolverParams& params) {
  if (scene_name == "two_spheres") {
    params.beta = 0.0;
    params.slop = 0.0;
    params.restitution = 1.0;
    params.mu = 0.0;
    params.spheres_only = true;
    params.frictionless = true;
  } else if (scene_name == "spheres_cloud_10k" ||
             scene_name == "spheres_cloud_50k") {
    params.mu = 0.0;
    params.spheres_only = true;
    params.frictionless = true;
  } else if (scene_name == "spheres_cloud_10k_fric") {
    params.mu = (params.mu <= 0.0) ? 0.5 : params.mu;
    params.spheres_only = true;
    params.frictionless = false;
  }
}

struct CliOptions {
  bool run_cli = false;
  bool run_benchmark = false;
  std::string solver = "auto";
  std::string scene = "two_spheres";
  int iterations = 10;
  int steps = 30;
  double dt = 1.0 / 60.0;
  int threads = 1;
  std::string csv_path = default_results_csv_path();
  std::vector<int> sizes;
  std::vector<std::string> solver_list;
  std::vector<int> tile_sizes;
  int tile_rows = -1;
  bool spheres_only = false;
  bool frictionless = false;
};

int parse_int_default(const std::string& value, int fallback) {
  try {
    return std::stoi(value);
  } catch (...) {
    return fallback;
  }
}

double parse_double_default(const std::string& value, double fallback) {
  try {
    return std::stod(value);
  } catch (...) {
    return fallback;
  }
}

std::string trim_copy(const std::string& value) {
  const auto begin = value.find_first_not_of(" \t\r\n");
  if (begin == std::string::npos) {
    return std::string();
  }
  const auto end = value.find_last_not_of(" \t\r\n");
  return value.substr(begin, end - begin + 1);
}

std::vector<std::string> split_csv(const std::string& value) {
  std::vector<std::string> tokens;
  std::stringstream ss(value);
  std::string item;
  while (std::getline(ss, item, ',')) {
    const std::string trimmed = trim_copy(item);
    if (!trimmed.empty()) {
      tokens.push_back(trimmed);
    }
  }
  return tokens;
}

std::string current_commit_sha() {
  static std::string cached;
  static bool initialized = false;
  if (initialized) {
    return cached;
  }
  initialized = true;
  if (const char* env = std::getenv("GITHUB_SHA")) {
    const std::string sha_env = trim_copy(env);
    if (!sha_env.empty()) {
      cached = sha_env.substr(0, std::min<std::size_t>(sha_env.size(), 12));
    }
  }
  if (!cached.empty()) {
    return cached;
  }
  std::array<char, 64> buffer{};
  if (FILE* pipe = ::popen("git rev-parse --short HEAD", "r")) {
    if (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe)) {
      cached = trim_copy(buffer.data());
    }
    ::pclose(pipe);
  }
  return cached;
}

std::string normalize_solver_name(std::string name) {
  if (name == "scalar_cached") {
    return "cached";
  }
  if (name == "scalar_soa" || name == "soa_simd" || name == "soa_mt") {
    return "soa";
  }
  if (name == "scalar_soa_vectorized" || name == "soa_vec" ||
      name == "soa_vectorized" || name == "vec_soa") {
    return "vec_soa";
  }
  if (name == "baseline_vec") {
    return "baseline";
  }
  return name;
}

bool make_scene_by_name(const std::string& name, Scene* scene) {
  if (name == "two_spheres") {
    *scene = make_two_spheres_head_on();
  } else if (name == "spheres_cloud_1024") {
    *scene = make_spheres_box_cloud(1024);
  } else if (name == "spheres_cloud_4096") {
    *scene = make_spheres_box_cloud(4096);
  } else if (name == "spheres_cloud_8192") {
    *scene = make_spheres_box_cloud(8192);
  } else if (name == "spheres_cloud_10k") {
    *scene = make_spheres_box_cloud(10000);
  } else if (name == "spheres_cloud_50k") {
    *scene = make_spheres_box_cloud(50000);
  } else if (name == "spheres_cloud_10k_fric") {
    *scene = make_spheres_box_cloud(10000);
  } else if (name == "box_stack_4") {
    *scene = make_box_stack(4);
  } else if (name == "box_stack") {
    *scene = make_box_stack(8);
  } else if (name == "pendulum") {
    *scene = make_pendulum(1);
  } else if (name == "chain_64") {
    *scene = make_chain_64();
  } else if (name == "rope_256") {
    *scene = make_rope_256();
  } else {
    return false;
  }
  return true;
}

bool make_scene_with_size(const std::string& name,
                          int size,
                          Scene* scene,
                          std::string* resolved_name) {
  if (name == "spheres_cloud") {
    const int count = (size > 0) ? size : 1024;
    *scene = make_spheres_box_cloud(count);
    if (resolved_name) {
      *resolved_name = "spheres_cloud_" + std::to_string(count);
    }
    return true;
  }
  if (size > 0) {
    const std::string sized = name + "_" + std::to_string(size);
    if (make_scene_by_name(sized, scene)) {
      if (resolved_name) {
        *resolved_name = sized;
      }
      return true;
    }
  }
  if (make_scene_by_name(name, scene)) {
    if (resolved_name) {
      *resolved_name = name;
    }
    return true;
  }
  return false;
}

CliOptions parse_cli_options(int argc, char** argv, std::vector<char*>& passthrough) {
  CliOptions opts;
  passthrough.push_back(argv[0]);
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg.rfind("--solver=", 0) == 0) {
      opts.run_cli = true;
      opts.solver = arg.substr(9);
    } else if (arg.rfind("--scene=", 0) == 0) {
      opts.run_cli = true;
      opts.scene = arg.substr(8);
    } else if (arg.rfind("--iters=", 0) == 0) {
      opts.run_cli = true;
      opts.iterations = std::max(1, parse_int_default(arg.substr(8), opts.iterations));
    } else if (arg.rfind("--steps=", 0) == 0) {
      opts.run_cli = true;
      opts.steps = std::max(1, parse_int_default(arg.substr(8), opts.steps));
    } else if (arg.rfind("--dt=", 0) == 0) {
      opts.run_cli = true;
      opts.dt = parse_double_default(arg.substr(5), opts.dt);
    } else if (arg.rfind("--threads=", 0) == 0) {
      opts.run_cli = true;
      opts.threads = std::max(1, parse_int_default(arg.substr(10), opts.threads));
    } else if (arg.rfind("--sizes=", 0) == 0) {
      opts.run_cli = true;
      opts.sizes.clear();
      for (const std::string& token : split_csv(arg.substr(8))) {
        const int value = parse_int_default(token, -1);
        if (value > 0) {
          opts.sizes.push_back(value);
        }
      }
    } else if (arg.rfind("--solvers=", 0) == 0) {
      opts.run_cli = true;
      opts.solver_list = split_csv(arg.substr(10));
    } else if (arg.rfind("--tile-sizes=", 0) == 0) {
      opts.run_cli = true;
      opts.tile_sizes.clear();
      for (const std::string& token : split_csv(arg.substr(13))) {
        const int value = parse_int_default(token, -1);
        if (value > 0) {
          opts.tile_sizes.push_back(value);
        }
      }
    } else if (arg.rfind("--tile_rows=", 0) == 0) {
      opts.run_cli = true;
      opts.tile_rows = std::max(1, parse_int_default(arg.substr(12), 128));
    } else if (arg == "--spheres-only") {
      opts.run_cli = true;
      opts.spheres_only = true;
    } else if (arg == "--frictionless") {
      opts.run_cli = true;
      opts.frictionless = true;
    } else if (arg.rfind("--csv=", 0) == 0) {
      opts.csv_path = arg.substr(6);
    } else if (arg == "--benchmark") {
      opts.run_benchmark = true;
    } else {
      passthrough.push_back(argv[i]);
    }
  }
  return opts;
}

void print_run_summary(const BenchmarkResult& result) {
  std::cout << "scene=" << result.scene << ", solver=" << result.solver
            << ", steps=" << result.steps << ", ms_per_step=" << std::fixed
            << std::setprecision(3) << result.ms_per_step << std::defaultfloat
            << ", drift_max=" << result.drift_max << ", contacts="
            << result.contacts << "\n";
  if (result.has_soa_timings) {
    std::ostringstream timings;
    timings << "    SoA timings (ms/step): contact=" << std::fixed
            << std::setprecision(3) << result.soa_timings.contact_prep_ms
            << ", row=" << result.soa_timings.row_build_ms
            << ", joint_build=" << result.soa_timings.joint_distance_build_ms
            << ", joint_pack=" << result.soa_timings.joint_pack_ms
            << ", solver=" << result.soa_timings.solver_total_ms
            << " [warm=" << result.soa_timings.solver_warmstart_ms
            << ", iter=" << result.soa_timings.solver_iterations_ms
            << ", integ=" << result.soa_timings.solver_integrate_ms
            << "], scatter=" << result.soa_timings.scatter_ms
            << ", total=" << result.soa_timings.total_step_ms;
    std::cout << timings.str() << '\n';
    if (!result.soa_debug_summary.empty()) {
      std::cout << "    SoA debug: " << result.soa_debug_summary << '\n';
    }
  }
}

enum class SoaSolverVariant {
  kLegacy,
  kVectorized,
};

BenchmarkResult run_soa_variant_result(const std::string& scene_name,
                                       const Scene& base_scene,
                                       const SoaParams& params,
                                       int steps,
                                       double ms_per_step_hint,
                                       SoaSolverVariant variant);

BenchmarkResult run_soa_result(const std::string& scene_name,
                               const Scene& base_scene,
                               const SoaParams& params,
                               int steps,
                               double ms_per_step_hint);

BenchmarkResult run_soa_result(const std::string& scene_name,
                               const Scene& base_scene,
                               const SolverParams& params,
                               int steps,
                               double ms_per_step_hint);

BenchmarkResult run_soa_vectorized_result(const std::string& scene_name,
                                          const Scene& base_scene,
                                          const SoaParams& params,
                                          int steps,
                                          double ms_per_step_hint);

BenchmarkResult run_soa_vectorized_result(const std::string& scene_name,
                                          const Scene& base_scene,
                                          const SolverParams& params,
                                          int steps,
                                          double ms_per_step_hint);

std::optional<BenchmarkResult> run_solver_case(const std::string& solver_name,
                                               const std::string& scene_name,
                                               const Scene& base_scene,
                                               int iterations,
                                               int steps,
                                               double dt,
                                               int threads,
                                               int tile_size_override,
                                               const CliOptions* overrides = nullptr) {
  const int safe_iterations = std::max(1, iterations);
  const int safe_steps = std::max(1, steps);
  const int safe_threads = std::max(1, threads);

  std::string normalized = normalize_solver_name(solver_name);

  std::vector<RigidBody> bodies = base_scene.bodies;
  std::vector<Contact> contacts = base_scene.contacts;
  std::vector<DistanceJoint> joints = base_scene.joints;
  refresh_contacts_from_state(bodies, contacts);
  const std::vector<RigidBody> pre = bodies;

  const auto start = std::chrono::steady_clock::now();
  bool simd_used = false;
  int threads_used = 1;

  if (normalized == "baseline") {
    BaselineParams params;
    params.iterations = safe_iterations;
    params.dt = dt;
    configure_baseline_params(scene_name, params);
    for (int step = 0; step < safe_steps; ++step) {
      build_contact_offsets_and_bias(bodies, contacts, params);
      solve_baseline(bodies, contacts, params);
    }
    refresh_contacts_from_state(bodies, contacts);
  } else if (normalized == "cached") {
    SolverParams params;
    params.iterations = safe_iterations;
    params.dt = dt;
    configure_solver_params(scene_name, params);
    if (overrides) {
      if (overrides->tile_rows > 0) {
        params.tile_rows = overrides->tile_rows;
      }
      params.spheres_only = overrides->spheres_only;
      params.frictionless = overrides->frictionless;
    }
    for (int step = 0; step < safe_steps; ++step) {
      build_contact_offsets_and_bias(bodies, contacts, params);
      build_distance_joint_rows(bodies, joints, params.dt);
      solve_scalar_cached(bodies, contacts, joints, params);
    }
    build_distance_joint_rows(bodies, joints, params.dt);
    refresh_contacts_from_state(bodies, contacts);
  } else if (normalized == "soa") {
    SoaParams params;
    params.iterations = safe_iterations;
    params.dt = dt;
    params.use_threads = (safe_threads > 1);
    params.thread_count = safe_threads;
    if (tile_size_override > 0) {
      params.tile_size = tile_size_override;
      params.max_contacts_per_tile = tile_size_override;
    }
    if (overrides) {
      if (overrides->tile_rows > 0) {
        params.tile_rows = overrides->tile_rows;
      }
      params.spheres_only = overrides->spheres_only;
      params.frictionless = overrides->frictionless;
    }
    configure_solver_params(scene_name, params);
    BenchmarkResult soa_result =
        run_soa_result(scene_name, base_scene, params, safe_steps, -1.0);
    return soa_result;
  } else if (normalized == "vec_soa") {
    SoaParams params;
    params.iterations = safe_iterations;
    params.dt = dt;
    params.use_threads = (safe_threads > 1);
    params.thread_count = safe_threads;
    if (tile_size_override > 0) {
      params.tile_size = tile_size_override;
      params.max_contacts_per_tile = tile_size_override;
    }
    if (overrides) {
      if (overrides->tile_rows > 0) {
        params.tile_rows = overrides->tile_rows;
      }
      params.spheres_only = overrides->spheres_only;
      params.frictionless = overrides->frictionless;
    }
    configure_solver_params(scene_name, params);
    BenchmarkResult soa_result = run_soa_vectorized_result(scene_name, base_scene,
                                                       params, safe_steps, -1.0);
    return soa_result;
  } else {
    std::cerr << "Unknown solver: " << solver_name << "\n";
    return std::nullopt;
  }

  const auto end = std::chrono::steady_clock::now();
  const double total_seconds =
      std::chrono::duration<double>(end - start).count();
  const double ms_per_step = safe_steps > 0
                                 ? (total_seconds / static_cast<double>(safe_steps)) * 1e3
                                 : 0.0;

  BenchmarkResult result;
  result.scene = scene_name;
  result.solver = normalized;
  result.iterations = safe_iterations;
  result.steps = safe_steps;
  result.dt = dt;
  result.bodies = base_scene.bodies.size();
  result.contacts = base_scene.contacts.size();
  result.joints = base_scene.joints.size();
  result.ms_per_step = ms_per_step;
  result.drift_max = directional_momentum_drift(pre, bodies).max_abs;
  result.penetration_linf = constraint_penetration_Linf(contacts);
  result.energy_drift = energy_drift(pre, bodies);
  result.cone_consistency = cone_consistency(contacts);
  result.joint_Linf = joint_error_Linf(joints);
  result.simd = simd_used;
  result.threads = threads_used;
  result.tile_size = (tile_size_override > 0) ? tile_size_override : 0;
  result.commit_sha = current_commit_sha();
  return result;
}

void run_default_suite(const std::string& csv_path) {
  struct QuickCase {
    const char* scene;
    const char* solver;
    int iterations;
    int steps;
  };

  constexpr double kDefaultDt = 1.0 / 60.0;
  std::vector<QuickCase> cases = {
      {"two_spheres", "baseline", 10, 1},
      {"two_spheres", "cached", 10, 1},
      {"two_spheres", "soa", 10, 1},
      {"two_spheres", "vec_soa", 10, 1},
      {"spheres_cloud_1024", "baseline", 10, 30},
      {"spheres_cloud_1024", "cached", 10, 30},
      {"spheres_cloud_1024", "soa", 10, 30},
      {"spheres_cloud_1024", "vec_soa", 10, 30},
      {"box_stack_4", "baseline", 10, 30},
      {"box_stack_4", "cached", 10, 30},
      {"box_stack_4", "soa", 10, 30},
      {"box_stack_4", "vec_soa", 10, 30},
      {"spheres_cloud_10k", "baseline", 10, 30},
      {"spheres_cloud_10k", "cached", 10, 30},
      {"spheres_cloud_10k", "soa", 10, 30},
      {"spheres_cloud_10k", "vec_soa", 10, 30},
      {"spheres_cloud_50k", "baseline", 10, 30},
      {"spheres_cloud_50k", "cached", 10, 30},
      {"spheres_cloud_50k", "soa", 10, 30},
      {"spheres_cloud_50k", "vec_soa", 10, 30},
      {"spheres_cloud_10k_fric", "baseline", 10, 30},
      {"spheres_cloud_10k_fric", "cached", 10, 30},
      {"spheres_cloud_10k_fric", "soa", 10, 30},
      {"spheres_cloud_10k_fric", "vec_soa", 10, 30},
  };

  const char* run_large_env = std::getenv("RUN_LARGE");
  if (run_large_env && std::string(run_large_env) == "1") {
    cases.push_back({"spheres_cloud_8192", "cached", 10, 30});
    cases.push_back({"spheres_cloud_8192", "soa", 10, 30});
    cases.push_back({"spheres_cloud_8192", "vec_soa", 10, 30});
  }

  std::vector<BenchmarkResult> results;
  results.reserve(cases.size());

  for (const QuickCase& qc : cases) {
    Scene scene;
    if (!make_scene_by_name(qc.scene, &scene)) {
      std::cerr << "Skipping unknown scene: " << qc.scene << "\n";
      continue;
    }
    auto maybe = run_solver_case(qc.solver, qc.scene, scene, qc.iterations,
                                 qc.steps, kDefaultDt, 1, -1);
    if (!maybe.has_value()) {
      continue;
    }
    results.push_back(*maybe);
    print_run_summary(results.back());
  }

  if (!results.empty()) {
    g_results = results;
    g_results_csv_path = csv_path;
    write_results_csv();
    g_results.clear();
  }
}

bool run_cli_mode(const CliOptions& opts) {
  const int steps = std::max(1, opts.steps);
  const int iterations = std::max(1, opts.iterations);
  const double dt = opts.dt;
  const int threads = std::max(1, opts.threads);

  std::vector<std::string> solvers;
  if (!opts.solver_list.empty()) {
    solvers = opts.solver_list;
  } else if (opts.solver == "auto") {
    solvers = {"baseline", "cached", "soa", "vec_soa"};
  } else {
    solvers.push_back(opts.solver);
  }

  if (solvers.empty()) {
    std::cerr << "No solvers specified.\n";
    return false;
  }

  struct SceneRequest {
    std::string name;
    Scene scene;
  };

  std::vector<SceneRequest> scenes;
  if (!opts.sizes.empty()) {
    for (int size : opts.sizes) {
      Scene scene;
      std::string label;
      if (!make_scene_with_size(opts.scene, size, &scene, &label)) {
        std::cerr << "Skipping unknown scene: " << opts.scene << " size=" << size
                  << "\n";
        continue;
      }
      scenes.push_back({label, std::move(scene)});
    }
  } else {
    Scene scene;
    std::string label;
    if (!make_scene_with_size(opts.scene, -1, &scene, &label)) {
      std::cerr << "Unknown scene: " << opts.scene << "\n";
      return false;
    }
    scenes.push_back({label, std::move(scene)});
  }

  if (scenes.empty()) {
    std::cerr << "No valid scenes to run.\n";
    return false;
  }

  std::vector<int> tile_sizes = opts.tile_sizes;
  if (tile_sizes.empty()) {
    tile_sizes.push_back(-1);
  }

  std::vector<BenchmarkResult> results;

  for (const SceneRequest& req : scenes) {
    for (const std::string& solver_name : solvers) {
      const std::string normalized = normalize_solver_name(solver_name);
      const bool uses_tiles = (normalized == "soa" || normalized == "vec_soa");
      if (uses_tiles) {
        for (int tile_size : tile_sizes) {
          const int override_tile = (tile_size > 0) ? tile_size : -1;
          auto maybe = run_solver_case(solver_name, req.name, req.scene, iterations,
                                       steps, dt, threads, override_tile, &opts);
          if (!maybe.has_value()) {
            std::cerr << "Skipping solver: " << solver_name
                      << " for scene " << req.name << "\n";
            continue;
          }
          results.push_back(*maybe);
          print_run_summary(results.back());
        }
      } else {
        auto maybe = run_solver_case(solver_name, req.name, req.scene, iterations,
                                     steps, dt, threads, -1, &opts);
        if (!maybe.has_value()) {
          std::cerr << "Skipping solver: " << solver_name
                    << " for scene " << req.name << "\n";
          continue;
        }
        results.push_back(*maybe);
        print_run_summary(results.back());
      }
    }
  }

  if (results.empty()) {
    std::cerr << "No runs executed.\n";
    return false;
  }

  g_results = results;
  g_results_csv_path = opts.csv_path;
  write_results_csv();
  g_results.clear();

  return true;
}

#if ADMC_HAVE_GBENCH
constexpr int kStepsPerRun = 60;

BenchmarkResult run_baseline_result(const std::string& scene_name,
                                    const Scene& base_scene,
                                    const BaselineParams& params,
                                    int steps,
                                    double ms_per_step) {
  std::vector<RigidBody> bodies = base_scene.bodies;
  std::vector<Contact> contacts = base_scene.contacts;
  const std::vector<RigidBody> pre = bodies;
  for (int i = 0; i < steps; ++i) {
    build_contact_offsets_and_bias(bodies, contacts, params);
    solve_baseline(bodies, contacts, params);
  }
  BenchmarkResult result;
  result.scene = scene_name;
  result.solver = "baseline_vec";
  result.iterations = params.iterations;
  result.steps = steps;
  result.dt = params.dt;
  result.bodies = base_scene.bodies.size();
  result.contacts = base_scene.contacts.size();
  result.joints = base_scene.joints.size();
  result.ms_per_step = ms_per_step;
  result.drift_max = directional_momentum_drift(pre, bodies).max_abs;
  result.penetration_linf = constraint_penetration_Linf(contacts);
  result.energy_drift = energy_drift(pre, bodies);
  result.cone_consistency = cone_consistency(contacts);
  result.joint_Linf = joint_error_Linf(base_scene.joints);
  result.simd = false;
  result.threads = 1;
  result.tile_size = 0;
  result.commit_sha = current_commit_sha();
  return result;
}

BenchmarkResult run_cached_result(const std::string& scene_name,
                                  const Scene& base_scene,
                                  const SolverParams& params,
                                  int steps,
                                  double ms_per_step) {
  std::vector<RigidBody> bodies = base_scene.bodies;
  std::vector<Contact> contacts = base_scene.contacts;
  std::vector<DistanceJoint> joints = base_scene.joints;
  const std::vector<RigidBody> pre = bodies;
  for (int i = 0; i < steps; ++i) {
    build_contact_offsets_and_bias(bodies, contacts, params);
    build_distance_joint_rows(bodies, joints, params.dt);
    solve_scalar_cached(bodies, contacts, joints, params);
  }
  build_distance_joint_rows(bodies, joints, params.dt);
  BenchmarkResult result;
  result.scene = scene_name;
  result.solver = "scalar_cached";
  result.iterations = params.iterations;
  result.steps = steps;
  result.dt = params.dt;
  result.bodies = base_scene.bodies.size();
  result.contacts = base_scene.contacts.size();
  result.joints = base_scene.joints.size();
  result.ms_per_step = ms_per_step;
  result.drift_max = directional_momentum_drift(pre, bodies).max_abs;
  result.penetration_linf = constraint_penetration_Linf(contacts);
  result.energy_drift = energy_drift(pre, bodies);
  result.cone_consistency = cone_consistency(contacts);
  result.joint_Linf = joint_error_Linf(joints);
  result.simd = false;
  result.threads = 1;
  result.tile_size = 0;
  result.commit_sha = current_commit_sha();
  return result;
}

template <typename StepFn>
double run_benchmark_loop(benchmark::State& state,
                          const Scene& base_scene,
                          const StepFn& step_fn) {
  double total_elapsed = 0.0;
  int iterations = 0;
  for (auto _ : state) {
    std::vector<RigidBody> bodies = base_scene.bodies;
    std::vector<Contact> contacts = base_scene.contacts;
    std::vector<DistanceJoint> joints = base_scene.joints;
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < kStepsPerRun; ++step) {
      step_fn(bodies, contacts, joints);
    }
    const auto end = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(end - start).count();
    total_elapsed += elapsed;
    ++iterations;
    state.SetIterationTime(elapsed);
  }
  return iterations > 0 ? (total_elapsed / iterations / static_cast<double>(kStepsPerRun)) * 1e3
                        : 0.0;
}

void update_counters(benchmark::State& state,
                     const Scene& scene,
                     double ms_per_step) {
  state.counters["bodies"] = benchmark::Counter(
      static_cast<double>(scene.bodies.size()), benchmark::Counter::kAvgThreads);
  state.counters["contacts"] = benchmark::Counter(
      static_cast<double>(scene.contacts.size()), benchmark::Counter::kAvgThreads);
  state.counters["joints"] = benchmark::Counter(
      static_cast<double>(scene.joints.size()), benchmark::Counter::kAvgThreads);
  state.counters["ms_per_step"] =
      benchmark::Counter(ms_per_step, benchmark::Counter::kAvgThreads);
}

void BenchSpheresCloudBaseline4096(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_spheres_box_cloud(4096);
  BaselineParams params;
  params.beta = 0.2;
  params.slop = 0.005;
  params.restitution = 0.0;
  params.iterations = 10;
  params.dt = 1.0 / 60.0;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene,
      [&](std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
          std::vector<DistanceJoint>& joints) {
        (void)joints;
        build_contact_offsets_and_bias(bodies, contacts, params);
        solve_baseline(bodies, contacts, params);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_baseline_result("spheres_box_cloud_4096", base_scene, params,
                                      kStepsPerRun, ms_per_step));
    recorded = true;
  }
}

void BenchSpheresCloudCached4096(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_spheres_box_cloud(4096);
  SolverParams params;
  params.beta = 0.2;
  params.slop = 0.005;
  params.restitution = 0.0;
  params.iterations = 10;
  params.dt = 1.0 / 60.0;
  params.mu = 0.5;
  params.warm_start = true;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene,
      [&](std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
          std::vector<DistanceJoint>& joints) {
        build_contact_offsets_and_bias(bodies, contacts, params);
        build_distance_joint_rows(bodies, joints, params.dt);
        solve_scalar_cached(bodies, contacts, joints, params);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_cached_result("spheres_box_cloud_4096", base_scene, params,
                                    kStepsPerRun, ms_per_step));
    recorded = true;
  }
}

void BenchSpheresCloudSoA4096(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_spheres_box_cloud(4096);
  SolverParams params;
  params.beta = 0.2;
  params.slop = 0.005;
  params.restitution = 0.0;
  params.iterations = 10;
  params.dt = 1.0 / 60.0;
  params.mu = 0.5;
  params.warm_start = true;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene,
      [&, rows = RowSOA{}, joint_rows = JointSOA{}](
          std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
          std::vector<DistanceJoint>& joints) mutable {
        build_contact_offsets_and_bias(bodies, contacts, params);
        build_soa(bodies, contacts, params, rows);
        build_distance_joint_rows(bodies, joints, params.dt);
        build_joint_soa(bodies, joints, params.dt, joint_rows);
        solve_scalar_soa(bodies, contacts, rows, joint_rows, params);
        scatter_joint_impulses(joint_rows, joints);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_soa_result("spheres_box_cloud_4096", base_scene, params,
                                 kStepsPerRun, ms_per_step));
    recorded = true;
  }
}

void BenchSpheresCloudSoA4096Fourth(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_spheres_box_cloud(4096);
  SolverParams params;
  params.beta = 0.2;
  params.slop = 0.005;
  params.restitution = 0.0;
  params.iterations = 10;
  params.dt = 1.0 / 60.0;
  params.mu = 0.5;
  params.warm_start = true;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene,
      [&, rows = RowSOA{}, joint_rows = JointSOA{}](
          std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
          std::vector<DistanceJoint>& joints) mutable {
        build_contact_offsets_and_bias(bodies, contacts, params);
        build_soa(bodies, contacts, params, rows);
        build_distance_joint_rows(bodies, joints, params.dt);
        build_joint_soa(bodies, joints, params.dt, joint_rows);
        solve_scalar_soa_vectorized(bodies, contacts, rows, joint_rows, params);
        scatter_joint_impulses(joint_rows, joints);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_soa_vectorized_result("spheres_box_cloud_4096", base_scene,
                                        params, kStepsPerRun, ms_per_step));
    recorded = true;
  }
}

void BenchSpheresCloudBaseline8192(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_spheres_box_cloud(8192);
  BaselineParams params;
  params.beta = 0.2;
  params.slop = 0.005;
  params.restitution = 0.0;
  params.iterations = 10;
  params.dt = 1.0 / 60.0;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene,
      [&](std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
          std::vector<DistanceJoint>& joints) {
        (void)joints;
        build_contact_offsets_and_bias(bodies, contacts, params);
        solve_baseline(bodies, contacts, params);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_baseline_result("spheres_box_cloud_8192", base_scene, params,
                                      kStepsPerRun, ms_per_step));
    recorded = true;
  }
}

void BenchSpheresCloudCached8192(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_spheres_box_cloud(8192);
  SolverParams params;
  params.beta = 0.2;
  params.slop = 0.005;
  params.restitution = 0.0;
  params.iterations = 10;
  params.dt = 1.0 / 60.0;
  params.mu = 0.5;
  params.warm_start = true;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene,
      [&](std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
          std::vector<DistanceJoint>& joints) {
        build_contact_offsets_and_bias(bodies, contacts, params);
        build_distance_joint_rows(bodies, joints, params.dt);
        solve_scalar_cached(bodies, contacts, joints, params);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_cached_result("spheres_box_cloud_8192", base_scene, params,
                                    kStepsPerRun, ms_per_step));
    recorded = true;
  }
}

void BenchSpheresCloudSoA8192Fourth(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_spheres_box_cloud(8192);
  SolverParams params;
  params.beta = 0.2;
  params.slop = 0.005;
  params.restitution = 0.0;
  params.iterations = 10;
  params.dt = 1.0 / 60.0;
  params.mu = 0.5;
  params.warm_start = true;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene,
      [&, rows = RowSOA{}, joint_rows = JointSOA{}](
          std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
          std::vector<DistanceJoint>& joints) mutable {
        build_contact_offsets_and_bias(bodies, contacts, params);
        build_soa(bodies, contacts, params, rows);
        build_distance_joint_rows(bodies, joints, params.dt);
        build_joint_soa(bodies, joints, params.dt, joint_rows);
        solve_scalar_soa_vectorized(bodies, contacts, rows, joint_rows, params);
        scatter_joint_impulses(joint_rows, joints);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_soa_vectorized_result("spheres_box_cloud_8192", base_scene,
                                        params, kStepsPerRun, ms_per_step));
    recorded = true;
  }
}

void BenchSpheresCloudSoA8192(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_spheres_box_cloud(8192);
  SolverParams params;
  params.beta = 0.2;
  params.slop = 0.005;
  params.restitution = 0.0;
  params.iterations = 10;
  params.dt = 1.0 / 60.0;
  params.mu = 0.5;
  params.warm_start = true;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene,
      [&, rows = RowSOA{}, joint_rows = JointSOA{}](
          std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
          std::vector<DistanceJoint>& joints) mutable {
        build_contact_offsets_and_bias(bodies, contacts, params);
        build_soa(bodies, contacts, params, rows);
        build_distance_joint_rows(bodies, joints, params.dt);
        build_joint_soa(bodies, joints, params.dt, joint_rows);
        solve_scalar_soa(bodies, contacts, rows, joint_rows, params);
        scatter_joint_impulses(joint_rows, joints);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_soa_result("spheres_box_cloud_8192", base_scene, params,
                                 kStepsPerRun, ms_per_step));
    recorded = true;
  }
}

void BenchBoxStackBaseline(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_box_stack(8);
  BaselineParams params;
  params.beta = 0.2;
  params.slop = 0.005;
  params.restitution = 0.0;
  params.iterations = 10;
  params.dt = 1.0 / 60.0;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene,
      [&](std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
          std::vector<DistanceJoint>& joints) {
        (void)joints;
        build_contact_offsets_and_bias(bodies, contacts, params);
        solve_baseline(bodies, contacts, params);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_baseline_result("box_stack_8", base_scene, params, kStepsPerRun,
                                      ms_per_step));
    recorded = true;
  }
}

void BenchBoxStackCached(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_box_stack(8);
  SolverParams params;
  params.beta = 0.2;
  params.slop = 0.005;
  params.restitution = 0.0;
  params.iterations = 10;
  params.dt = 1.0 / 60.0;
  params.mu = 0.5;
  params.warm_start = true;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene,
      [&](std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
          std::vector<DistanceJoint>& joints) {
        build_contact_offsets_and_bias(bodies, contacts, params);
        build_distance_joint_rows(bodies, joints, params.dt);
        solve_scalar_cached(bodies, contacts, joints, params);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_cached_result("box_stack_8", base_scene, params, kStepsPerRun,
                                    ms_per_step));
    recorded = true;
  }
}

void BenchBoxStackSoA(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_box_stack(8);
  SolverParams params;
  params.beta = 0.2;
  params.slop = 0.005;
  params.restitution = 0.0;
  params.iterations = 10;
  params.dt = 1.0 / 60.0;
  params.mu = 0.5;
  params.warm_start = true;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene,
      [&, rows = RowSOA{}, joint_rows = JointSOA{}](
          std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
          std::vector<DistanceJoint>& joints) mutable {
        build_contact_offsets_and_bias(bodies, contacts, params);
        build_soa(bodies, contacts, params, rows);
        build_distance_joint_rows(bodies, joints, params.dt);
        build_joint_soa(bodies, joints, params.dt, joint_rows);
        solve_scalar_soa(bodies, contacts, rows, joint_rows, params);
        scatter_joint_impulses(joint_rows, joints);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_soa_result("box_stack_8", base_scene, params, kStepsPerRun,
                                 ms_per_step));
    recorded = true;
  }
}

void BenchBoxStackSoAFourth(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_box_stack(8);
  SolverParams params;
  params.beta = 0.2;
  params.slop = 0.005;
  params.restitution = 0.0;
  params.iterations = 10;
  params.dt = 1.0 / 60.0;
  params.mu = 0.5;
  params.warm_start = true;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene,
      [&, rows = RowSOA{}, joint_rows = JointSOA{}](
          std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
          std::vector<DistanceJoint>& joints) mutable {
        build_contact_offsets_and_bias(bodies, contacts, params);
        build_soa(bodies, contacts, params, rows);
        build_distance_joint_rows(bodies, joints, params.dt);
        build_joint_soa(bodies, joints, params.dt, joint_rows);
        solve_scalar_soa_vectorized(bodies, contacts, rows, joint_rows, params);
        scatter_joint_impulses(joint_rows, joints);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_soa_vectorized_result("box_stack_8", base_scene, params,
                                        kStepsPerRun, ms_per_step));
    recorded = true;
  }
}

void BenchPendulumCached(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_pendulum(1);
  SolverParams params;
  params.beta = 0.2;
  params.slop = 0.0;
  params.restitution = 0.0;
  params.iterations = 40;
  params.dt = 1.0 / 120.0;
  params.mu = 0.0;
  params.warm_start = true;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene,
      [&](std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
          std::vector<DistanceJoint>& joints) {
        (void)contacts;
        build_distance_joint_rows(bodies, joints, params.dt);
        solve_scalar_cached(bodies, contacts, joints, params);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_cached_result("pendulum", base_scene, params, kStepsPerRun,
                                    ms_per_step));
    recorded = true;
  }
}

void BenchPendulumSoA(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_pendulum(1);
  SolverParams params;
  params.beta = 0.2;
  params.slop = 0.0;
  params.restitution = 0.0;
  params.iterations = 40;
  params.dt = 1.0 / 120.0;
  params.mu = 0.0;
  params.warm_start = true;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene,
      [&, rows = RowSOA{}, joint_rows = JointSOA{}](
          std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
          std::vector<DistanceJoint>& joints) mutable {
        build_soa(bodies, contacts, params, rows);
        build_distance_joint_rows(bodies, joints, params.dt);
        build_joint_soa(bodies, joints, params.dt, joint_rows);
        solve_scalar_soa(bodies, contacts, rows, joint_rows, params);
        scatter_joint_impulses(joint_rows, joints);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_soa_result("pendulum", base_scene, params, kStepsPerRun,
                                 ms_per_step));
    recorded = true;
  }
}

void BenchPendulumSoAFourth(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_pendulum(1);
  SolverParams params;
  params.beta = 0.1;
  params.slop = 0.0025;
  params.restitution = 0.0;
  params.iterations = 10;
  params.dt = 1.0 / 120.0;
  params.mu = 0.3;
  params.warm_start = true;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene,
      [&, rows = RowSOA{}, joint_rows = JointSOA{}](
          std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
          std::vector<DistanceJoint>& joints) mutable {
        build_contact_offsets_and_bias(bodies, contacts, params);
        build_soa(bodies, contacts, params, rows);
        build_distance_joint_rows(bodies, joints, params.dt);
        build_joint_soa(bodies, joints, params.dt, joint_rows);
        solve_scalar_soa_vectorized(bodies, contacts, rows, joint_rows, params);
        scatter_joint_impulses(joint_rows, joints);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_soa_vectorized_result("pendulum", base_scene, params,
                                        kStepsPerRun, ms_per_step));
    recorded = true;
  }
}

void BenchChainCached(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_chain_64();
  SolverParams params;
  params.beta = 0.2;
  params.slop = 0.0;
  params.restitution = 0.0;
  params.iterations = 40;
  params.dt = 1.0 / 120.0;
  params.mu = 0.0;
  params.warm_start = true;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene,
      [&](std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
          std::vector<DistanceJoint>& joints) {
        build_distance_joint_rows(bodies, joints, params.dt);
        solve_scalar_cached(bodies, contacts, joints, params);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_cached_result("chain_64", base_scene, params, kStepsPerRun,
                                    ms_per_step));
    recorded = true;
  }
}

void BenchChainSoA(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_chain_64();
  SolverParams params;
  params.beta = 0.2;
  params.slop = 0.0;
  params.restitution = 0.0;
  params.iterations = 40;
  params.dt = 1.0 / 120.0;
  params.mu = 0.0;
  params.warm_start = true;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene,
      [&, rows = RowSOA{}, joint_rows = JointSOA{}](
          std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
          std::vector<DistanceJoint>& joints) mutable {
        build_soa(bodies, contacts, params, rows);
        build_distance_joint_rows(bodies, joints, params.dt);
        build_joint_soa(bodies, joints, params.dt, joint_rows);
        solve_scalar_soa(bodies, contacts, rows, joint_rows, params);
        scatter_joint_impulses(joint_rows, joints);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_soa_result("chain_64", base_scene, params, kStepsPerRun,
                                 ms_per_step));
    recorded = true;
  }
}

void BenchChainSoAFourth(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_chain_64();
  SolverParams params;
  params.beta = 0.05;
  params.slop = 0.001;
  params.restitution = 0.0;
  params.iterations = 10;
  params.dt = 1.0 / 120.0;
  params.mu = 0.0;
  params.warm_start = true;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene,
      [&, rows = RowSOA{}, joint_rows = JointSOA{}](
          std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
          std::vector<DistanceJoint>& joints) mutable {
        build_contact_offsets_and_bias(bodies, contacts, params);
        build_soa(bodies, contacts, params, rows);
        build_distance_joint_rows(bodies, joints, params.dt);
        build_joint_soa(bodies, joints, params.dt, joint_rows);
        solve_scalar_soa_vectorized(bodies, contacts, rows, joint_rows, params);
        scatter_joint_impulses(joint_rows, joints);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_soa_vectorized_result("chain_64", base_scene, params,
                                        kStepsPerRun, ms_per_step));
    recorded = true;
  }
}

void BenchRopeCached(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_rope_256();
  SolverParams params;
  params.beta = 0.0;
  params.slop = 0.0;
  params.restitution = 0.0;
  params.iterations = 40;
  params.dt = 1.0 / 180.0;
  params.mu = 0.0;
  params.warm_start = true;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene,
      [&](std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
          std::vector<DistanceJoint>& joints) {
        build_distance_joint_rows(bodies, joints, params.dt);
        solve_scalar_cached(bodies, contacts, joints, params);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_cached_result("rope_256", base_scene, params, kStepsPerRun,
                                    ms_per_step));
    recorded = true;
  }
}

void BenchRopeSoA(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_rope_256();
  SolverParams params;
  params.beta = 0.0;
  params.slop = 0.0;
  params.restitution = 0.0;
  params.iterations = 40;
  params.dt = 1.0 / 180.0;
  params.mu = 0.0;
  params.warm_start = true;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene,
      [&, rows = RowSOA{}, joint_rows = JointSOA{}](
          std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
          std::vector<DistanceJoint>& joints) mutable {
        build_soa(bodies, contacts, params, rows);
        build_distance_joint_rows(bodies, joints, params.dt);
        build_joint_soa(bodies, joints, params.dt, joint_rows);
        solve_scalar_soa(bodies, contacts, rows, joint_rows, params);
        scatter_joint_impulses(joint_rows, joints);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_soa_result("rope_256", base_scene, params, kStepsPerRun,
                                 ms_per_step));
    recorded = true;
  }
}

void BenchRopeSoAFourth(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_rope_256();
  SolverParams params;
  params.beta = 0.05;
  params.slop = 0.001;
  params.restitution = 0.0;
  params.iterations = 10;
  params.dt = 1.0 / 120.0;
  params.mu = 0.0;
  params.warm_start = true;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene,
      [&, rows = RowSOA{}, joint_rows = JointSOA{}](
          std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
          std::vector<DistanceJoint>& joints) mutable {
        build_contact_offsets_and_bias(bodies, contacts, params);
        build_soa(bodies, contacts, params, rows);
        build_distance_joint_rows(bodies, joints, params.dt);
        build_joint_soa(bodies, joints, params.dt, joint_rows);
        solve_scalar_soa_vectorized(bodies, contacts, rows, joint_rows, params);
        scatter_joint_impulses(joint_rows, joints);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_soa_vectorized_result("rope_256", base_scene, params,
                                        kStepsPerRun, ms_per_step));
    recorded = true;
  }
}

#endif  // ADMC_HAVE_GBENCH

namespace {

using SoaSolveFn = void (*)(std::vector<RigidBody>&,
                            std::vector<Contact>&,
                            RowSOA&,
                            JointSOA&,
                            const SoaParams&,
                            SolverDebugInfo*);

SoaSolveFn select_solver(SoaSolverVariant variant) {
  switch (variant) {
    case SoaSolverVariant::kLegacy:
      return [](std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
                RowSOA& rows, JointSOA& joints, const SoaParams& params,
                SolverDebugInfo* debug_info) {
        solve_scalar_soa(bodies, contacts, rows, joints, params, debug_info);
      };
    case SoaSolverVariant::kVectorized:
      return [](std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
                RowSOA& rows, JointSOA& joints, const SoaParams& params,
                SolverDebugInfo* debug_info) {
        solve_scalar_soa_vectorized(bodies, contacts, rows, joints, params,
                                debug_info);
      };
  }
  return nullptr;
}

const char* solver_label(SoaSolverVariant variant) {
  switch (variant) {
    case SoaSolverVariant::kLegacy:
      return "scalar_soa";
    case SoaSolverVariant::kVectorized:
      return "vec_soa";
  }
  return "scalar_soa";
}

}  // namespace

BenchmarkResult run_soa_variant_result(const std::string& scene_name,
                                       const Scene& base_scene,
                                       const SoaParams& params,
                                       int steps,
                                       double ms_per_step_hint,
                                       SoaSolverVariant variant) {
  std::vector<RigidBody> bodies = base_scene.bodies;
  std::vector<Contact> contacts = base_scene.contacts;
  std::vector<DistanceJoint> joints = base_scene.joints;
  const std::vector<RigidBody> pre = bodies;
  SolverDebugInfo aggregate_debug;
  SoaTimingBreakdown total_timings;
  aggregate_debug.reset();
  RowSOA rows;
  JointSOA joint_rows;
  SoaSolveFn solver = select_solver(variant);
  if (!solver) {
    throw std::runtime_error("Unknown SoA solver variant");
  }
  for (int i = 0; i < steps; ++i) {
    SolverDebugInfo step_debug;
    SoaTimingBreakdown step_timings;
    const auto step_begin = Clock::now();

    const auto contact_begin = Clock::now();
    build_contact_offsets_and_bias(bodies, contacts, params);
    const auto contact_end = Clock::now();
    step_timings.contact_prep_ms += elapsed_ms(contact_begin, contact_end);

    const auto row_begin = Clock::now();
    build_soa(bodies, contacts, params, rows);
    const auto row_end = Clock::now();
    step_timings.row_build_ms += elapsed_ms(row_begin, row_end);

    const auto joint_distance_begin = Clock::now();
    build_distance_joint_rows(bodies, joints, params.dt);
    const auto joint_distance_end = Clock::now();
    step_timings.joint_distance_build_ms +=
        elapsed_ms(joint_distance_begin, joint_distance_end);

    const auto joint_pack_begin = Clock::now();
    build_joint_soa(bodies, joints, params.dt, joint_rows);
    const auto joint_pack_end = Clock::now();
    step_timings.joint_pack_ms += elapsed_ms(joint_pack_begin, joint_pack_end);

    const auto solver_begin = Clock::now();
    solver(bodies, contacts, rows, joint_rows, params, &step_debug);
    const auto solver_end = Clock::now();
    double solver_ms = step_debug.timings.solver_total_ms;
    if (solver_ms <= 0.0) {
      solver_ms = elapsed_ms(solver_begin, solver_end);
    }
    step_timings.solver_total_ms += solver_ms;
    step_timings.solver_warmstart_ms += step_debug.timings.solver_warmstart_ms;
    step_timings.solver_iterations_ms +=
        step_debug.timings.solver_iterations_ms;
    step_timings.solver_integrate_ms += step_debug.timings.solver_integrate_ms;

    const auto scatter_begin = Clock::now();
    scatter_joint_impulses(joint_rows, joints);
    const auto scatter_end = Clock::now();
    step_timings.scatter_ms += elapsed_ms(scatter_begin, scatter_end);

    const auto step_end = Clock::now();
    step_timings.total_step_ms += elapsed_ms(step_begin, step_end);

    aggregate_debug.accumulate(step_debug);
    total_timings.accumulate(step_timings);
  }
  build_distance_joint_rows(bodies, joints, params.dt);
  BenchmarkResult result;
  result.scene = scene_name;
  result.solver = solver_label(variant);
  result.iterations = params.iterations;
  result.steps = steps;
  result.dt = params.dt;
  result.bodies = base_scene.bodies.size();
  result.contacts = base_scene.contacts.size();
  result.joints = base_scene.joints.size();
  result.drift_max = directional_momentum_drift(pre, bodies).max_abs;
  result.penetration_linf = constraint_penetration_Linf(contacts);
  result.energy_drift = energy_drift(pre, bodies);
  result.cone_consistency = cone_consistency(contacts);
  result.joint_Linf = joint_error_Linf(joints);
  result.simd = params.use_simd;
  result.threads = params.use_threads ? std::max(1, params.thread_count) : 1;
  result.tile_size = params.tile_size;
  result.commit_sha = current_commit_sha();
  const double derived_ms = (steps > 0)
                                ? (total_timings.total_step_ms / static_cast<double>(steps))
                                : 0.0;
  result.ms_per_step = (ms_per_step_hint >= 0.0) ? ms_per_step_hint : derived_ms;
  if (steps > 0) {
    result.has_soa_timings = true;
    result.soa_timings = total_timings;
    result.soa_timings.scale(1.0 / steps);
    SolverDebugInfo debug_avg = aggregate_debug;
    debug_avg.timings = result.soa_timings;
    result.soa_debug_summary = solver_debug_summary(debug_avg);
  }
  return result;
}

BenchmarkResult run_soa_result(const std::string& scene_name,
                               const Scene& base_scene,
                               const SoaParams& params,
                               int steps,
                               double ms_per_step_hint) {
  return run_soa_variant_result(scene_name, base_scene, params, steps,
                                ms_per_step_hint, SoaSolverVariant::kLegacy);
}

BenchmarkResult run_soa_vectorized_result(const std::string& scene_name,
                                      const Scene& base_scene,
                                      const SoaParams& params,
                                      int steps,
                                      double ms_per_step_hint) {
  return run_soa_variant_result(scene_name, base_scene, params, steps,
                                ms_per_step_hint, SoaSolverVariant::kVectorized);
}

BenchmarkResult run_soa_result(const std::string& scene_name,
                               const Scene& base_scene,
                               const SolverParams& params,
                               int steps,
                               double ms_per_step_hint) {
  SoaParams derived;
  static_cast<SolverParams&>(derived) = params;
  return run_soa_result(scene_name, base_scene, derived, steps, ms_per_step_hint);
}

BenchmarkResult run_soa_vectorized_result(const std::string& scene_name,
                                      const Scene& base_scene,
                                      const SolverParams& params,
                                      int steps,
                                      double ms_per_step_hint) {
  SoaParams derived;
  static_cast<SolverParams&>(derived) = params;
  return run_soa_vectorized_result(scene_name, base_scene, derived, steps,
                               ms_per_step_hint);
}


void write_results_csv() {
  if (g_results.empty()) {
    return;
  }
  const std::filesystem::path path(g_results_csv_path);
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  const bool exists = std::filesystem::exists(path);
  bool needs_header = !exists;
  if (exists) {
    std::ifstream existing(path);
    needs_header = !existing.good() ||
                   existing.peek() == std::ifstream::traits_type::eof();
  }
  std::ofstream out(path, std::ios::app);
  if (!out) {
    std::cerr << "Failed to open CSV for writing: " << path << "\n";
    return;
  }
  if (needs_header) {
    out << benchcsv::kHeader << '\n';
  }
  out << std::setprecision(12);
  for (const BenchmarkResult& r : g_results) {
    out << r.scene << ',' << r.solver << ',' << r.iterations << ',' << r.steps << ','
        << r.bodies << ',' << r.contacts << ',' << r.joints << ',' << r.tile_size
        << ',' << r.ms_per_step << ',' << r.drift_max << ',' << r.penetration_linf
        << ',' << r.energy_drift << ',' << r.cone_consistency << ','
        << (r.simd ? 1 : 0) << ',' << r.threads << ',';
    if (!r.commit_sha.empty()) {
      out << r.commit_sha;
    }
    out << '\n';
  }
}

void print_results_table() {
  if (g_results.empty()) {
    return;
  }
  std::cout << "\nBenchmark Summary\n";
  std::cout << std::left << std::setw(24) << "scene" << std::setw(16) << "solver"
            << std::setw(12) << "ms/step" << std::setw(14) << "drift" << std::setw(16)
            << "energy" << std::setw(18) << "cone" << std::setw(14) << "joint" << '\n';
  for (const BenchmarkResult& r : g_results) {
    std::cout << std::left << std::setw(24) << r.scene << std::setw(16) << r.solver
              << std::setw(12) << std::fixed << std::setprecision(3) << r.ms_per_step
              << std::setw(14) << std::scientific << std::setprecision(3)
              << r.drift_max << std::setw(16) << r.energy_drift << std::setw(18)
              << r.cone_consistency << std::setw(14) << r.joint_Linf << '\n';
    if (r.has_soa_timings) {
      std::ostringstream timings;
      timings << "    SoA timings (ms/step): contact=" << std::fixed
              << std::setprecision(3) << r.soa_timings.contact_prep_ms
              << ", row=" << r.soa_timings.row_build_ms
              << ", joint_build=" << r.soa_timings.joint_distance_build_ms
              << ", joint_pack=" << r.soa_timings.joint_pack_ms
              << ", solver=" << r.soa_timings.solver_total_ms
              << " [warm=" << r.soa_timings.solver_warmstart_ms
              << ", iter=" << r.soa_timings.solver_iterations_ms
              << ", integ=" << r.soa_timings.solver_integrate_ms
              << "], scatter=" << r.soa_timings.scatter_ms
              << ", total=" << r.soa_timings.total_step_ms;
      std::cout << timings.str() << '\n';
      if (!r.soa_debug_summary.empty()) {
        std::cout << "    SoA debug: " << r.soa_debug_summary << '\n';
      }
    }
  }
  std::cout << std::defaultfloat;
}

void write_microbench_csv() {
  if (g_micro_results.empty()) {
    return;
  }
  const std::filesystem::path path(g_micro_csv_path);
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  const bool exists = std::filesystem::exists(path);
  std::ofstream out(path, std::ios::app);
  if (!exists || out.tellp() == 0) {
    out << "kernel,variant,lane,threads,N_rows,ns_per_row\n";
  }
  for (const MicrobenchResult& r : g_micro_results) {
    out << r.kernel << ',' << r.variant << ',' << r.lane << ',' << r.threads << ','
        << r.rows << ',' << r.ns_per_row << '\n';
  }
}

#if ADMC_HAVE_GBENCH
void MicrobenchUpdateNormal(benchmark::State& state) {
  const bool use_simd = state.range(0) != 0;
  const int rows = 1024;
  std::vector<double> target(rows, 1.0);
  std::vector<double> v_rel(rows, 0.2);
  std::vector<double> k(rows, 2.0);
  std::vector<double> impulses(rows, 0.0);
  const int width = use_simd ? std::max(1, soa::kLane) : 1;

  double total_elapsed = 0.0;
  int iterations = 0;
  for (auto _ : state) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < rows; i += width) {
      const int count = std::min(width, rows - i);
      if (use_simd) {
        soa_simd::update_normal_batch(target.data() + i, v_rel.data() + i,
                                      k.data() + i, impulses.data() + i, count);
      } else {
        for (int lane = 0; lane < count; ++lane) {
          const int idx = i + lane;
          const double delta = (target[idx] - v_rel[idx]) / k[idx];
          impulses[idx] += delta;
        }
      }
    }
    auto end = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(end - start).count();
    total_elapsed += elapsed;
    ++iterations;
    state.SetIterationTime(elapsed);
  }

  if (state.thread_index == 0 && iterations > 0) {
    MicrobenchResult result;
    result.kernel = "SOA_UpdateNormal";
    result.variant = use_simd ? "simd" : "scalar";
    result.lane = use_simd ? std::max(1, soa::kLane) : 1;
    result.threads = 1;
    result.rows = rows;
    result.ns_per_row =
        (total_elapsed / static_cast<double>(iterations)) * 1e9 /
        static_cast<double>(rows);
    record_micro_result(result);
  }
}

void MicrobenchUpdateTangent(benchmark::State& state) {
  const bool use_simd = state.range(0) != 0;
  const int rows = 1024;
  std::vector<double> target(rows, 0.5);
  std::vector<double> v_rel(rows, 0.1);
  std::vector<double> k(rows, 3.0);
  std::vector<double> impulses(rows, 0.0);
  const int width = use_simd ? std::max(1, soa::kLane) : 1;

  double total_elapsed = 0.0;
  int iterations = 0;
  for (auto _ : state) {
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < rows; i += width) {
      const int count = std::min(width, rows - i);
      if (use_simd) {
        soa_simd::update_tangent_batch(target.data() + i, v_rel.data() + i,
                                       k.data() + i, impulses.data() + i, count);
      } else {
        for (int lane = 0; lane < count; ++lane) {
          const int idx = i + lane;
          const double delta = (target[idx] - v_rel[idx]) / k[idx];
          impulses[idx] += delta;
        }
      }
    }
    auto end = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(end - start).count();
    total_elapsed += elapsed;
    ++iterations;
    state.SetIterationTime(elapsed);
  }

  if (state.thread_index == 0 && iterations > 0) {
    MicrobenchResult result;
    result.kernel = "SOA_UpdateTangents";
    result.variant = use_simd ? "simd" : "scalar";
    result.lane = use_simd ? std::max(1, soa::kLane) : 1;
    result.threads = 1;
    result.rows = rows;
    result.ns_per_row =
        (total_elapsed / static_cast<double>(iterations)) * 1e9 /
        static_cast<double>(rows);
    record_micro_result(result);
  }
}

void MicrobenchApplyImpulses(benchmark::State& state) {
  const bool use_simd = state.range(0) != 0;
  const int rows_count = 256;
  RowSOA rows;
  rows.N = rows_count;
  rows.a.assign(rows_count, 0);
  rows.b.assign(rows_count, 1);
  rows.nx.assign(rows_count, 0.0);
  rows.ny.assign(rows_count, 1.0);
  rows.nz.assign(rows_count, 0.0);
  rows.t1x.assign(rows_count, 1.0);
  rows.t1y.assign(rows_count, 0.0);
  rows.t1z.assign(rows_count, 0.0);
  rows.t2x.assign(rows_count, 0.0);
  rows.t2y.assign(rows_count, 0.0);
  rows.t2z.assign(rows_count, 1.0);
  rows.rax.assign(rows_count, 0.0);
  rows.ray.assign(rows_count, 0.0);
  rows.raz.assign(rows_count, 0.0);
  rows.rbx.assign(rows_count, 0.0);
  rows.rby.assign(rows_count, 0.0);
  rows.rbz.assign(rows_count, 0.0);
  rows.raxn_x.assign(rows_count, 0.0);
  rows.raxn_y.assign(rows_count, 0.0);
  rows.raxn_z.assign(rows_count, 0.0);
  rows.rbxn_x.assign(rows_count, 0.0);
  rows.rbxn_y.assign(rows_count, 0.0);
  rows.rbxn_z.assign(rows_count, 0.0);
  rows.raxt1_x.assign(rows_count, 0.0);
  rows.raxt1_y.assign(rows_count, 0.0);
  rows.raxt1_z.assign(rows_count, 0.0);
  rows.rbxt1_x.assign(rows_count, 0.0);
  rows.rbxt1_y.assign(rows_count, 0.0);
  rows.rbxt1_z.assign(rows_count, 0.0);
  rows.raxt2_x.assign(rows_count, 0.0);
  rows.raxt2_y.assign(rows_count, 0.0);
  rows.raxt2_z.assign(rows_count, 0.0);
  rows.rbxt2_x.assign(rows_count, 0.0);
  rows.rbxt2_y.assign(rows_count, 0.0);
  rows.rbxt2_z.assign(rows_count, 0.0);
  rows.TWn_a_x.assign(rows_count, 0.0);
  rows.TWn_a_y.assign(rows_count, 0.0);
  rows.TWn_a_z.assign(rows_count, 0.0);
  rows.TWn_b_x.assign(rows_count, 0.0);
  rows.TWn_b_y.assign(rows_count, 0.0);
  rows.TWn_b_z.assign(rows_count, 0.0);
  rows.TWt1_a_x.assign(rows_count, 0.0);
  rows.TWt1_a_y.assign(rows_count, 0.0);
  rows.TWt1_a_z.assign(rows_count, 0.0);
  rows.TWt1_b_x.assign(rows_count, 0.0);
  rows.TWt1_b_y.assign(rows_count, 0.0);
  rows.TWt1_b_z.assign(rows_count, 0.0);
  rows.TWt2_a_x.assign(rows_count, 0.0);
  rows.TWt2_a_y.assign(rows_count, 0.0);
  rows.TWt2_a_z.assign(rows_count, 0.0);
  rows.TWt2_b_x.assign(rows_count, 0.0);
  rows.TWt2_b_y.assign(rows_count, 0.0);
  rows.TWt2_b_z.assign(rows_count, 0.0);
  rows.indices.resize(rows_count);
  for (int i = 0; i < rows_count; ++i) {
    rows.indices[i] = i;
  }

  std::vector<double> delta_jn(rows_count, 0.05);
  std::vector<double> delta_jt1(rows_count, 0.01);
  std::vector<double> delta_jt2(rows_count, 0.015);

  std::vector<RigidBody> base_bodies(2);
  base_bodies[0].invMass = 1.0;
  base_bodies[1].invMass = 1.0;
  base_bodies[0].invInertiaLocal = math::Mat3::identity();
  base_bodies[1].invInertiaLocal = math::Mat3::identity();
  base_bodies[0].syncDerived();
  base_bodies[1].syncDerived();

  const int width = use_simd ? std::max(1, soa::kLane) : 1;
  double total_elapsed = 0.0;
  int iterations = 0;
  for (auto _ : state) {
    std::vector<RigidBody> bodies = base_bodies;
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < rows_count; i += width) {
      const int count = std::min(width, rows_count - i);
      if (use_simd) {
        soa_simd::apply_impulses_batch(bodies, rows, delta_jn.data() + i,
                                       delta_jt1.data() + i,
                                       delta_jt2.data() + i, i, count);
      } else {
        for (int lane = 0; lane < count; ++lane) {
          const int idx = i + lane;
          RigidBody& A = bodies[rows.a[idx]];
          RigidBody& B = bodies[rows.b[idx]];
          const double dj_n = delta_jn[idx];
          const double dj_t1 = delta_jt1[idx];
          const double dj_t2 = delta_jt2[idx];

          const double impulse_x = rows.nx[idx] * dj_n + rows.t1x[idx] * dj_t1 +
                                   rows.t2x[idx] * dj_t2;
          const double impulse_y = rows.ny[idx] * dj_n + rows.t1y[idx] * dj_t1 +
                                   rows.t2y[idx] * dj_t2;
          const double impulse_z = rows.nz[idx] * dj_n + rows.t1z[idx] * dj_t1 +
                                   rows.t2z[idx] * dj_t2;

          A.v.x -= impulse_x * A.invMass;
          A.v.y -= impulse_y * A.invMass;
          A.v.z -= impulse_z * A.invMass;
          B.v.x += impulse_x * B.invMass;
          B.v.y += impulse_y * B.invMass;
          B.v.z += impulse_z * B.invMass;

          const double dw_ax = dj_n * rows.TWn_a_x[idx] +
                               dj_t1 * rows.TWt1_a_x[idx] +
                               dj_t2 * rows.TWt2_a_x[idx];
          const double dw_ay = dj_n * rows.TWn_a_y[idx] +
                               dj_t1 * rows.TWt1_a_y[idx] +
                               dj_t2 * rows.TWt2_a_y[idx];
          const double dw_az = dj_n * rows.TWn_a_z[idx] +
                               dj_t1 * rows.TWt1_a_z[idx] +
                               dj_t2 * rows.TWt2_a_z[idx];
          const double dw_bx = dj_n * rows.TWn_b_x[idx] +
                               dj_t1 * rows.TWt1_b_x[idx] +
                               dj_t2 * rows.TWt2_b_x[idx];
          const double dw_by = dj_n * rows.TWn_b_y[idx] +
                               dj_t1 * rows.TWt1_b_y[idx] +
                               dj_t2 * rows.TWt2_b_y[idx];
          const double dw_bz = dj_n * rows.TWn_b_z[idx] +
                               dj_t1 * rows.TWt1_b_z[idx] +
                               dj_t2 * rows.TWt2_b_z[idx];

          A.w.x -= dw_ax;
          A.w.y -= dw_ay;
          A.w.z -= dw_az;
          B.w.x += dw_bx;
          B.w.y += dw_by;
          B.w.z += dw_bz;
        }
      }
    }
    auto end = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(end - start).count();
    total_elapsed += elapsed;
    ++iterations;
    state.SetIterationTime(elapsed);
  }

  if (state.thread_index == 0 && iterations > 0) {
    MicrobenchResult result;
    result.kernel = "SOA_ApplyImpulses";
    result.variant = use_simd ? "simd" : "scalar";
    result.lane = use_simd ? std::max(1, soa::kLane) : 1;
    result.threads = 1;
    result.rows = rows_count;
    result.ns_per_row =
        (total_elapsed / static_cast<double>(iterations)) * 1e9 /
        static_cast<double>(rows_count);
    record_micro_result(result);
  }
}

void MicrobenchSoaMtScaling(benchmark::State& state) {
  const int threads = std::max(1, static_cast<int>(state.range(0)));
  const Scene base_scene = make_spheres_box_cloud(256);
  SoaParams params;
  params.iterations = 1;
  params.dt = 1.0 / 60.0;
  params.use_simd = true;
  params.use_threads = true;
  params.thread_count = threads;

  double total_elapsed = 0.0;
  int iterations = 0;
  RowSOA rows;
  JointSOA joint_rows;
  for (auto _ : state) {
    std::vector<RigidBody> bodies = base_scene.bodies;
    std::vector<Contact> contacts = base_scene.contacts;
    std::vector<DistanceJoint> joints = base_scene.joints;
    build_contact_offsets_and_bias(bodies, contacts, params);
    build_soa(bodies, contacts, params, rows);
    build_distance_joint_rows(bodies, joints, params.dt);
    build_joint_soa(bodies, joints, params.dt, joint_rows);
    auto start = std::chrono::steady_clock::now();
    solve_scalar_soa(bodies, contacts, rows, joint_rows, params);
    auto end = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(end - start).count();
    total_elapsed += elapsed;
    ++iterations;
    state.SetIterationTime(elapsed);
  }

  if (state.thread_index == 0 && iterations > 0) {
    const int rows_processed = std::max<std::size_t>(1, base_scene.contacts.size());
    MicrobenchResult result;
    result.kernel = "SOA_MT_Scaling";
    result.variant = "threads_" + std::to_string(threads);
    result.lane = std::max(1, soa::kLane);
    result.threads = threads;
    result.rows = rows_processed;
    result.ns_per_row =
        (total_elapsed / static_cast<double>(iterations)) * 1e9 /
        static_cast<double>(rows_processed);
    record_micro_result(result);
  }
}
#endif  // ADMC_HAVE_GBENCH

}  // namespace

#if ADMC_HAVE_GBENCH

BENCHMARK(BenchSpheresCloudBaseline4096)->UseManualTime();
BENCHMARK(BenchSpheresCloudCached4096)->UseManualTime();
BENCHMARK(BenchSpheresCloudSoA4096)->UseManualTime();
BENCHMARK(BenchSpheresCloudSoA4096Fourth)->UseManualTime();
BENCHMARK(BenchSpheresCloudBaseline8192)->UseManualTime();
BENCHMARK(BenchSpheresCloudCached8192)->UseManualTime();
BENCHMARK(BenchSpheresCloudSoA8192)->UseManualTime();
BENCHMARK(BenchSpheresCloudSoA8192Fourth)->UseManualTime();
BENCHMARK(BenchBoxStackBaseline)->UseManualTime();
BENCHMARK(BenchBoxStackCached)->UseManualTime();
BENCHMARK(BenchBoxStackSoA)->UseManualTime();
BENCHMARK(BenchBoxStackSoAFourth)->UseManualTime();
BENCHMARK(BenchPendulumCached)->UseManualTime();
BENCHMARK(BenchPendulumSoA)->UseManualTime();
BENCHMARK(BenchPendulumSoAFourth)->UseManualTime();
BENCHMARK(BenchChainCached)->UseManualTime();
BENCHMARK(BenchChainSoA)->UseManualTime();
BENCHMARK(BenchChainSoAFourth)->UseManualTime();
BENCHMARK(BenchRopeCached)->UseManualTime();
BENCHMARK(BenchRopeSoA)->UseManualTime();
BENCHMARK(BenchRopeSoAFourth)->UseManualTime();
BENCHMARK(MicrobenchUpdateNormal)->Arg(0)->Arg(1)->UseManualTime();
BENCHMARK(MicrobenchUpdateTangent)->Arg(0)->Arg(1)->UseManualTime();
BENCHMARK(MicrobenchApplyImpulses)->Arg(0)->Arg(1)->UseManualTime();
BENCHMARK(MicrobenchSoaMtScaling)->Arg(1)->Arg(2)->Arg(4)->Arg(8)->UseManualTime();

#endif  // ADMC_HAVE_GBENCH

int main(int argc, char** argv) {
  std::vector<char*> passthrough;
  CliOptions opts = parse_cli_options(argc, argv, passthrough);
  if (opts.run_cli) {
    if (!run_cli_mode(opts)) {
      return 1;
    }
    return 0;
  }

#if ADMC_HAVE_GBENCH
  if (opts.run_benchmark) {
    passthrough.push_back(nullptr);
    int bench_argc = static_cast<int>(passthrough.size()) - 1;
    benchmark::Initialize(&bench_argc, passthrough.data());
    benchmark::RunSpecifiedBenchmarks();
    write_results_csv();
    write_microbench_csv();
    print_results_table();
    benchmark::Shutdown();
    return 0;
  }
#else
  if (opts.run_benchmark) {
    std::cerr << "Google Benchmark support disabled at build time; running quick suite instead.\n";
  }
#endif

  run_default_suite(opts.csv_path);
  return 0;
}
