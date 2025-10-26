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
#include "solver_scalar_soa_mt.hpp"
#include "solver_scalar_soa_simd.hpp"
#include "soa_pack.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <optional>
#include <random>
#include <string>
#include <thread>
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
  double ms_per_step = 0.0;
  double drift_max = 0.0;
  double penetration_linf = 0.0;
  double energy_drift = 0.0;
  double cone_consistency = 0.0;
  double joint_Linf = 0.0;
  bool simd = false;
  int threads = 1;
};

std::vector<BenchmarkResult> g_results;
std::string g_results_csv_path = "results/results.csv";

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

struct CliOptions {
  bool run_cli = false;
  bool run_benchmark = false;
  std::string solver = "auto";
  std::string scene = "two_spheres";
  int iterations = 10;
  int steps = 30;
  double dt = 1.0 / 60.0;
  int threads = 1;
  std::string csv_path = "results/results.csv";
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

bool make_scene_by_name(const std::string& name, Scene* scene) {
  if (name == "two_spheres") {
    *scene = make_two_spheres_head_on();
  } else if (name == "spheres_cloud_1024") {
    *scene = make_spheres_box_cloud(1024);
  } else if (name == "spheres_cloud_4096") {
    *scene = make_spheres_box_cloud(4096);
  } else if (name == "spheres_cloud_8192") {
    *scene = make_spheres_box_cloud(8192);
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
}

std::optional<BenchmarkResult> run_solver_case(const std::string& solver_name,
                                               const std::string& scene_name,
                                               const Scene& base_scene,
                                               int iterations,
                                               int steps,
                                               double dt,
                                               int threads) {
  const int safe_iterations = std::max(1, iterations);
  const int safe_steps = std::max(1, steps);
  const int safe_threads = std::max(1, threads);

  std::string normalized = solver_name;
  if (normalized == "scalar_cached") {
    normalized = "cached";
  } else if (normalized == "scalar_soa" || normalized == "soa_simd" ||
             normalized == "soa_mt") {
    normalized = "soa";
  } else if (normalized == "baseline_vec") {
    normalized = "baseline";
  }

  std::vector<RigidBody> bodies = base_scene.bodies;
  std::vector<Contact> contacts = base_scene.contacts;
  std::vector<DistanceJoint> joints = base_scene.joints;
  const std::vector<RigidBody> pre = bodies;

  const auto start = std::chrono::steady_clock::now();

  if (normalized == "baseline") {
    BaselineParams params;
    params.iterations = safe_iterations;
    params.dt = dt;
    for (int step = 0; step < safe_steps; ++step) {
      build_contact_offsets_and_bias(bodies, contacts, params);
      solve_baseline(bodies, contacts, params);
    }
  } else if (normalized == "cached") {
    SolverParams params;
    params.iterations = safe_iterations;
    params.dt = dt;
    for (int step = 0; step < safe_steps; ++step) {
      build_contact_offsets_and_bias(bodies, contacts, params);
      build_distance_joint_rows(bodies, joints, params.dt);
      solve_scalar_cached(bodies, contacts, joints, params);
    }
    build_distance_joint_rows(bodies, joints, params.dt);
  } else if (normalized == "soa") {
    SolverParams params;
    params.iterations = safe_iterations;
    params.dt = dt;
    for (int step = 0; step < safe_steps; ++step) {
      build_contact_offsets_and_bias(bodies, contacts, params);
      RowSOA rows = build_soa(bodies, contacts, params);
      solve_scalar_soa(bodies, contacts, rows, params);
    }
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
  result.simd = (normalized == "soa");
  result.threads = (normalized == "soa") ? safe_threads : 1;
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
  const QuickCase cases[] = {
      {"two_spheres", "baseline", 10, 10},
      {"two_spheres", "cached", 10, 10},
      {"two_spheres", "soa", 10, 10},
      {"spheres_cloud_1024", "baseline", 10, 30},
      {"spheres_cloud_1024", "cached", 10, 30},
      {"spheres_cloud_1024", "soa", 10, 30},
      {"box_stack_4", "baseline", 10, 30},
      {"box_stack_4", "cached", 10, 30},
      {"box_stack_4", "soa", 10, 30},
  };

  std::vector<BenchmarkResult> results;
  results.reserve(std::size(cases));

  for (const QuickCase& qc : cases) {
    Scene scene;
    if (!make_scene_by_name(qc.scene, &scene)) {
      std::cerr << "Skipping unknown scene: " << qc.scene << "\n";
      continue;
    }
    auto maybe = run_solver_case(qc.solver, qc.scene, scene, qc.iterations,
                                 qc.steps, kDefaultDt, 1);
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
  Scene base_scene;
  if (!make_scene_by_name(opts.scene, &base_scene)) {
    std::cerr << "Unknown scene: " << opts.scene << "\n";
    return false;
  }

  const int steps = std::max(1, opts.steps);
  const int iterations = std::max(1, opts.iterations);
  const double dt = opts.dt;
  const int threads = std::max(1, opts.threads);

  std::vector<std::string> solvers;
  if (opts.solver == "auto") {
    solvers = {"baseline", "cached", "soa"};
  } else {
    solvers.push_back(opts.solver);
  }

  std::vector<BenchmarkResult> results;
  results.reserve(solvers.size());

  for (const std::string& solver_name : solvers) {
    auto maybe =
        run_solver_case(solver_name, opts.scene, base_scene, iterations, steps,
                        dt, threads);
    if (!maybe.has_value()) {
      std::cerr << "Skipping solver: " << solver_name << "\n";
      continue;
    }
    results.push_back(*maybe);
    print_run_summary(results.back());
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
  return result;
}

BenchmarkResult run_soa_result(const std::string& scene_name,
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
    RowSOA rows = build_soa(bodies, contacts, params);
    build_distance_joint_rows(bodies, joints, params.dt);
    JointSOA joint_rows = build_joint_soa(bodies, joints, params.dt);
    solve_scalar_soa(bodies, contacts, rows, joint_rows, params);
    scatter_joint_impulses(joint_rows, joints);
  }
  build_distance_joint_rows(bodies, joints, params.dt);
  BenchmarkResult result;
  result.scene = scene_name;
  result.solver = "scalar_soa";
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
      [&](std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
          std::vector<DistanceJoint>& joints) {
        build_contact_offsets_and_bias(bodies, contacts, params);
        RowSOA rows = build_soa(bodies, contacts, params);
        build_distance_joint_rows(bodies, joints, params.dt);
        JointSOA joint_rows = build_joint_soa(bodies, joints, params.dt);
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
      [&](std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
          std::vector<DistanceJoint>& joints) {
        build_contact_offsets_and_bias(bodies, contacts, params);
        RowSOA rows = build_soa(bodies, contacts, params);
        build_distance_joint_rows(bodies, joints, params.dt);
        JointSOA joint_rows = build_joint_soa(bodies, joints, params.dt);
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
      [&](std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
          std::vector<DistanceJoint>& joints) {
        build_contact_offsets_and_bias(bodies, contacts, params);
        RowSOA rows = build_soa(bodies, contacts, params);
        build_distance_joint_rows(bodies, joints, params.dt);
        JointSOA joint_rows = build_joint_soa(bodies, joints, params.dt);
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
      [&](std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
          std::vector<DistanceJoint>& joints) {
        RowSOA rows = build_soa(bodies, contacts, params);
        build_distance_joint_rows(bodies, joints, params.dt);
        JointSOA joint_rows = build_joint_soa(bodies, joints, params.dt);
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
      [&](std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
          std::vector<DistanceJoint>& joints) {
        RowSOA rows = build_soa(bodies, contacts, params);
        build_distance_joint_rows(bodies, joints, params.dt);
        JointSOA joint_rows = build_joint_soa(bodies, joints, params.dt);
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
      [&](std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
          std::vector<DistanceJoint>& joints) {
        RowSOA rows = build_soa(bodies, contacts, params);
        build_distance_joint_rows(bodies, joints, params.dt);
        JointSOA joint_rows = build_joint_soa(bodies, joints, params.dt);
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

#endif  // ADMC_HAVE_GBENCH

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
  for (const BenchmarkResult& r : g_results) {
    out << r.scene << ',' << r.solver << ',' << r.iterations << ',' << r.steps << ','
        << r.bodies << ',' << r.contacts << ',' << r.joints << ',' << r.ms_per_step
        << ',' << r.drift_max << ',' << r.penetration_linf << ',' << r.energy_drift
        << ',' << r.cone_consistency << ',' << (r.simd ? 1 : 0) << ',' << r.threads
        << '\n';
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
  rows.a.assign(rows_count, 0);
  rows.b.assign(rows_count, 1);
  rows.n.assign(rows_count, math::Vec3(0.0, 1.0, 0.0));
  rows.t1.assign(rows_count, math::Vec3(1.0, 0.0, 0.0));
  rows.t2.assign(rows_count, math::Vec3(0.0, 0.0, 1.0));
  rows.ra.assign(rows_count, math::Vec3(0.0, 0.0, 0.0));
  rows.rb.assign(rows_count, math::Vec3(0.0, 0.0, 0.0));
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
          const math::Vec3 Pn = rows.n[idx] * delta_jn[idx];
          const math::Vec3 Pt1 = rows.t1[idx] * delta_jt1[idx];
          const math::Vec3 Pt2 = rows.t2[idx] * delta_jt2[idx];
          const math::Vec3 impulse = Pn + Pt1 + Pt2;
          bodies[rows.a[idx]].applyImpulse(-impulse, rows.ra[idx]);
          bodies[rows.b[idx]].applyImpulse(impulse, rows.rb[idx]);
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
  for (auto _ : state) {
    std::vector<RigidBody> bodies = base_scene.bodies;
    std::vector<Contact> contacts = base_scene.contacts;
    std::vector<DistanceJoint> joints = base_scene.joints;
    build_contact_offsets_and_bias(bodies, contacts, params);
    RowSOA rows = build_soa(bodies, contacts, params);
    build_distance_joint_rows(bodies, joints, params.dt);
    JointSOA joint_rows = build_joint_soa(bodies, joints, params.dt);
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
BENCHMARK(BenchSpheresCloudBaseline8192)->UseManualTime();
BENCHMARK(BenchSpheresCloudCached8192)->UseManualTime();
BENCHMARK(BenchSpheresCloudSoA8192)->UseManualTime();
BENCHMARK(BenchBoxStackBaseline)->UseManualTime();
BENCHMARK(BenchBoxStackCached)->UseManualTime();
BENCHMARK(BenchBoxStackSoA)->UseManualTime();
BENCHMARK(BenchPendulumCached)->UseManualTime();
BENCHMARK(BenchPendulumSoA)->UseManualTime();
BENCHMARK(BenchChainCached)->UseManualTime();
BENCHMARK(BenchChainSoA)->UseManualTime();
BENCHMARK(BenchRopeCached)->UseManualTime();
BENCHMARK(BenchRopeSoA)->UseManualTime();
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
