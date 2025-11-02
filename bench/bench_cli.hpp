// bench/bench_cli.hpp
#pragma once
#include <algorithm>
#include <cstdlib>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <thread>

namespace bench {

struct BenchConfig {
  // Scenes/solvers
  std::vector<std::string> scenes;   // default seeded if empty
  std::vector<std::string> solvers;  // default seeded if empty
  std::vector<int>         sizes;    // for parametric scenes like "spheres_cloud"
  std::vector<int>         tile_sizes;
  int                      tile_rows = -1;

  // Run parameters
  int    iterations = 10;
  int    steps      = 30;          // overridden for some scenes
  double dt         = 1.0 / 60.0;

  // Threading
  int                      threads = 1;            // legacy single value
  std::vector<int>         threads_list;           // NEW: sweep runs
  bool                     deterministic = false;  // forces single-thread

  // Solver behavior
  bool   spheres_only = false;
  bool   frictionless = false;
  double convergence_threshold = -1.0;

  // Output
  bool        csv_mode = true;
  std::string csv_path;

  // Optional Google Benchmark interop
  bool run_benchmark = false;

  // Raw passthrough for GBENCH
  std::vector<char*> passthrough;
};

inline std::string trim_copy(const std::string& s) {
  const auto b = s.find_first_not_of(" \t\r\n");
  if (b == std::string::npos) return {};
  const auto e = s.find_last_not_of(" \t\r\n");
  return s.substr(b, e - b + 1);
}

inline std::vector<std::string> split_csv_str(const std::string& value) {
  std::vector<std::string> out;
  std::string tok;
  for (size_t i = 0; i <= value.size(); ++i) {
    if (i == value.size() || value[i] == ',') {
      tok = trim_copy(tok);
      if (!tok.empty()) out.push_back(tok);
      tok.clear();
    } else {
      tok.push_back(value[i]);
    }
  }
  return out;
}

inline int parse_int_default(const std::string& s, int fallback) {
  try { return std::stoi(s); } catch (...) { return fallback; }
}
inline double parse_double_default(const std::string& s, double fallback) {
  try { return std::stod(s); } catch (...) { return fallback; }
}

// --- defaults identical to the “quick” suite in the monolith ----------------
inline std::vector<std::string> default_solvers() {
  return {"baseline", "cached", "soa", "soa_native", "soa_parallel", "vec_soa"};
}
inline std::vector<std::string> default_scenes_quick() {
  return {
    "two_spheres",
    "spheres_cloud_1024",
    "box_stack_4",
    "spheres_cloud_10k",
    "spheres_cloud_50k",
    "spheres_cloud_10k_fric"
  };
}

inline void seed_defaults(BenchConfig& cfg,
                          const char* preset = nullptr,
                          const char* run_large_env = nullptr) {
  if (cfg.solvers.empty())
    cfg.solvers = default_solvers();

  if (cfg.scenes.empty()) {
    cfg.scenes = default_scenes_quick();
    const bool want_full = (preset && std::string(preset) == "full");
    const bool want_large = (run_large_env && std::string(run_large_env) == "1");
    if (want_full || want_large) {
      cfg.scenes.push_back("spheres_cloud_8192");
    }
  }

  // Threads sweep defaults
#if defined(ADMC_DETERMINISTIC) && !defined(ADMC_ALLOW_PARALLEL_IN_BENCH)
  const bool force_single = true;
#else
  const bool force_single = cfg.deterministic;
#endif

  if (cfg.threads_list.empty()) {
    if (force_single) {
      cfg.threads_list = {1};
    } else {
      unsigned hc = std::thread::hardware_concurrency();
      if (hc == 0) hc = 8; // safe fallback
      // Keep legacy single override if user gave --threads
      const int legacy = std::max(1, cfg.threads);
      if (legacy == 1) {
        cfg.threads_list = {1, static_cast<int>(hc)};
      } else {
        cfg.threads_list = {1, legacy};
      }
    }
  }
}

// Normalizes aliases to stable solver keys used by the runner
inline std::string normalize_solver(std::string s) {
  if (s == "scalar_cached") return "cached";
  if (s == "scalar_soa" || s == "soa_simd" || s == "soa_mt") return "soa";
  if (s == "scalar_soa_native" || s == "soa_native" || s == "native_soa") return "soa_native";
  if (s == "scalar_soa_parallel" || s == "soa_parallel" || s == "parallel_soa")
    return "soa_parallel";
  if (s == "scalar_soa_vectorized" || s == "soa_vec" ||
      s == "soa_vectorized" || s == "vec_soa") return "vec_soa";
  if (s == "baseline_vec") return "baseline";
  return s;
}

inline BenchConfig parse_cli(int argc, char** argv, std::vector<char*>& passthrough_out) {
  BenchConfig cfg;
  cfg.passthrough.push_back(argv[0]);

  const char* preset_env     = std::getenv("BENCH_PRESET"); // optional
  const char* run_large_env  = std::getenv("RUN_LARGE");    // legacy compat

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto starts = [&](std::string_view p) { return arg.rfind(p, 0) == 0; };

    if (starts("--scene=")) {
      const auto v = arg.substr(8);
      cfg.scenes = {v};
    } else if (starts("--scenes=")) {
      cfg.scenes = split_csv_str(arg.substr(9));
    } else if (starts("--solvers=")) {
      cfg.solvers.clear();
      for (auto& s : split_csv_str(arg.substr(10))) {
        cfg.solvers.push_back(normalize_solver(s));
      }
    } else if (starts("--sizes=")) {
      cfg.sizes.clear();
      for (auto& t : split_csv_str(arg.substr(8))) {
        const int v = parse_int_default(t, -1);
        if (v > 0) cfg.sizes.push_back(v);
      }
    } else if (starts("--iters=")) {
      cfg.iterations = std::max(1, parse_int_default(arg.substr(8), cfg.iterations));
    } else if (starts("--steps=")) {
      cfg.steps = std::max(1, parse_int_default(arg.substr(8), cfg.steps));
    } else if (starts("--dt=")) {
      cfg.dt = parse_double_default(arg.substr(5), cfg.dt);
    } else if (starts("--threads=")) {
      cfg.threads = std::max(1, parse_int_default(arg.substr(10), cfg.threads));
    } else if (starts("--threads-list=")) { // NEW
      cfg.threads_list.clear();
      for (auto& t : split_csv_str(arg.substr(15))) {
        const int v = parse_int_default(t, -1);
        if (v > 0) cfg.threads_list.push_back(v);
      }
    } else if (starts("--tile-sizes=")) {
      cfg.tile_sizes.clear();
      for (auto& t : split_csv_str(arg.substr(13))) {
        const int v = parse_int_default(t, -1);
        if (v > 0) cfg.tile_sizes.push_back(v);
      }
    } else if (starts("--tile_rows=")) {
      cfg.tile_rows = std::max(1, parse_int_default(arg.substr(12), 128));
    } else if (arg == "--spheres-only") {
      cfg.spheres_only = true;
    } else if (arg == "--frictionless") {
      cfg.frictionless = true;
    } else if (arg == "--deterministic") {
      cfg.deterministic = true;
      cfg.threads = 1;
      cfg.threads_list.clear(); // will be seeded to {1}
    } else if (starts("--convergence-threshold=")) {
      cfg.convergence_threshold = parse_double_default(arg.substr(24), cfg.convergence_threshold);
    } else if (starts("--csv=")) {
      cfg.csv_mode = true; cfg.csv_path = arg.substr(6);
    } else if (arg == "--no-csv") {
      cfg.csv_mode = false;
    } else if (starts("--preset=")) {
      preset_env = arg.substr(9).c_str();
    } else if (arg == "--benchmark") {
      cfg.run_benchmark = true;
    } else {
      // passthrough to Google Benchmark
      cfg.passthrough.push_back(argv[i]);
    }
  }

  seed_defaults(cfg, preset_env, run_large_env);

  // Normalize solvers in case user gave aliases via --solvers
  for (auto& s : cfg.solvers) s = normalize_solver(s);

  // Expose passthrough
  passthrough_out = cfg.passthrough;
  return cfg;
}

} // namespace bench
