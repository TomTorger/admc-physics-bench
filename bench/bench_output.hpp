#pragma once
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "bench_cli.hpp"
#include "bench_runner.hpp"        // for bench::BenchResult + SoaTimingBreakdown
#include "bench_csv_schema.hpp"    // for benchcsv::kHeader

namespace bench {

// Same default path logic as your current bench
inline std::string default_results_csv_path() {
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

// One-line summary identical to the monolithic print
inline void print_result_line(const BenchResult& r,
                              const std::optional<double>& ms_per_step_1t = std::nullopt) {
  std::cout << "scene=" << r.scene
            << ", solver=" << r.solver
            << ", steps=" << r.steps
            << ", ms_per_step=" << std::fixed << std::setprecision(3) << r.ms_per_step
            << std::defaultfloat
            << ", drift_max=" << r.drift_max
            << ", contacts=" << r.contacts;

  if (r.threads > 1) {
    std::cout << ", threads=" << r.threads;
    if (ms_per_step_1t && *ms_per_step_1t > 0.0) {
      const double speedup = *ms_per_step_1t / r.ms_per_step;
      std::cout << ", speedup_vs_1t=" << std::fixed << std::setprecision(2) << speedup
                << std::defaultfloat;
    }
  }
  std::cout << "\n";

  if (r.has_soa_timings) {
    std::ostringstream timings;
    timings << "    SoA timings (ms/step): "
            << "contact=" << std::fixed << std::setprecision(3) << r.soa_timings.contact_prep_ms
            << ", row=" << r.soa_timings.row_build_ms
            << ", joint_build=" << r.soa_timings.joint_distance_build_ms
            << ", joint_pack=" << r.soa_timings.joint_pack_ms
            << ", solver=" << r.soa_timings.solver_total_ms
            << " [warm=" << r.soa_timings.solver_warmstart_ms
            << ", iter=" << r.soa_timings.solver_iterations_ms
            << ", integ=" << r.soa_timings.solver_integrate_ms
            << "], scatter=" << r.soa_timings.scatter_ms;
    if (r.soa_parallel_stage_ms > 0.0 || r.soa_parallel_scatter_ms > 0.0) {
      timings << ", par_stage=" << r.soa_parallel_stage_ms
              << ", par_scatter=" << r.soa_parallel_scatter_ms;
    }
    timings << ", total=" << r.soa_timings.total_step_ms;
    std::cout << timings.str() << '\n';

    if (!r.soa_debug_summary.empty()) {
      std::cout << "    SoA debug: " << r.soa_debug_summary << '\n';
    }
  }
}

// Table summary (optional pretty footer)
inline void print_summary_table(const std::vector<BenchResult>& results) {
  if (results.empty()) return;

  std::cout << "\nBenchmark Summary\n";
  std::cout << std::left
            << std::setw(24) << "scene"
            << std::setw(16) << "solver"
            << std::setw(12) << "ms/step"
            << std::setw(14) << "drift"
            << std::setw(16) << "energy"
            << std::setw(18) << "cone"
            << std::setw(14) << "joint" << '\n';

  for (const auto& r : results) {
    std::cout << std::left
              << std::setw(24) << r.scene
              << std::setw(16) << r.solver
              << std::setw(12) << std::fixed << std::setprecision(3) << r.ms_per_step
              << std::setw(14) << std::scientific << std::setprecision(3) << r.drift_max
              << std::setw(16) << r.energy_drift
              << std::setw(18) << r.cone_consistency
              << std::setw(14) << r.joint_Linf
              << std::defaultfloat << '\n';

    if (r.has_soa_timings) {
      std::ostringstream timings;
      timings << "    SoA timings (ms/step): "
              << "contact=" << std::fixed << std::setprecision(3) << r.soa_timings.contact_prep_ms
              << ", row=" << r.soa_timings.row_build_ms
              << ", joint_build=" << r.soa_timings.joint_distance_build_ms
              << ", joint_pack=" << r.soa_timings.joint_pack_ms
              << ", solver=" << r.soa_timings.solver_total_ms
              << " [warm=" << r.soa_timings.solver_warmstart_ms
              << ", iter=" << r.soa_timings.solver_iterations_ms
              << ", integ=" << r.soa_timings.solver_integrate_ms
              << "], scatter=" << r.soa_timings.scatter_ms;
      if (r.soa_parallel_stage_ms > 0.0 || r.soa_parallel_scatter_ms > 0.0) {
        timings << ", par_stage=" << r.soa_parallel_stage_ms
                << ", par_scatter=" << r.soa_parallel_scatter_ms;
      }
      timings << ", total=" << r.soa_timings.total_step_ms;
      std::cout << timings.str() << '\n';

      if (!r.soa_debug_summary.empty()) {
        std::cout << "    SoA debug: " << r.soa_debug_summary << '\n';
      }
    }
  }
  std::cout << std::defaultfloat;
}

// CSV writer compatible with your existing schema/order
inline void write_results_csv(const std::vector<BenchResult>& results,
                              std::string csv_path) {
  if (results.empty()) return;

  if (csv_path.empty()) {
    csv_path = default_results_csv_path();
  }
  const std::filesystem::path path(csv_path);
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
  for (const auto& r : results) {
    out << r.scene << ',' << r.solver << ','
        << r.iterations << ',' << r.steps << ','
        << r.bodies << ',' << r.contacts << ',' << r.joints << ','
        << r.tile_size << ','
        << r.ms_per_step << ','
        << r.drift_max << ','
        << r.penetration_linf << ','
        << r.energy_drift << ','
        << r.cone_consistency << ','
        << (r.simd ? 1 : 0) << ','
        << r.threads << ',';
    if (!r.commit_sha.empty()) out << r.commit_sha;
    out << '\n';
  }
}

} // namespace bench
