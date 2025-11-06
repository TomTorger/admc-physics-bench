#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "bench_cli.hpp"
#include "bench_runner.hpp"        // for bench::BenchResult + SoaTimingBreakdown
#include "bench_csv_schema.hpp"    // for benchcsv::kHeader

namespace bench {

namespace detail {

inline std::string canonical_solver_key(std::string_view solver) {
  if (solver == "scalar_soa") return "soa";
  if (solver == "scalar_soa_native") return "soa_native";
  if (solver == "scalar_soa_parallel") return "soa_parallel";
  if (solver == "scalar_cached") return "cached";
  if (solver == "scalar_baseline") return "baseline";
  return std::string(solver);
}

inline std::string solver_display_name(std::string_view key) {
  if (key == "baseline") return "Baseline";
  if (key == "cached") return "Cached";
  if (key == "soa") return "Scalar-SoA";
  if (key == "soa_native") return "SoA-Native";
  if (key == "soa_parallel") return "SoA-Parallel";
  if (key == "vec_soa") return "Vec-SoA";
  return std::string(key);
}

inline std::string format_fixed3(double value) {
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(3) << value;
  return oss.str();
}

inline std::string truncate_scene(const std::string& name, int width) {
  if (width <= 0) return "";
  if (static_cast<int>(name.size()) <= width) return name;
  if (width <= 3) return name.substr(0, static_cast<std::size_t>(width));
  return name.substr(0, static_cast<std::size_t>(width - 3)) + "...";
}

inline int detect_terminal_columns(const BenchConfig& cfg) {
  if (cfg.columns > 0) return cfg.columns;
  if (const char* env = std::getenv("COLUMNS")) {
    const int v = std::atoi(env);
    if (v > 0) return v;
  }
  return 100;
}

inline void print_separator_row(const std::vector<int>& widths) {
  bool first = true;
  for (int w : widths) {
    if (!first) std::cout << ' ';
    first = false;
    std::cout << std::string(std::max(0, w), '-');
  }
  std::cout << '\n';
}

} // namespace detail

using detail::canonical_solver_key;
using detail::detect_terminal_columns;
using detail::format_fixed3;
using detail::print_separator_row;
using detail::solver_display_name;
using detail::truncate_scene;

inline void print_machine_line(const BenchResult& r,
                               BenchConfig::TimingsMode mode) {
  if (mode == BenchConfig::TimingsMode::Off) return;

  const auto timings = r.has_soa_timings ? r.soa_timings : SoaTimingBreakdown{};

  switch (mode) {
    case BenchConfig::TimingsMode::Min: {
      static bool header_printed = false;
      if (!header_printed) {
        std::cout << "scene,solver,threads,steps,ms_per_step,"
                     "contact_ms,row_ms,solver_ms,warm_ms,iter_ms,integ_ms\n";
        header_printed = true;
      }
      std::cout << r.scene << ',' << r.solver << ',' << r.threads << ','
                << r.steps << ',' << format_fixed3(r.ms_per_step) << ','
                << format_fixed3(timings.contact_prep_ms) << ','
                << format_fixed3(timings.row_build_ms) << ','
                << format_fixed3(timings.solver_total_ms) << ','
                << format_fixed3(timings.solver_warmstart_ms) << ','
                << format_fixed3(timings.solver_iterations_ms) << ','
                << format_fixed3(timings.solver_integrate_ms) << '\n';
      break;
    }
    case BenchConfig::TimingsMode::Wide: {
      static bool header_printed = false;
      if (!header_printed) {
        std::cout << "scene,solver,threads,steps,ms,contact,row,"
                     "joint_build,joint_pack,solver,warm,iter,integ,scatter\n";
        header_printed = true;
      }
      std::cout << r.scene << ',' << r.solver << ',' << r.threads << ','
                << r.steps << ',' << format_fixed3(r.ms_per_step) << ','
                << format_fixed3(timings.contact_prep_ms) << ','
                << format_fixed3(timings.row_build_ms) << ','
                << format_fixed3(timings.joint_distance_build_ms) << ','
                << format_fixed3(timings.joint_pack_ms) << ','
                << format_fixed3(timings.solver_total_ms) << ','
                << format_fixed3(timings.solver_warmstart_ms) << ','
                << format_fixed3(timings.solver_iterations_ms) << ','
                << format_fixed3(timings.solver_integrate_ms) << ','
                << format_fixed3(timings.scatter_ms) << '\n';
      break;
    }
    case BenchConfig::TimingsMode::Json: {
      std::ostringstream oss;
      oss << "{\"scene\":\"" << r.scene << "\","
          << "\"solver\":\"" << r.solver << "\","
          << "\"threads\":" << r.threads << ","
          << "\"steps\":" << r.steps << ","
          << "\"ms\":" << format_fixed3(r.ms_per_step) << ","
          << "\"stages\":{"
          << "\"contact\":" << format_fixed3(timings.contact_prep_ms) << ','
          << "\"row\":" << format_fixed3(timings.row_build_ms) << ','
          << "\"joint_build\":" << format_fixed3(timings.joint_distance_build_ms) << ','
          << "\"joint_pack\":" << format_fixed3(timings.joint_pack_ms) << ','
          << "\"scatter\":" << format_fixed3(timings.scatter_ms)
          << "},"
          << "\"solver\":{"
          << "\"total\":" << format_fixed3(timings.solver_total_ms) << ','
          << "\"warm\":" << format_fixed3(timings.solver_warmstart_ms) << ','
          << "\"iter\":" << format_fixed3(timings.solver_iterations_ms) << ','
          << "\"integ\":" << format_fixed3(timings.solver_integrate_ms)
          << "}"
          << "}\n";
      std::cout << oss.str();
      break;
    }
    case BenchConfig::TimingsMode::Off:
      break;
  }
}

inline void print_compact_report(const std::vector<BenchResult>& results,
                                 const BenchConfig& cfg) {
  if (results.empty()) {
    std::cout << "\nNo benchmark results.\n";
    return;
  }

  const int columns = detect_terminal_columns(cfg);
  const std::vector<std::string> headline_order = {
      "baseline", "cached", "soa", "soa_native", "vec_soa"};

  struct SolverBucket {
    double ms = std::numeric_limits<double>::infinity();
    const BenchResult* result = nullptr;
  };

  struct SceneAggregate {
    std::string scene;
    std::size_t contacts = 0;
    std::unordered_map<std::string, SolverBucket> best;
  };

  std::vector<SceneAggregate> scene_aggs;
  std::unordered_map<std::string, std::size_t> scene_index;
  std::unordered_map<std::string, std::vector<const BenchResult*>> solver_rows;

  for (const auto& r : results) {
    const std::string solver_key = canonical_solver_key(r.solver);

    auto& per_solver = solver_rows[solver_key];
    bool merged = false;
    for (auto& existing : per_solver) {
      if (existing->scene == r.scene && existing->threads == r.threads) {
        if (r.ms_per_step < existing->ms_per_step) {
          existing = &r;
        }
        merged = true;
        break;
      }
    }
    if (!merged) {
      per_solver.push_back(&r);
    }

    auto [idx_it, inserted] =
        scene_index.emplace(r.scene, static_cast<std::size_t>(scene_aggs.size()));
    if (inserted) {
      SceneAggregate agg;
      agg.scene = r.scene;
      agg.contacts = r.contacts;
      scene_aggs.push_back(std::move(agg));
    }
    SceneAggregate& agg = scene_aggs[idx_it->second];
    if (r.contacts > 0) {
      agg.contacts = r.contacts;
    }
    auto& bucket = agg.best[solver_key];
    if (r.ms_per_step < bucket.ms) {
      bucket.ms = r.ms_per_step;
      bucket.result = &r;
    }
  }

  // Headline table ----------------------------------------------------------
  {
    int scene_w = 21;
    const int contacts_w = 8;
    int solver_w = 10;
    const int best_w = 7;
    int winner_w = 14;
    const int solver_count = static_cast<int>(headline_order.size());
    const int min_scene = 12;
    const int min_solver = 8;
    const int min_winner = 10;

    auto total_width = [&](int sw, int solw, int winw) {
      const int column_count = solver_count + 4;
      const int spaces = column_count - 1;
      return sw + contacts_w + solver_count * solw + best_w + winw + spaces;
    };

    while (true) {
      const int used = total_width(scene_w, solver_w, winner_w);
      if (used <= columns || (scene_w == min_scene && solver_w == min_solver &&
                              winner_w == min_winner)) {
        break;
      }
      if (scene_w > min_scene) {
        --scene_w;
        continue;
      }
      if (solver_w > min_solver) {
        --solver_w;
        continue;
      }
      if (winner_w > min_winner) {
        --winner_w;
        continue;
      }
      break;
    }

    std::cout << "\nAt-a-glance (best ms/step per solver & scene)\n";

    std::cout << std::left << std::setw(scene_w) << "Scene"
              << ' ' << std::right << std::setw(contacts_w) << "Contacts";
    for (const auto& key : headline_order) {
      std::cout << ' ' << std::right << std::setw(solver_w)
                << solver_display_name(key);
    }
    std::cout << ' ' << std::right << std::setw(best_w) << "Best"
              << ' ' << std::left << std::setw(winner_w) << "Winner" << '\n';

    std::vector<int> widths;
    widths.reserve(headline_order.size() + 4);
    widths.push_back(scene_w);
    widths.push_back(contacts_w);
    for (int i = 0; i < solver_count; ++i) widths.push_back(solver_w);
    widths.push_back(best_w);
    widths.push_back(winner_w);
    print_separator_row(widths);

    for (const auto& agg : scene_aggs) {
      const std::string scene_cell = truncate_scene(agg.scene, scene_w);
      const std::string contacts_cell =
          agg.contacts > 0 ? std::to_string(agg.contacts) : std::string("--");

      std::cout << std::left << std::setw(scene_w) << scene_cell
                << ' ' << std::right << std::setw(contacts_w) << contacts_cell;

      for (const auto& key : headline_order) {
        std::string cell = "--";
        auto it = agg.best.find(key);
        if (it != agg.best.end() && it->second.result) {
          cell = format_fixed3(it->second.ms);
        }
        std::cout << ' ' << std::right << std::setw(solver_w) << cell;
      }

      double best_ms = std::numeric_limits<double>::infinity();
      std::string winner = "--";
      for (const auto& [key, bucket] : agg.best) {
        if (bucket.result && bucket.ms < best_ms) {
          best_ms = bucket.ms;
          winner = solver_display_name(key);
        }
      }

      const std::string best_cell =
          std::isfinite(best_ms) ? format_fixed3(best_ms) : std::string("--");

      std::cout << ' ' << std::right << std::setw(best_w) << best_cell
                << ' ' << std::left << std::setw(winner_w) << winner << '\n';
    }
  }

  // Solver-by-solver breakdown ----------------------------------------------
  std::vector<std::string> solver_order;
  solver_order.reserve(solver_rows.size());
  for (const auto& key : headline_order) {
    if (auto it = solver_rows.find(key); it != solver_rows.end() && !it->second.empty()) {
      solver_order.push_back(key);
    }
  }
  std::vector<std::string> extras;
  extras.reserve(solver_rows.size());
  for (const auto& [key, rows] : solver_rows) {
    if (std::find(solver_order.begin(), solver_order.end(), key) == solver_order.end() &&
        !rows.empty()) {
      extras.push_back(key);
    }
  }
  std::sort(extras.begin(), extras.end());
  solver_order.insert(solver_order.end(), extras.begin(), extras.end());

  auto scene_rank = [&](const std::string& scene) -> std::size_t {
    if (auto it = scene_index.find(scene); it != scene_index.end()) {
      return it->second;
    }
    return std::numeric_limits<std::size_t>::max();
  };

  for (const auto& solver_key : solver_order) {
    auto rows = solver_rows[solver_key];
    if (rows.empty()) continue;

    std::sort(rows.begin(), rows.end(),
              [&](const BenchResult* a, const BenchResult* b) {
                const std::size_t ra = scene_rank(a->scene);
                const std::size_t rb = scene_rank(b->scene);
                if (ra != rb) return ra < rb;
                if (a->threads != b->threads) return a->threads < b->threads;
                if (a->ms_per_step != b->ms_per_step) return a->ms_per_step < b->ms_per_step;
                return a < b;
              });

    const bool has_stage =
        std::any_of(rows.begin(), rows.end(),
                    [](const BenchResult* r) { return r->has_soa_timings; });

    const int stage_cols = has_stage ? 10 : 1;
    const int column_count = 2 + stage_cols;
    int scene_w = std::min(21, std::max(12, columns / 4));
    int threads_w = 6;
    int stage_w = 8;
    const int spaces = column_count - 1;

    auto recompute_stage_width = [&]() {
      int remaining = columns - scene_w - threads_w - spaces;
      if (remaining < stage_cols * 7) {
        int deficit = stage_cols * 7 - remaining;
        int reduce_scene = std::min(deficit, scene_w - 12);
        scene_w -= reduce_scene;
        remaining += reduce_scene;
        deficit -= reduce_scene;
        if (deficit > 0) {
          int reduce_threads = std::min(deficit, threads_w - 4);
          threads_w -= reduce_threads;
          remaining += reduce_threads;
          deficit -= reduce_threads;
        }
      }
      remaining = std::max(remaining, stage_cols * 7);
      stage_w = std::min(12, remaining / stage_cols);
      stage_w = std::max(stage_w, 7);
    };

    recompute_stage_width();
    if (!has_stage) {
      int remaining = columns - scene_w - threads_w - spaces;
      if (remaining < 7) {
        int deficit = 7 - remaining;
        int reduce_scene = std::min(deficit, scene_w - 12);
        scene_w -= reduce_scene;
        remaining += reduce_scene;
        deficit -= reduce_scene;
        if (deficit > 0) {
          int reduce_threads = std::min(deficit, threads_w - 4);
          threads_w -= reduce_threads;
          remaining += reduce_threads;
        }
      }
      stage_w = std::max(7, columns - scene_w - threads_w - spaces);
    }

    std::cout << "\n" << solver_display_name(solver_key) << " (ms/step)\n";

    std::cout << std::left << std::setw(scene_w) << "Scene"
              << ' ' << std::right << std::setw(threads_w) << "Threads";

    std::vector<std::string> stage_headers;
    if (has_stage) {
      stage_headers = {"Contact", "Row", "J-Build", "J-Pack", "Solver",
                       "Warm",    "Iter", "Integ",   "Scatter", "Total"};
    } else {
      stage_headers = {"Total"};
    }

    for (const auto& header : stage_headers) {
      std::cout << ' ' << std::right << std::setw(stage_w) << header;
    }
    std::cout << '\n';

    std::vector<int> widths;
    widths.reserve(stage_headers.size() + 2);
    widths.push_back(scene_w);
    widths.push_back(threads_w);
    for (std::size_t i = 0; i < stage_headers.size(); ++i) widths.push_back(stage_w);
    print_separator_row(widths);

    for (const BenchResult* row : rows) {
      const std::string scene_cell = truncate_scene(row->scene, scene_w);
      std::cout << std::left << std::setw(scene_w) << scene_cell
                << ' ' << std::right << std::setw(threads_w) << row->threads;

      if (has_stage) {
        const auto t = row->has_soa_timings ? row->soa_timings : SoaTimingBreakdown{};
        const std::array<double, 10> values = {
            t.contact_prep_ms,       t.row_build_ms,          t.joint_distance_build_ms,
            t.joint_pack_ms,         t.solver_total_ms,       t.solver_warmstart_ms,
            t.solver_iterations_ms,  t.solver_integrate_ms,   t.scatter_ms,
            row->ms_per_step};
        for (double v : values) {
          std::cout << ' ' << std::right << std::setw(stage_w) << format_fixed3(v);
        }
      } else {
        std::cout << ' ' << std::right << std::setw(stage_w)
                  << format_fixed3(row->ms_per_step);
      }

      std::cout << '\n';

      print_machine_line(*row, cfg.timings_mode);
    }
  }

  std::cout << std::right;
  std::cout << '\n';
}

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
