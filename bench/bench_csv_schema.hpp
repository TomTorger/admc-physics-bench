#pragma once
#include <string>

namespace benchcsv {

// 🔖 Centralized CSV schema used by all output code.
// Mirrors the columns printed in your existing benchmark CSV files.
// If you ever evolve the format (e.g., add “speedup_vs_baseline”), update it here only.

inline constexpr const char* kHeader =
  "scene,solver,iterations,steps,bodies,contacts,joints,tile_size,"
  "ms_per_step,drift_max,penetration_linf,energy_drift,cone_consistency,"
  "simd,threads,commit_sha";

// Optionally, keep a version string for schema tracking
inline constexpr const char* kVersion = "1.0.0";

// Simple check utility (if you ever parse existing CSVs for regression testing)
inline bool validate_header(const std::string& header) {
  return header.find("scene,solver") != std::string::npos &&
         header.find("ms_per_step") != std::string::npos;
}

} // namespace benchcsv
