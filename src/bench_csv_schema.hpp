#pragma once
#include <string_view>

namespace benchcsv {
inline constexpr std::string_view kHeader =
    "scene,solver,iterations,steps,N_bodies,N_contacts,N_joints,tile_size,"
    "ms_per_step,drift_max,Linf_penetration,energy_drift,cone_consistency,"
    "simd,threads,commit_sha";
} // namespace benchcsv
