#pragma once
#include <string_view>

namespace benchcsv {
inline constexpr std::string_view kHeader =
    "scene,solver,iterations,steps,N_bodies,N_contacts,N_joints,"
    "ms_per_step,drift_max,Linf_penetration,energy_drift,cone_consistency,"
    "simd,threads,soa_contact_ms,soa_row_ms,soa_joint_distance_ms,soa_joint_pack_ms,"
    "soa_solver_ms,soa_solver_warm_ms,soa_solver_iter_ms,soa_solver_integrate_ms,"
    "soa_scatter_ms,soa_step_ms";
} // namespace benchcsv
