#pragma once

#include "solver_scalar_soa.hpp"

#include <vector>

struct SoaNativeStats {
  double staging_ms = 0.0;
  double warmstart_ms = 0.0;
  double normal_ms = 0.0;
  double friction_ms = 0.0;
  double writeback_ms = 0.0;

  double solver_total_ms() const {
    return staging_ms + warmstart_ms + normal_ms + friction_ms + writeback_ms;
  }

  SoaTimingBreakdown to_breakdown() const {
    SoaTimingBreakdown out;
    out.solver_warmstart_ms = warmstart_ms;
    out.solver_iterations_ms = normal_ms + friction_ms;
    out.solver_integrate_ms = writeback_ms;
    out.solver_total_ms = solver_total_ms();
    out.total_step_ms = out.solver_total_ms;
    return out;
  }
};

void solve_scalar_soa_native(std::vector<RigidBody>& bodies,
                             std::vector<Contact>& contacts,
                             RowSOA& rows,
                             JointSOA& joints,
                             const SoaParams& params,
                             SolverDebugInfo* debug_info = nullptr);

void solve_scalar_soa_native(std::vector<RigidBody>& bodies,
                             std::vector<Contact>& contacts,
                             RowSOA& rows,
                             const SoaParams& params,
                             SolverDebugInfo* debug_info = nullptr);

void solve_scalar_soa_native(std::vector<RigidBody>& bodies,
                             std::vector<Contact>& contacts,
                             RowSOA& rows,
                             JointSOA& joints,
                             const SolverParams& params,
                             SolverDebugInfo* debug_info = nullptr);

void solve_scalar_soa_native(std::vector<RigidBody>& bodies,
                             std::vector<Contact>& contacts,
                             RowSOA& rows,
                             const SolverParams& params,
                             SolverDebugInfo* debug_info = nullptr);
