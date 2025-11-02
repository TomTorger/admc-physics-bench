#pragma once

#include "solver_scalar_soa.hpp"
#include "solver_scalar_soa_native.hpp"

namespace admc {

//! Parallel wrapper around the native scalar SoA solver.
//! Returns true when this function completed the solve (either in parallel or
//! via its internal fallback). A false return indicates the caller must invoke
//! the sequential solver.
bool solve_scalar_soa_parallel(std::vector<RigidBody>& bodies,
                               std::vector<Contact>& contacts,
                               RowSOA& rows,
                               JointSOA& joints,
                               const SoaParams& params,
                               SolverDebugInfo* debug_info = nullptr);

bool solve_scalar_soa_parallel(std::vector<RigidBody>& bodies,
                               std::vector<Contact>& contacts,
                               RowSOA& rows,
                               const SoaParams& params,
                               SolverDebugInfo* debug_info = nullptr);

// Backwards compatibility shim for existing callers.
inline bool solve_scalar_soa_native_parallel(std::vector<RigidBody>& bodies,
                                             std::vector<Contact>& contacts,
                                             RowSOA& rows,
                                             JointSOA& joints,
                                             const SoaParams& params,
                                             SolverDebugInfo* debug_info = nullptr) {
  return solve_scalar_soa_parallel(bodies, contacts, rows, joints, params, debug_info);
}

inline bool solve_scalar_soa_native_parallel(std::vector<RigidBody>& bodies,
                                             std::vector<Contact>& contacts,
                                             RowSOA& rows,
                                             const SoaParams& params,
                                             SolverDebugInfo* debug_info = nullptr) {
  return solve_scalar_soa_parallel(bodies, contacts, rows, params, debug_info);
}

}  // namespace admc
