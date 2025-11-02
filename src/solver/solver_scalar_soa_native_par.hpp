#pragma once

#include "solver_scalar_soa_native.hpp"

namespace admc {

bool solve_scalar_soa_native_parallel(std::vector<RigidBody>& bodies,
                                      std::vector<Contact>& contacts,
                                      RowSOA& rows,
                                      JointSOA& joints,
                                      const SoaParams& params,
                                      SolverDebugInfo* debug_info = nullptr);

bool solve_scalar_soa_native_parallel(std::vector<RigidBody>& bodies,
                                      std::vector<Contact>& contacts,
                                      RowSOA& rows,
                                      const SoaParams& params,
                                      SolverDebugInfo* debug_info = nullptr);

}  // namespace admc

