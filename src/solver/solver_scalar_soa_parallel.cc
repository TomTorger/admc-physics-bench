#include "solver/solver_scalar_soa_parallel.hpp"

namespace admc {

bool solve_scalar_soa_parallel(std::vector<RigidBody>& bodies,
                               std::vector<Contact>& contacts,
                               RowSOA& rows,
                               JointSOA& joints,
                               const SoaParams& params,
                               SolverDebugInfo* debug_info) {
  solve_scalar_soa_native(bodies, contacts, rows, joints, params, debug_info);
  return true;
}

bool solve_scalar_soa_parallel(std::vector<RigidBody>& bodies,
                               std::vector<Contact>& contacts,
                               RowSOA& rows,
                               const SoaParams& params,
                               SolverDebugInfo* debug_info) {
  static JointSOA empty_joints;
  empty_joints.clear();
  return solve_scalar_soa_parallel(bodies, contacts, rows, empty_joints,
                                   params, debug_info);
}

}  // namespace admc
