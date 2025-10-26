#include "solver_scalar_soa_mt.hpp"

#include "solver_scalar_soa.hpp"

void solve_scalar_soa_mt(std::vector<RigidBody>& bodies,
                         std::vector<Contact>& contacts,
                         RowSOA& rows,
                         JointSOA& joints,
                         const SoaParams& params,
                         SolverDebugInfo* debug_info) {
  solve_scalar_soa_scalar(bodies, contacts, rows, joints, params, debug_info);
}

void solve_scalar_soa_mt(std::vector<RigidBody>& bodies,
                         std::vector<Contact>& contacts,
                         RowSOA& rows,
                         const SoaParams& params,
                         SolverDebugInfo* debug_info) {
  solve_scalar_soa_scalar(bodies, contacts, rows, params, debug_info);
}
