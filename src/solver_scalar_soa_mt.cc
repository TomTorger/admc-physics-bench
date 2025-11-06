#include "solver_scalar_soa_mt.hpp"

#include "solver_scalar_soa.hpp"
#include "solver_scalar_soa_native.hpp"

namespace {

void disable_parallel(SoaParams& params) {
  params.use_threads = false;
  params.thread_count = 1;
}

}  // namespace

void solve_scalar_soa_mt(std::vector<RigidBody>& bodies,
                         std::vector<Contact>& contacts,
                         RowSOA& rows,
                         JointSOA& joints,
                         const SoaParams& params,
                         SolverDebugInfo* debug_info) {
  SoaParams local = params;
#if defined(ADMC_ENABLE_PARALLEL) && !defined(ADMC_DETERMINISTIC)
  solve_scalar_soa_native(bodies, contacts, rows, joints, local, debug_info);
#else
  disable_parallel(local);
  solve_scalar_soa_scalar(bodies, contacts, rows, joints, local, debug_info);
#endif
}

void solve_scalar_soa_mt(std::vector<RigidBody>& bodies,
                         std::vector<Contact>& contacts,
                         RowSOA& rows,
                         const SoaParams& params,
                         SolverDebugInfo* debug_info) {
  static JointSOA empty_joints;
  empty_joints.clear();
  solve_scalar_soa_mt(bodies, contacts, rows, empty_joints, params,
                      debug_info);
}
