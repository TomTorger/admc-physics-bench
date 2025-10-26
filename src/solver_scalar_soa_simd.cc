#include "solver_scalar_soa_simd.hpp"

#include "solver_scalar_soa.hpp"
#include "types.hpp"

#include <algorithm>

namespace soa_simd {

void update_normal_batch(const double* target,
                         const double* v_rel,
                         const double* k,
                         double* impulses,
                         int count) {
  for (int i = 0; i < count; ++i) {
    const double delta = (target[i] - v_rel[i]) / k[i];
    impulses[i] += delta;
  }
}

void update_tangent_batch(const double* target,
                          const double* v_rel,
                          const double* k,
                          double* impulses,
                          int count) {
  for (int i = 0; i < count; ++i) {
    const double delta = (target[i] - v_rel[i]) / k[i];
    impulses[i] += delta;
  }
}

void apply_impulses_batch(std::vector<RigidBody>& bodies,
                          const RowSOA& rows,
                          const double* delta_jn,
                          const double* delta_jt1,
                          const double* delta_jt2,
                          int start,
                          int count) {
  for (int lane = 0; lane < count; ++lane) {
    const int idx = start + lane;
    if (idx >= static_cast<int>(rows.size())) {
      break;
    }
    const int ia = rows.a[idx];
    const int ib = rows.b[idx];
    if (ia < 0 || ib < 0 || ia >= static_cast<int>(bodies.size()) ||
        ib >= static_cast<int>(bodies.size())) {
      continue;
    }

    RigidBody& A = bodies[ia];
    RigidBody& B = bodies[ib];
    const math::Vec3 Pn = rows.n[idx] * delta_jn[lane];
    const math::Vec3 Pt1 = rows.t1[idx] * delta_jt1[lane];
    const math::Vec3 Pt2 = rows.t2[idx] * delta_jt2[lane];
    const math::Vec3 impulse = Pn + Pt1 + Pt2;
    A.applyImpulse(-impulse, rows.ra[idx]);
    B.applyImpulse(impulse, rows.rb[idx]);
  }
}

}  // namespace soa_simd

void solve_scalar_soa_simd(std::vector<RigidBody>& bodies,
                           std::vector<Contact>& contacts,
                           RowSOA& rows,
                           JointSOA& joints,
                           const SoaParams& params,
                           SolverDebugInfo* debug_info) {
  solve_scalar_soa_scalar(bodies, contacts, rows, joints, params, debug_info);
}

void solve_scalar_soa_simd(std::vector<RigidBody>& bodies,
                           std::vector<Contact>& contacts,
                           RowSOA& rows,
                           const SoaParams& params,
                           SolverDebugInfo* debug_info) {
  solve_scalar_soa_scalar(bodies, contacts, rows, params, debug_info);
}
