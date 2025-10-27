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
    const double dj_n = delta_jn[lane];
    const double dj_t1 = delta_jt1[lane];
    const double dj_t2 = delta_jt2[lane];

    const double impulse_x = rows.nx[idx] * dj_n + rows.t1x[idx] * dj_t1 +
                             rows.t2x[idx] * dj_t2;
    const double impulse_y = rows.ny[idx] * dj_n + rows.t1y[idx] * dj_t1 +
                             rows.t2y[idx] * dj_t2;
    const double impulse_z = rows.nz[idx] * dj_n + rows.t1z[idx] * dj_t1 +
                             rows.t2z[idx] * dj_t2;

    A.v.x -= impulse_x * A.invMass;
    A.v.y -= impulse_y * A.invMass;
    A.v.z -= impulse_z * A.invMass;
    B.v.x += impulse_x * B.invMass;
    B.v.y += impulse_y * B.invMass;
    B.v.z += impulse_z * B.invMass;

    const double dw_ax = dj_n * rows.TWn_a_x[idx] +
                         dj_t1 * rows.TWt1_a_x[idx] +
                         dj_t2 * rows.TWt2_a_x[idx];
    const double dw_ay = dj_n * rows.TWn_a_y[idx] +
                         dj_t1 * rows.TWt1_a_y[idx] +
                         dj_t2 * rows.TWt2_a_y[idx];
    const double dw_az = dj_n * rows.TWn_a_z[idx] +
                         dj_t1 * rows.TWt1_a_z[idx] +
                         dj_t2 * rows.TWt2_a_z[idx];
    const double dw_bx = dj_n * rows.TWn_b_x[idx] +
                         dj_t1 * rows.TWt1_b_x[idx] +
                         dj_t2 * rows.TWt2_b_x[idx];
    const double dw_by = dj_n * rows.TWn_b_y[idx] +
                         dj_t1 * rows.TWt1_b_y[idx] +
                         dj_t2 * rows.TWt2_b_y[idx];
    const double dw_bz = dj_n * rows.TWn_b_z[idx] +
                         dj_t1 * rows.TWt1_b_z[idx] +
                         dj_t2 * rows.TWt2_b_z[idx];

    A.w.x -= dw_ax;
    A.w.y -= dw_ay;
    A.w.z -= dw_az;
    B.w.x += dw_bx;
    B.w.y += dw_by;
    B.w.z += dw_bz;
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
