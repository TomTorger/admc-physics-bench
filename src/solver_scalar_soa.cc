#include "solver_scalar_soa.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

namespace {
using math::Vec3;

Vec3 make_tangent(const Vec3& n) {
  if (std::fabs(n.x) < 0.57735026919) {
    return math::normalize_safe(math::cross(Vec3(1.0, 0.0, 0.0), n));
  }
  return math::normalize_safe(math::cross(Vec3(0.0, 1.0, 0.0), n));
}

Vec3 orthonormalize(const Vec3& n, const Vec3& t) {
  Vec3 tangent = t - math::dot(t, n) * n;
  tangent = math::normalize_safe(tangent);
  if (math::length2(tangent) <= math::kEps * math::kEps) {
    tangent = make_tangent(n);
  }
  return tangent;
}

SoaParams make_soa_params(const SolverParams& params) {
  SoaParams derived;
  static_cast<SolverParams&>(derived) = params;
  if (derived.thread_count <= 0) {
    derived.thread_count = 1;
  }
  if (derived.block_size <= 0) {
    derived.block_size = 1;
  }
  return derived;
}

}  // namespace

void solve_scalar_soa_simd(std::vector<RigidBody>& bodies,
                           std::vector<Contact>& contacts,
                           RowSOA& rows,
                           JointSOA& joints,
                           const SoaParams& params,
                           SolverDebugInfo* debug_info);

void solve_scalar_soa_simd(std::vector<RigidBody>& bodies,
                           std::vector<Contact>& contacts,
                           RowSOA& rows,
                           const SoaParams& params,
                           SolverDebugInfo* debug_info);

void solve_scalar_soa_mt(std::vector<RigidBody>& bodies,
                         std::vector<Contact>& contacts,
                         RowSOA& rows,
                         JointSOA& joints,
                         const SoaParams& params,
                         SolverDebugInfo* debug_info);

void solve_scalar_soa_mt(std::vector<RigidBody>& bodies,
                         std::vector<Contact>& contacts,
                         RowSOA& rows,
                         const SoaParams& params,
                         SolverDebugInfo* debug_info);

std::string solver_debug_summary(const SolverDebugInfo& info) {
  std::ostringstream oss;
  oss << "invalid_contacts=" << info.invalid_contact_indices
      << ", invalid_joints=" << info.invalid_joint_indices
      << ", warmstart_contacts=" << info.warmstart_contact_impulses
      << ", warmstart_joints=" << info.warmstart_joint_impulses
      << ", normal_clamps=" << info.normal_impulse_clamps
      << ", tangent_projections=" << info.tangent_projections
      << ", rope_clamps=" << info.rope_clamps
      << ", singular_joint_denoms=" << info.singular_joint_denominators;
  return oss.str();
}

RowSOA build_soa(const std::vector<RigidBody>& bodies,
                 const std::vector<Contact>& contacts,
                 const SoaParams& params) {
  RowSOA rows;
  const std::size_t capacity = contacts.size();
  auto resize_all = [&](std::size_t size) {
    rows.a.resize(size);
    rows.b.resize(size);
    rows.nx.resize(size);
    rows.ny.resize(size);
    rows.nz.resize(size);
    rows.t1x.resize(size);
    rows.t1y.resize(size);
    rows.t1z.resize(size);
    rows.t2x.resize(size);
    rows.t2y.resize(size);
    rows.t2z.resize(size);
    rows.rax.resize(size);
    rows.ray.resize(size);
    rows.raz.resize(size);
    rows.rbx.resize(size);
    rows.rby.resize(size);
    rows.rbz.resize(size);
    rows.raxn_x.resize(size);
    rows.raxn_y.resize(size);
    rows.raxn_z.resize(size);
    rows.rbxn_x.resize(size);
    rows.rbxn_y.resize(size);
    rows.rbxn_z.resize(size);
    rows.raxt1_x.resize(size);
    rows.raxt1_y.resize(size);
    rows.raxt1_z.resize(size);
    rows.rbxt1_x.resize(size);
    rows.rbxt1_y.resize(size);
    rows.rbxt1_z.resize(size);
    rows.raxt2_x.resize(size);
    rows.raxt2_y.resize(size);
    rows.raxt2_z.resize(size);
    rows.rbxt2_x.resize(size);
    rows.rbxt2_y.resize(size);
    rows.rbxt2_z.resize(size);
    rows.TWn_a_x.resize(size);
    rows.TWn_a_y.resize(size);
    rows.TWn_a_z.resize(size);
    rows.TWn_b_x.resize(size);
    rows.TWn_b_y.resize(size);
    rows.TWn_b_z.resize(size);
    rows.TWt1_a_x.resize(size);
    rows.TWt1_a_y.resize(size);
    rows.TWt1_a_z.resize(size);
    rows.TWt1_b_x.resize(size);
    rows.TWt1_b_y.resize(size);
    rows.TWt1_b_z.resize(size);
    rows.TWt2_a_x.resize(size);
    rows.TWt2_a_y.resize(size);
    rows.TWt2_a_z.resize(size);
    rows.TWt2_b_x.resize(size);
    rows.TWt2_b_y.resize(size);
    rows.TWt2_b_z.resize(size);
    rows.k_n.resize(size);
    rows.k_t1.resize(size);
    rows.k_t2.resize(size);
    rows.inv_k_n.resize(size);
    rows.inv_k_t1.resize(size);
    rows.inv_k_t2.resize(size);
    rows.mu.resize(size);
    rows.e.resize(size);
    rows.bias.resize(size);
    rows.bounce.resize(size);
    rows.C.resize(size);
    rows.jn.resize(size);
    rows.jt1.resize(size);
    rows.jt2.resize(size);
    rows.indices.resize(size);
  };
  resize_all(capacity);

  std::size_t write_index = 0;

  for (std::size_t i = 0; i < contacts.size(); ++i) {
    const Contact& c = contacts[i];
    if (c.a < 0 || c.b < 0 || c.a >= static_cast<int>(bodies.size()) ||
        c.b >= static_cast<int>(bodies.size())) {
      continue;
    }

    const RigidBody& A = bodies[c.a];
    const RigidBody& B = bodies[c.b];

    Vec3 n = c.n;
    Vec3 t1 = c.t1;
    Vec3 t2 = c.t2;
    if (math::length2(n) <= math::kEps * math::kEps) {
      n = Vec3(1.0, 0.0, 0.0);
    }
    if (math::length2(t1) <= math::kEps * math::kEps) {
      t1 = make_tangent(n);
    }
    if (math::length2(t2) <= math::kEps * math::kEps) {
      t2 = math::normalize_safe(math::cross(n, t1));
    }

    Vec3 ra = c.ra;
    Vec3 rb = c.rb;
    if (math::length2(ra) <= math::kEps * math::kEps) {
      ra = c.p - A.x;
    }
    if (math::length2(rb) <= math::kEps * math::kEps) {
      rb = c.p - B.x;
    }

    const Vec3 ra_cross_n = c.ra_cross_n;
    const Vec3 rb_cross_n = c.rb_cross_n;
    const Vec3 ra_cross_t1 = c.ra_cross_t1;
    const Vec3 rb_cross_t1 = c.rb_cross_t1;
    const Vec3 ra_cross_t2 = c.ra_cross_t2;
    const Vec3 rb_cross_t2 = c.rb_cross_t2;

    const Vec3 TWn_a = A.invInertiaWorld * ra_cross_n;
    const Vec3 TWn_b = B.invInertiaWorld * rb_cross_n;
    const Vec3 TWt1_a = A.invInertiaWorld * ra_cross_t1;
    const Vec3 TWt1_b = B.invInertiaWorld * rb_cross_t1;
    const Vec3 TWt2_a = A.invInertiaWorld * ra_cross_t2;
    const Vec3 TWt2_b = B.invInertiaWorld * rb_cross_t2;

    const double k_n = (c.k_n > math::kEps) ? c.k_n : 1.0;
    const double k_t1 = (c.k_t1 > math::kEps) ? c.k_t1 : 1.0;
    const double k_t2 = (c.k_t2 > math::kEps) ? c.k_t2 : 1.0;

    const Vec3 va = A.v + math::cross(A.w, ra);
    const Vec3 vb = B.v + math::cross(B.w, rb);
    const double v_rel_n = math::dot(n, vb - va);
    const double restitution = std::max(c.e, params.restitution);
    const double bounce = (v_rel_n < 0.0) ? (-restitution * v_rel_n) : 0.0;
    const double bias = c.bias;
    const double mu = std::max(c.mu, params.mu);
    const double violation = (std::fabs(c.C) <= math::kEps) ? 0.0 : c.C;

    rows.indices[write_index] = static_cast<int>(i);
    rows.a[write_index] = c.a;
    rows.b[write_index] = c.b;
    rows.nx[write_index] = n.x;
    rows.ny[write_index] = n.y;
    rows.nz[write_index] = n.z;
    rows.t1x[write_index] = t1.x;
    rows.t1y[write_index] = t1.y;
    rows.t1z[write_index] = t1.z;
    rows.t2x[write_index] = t2.x;
    rows.t2y[write_index] = t2.y;
    rows.t2z[write_index] = t2.z;
    rows.rax[write_index] = ra.x;
    rows.ray[write_index] = ra.y;
    rows.raz[write_index] = ra.z;
    rows.rbx[write_index] = rb.x;
    rows.rby[write_index] = rb.y;
    rows.rbz[write_index] = rb.z;
    rows.raxn_x[write_index] = ra_cross_n.x;
    rows.raxn_y[write_index] = ra_cross_n.y;
    rows.raxn_z[write_index] = ra_cross_n.z;
    rows.rbxn_x[write_index] = rb_cross_n.x;
    rows.rbxn_y[write_index] = rb_cross_n.y;
    rows.rbxn_z[write_index] = rb_cross_n.z;
    rows.raxt1_x[write_index] = ra_cross_t1.x;
    rows.raxt1_y[write_index] = ra_cross_t1.y;
    rows.raxt1_z[write_index] = ra_cross_t1.z;
    rows.rbxt1_x[write_index] = rb_cross_t1.x;
    rows.rbxt1_y[write_index] = rb_cross_t1.y;
    rows.rbxt1_z[write_index] = rb_cross_t1.z;
    rows.raxt2_x[write_index] = ra_cross_t2.x;
    rows.raxt2_y[write_index] = ra_cross_t2.y;
    rows.raxt2_z[write_index] = ra_cross_t2.z;
    rows.rbxt2_x[write_index] = rb_cross_t2.x;
    rows.rbxt2_y[write_index] = rb_cross_t2.y;
    rows.rbxt2_z[write_index] = rb_cross_t2.z;
    rows.TWn_a_x[write_index] = TWn_a.x;
    rows.TWn_a_y[write_index] = TWn_a.y;
    rows.TWn_a_z[write_index] = TWn_a.z;
    rows.TWn_b_x[write_index] = TWn_b.x;
    rows.TWn_b_y[write_index] = TWn_b.y;
    rows.TWn_b_z[write_index] = TWn_b.z;
    rows.TWt1_a_x[write_index] = TWt1_a.x;
    rows.TWt1_a_y[write_index] = TWt1_a.y;
    rows.TWt1_a_z[write_index] = TWt1_a.z;
    rows.TWt1_b_x[write_index] = TWt1_b.x;
    rows.TWt1_b_y[write_index] = TWt1_b.y;
    rows.TWt1_b_z[write_index] = TWt1_b.z;
    rows.TWt2_a_x[write_index] = TWt2_a.x;
    rows.TWt2_a_y[write_index] = TWt2_a.y;
    rows.TWt2_a_z[write_index] = TWt2_a.z;
    rows.TWt2_b_x[write_index] = TWt2_b.x;
    rows.TWt2_b_y[write_index] = TWt2_b.y;
    rows.TWt2_b_z[write_index] = TWt2_b.z;
    rows.k_n[write_index] = k_n;
    rows.k_t1[write_index] = k_t1;
    rows.k_t2[write_index] = k_t2;
    rows.inv_k_n[write_index] = 1.0 / k_n;
    rows.inv_k_t1[write_index] = 1.0 / k_t1;
    rows.inv_k_t2[write_index] = 1.0 / k_t2;
    const bool allow_warm_start = params.warm_start && (violation < -params.slop);
    rows.jn[write_index] = allow_warm_start ? c.jn : 0.0;
    rows.jt1[write_index] = allow_warm_start ? c.jt1 : 0.0;
    rows.jt2[write_index] = allow_warm_start ? c.jt2 : 0.0;
    rows.mu[write_index] = mu;
    rows.e[write_index] = restitution;
    rows.bias[write_index] = bias;
    rows.bounce[write_index] = bounce;
    rows.C[write_index] = violation;

    ++write_index;
  }

  const std::size_t valid_rows = write_index;
  resize_all(valid_rows);
  rows.N = static_cast<int>(valid_rows);
  return rows;
}

RowSOA build_soa(const std::vector<RigidBody>& bodies,
                 const std::vector<Contact>& contacts,
                 const SolverParams& params) {
  return build_soa(bodies, contacts, make_soa_params(params));
}

void solve_scalar_soa_scalar(std::vector<RigidBody>& bodies,
                             std::vector<Contact>& contacts,
                             RowSOA& rows,
                             JointSOA& joints,
                             const SoaParams& params,
                             SolverDebugInfo* debug_info) {
  const int iterations = std::max(1, params.iterations);

  if (debug_info) {
    debug_info->reset();
  }

  for (RigidBody& body : bodies) {
    body.syncDerived();
  }

  if (!params.warm_start) {
    std::fill(rows.jn.begin(), rows.jn.end(), 0.0);
    std::fill(rows.jt1.begin(), rows.jt1.end(), 0.0);
    std::fill(rows.jt2.begin(), rows.jt2.end(), 0.0);
    std::fill(joints.j.begin(), joints.j.end(), 0.0);
  } else {
    for (std::size_t i = 0; i < rows.size(); ++i) {
      const int ia = rows.a[i];
      const int ib = rows.b[i];
      if (ia < 0 || ib < 0 || ia >= static_cast<int>(bodies.size()) ||
          ib >= static_cast<int>(bodies.size())) {
        if (debug_info) {
          ++debug_info->invalid_contact_indices;
        }
        continue;
      }

      RigidBody& A = bodies[ia];
      RigidBody& B = bodies[ib];

      const double jn = rows.jn[i];
      const double jt1 = rows.jt1[i];
      const double jt2 = rows.jt2[i];

      const double impulse_x = rows.nx[i] * jn + rows.t1x[i] * jt1 +
                               rows.t2x[i] * jt2;
      const double impulse_y = rows.ny[i] * jn + rows.t1y[i] * jt1 +
                               rows.t2y[i] * jt2;
      const double impulse_z = rows.nz[i] * jn + rows.t1z[i] * jt1 +
                               rows.t2z[i] * jt2;
      const double impulse_len2 = impulse_x * impulse_x + impulse_y * impulse_y +
                                  impulse_z * impulse_z;
      if (impulse_len2 <= math::kEps * math::kEps) {
        continue;
      }
      if (debug_info) {
        ++debug_info->warmstart_contact_impulses;
      }

      A.v.x -= impulse_x * A.invMass;
      A.v.y -= impulse_y * A.invMass;
      A.v.z -= impulse_z * A.invMass;
      B.v.x += impulse_x * B.invMass;
      B.v.y += impulse_y * B.invMass;
      B.v.z += impulse_z * B.invMass;

      const double dw_ax = jn * rows.TWn_a_x[i] + jt1 * rows.TWt1_a_x[i] +
                           jt2 * rows.TWt2_a_x[i];
      const double dw_ay = jn * rows.TWn_a_y[i] + jt1 * rows.TWt1_a_y[i] +
                           jt2 * rows.TWt2_a_y[i];
      const double dw_az = jn * rows.TWn_a_z[i] + jt1 * rows.TWt1_a_z[i] +
                           jt2 * rows.TWt2_a_z[i];
      const double dw_bx = jn * rows.TWn_b_x[i] + jt1 * rows.TWt1_b_x[i] +
                           jt2 * rows.TWt2_b_x[i];
      const double dw_by = jn * rows.TWn_b_y[i] + jt1 * rows.TWt1_b_y[i] +
                           jt2 * rows.TWt2_b_y[i];
      const double dw_bz = jn * rows.TWn_b_z[i] + jt1 * rows.TWt1_b_z[i] +
                           jt2 * rows.TWt2_b_z[i];

      A.w.x -= dw_ax;
      A.w.y -= dw_ay;
      A.w.z -= dw_az;
      B.w.x += dw_bx;
      B.w.y += dw_by;
      B.w.z += dw_bz;
    }

    for (std::size_t i = 0; i < joints.size(); ++i) {
      const int ia = joints.a[i];
      const int ib = joints.b[i];
      if (ia < 0 || ib < 0 || ia >= static_cast<int>(bodies.size()) ||
          ib >= static_cast<int>(bodies.size())) {
        if (debug_info) {
          ++debug_info->invalid_joint_indices;
        }
        continue;
      }

      if (std::fabs(joints.j[i]) <= math::kEps) {
        continue;
      }

      RigidBody& A = bodies[ia];
      RigidBody& B = bodies[ib];
      const Vec3 impulse = joints.d[i] * joints.j[i];
      if (debug_info) {
        ++debug_info->warmstart_joint_impulses;
      }
      A.applyImpulse(-impulse, joints.ra[i]);
      B.applyImpulse(impulse, joints.rb[i]);
    }
  }

  for (int it = 0; it < iterations; ++it) {
    for (std::size_t i = 0; i < rows.size(); ++i) {
      const int ia = rows.a[i];
      const int ib = rows.b[i];
      if (ia < 0 || ib < 0 || ia >= static_cast<int>(bodies.size()) ||
          ib >= static_cast<int>(bodies.size())) {
        if (debug_info) {
          ++debug_info->invalid_contact_indices;
        }
        continue;
      }

      RigidBody& A = bodies[ia];
      RigidBody& B = bodies[ib];

      const double invMassA = A.invMass;
      const double invMassB = B.invMass;
      // Compute relative linear velocities once so the normal and tangential
      // solves can reuse the deltas without repeating subtractions.
      const double dvx = B.v.x - A.v.x;
      const double dvy = B.v.y - A.v.y;
      const double dvz = B.v.z - A.v.z;
      const double wAx = A.w.x;
      const double wAy = A.w.y;
      const double wAz = A.w.z;
      const double wBx = B.w.x;
      const double wBy = B.w.y;
      const double wBz = B.w.z;

      const double nx = rows.nx[i];
      const double ny = rows.ny[i];
      const double nz = rows.nz[i];

      const double raxn_x = rows.raxn_x[i];
      const double raxn_y = rows.raxn_y[i];
      const double raxn_z = rows.raxn_z[i];
      const double rbxn_x = rows.rbxn_x[i];
      const double rbxn_y = rows.rbxn_y[i];
      const double rbxn_z = rows.rbxn_z[i];
      const double wA_dot_raxn = wAx * raxn_x + wAy * raxn_y + wAz * raxn_z;
      const double wB_dot_rbxn = wBx * rbxn_x + wBy * rbxn_y + wBz * rbxn_z;

      const double v_rel_n = nx * dvx + ny * dvy + nz * dvz +
                             wB_dot_rbxn - wA_dot_raxn;

      const double rhs = -(v_rel_n + rows.bias[i] - rows.bounce[i]);

      // Multiplying by the cached reciprocal avoids a divide in this hot loop;
      // degenerate denominators were clamped during row construction.
      const double delta_jn = rhs * rows.inv_k_n[i];
      const double jn_old = rows.jn[i];
      double jn_candidate = jn_old + delta_jn;
      if (jn_candidate < 0.0) {
        if (debug_info) {
          ++debug_info->normal_impulse_clamps;
        }
        jn_candidate = 0.0;
      }
      rows.jn[i] = jn_candidate;
      const double applied_n = rows.jn[i] - jn_old;
      if (std::fabs(applied_n) > math::kEps) {
        const double impulse_x = applied_n * nx;
        const double impulse_y = applied_n * ny;
        const double impulse_z = applied_n * nz;

        A.v.x -= impulse_x * invMassA;
        A.v.y -= impulse_y * invMassA;
        A.v.z -= impulse_z * invMassA;
        B.v.x += impulse_x * invMassB;
        B.v.y += impulse_y * invMassB;
        B.v.z += impulse_z * invMassB;

        A.w.x -= applied_n * rows.TWn_a_x[i];
        A.w.y -= applied_n * rows.TWn_a_y[i];
        A.w.z -= applied_n * rows.TWn_a_z[i];
        B.w.x += applied_n * rows.TWn_b_x[i];
        B.w.y += applied_n * rows.TWn_b_y[i];
        B.w.z += applied_n * rows.TWn_b_z[i];
      }

      const double t1x = rows.t1x[i];
      const double t1y = rows.t1y[i];
      const double t1z = rows.t1z[i];
      const double t2x = rows.t2x[i];
      const double t2y = rows.t2y[i];
      const double t2z = rows.t2z[i];

      const double raxt1_x = rows.raxt1_x[i];
      const double raxt1_y = rows.raxt1_y[i];
      const double raxt1_z = rows.raxt1_z[i];
      const double rbxt1_x = rows.rbxt1_x[i];
      const double rbxt1_y = rows.rbxt1_y[i];
      const double rbxt1_z = rows.rbxt1_z[i];
      const double raxt2_x = rows.raxt2_x[i];
      const double raxt2_y = rows.raxt2_y[i];
      const double raxt2_z = rows.raxt2_z[i];
      const double rbxt2_x = rows.rbxt2_x[i];
      const double rbxt2_y = rows.rbxt2_y[i];
      const double rbxt2_z = rows.rbxt2_z[i];

      const double dvx_post = B.v.x - A.v.x;
      const double dvy_post = B.v.y - A.v.y;
      const double dvz_post = B.v.z - A.v.z;

      const double wAx_post = A.w.x;
      const double wAy_post = A.w.y;
      const double wAz_post = A.w.z;
      const double wBx_post = B.w.x;
      const double wBy_post = B.w.y;
      const double wBz_post = B.w.z;

      const double wA_dot_raxt1 =
          wAx_post * raxt1_x + wAy_post * raxt1_y + wAz_post * raxt1_z;
      const double wB_dot_rbxt1 =
          wBx_post * rbxt1_x + wBy_post * rbxt1_y + wBz_post * rbxt1_z;
      const double wA_dot_raxt2 =
          wAx_post * raxt2_x + wAy_post * raxt2_y + wAz_post * raxt2_z;
      const double wB_dot_rbxt2 =
          wBx_post * rbxt2_x + wBy_post * rbxt2_y + wBz_post * rbxt2_z;

      const double v_rel_t1 = t1x * dvx_post + t1y * dvy_post + t1z * dvz_post +
                              wB_dot_rbxt1 - wA_dot_raxt1;

      const double v_rel_t2 = t2x * dvx_post + t2y * dvy_post + t2z * dvz_post +
                              wB_dot_rbxt2 - wA_dot_raxt2;

      double jt1_candidate =
          rows.jt1[i] + (-v_rel_t1) * rows.inv_k_t1[i];

      double jt2_candidate =
          rows.jt2[i] + (-v_rel_t2) * rows.inv_k_t2[i];

      const double friction_max = rows.mu[i] * std::max(rows.jn[i], 0.0);
      const double jt_mag =
          std::sqrt(jt1_candidate * jt1_candidate + jt2_candidate * jt2_candidate);
      double scale = 1.0;
      if (jt_mag > friction_max && jt_mag > math::kEps) {
        scale = (friction_max > 0.0) ? (friction_max / jt_mag) : 0.0;
        if (debug_info) {
          ++debug_info->tangent_projections;
        }
      }

      jt1_candidate *= scale;
      jt2_candidate *= scale;

      const double delta_jt1 = jt1_candidate - rows.jt1[i];
      const double delta_jt2 = jt2_candidate - rows.jt2[i];
      rows.jt1[i] = jt1_candidate;
      rows.jt2[i] = jt2_candidate;

      if (std::fabs(delta_jt1) > math::kEps || std::fabs(delta_jt2) > math::kEps) {
        const double impulse_x = delta_jt1 * t1x + delta_jt2 * t2x;
        const double impulse_y = delta_jt1 * t1y + delta_jt2 * t2y;
        const double impulse_z = delta_jt1 * t1z + delta_jt2 * t2z;

        A.v.x -= impulse_x * A.invMass;
        A.v.y -= impulse_y * A.invMass;
        A.v.z -= impulse_z * A.invMass;
        B.v.x += impulse_x * B.invMass;
        B.v.y += impulse_y * B.invMass;
        B.v.z += impulse_z * B.invMass;

        A.w.x -= delta_jt1 * rows.TWt1_a_x[i] + delta_jt2 * rows.TWt2_a_x[i];
        A.w.y -= delta_jt1 * rows.TWt1_a_y[i] + delta_jt2 * rows.TWt2_a_y[i];
        A.w.z -= delta_jt1 * rows.TWt1_a_z[i] + delta_jt2 * rows.TWt2_a_z[i];
        B.w.x += delta_jt1 * rows.TWt1_b_x[i] + delta_jt2 * rows.TWt2_b_x[i];
        B.w.y += delta_jt1 * rows.TWt1_b_y[i] + delta_jt2 * rows.TWt2_b_y[i];
        B.w.z += delta_jt1 * rows.TWt1_b_z[i] + delta_jt2 * rows.TWt2_b_z[i];
      }
    }

    for (std::size_t i = 0; i < joints.size(); ++i) {
      const int ia = joints.a[i];
      const int ib = joints.b[i];
      if (ia < 0 || ib < 0 || ia >= static_cast<int>(bodies.size()) ||
          ib >= static_cast<int>(bodies.size())) {
        if (debug_info) {
          ++debug_info->invalid_joint_indices;
        }
        continue;
      }

      const double denom = joints.k[i] + joints.gamma[i];
      if (denom <= math::kEps) {
        if (debug_info) {
          ++debug_info->singular_joint_denominators;
        }
        continue;
      }

      RigidBody& A = bodies[ia];
      RigidBody& B = bodies[ib];
      const Vec3 va = A.v + math::cross(A.w, joints.ra[i]);
      const Vec3 vb = B.v + math::cross(B.w, joints.rb[i]);
      const double v_rel_d = math::dot(joints.d[i], vb - va);

      double j_new = joints.j[i] - (v_rel_d + joints.bias[i]) / denom;
      if (joints.rope[i] && j_new < 0.0) {
        if (debug_info) {
          ++debug_info->rope_clamps;
        }
        j_new = 0.0;
      }

      const double applied = j_new - joints.j[i];
      joints.j[i] = j_new;

      if (std::fabs(applied) > math::kEps) {
        const Vec3 impulse = applied * joints.d[i];
        A.applyImpulse(-impulse, joints.ra[i]);
        B.applyImpulse(impulse, joints.rb[i]);
      }
    }
  }

  for (std::size_t i = 0; i < rows.size(); ++i) {
    const int idx = rows.indices[i];
    if (idx < 0 || idx >= static_cast<int>(contacts.size())) {
      if (debug_info) {
        ++debug_info->invalid_contact_indices;
      }
      continue;
    }
    Contact& c = contacts[static_cast<std::size_t>(idx)];
    c.a = rows.a[i];
    c.b = rows.b[i];
    c.n = Vec3(rows.nx[i], rows.ny[i], rows.nz[i]);
    c.t1 = Vec3(rows.t1x[i], rows.t1y[i], rows.t1z[i]);
    c.t2 = Vec3(rows.t2x[i], rows.t2y[i], rows.t2z[i]);
    c.ra = Vec3(rows.rax[i], rows.ray[i], rows.raz[i]);
    c.rb = Vec3(rows.rbx[i], rows.rby[i], rows.rbz[i]);
    c.ra_cross_n = Vec3(rows.raxn_x[i], rows.raxn_y[i], rows.raxn_z[i]);
    c.rb_cross_n = Vec3(rows.rbxn_x[i], rows.rbxn_y[i], rows.rbxn_z[i]);
    c.ra_cross_t1 = Vec3(rows.raxt1_x[i], rows.raxt1_y[i], rows.raxt1_z[i]);
    c.rb_cross_t1 = Vec3(rows.rbxt1_x[i], rows.rbxt1_y[i], rows.rbxt1_z[i]);
    c.ra_cross_t2 = Vec3(rows.raxt2_x[i], rows.raxt2_y[i], rows.raxt2_z[i]);
    c.rb_cross_t2 = Vec3(rows.rbxt2_x[i], rows.rbxt2_y[i], rows.rbxt2_z[i]);
    c.k_n = rows.k_n[i];
    c.k_t1 = rows.k_t1[i];
    c.k_t2 = rows.k_t2[i];
    c.jn = rows.jn[i];
    c.jt1 = rows.jt1[i];
    c.jt2 = rows.jt2[i];
    c.mu = rows.mu[i];
    c.e = rows.e[i];
    c.bias = rows.bias[i];
    c.bounce = rows.bounce[i];
    c.C = rows.C[i];
  }

  for (RigidBody& body : bodies) {
    body.integrate(params.dt);
  }
}

void solve_scalar_soa_scalar(std::vector<RigidBody>& bodies,
                             std::vector<Contact>& contacts,
                             RowSOA& rows,
                             const SoaParams& params,
                             SolverDebugInfo* debug_info) {
  static JointSOA empty_joints;
  empty_joints.a.clear();
  empty_joints.b.clear();
  empty_joints.d.clear();
  empty_joints.ra.clear();
  empty_joints.rb.clear();
  empty_joints.k.clear();
  empty_joints.gamma.clear();
  empty_joints.bias.clear();
  empty_joints.j.clear();
  empty_joints.rope.clear();
  empty_joints.C.clear();
  empty_joints.indices.clear();
  solve_scalar_soa_scalar(bodies, contacts, rows, empty_joints, params, debug_info);
}

void solve_scalar_soa(std::vector<RigidBody>& bodies,
                      std::vector<Contact>& contacts,
                      RowSOA& rows,
                      JointSOA& joints,
                      const SoaParams& params,
                      SolverDebugInfo* debug_info) {
  SoaParams effective = params;
  if (effective.thread_count <= 0) {
    effective.thread_count = 1;
  }
  if (effective.block_size <= 0) {
    effective.block_size = 1;
  }

#if defined(ADMC_USE_THREADS)
  if (effective.use_threads && effective.thread_count > 1) {
    solve_scalar_soa_mt(bodies, contacts, rows, joints, effective, debug_info);
    return;
  }
#endif

#if defined(ADMC_USE_AVX2) || defined(ADMC_USE_NEON)
  if (effective.use_simd) {
    solve_scalar_soa_simd(bodies, contacts, rows, joints, effective, debug_info);
    return;
  }
#endif

  solve_scalar_soa_scalar(bodies, contacts, rows, joints, effective, debug_info);
}

void solve_scalar_soa(std::vector<RigidBody>& bodies,
                      std::vector<Contact>& contacts,
                      RowSOA& rows,
                      JointSOA& joints,
                      const SolverParams& params,
                      SolverDebugInfo* debug_info) {
  solve_scalar_soa(bodies, contacts, rows, joints, make_soa_params(params),
                   debug_info);
}

void solve_scalar_soa(std::vector<RigidBody>& bodies,
                      std::vector<Contact>& contacts,
                      RowSOA& rows,
                      const SoaParams& params,
                      SolverDebugInfo* debug_info) {
  SoaParams effective = params;
  if (effective.thread_count <= 0) {
    effective.thread_count = 1;
  }
  if (effective.block_size <= 0) {
    effective.block_size = 1;
  }

#if defined(ADMC_USE_THREADS)
  if (effective.use_threads && effective.thread_count > 1) {
    solve_scalar_soa_mt(bodies, contacts, rows, effective, debug_info);
    return;
  }
#endif

#if defined(ADMC_USE_AVX2) || defined(ADMC_USE_NEON)
  if (effective.use_simd) {
    solve_scalar_soa_simd(bodies, contacts, rows, effective, debug_info);
    return;
  }
#endif

  solve_scalar_soa_scalar(bodies, contacts, rows, effective, debug_info);
}

void solve_scalar_soa(std::vector<RigidBody>& bodies,
                      std::vector<Contact>& contacts,
                      RowSOA& rows,
                      const SolverParams& params,
                      SolverDebugInfo* debug_info) {
  solve_scalar_soa(bodies, contacts, rows, make_soa_params(params), debug_info);
}
