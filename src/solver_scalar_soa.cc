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
  Vec3 tangent = math::normalize_safe(t);
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
  rows.a.reserve(capacity);
  rows.b.reserve(capacity);
  rows.nx.reserve(capacity);
  rows.ny.reserve(capacity);
  rows.nz.reserve(capacity);
  rows.t1x.reserve(capacity);
  rows.t1y.reserve(capacity);
  rows.t1z.reserve(capacity);
  rows.t2x.reserve(capacity);
  rows.t2y.reserve(capacity);
  rows.t2z.reserve(capacity);
  rows.rax.reserve(capacity);
  rows.ray.reserve(capacity);
  rows.raz.reserve(capacity);
  rows.rbx.reserve(capacity);
  rows.rby.reserve(capacity);
  rows.rbz.reserve(capacity);
  rows.raxn_x.reserve(capacity);
  rows.raxn_y.reserve(capacity);
  rows.raxn_z.reserve(capacity);
  rows.rbxn_x.reserve(capacity);
  rows.rbxn_y.reserve(capacity);
  rows.rbxn_z.reserve(capacity);
  rows.raxt1_x.reserve(capacity);
  rows.raxt1_y.reserve(capacity);
  rows.raxt1_z.reserve(capacity);
  rows.rbxt1_x.reserve(capacity);
  rows.rbxt1_y.reserve(capacity);
  rows.rbxt1_z.reserve(capacity);
  rows.raxt2_x.reserve(capacity);
  rows.raxt2_y.reserve(capacity);
  rows.raxt2_z.reserve(capacity);
  rows.rbxt2_x.reserve(capacity);
  rows.rbxt2_y.reserve(capacity);
  rows.rbxt2_z.reserve(capacity);
  rows.TWn_a_x.reserve(capacity);
  rows.TWn_a_y.reserve(capacity);
  rows.TWn_a_z.reserve(capacity);
  rows.TWn_b_x.reserve(capacity);
  rows.TWn_b_y.reserve(capacity);
  rows.TWn_b_z.reserve(capacity);
  rows.TWt1_a_x.reserve(capacity);
  rows.TWt1_a_y.reserve(capacity);
  rows.TWt1_a_z.reserve(capacity);
  rows.TWt1_b_x.reserve(capacity);
  rows.TWt1_b_y.reserve(capacity);
  rows.TWt1_b_z.reserve(capacity);
  rows.TWt2_a_x.reserve(capacity);
  rows.TWt2_a_y.reserve(capacity);
  rows.TWt2_a_z.reserve(capacity);
  rows.TWt2_b_x.reserve(capacity);
  rows.TWt2_b_y.reserve(capacity);
  rows.TWt2_b_z.reserve(capacity);
  rows.k_n.reserve(capacity);
  rows.k_t1.reserve(capacity);
  rows.k_t2.reserve(capacity);
  rows.mu.reserve(capacity);
  rows.e.reserve(capacity);
  rows.bias.reserve(capacity);
  rows.bounce.reserve(capacity);
  rows.C.reserve(capacity);
  rows.jn.reserve(capacity);
  rows.jt1.reserve(capacity);
  rows.jt2.reserve(capacity);
  rows.indices.reserve(capacity);

  const double beta_dt = (params.dt > math::kEps) ? (params.beta / params.dt) : 0.0;

  for (std::size_t i = 0; i < contacts.size(); ++i) {
    const Contact& c = contacts[i];
    if (c.a < 0 || c.b < 0 || c.a >= static_cast<int>(bodies.size()) ||
        c.b >= static_cast<int>(bodies.size())) {
      continue;
    }

    const RigidBody& A = bodies[c.a];
    const RigidBody& B = bodies[c.b];

    Vec3 n = math::normalize_safe(c.n);
    Vec3 t1 = orthonormalize(n, c.t1);
    Vec3 t2 = math::cross(n, t1);
    t2 = math::normalize_safe(t2);
    if (math::length2(t2) <= math::kEps * math::kEps) {
      t1 = make_tangent(n);
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

    double violation = c.C;
    if (std::fabs(violation) <= math::kEps) {
      violation = 0.0;
    }
    const double bias = -beta_dt * std::max(0.0, -violation - params.slop);

    const double restitution = std::max(c.e, params.restitution);
    const double mu = std::max(c.mu, params.mu);

    const Vec3 ra_cross_n = math::cross(ra, n);
    const Vec3 rb_cross_n = math::cross(rb, n);
    const Vec3 ra_cross_t1 = math::cross(ra, t1);
    const Vec3 rb_cross_t1 = math::cross(rb, t1);
    const Vec3 ra_cross_t2 = math::cross(ra, t2);
    const Vec3 rb_cross_t2 = math::cross(rb, t2);

    const Vec3 TWn_a = A.invInertiaWorld * ra_cross_n;
    const Vec3 TWn_b = B.invInertiaWorld * rb_cross_n;
    double k_n = A.invMass + B.invMass;
    k_n += math::dot(ra_cross_n, TWn_a) + math::dot(rb_cross_n, TWn_b);
    if (k_n <= math::kEps) {
      k_n = 1.0;
    }

    const Vec3 TWt1_a = A.invInertiaWorld * ra_cross_t1;
    const Vec3 TWt1_b = B.invInertiaWorld * rb_cross_t1;
    double k_t1 = A.invMass + B.invMass;
    k_t1 += math::dot(ra_cross_t1, TWt1_a) + math::dot(rb_cross_t1, TWt1_b);
    if (k_t1 <= math::kEps) {
      k_t1 = 1.0;
    }

    const Vec3 TWt2_a = A.invInertiaWorld * ra_cross_t2;
    const Vec3 TWt2_b = B.invInertiaWorld * rb_cross_t2;
    double k_t2 = A.invMass + B.invMass;
    k_t2 += math::dot(ra_cross_t2, TWt2_a) + math::dot(rb_cross_t2, TWt2_b);
    if (k_t2 <= math::kEps) {
      k_t2 = 1.0;
    }

    const Vec3 va = A.v + math::cross(A.w, ra);
    const Vec3 vb = B.v + math::cross(B.w, rb);
    const double v_rel_n = math::dot(n, vb - va);
    const double bounce = (v_rel_n < 0.0) ? (-restitution * v_rel_n) : 0.0;

    rows.indices.push_back(static_cast<int>(i));
    rows.a.push_back(c.a);
    rows.b.push_back(c.b);
    rows.nx.push_back(n.x);
    rows.ny.push_back(n.y);
    rows.nz.push_back(n.z);
    rows.t1x.push_back(t1.x);
    rows.t1y.push_back(t1.y);
    rows.t1z.push_back(t1.z);
    rows.t2x.push_back(t2.x);
    rows.t2y.push_back(t2.y);
    rows.t2z.push_back(t2.z);
    rows.rax.push_back(ra.x);
    rows.ray.push_back(ra.y);
    rows.raz.push_back(ra.z);
    rows.rbx.push_back(rb.x);
    rows.rby.push_back(rb.y);
    rows.rbz.push_back(rb.z);
    rows.raxn_x.push_back(ra_cross_n.x);
    rows.raxn_y.push_back(ra_cross_n.y);
    rows.raxn_z.push_back(ra_cross_n.z);
    rows.rbxn_x.push_back(rb_cross_n.x);
    rows.rbxn_y.push_back(rb_cross_n.y);
    rows.rbxn_z.push_back(rb_cross_n.z);
    rows.raxt1_x.push_back(ra_cross_t1.x);
    rows.raxt1_y.push_back(ra_cross_t1.y);
    rows.raxt1_z.push_back(ra_cross_t1.z);
    rows.rbxt1_x.push_back(rb_cross_t1.x);
    rows.rbxt1_y.push_back(rb_cross_t1.y);
    rows.rbxt1_z.push_back(rb_cross_t1.z);
    rows.raxt2_x.push_back(ra_cross_t2.x);
    rows.raxt2_y.push_back(ra_cross_t2.y);
    rows.raxt2_z.push_back(ra_cross_t2.z);
    rows.rbxt2_x.push_back(rb_cross_t2.x);
    rows.rbxt2_y.push_back(rb_cross_t2.y);
    rows.rbxt2_z.push_back(rb_cross_t2.z);
    rows.TWn_a_x.push_back(TWn_a.x);
    rows.TWn_a_y.push_back(TWn_a.y);
    rows.TWn_a_z.push_back(TWn_a.z);
    rows.TWn_b_x.push_back(TWn_b.x);
    rows.TWn_b_y.push_back(TWn_b.y);
    rows.TWn_b_z.push_back(TWn_b.z);
    rows.TWt1_a_x.push_back(TWt1_a.x);
    rows.TWt1_a_y.push_back(TWt1_a.y);
    rows.TWt1_a_z.push_back(TWt1_a.z);
    rows.TWt1_b_x.push_back(TWt1_b.x);
    rows.TWt1_b_y.push_back(TWt1_b.y);
    rows.TWt1_b_z.push_back(TWt1_b.z);
    rows.TWt2_a_x.push_back(TWt2_a.x);
    rows.TWt2_a_y.push_back(TWt2_a.y);
    rows.TWt2_a_z.push_back(TWt2_a.z);
    rows.TWt2_b_x.push_back(TWt2_b.x);
    rows.TWt2_b_y.push_back(TWt2_b.y);
    rows.TWt2_b_z.push_back(TWt2_b.z);
    rows.k_n.push_back(k_n);
    rows.k_t1.push_back(k_t1);
    rows.k_t2.push_back(k_t2);
    const bool allow_warm_start = params.warm_start && (violation < -params.slop);
    rows.jn.push_back(allow_warm_start ? c.jn : 0.0);
    rows.jt1.push_back(allow_warm_start ? c.jt1 : 0.0);
    rows.jt2.push_back(allow_warm_start ? c.jt2 : 0.0);
    rows.mu.push_back(mu);
    rows.e.push_back(restitution);
    rows.bias.push_back(bias);
    rows.bounce.push_back(bounce);
    rows.C.push_back(violation);
  }

  rows.N = static_cast<int>(rows.a.size());
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

      const double nx = rows.nx[i];
      const double ny = rows.ny[i];
      const double nz = rows.nz[i];

      const double v_rel_n =
          nx * (B.v.x - A.v.x) + ny * (B.v.y - A.v.y) + nz * (B.v.z - A.v.z) +
          (B.w.x * rows.rbxn_x[i] + B.w.y * rows.rbxn_y[i] +
           B.w.z * rows.rbxn_z[i]) -
          (A.w.x * rows.raxn_x[i] + A.w.y * rows.raxn_y[i] +
           A.w.z * rows.raxn_z[i]);

      const double rhs = -(v_rel_n + rows.bias[i] - rows.bounce[i]);

      double delta_jn = rhs;
      if (rows.k_n[i] > math::kEps) {
        delta_jn /= rows.k_n[i];
      } else {
        delta_jn = 0.0;
      }
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

        A.v.x -= impulse_x * A.invMass;
        A.v.y -= impulse_y * A.invMass;
        A.v.z -= impulse_z * A.invMass;
        B.v.x += impulse_x * B.invMass;
        B.v.y += impulse_y * B.invMass;
        B.v.z += impulse_z * B.invMass;

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

      const double v_rel_t1 =
          t1x * (B.v.x - A.v.x) + t1y * (B.v.y - A.v.y) +
          t1z * (B.v.z - A.v.z) +
          (B.w.x * rows.rbxt1_x[i] + B.w.y * rows.rbxt1_y[i] +
           B.w.z * rows.rbxt1_z[i]) -
          (A.w.x * rows.raxt1_x[i] + A.w.y * rows.raxt1_y[i] +
           A.w.z * rows.raxt1_z[i]);

      const double v_rel_t2 =
          t2x * (B.v.x - A.v.x) + t2y * (B.v.y - A.v.y) +
          t2z * (B.v.z - A.v.z) +
          (B.w.x * rows.rbxt2_x[i] + B.w.y * rows.rbxt2_y[i] +
           B.w.z * rows.rbxt2_z[i]) -
          (A.w.x * rows.raxt2_x[i] + A.w.y * rows.raxt2_y[i] +
           A.w.z * rows.raxt2_z[i]);

      double jt1_candidate = rows.jt1[i];
      if (rows.k_t1[i] > math::kEps) {
        jt1_candidate += (-v_rel_t1) / rows.k_t1[i];
      }

      double jt2_candidate = rows.jt2[i];
      if (rows.k_t2[i] > math::kEps) {
        jt2_candidate += (-v_rel_t2) / rows.k_t2[i];
      }

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
