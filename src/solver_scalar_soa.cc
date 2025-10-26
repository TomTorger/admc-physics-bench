#include "solver_scalar_soa.hpp"

#include <algorithm>
#include <cmath>

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

}  // namespace

RowSOA build_soa(const std::vector<RigidBody>& bodies,
                 const std::vector<Contact>& contacts,
                 const SolverParams& params) {
  RowSOA rows;
  rows.a.reserve(contacts.size());
  rows.b.reserve(contacts.size());
  rows.n.reserve(contacts.size());
  rows.t1.reserve(contacts.size());
  rows.t2.reserve(contacts.size());
  rows.ra.reserve(contacts.size());
  rows.rb.reserve(contacts.size());
  rows.ra_cross_n.reserve(contacts.size());
  rows.rb_cross_n.reserve(contacts.size());
  rows.ra_cross_t1.reserve(contacts.size());
  rows.rb_cross_t1.reserve(contacts.size());
  rows.ra_cross_t2.reserve(contacts.size());
  rows.rb_cross_t2.reserve(contacts.size());
  rows.k_n.reserve(contacts.size());
  rows.k_t1.reserve(contacts.size());
  rows.k_t2.reserve(contacts.size());
  rows.jn.reserve(contacts.size());
  rows.jt1.reserve(contacts.size());
  rows.jt2.reserve(contacts.size());
  rows.mu.reserve(contacts.size());
  rows.e.reserve(contacts.size());
  rows.bias.reserve(contacts.size());
  rows.penetration.reserve(contacts.size());
  rows.indices.reserve(contacts.size());

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

    double penetration = c.penetration;
    if (std::fabs(penetration) <= math::kEps) {
      penetration = 0.0;
    }
    double depth = 0.0;
    if (penetration < -params.slop) {
      depth = -penetration - params.slop;
    }
    const double bias = -beta_dt * depth;

    const double restitution = std::max(c.e, params.restitution);
    const double mu = std::max(c.mu, params.mu);

    const Vec3 ra_cross_n = math::cross(ra, n);
    const Vec3 rb_cross_n = math::cross(rb, n);
    const Vec3 ra_cross_t1 = math::cross(ra, t1);
    const Vec3 rb_cross_t1 = math::cross(rb, t1);
    const Vec3 ra_cross_t2 = math::cross(ra, t2);
    const Vec3 rb_cross_t2 = math::cross(rb, t2);

    const Vec3 Iwa_n = A.invInertiaWorld * ra_cross_n;
    const Vec3 Iwb_n = B.invInertiaWorld * rb_cross_n;
    double k_n = A.invMass + B.invMass;
    k_n += math::dot(ra_cross_n, Iwa_n) + math::dot(rb_cross_n, Iwb_n);
    if (k_n <= math::kEps) {
      k_n = 1.0;
    }

    const Vec3 Iwa_t1 = A.invInertiaWorld * ra_cross_t1;
    const Vec3 Iwb_t1 = B.invInertiaWorld * rb_cross_t1;
    double k_t1 = A.invMass + B.invMass;
    k_t1 += math::dot(ra_cross_t1, Iwa_t1) + math::dot(rb_cross_t1, Iwb_t1);
    if (k_t1 <= math::kEps) {
      k_t1 = 1.0;
    }

    const Vec3 Iwa_t2 = A.invInertiaWorld * ra_cross_t2;
    const Vec3 Iwb_t2 = B.invInertiaWorld * rb_cross_t2;
    double k_t2 = A.invMass + B.invMass;
    k_t2 += math::dot(ra_cross_t2, Iwa_t2) + math::dot(rb_cross_t2, Iwb_t2);
    if (k_t2 <= math::kEps) {
      k_t2 = 1.0;
    }

    rows.indices.push_back(static_cast<int>(i));
    rows.a.push_back(c.a);
    rows.b.push_back(c.b);
    rows.n.push_back(n);
    rows.t1.push_back(t1);
    rows.t2.push_back(t2);
    rows.ra.push_back(ra);
    rows.rb.push_back(rb);
    rows.ra_cross_n.push_back(ra_cross_n);
    rows.rb_cross_n.push_back(rb_cross_n);
    rows.ra_cross_t1.push_back(ra_cross_t1);
    rows.rb_cross_t1.push_back(rb_cross_t1);
    rows.ra_cross_t2.push_back(ra_cross_t2);
    rows.rb_cross_t2.push_back(rb_cross_t2);
    rows.k_n.push_back(k_n);
    rows.k_t1.push_back(k_t1);
    rows.k_t2.push_back(k_t2);
    rows.jn.push_back(params.warm_start ? c.jn : 0.0);
    rows.jt1.push_back(params.warm_start ? c.jt1 : 0.0);
    rows.jt2.push_back(params.warm_start ? c.jt2 : 0.0);
    rows.mu.push_back(mu);
    rows.e.push_back(restitution);
    rows.bias.push_back(bias);
    rows.penetration.push_back(penetration);
  }

  return rows;
}

void solve_scalar_soa(std::vector<RigidBody>& bodies,
                      std::vector<Contact>& contacts,
                      RowSOA& rows,
                      JointSOA& joints,
                      const SolverParams& params) {
  const int iterations = std::max(1, params.iterations);

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
        continue;
      }

      RigidBody& A = bodies[ia];
      RigidBody& B = bodies[ib];

      const Vec3 impulse = rows.n[i] * rows.jn[i] + rows.t1[i] * rows.jt1[i] +
                           rows.t2[i] * rows.jt2[i];
      if (math::length2(impulse) <= math::kEps) {
        continue;
      }
      A.applyImpulse(-impulse, rows.ra[i]);
      B.applyImpulse(impulse, rows.rb[i]);
    }

    for (std::size_t i = 0; i < joints.size(); ++i) {
      const int ia = joints.a[i];
      const int ib = joints.b[i];
      if (ia < 0 || ib < 0 || ia >= static_cast<int>(bodies.size()) ||
          ib >= static_cast<int>(bodies.size())) {
        continue;
      }

      if (std::fabs(joints.j[i]) <= math::kEps) {
        continue;
      }

      RigidBody& A = bodies[ia];
      RigidBody& B = bodies[ib];
      const Vec3 impulse = joints.d[i] * joints.j[i];
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
        continue;
      }

      RigidBody& A = bodies[ia];
      RigidBody& B = bodies[ib];

      const Vec3 va = A.v + math::cross(A.w, rows.ra[i]);
      const Vec3 vb = B.v + math::cross(B.w, rows.rb[i]);
      const Vec3 v_rel = vb - va;
      const double v_rel_n = math::dot(rows.n[i], v_rel);

      double target = rows.bias[i];
      if (v_rel_n < 0.0) {
        target += -rows.e[i] * v_rel_n;
      }

      const double delta_jn = (target - v_rel_n) / rows.k_n[i];
      const double jn_old = rows.jn[i];
      rows.jn[i] = std::max(0.0, rows.jn[i] + delta_jn);
      const double applied_n = rows.jn[i] - jn_old;
      if (std::fabs(applied_n) > math::kEps) {
        const Vec3 impulse_n = applied_n * rows.n[i];
        A.applyImpulse(-impulse_n, rows.ra[i]);
        B.applyImpulse(impulse_n, rows.rb[i]);
      }

      const Vec3 va_f = A.v + math::cross(A.w, rows.ra[i]);
      const Vec3 vb_f = B.v + math::cross(B.w, rows.rb[i]);
      const Vec3 v_rel_f = vb_f - va_f;
      const double v_rel_t1 = math::dot(rows.t1[i], v_rel_f);
      const double v_rel_t2 = math::dot(rows.t2[i], v_rel_f);

      double jt1_candidate = rows.jt1[i];
      if (rows.k_t1[i] > math::kEps) {
        jt1_candidate += (-v_rel_t1) / rows.k_t1[i];
      }

      double jt2_candidate = rows.jt2[i];
      if (rows.k_t2[i] > math::kEps) {
        jt2_candidate += (-v_rel_t2) / rows.k_t2[i];
      }

      const double friction_max = rows.mu[i] * std::max(rows.jn[i], 0.0);
      const double jt_mag = std::sqrt(jt1_candidate * jt1_candidate +
                                      jt2_candidate * jt2_candidate);
      double scale = 1.0;
      if (jt_mag > friction_max && jt_mag > math::kEps) {
        scale = (friction_max > 0.0) ? (friction_max / jt_mag) : 0.0;
      }

      jt1_candidate *= scale;
      jt2_candidate *= scale;

      const double delta_jt1 = jt1_candidate - rows.jt1[i];
      const double delta_jt2 = jt2_candidate - rows.jt2[i];
      rows.jt1[i] = jt1_candidate;
      rows.jt2[i] = jt2_candidate;

      if (std::fabs(delta_jt1) > math::kEps || std::fabs(delta_jt2) > math::kEps) {
        const Vec3 impulse_t = delta_jt1 * rows.t1[i] + delta_jt2 * rows.t2[i];
        A.applyImpulse(-impulse_t, rows.ra[i]);
        B.applyImpulse(impulse_t, rows.rb[i]);
      }
    }

    for (std::size_t i = 0; i < joints.size(); ++i) {
      const int ia = joints.a[i];
      const int ib = joints.b[i];
      if (ia < 0 || ib < 0 || ia >= static_cast<int>(bodies.size()) ||
          ib >= static_cast<int>(bodies.size())) {
        continue;
      }

      const double denom = joints.k[i] + joints.gamma[i];
      if (denom <= math::kEps) {
        continue;
      }

      RigidBody& A = bodies[ia];
      RigidBody& B = bodies[ib];
      const Vec3 va = A.v + math::cross(A.w, joints.ra[i]);
      const Vec3 vb = B.v + math::cross(B.w, joints.rb[i]);
      const double v_rel_d = math::dot(joints.d[i], vb - va);

      double j_new = joints.j[i] - (v_rel_d + joints.bias[i]) / denom;
      if (joints.rope[i] && j_new < 0.0) {
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
      continue;
    }
    Contact& c = contacts[static_cast<std::size_t>(idx)];
    c.a = rows.a[i];
    c.b = rows.b[i];
    c.n = rows.n[i];
    c.t1 = rows.t1[i];
    c.t2 = rows.t2[i];
    c.ra = rows.ra[i];
    c.rb = rows.rb[i];
    c.ra_cross_n = rows.ra_cross_n[i];
    c.rb_cross_n = rows.rb_cross_n[i];
    c.ra_cross_t1 = rows.ra_cross_t1[i];
    c.rb_cross_t1 = rows.rb_cross_t1[i];
    c.ra_cross_t2 = rows.ra_cross_t2[i];
    c.rb_cross_t2 = rows.rb_cross_t2[i];
    c.k_n = rows.k_n[i];
    c.k_t1 = rows.k_t1[i];
    c.k_t2 = rows.k_t2[i];
    c.jn = rows.jn[i];
    c.jt1 = rows.jt1[i];
    c.jt2 = rows.jt2[i];
    c.mu = rows.mu[i];
    c.e = rows.e[i];
    c.bias = rows.bias[i];
    c.penetration = rows.penetration[i];
  }

  for (RigidBody& body : bodies) {
    body.integrate(params.dt);
  }
}

void solve_scalar_soa(std::vector<RigidBody>& bodies,
                      std::vector<Contact>& contacts,
                      RowSOA& rows,
                      const SolverParams& params) {
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
  solve_scalar_soa(bodies, contacts, rows, empty_joints, params);
}
