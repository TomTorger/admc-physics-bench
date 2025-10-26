#include "solver_scalar_cached.hpp"

#include <algorithm>
#include <cmath>

namespace {
using math::Vec3;

Vec3 ensure_tangent(const Vec3& n, const Vec3& candidate) {
  const Vec3 t = math::normalize_safe(candidate);
  if (math::length2(t) > math::kEps * math::kEps) {
    return t;
  }
  Vec3 axis = std::fabs(n.x) > 0.707 ? Vec3(0.0, 1.0, 0.0) : Vec3(1.0, 0.0, 0.0);
  Vec3 tangent = math::cross(axis, n);
  return math::normalize_safe(tangent);
}

Vec3 generate_t1(const Vec3& n) {
  if (std::fabs(n.x) < 0.57735026919) {
    return math::normalize_safe(math::cross(Vec3(1.0, 0.0, 0.0), n));
  }
  return math::normalize_safe(math::cross(Vec3(0.0, 1.0, 0.0), n));
}

}  // namespace

void solve_scalar_cached(std::vector<RigidBody>& bodies,
                         std::vector<Contact>& contacts,
                         const SolverParams& params) {
  const int iterations = std::max(1, params.iterations);

  for (RigidBody& body : bodies) {
    body.syncDerived();
  }

  const double beta_dt = (params.dt > math::kEps) ? (params.beta / params.dt) : 0.0;

  for (Contact& c : contacts) {
    if (c.a < 0 || c.b < 0 || c.a >= static_cast<int>(bodies.size()) ||
        c.b >= static_cast<int>(bodies.size())) {
      continue;
    }

    RigidBody& A = bodies[c.a];
    RigidBody& B = bodies[c.b];

    c.n = math::normalize_safe(c.n);
    if (math::length2(c.t1) <= math::kEps * math::kEps) {
      c.t1 = generate_t1(c.n);
    } else {
      c.t1 = ensure_tangent(c.n, c.t1);
    }
    c.t2 = math::normalize_safe(math::cross(c.n, c.t1));

    c.ra = c.p - A.x;
    c.rb = c.p - B.x;

    if (std::fabs(c.penetration) <= math::kEps) {
      c.penetration = 0.0;
    }

    double depth = 0.0;
    if (c.penetration < -params.slop) {
      depth = -c.penetration - params.slop;
    }
    c.bias = -beta_dt * depth;
    c.e = std::max(c.e, params.restitution);
    c.mu = std::max(c.mu, params.mu);

    c.ra_cross_n = math::cross(c.ra, c.n);
    c.rb_cross_n = math::cross(c.rb, c.n);
    c.ra_cross_t1 = math::cross(c.ra, c.t1);
    c.rb_cross_t1 = math::cross(c.rb, c.t1);
    c.ra_cross_t2 = math::cross(c.ra, c.t2);
    c.rb_cross_t2 = math::cross(c.rb, c.t2);

    const Vec3 Iwa_n = A.invInertiaWorld * c.ra_cross_n;
    const Vec3 Iwb_n = B.invInertiaWorld * c.rb_cross_n;
    double k_n = A.invMass + B.invMass;
    k_n += math::dot(c.ra_cross_n, Iwa_n) + math::dot(c.rb_cross_n, Iwb_n);
    c.k_n = (k_n > math::kEps) ? k_n : 1.0;

    const Vec3 Iwa_t1 = A.invInertiaWorld * c.ra_cross_t1;
    const Vec3 Iwb_t1 = B.invInertiaWorld * c.rb_cross_t1;
    double k_t1 = A.invMass + B.invMass;
    k_t1 += math::dot(c.ra_cross_t1, Iwa_t1) + math::dot(c.rb_cross_t1, Iwb_t1);
    c.k_t1 = (k_t1 > math::kEps) ? k_t1 : 1.0;

    const Vec3 Iwa_t2 = A.invInertiaWorld * c.ra_cross_t2;
    const Vec3 Iwb_t2 = B.invInertiaWorld * c.rb_cross_t2;
    double k_t2 = A.invMass + B.invMass;
    k_t2 += math::dot(c.ra_cross_t2, Iwa_t2) + math::dot(c.rb_cross_t2, Iwb_t2);
    c.k_t2 = (k_t2 > math::kEps) ? k_t2 : 1.0;

    if (!params.warm_start) {
      c.jn = 0.0;
      c.jt1 = 0.0;
      c.jt2 = 0.0;
    }
  }

  if (params.warm_start) {
    for (Contact& c : contacts) {
      if (c.a < 0 || c.b < 0 || c.a >= static_cast<int>(bodies.size()) ||
          c.b >= static_cast<int>(bodies.size())) {
        continue;
      }
      RigidBody& A = bodies[c.a];
      RigidBody& B = bodies[c.b];
      const Vec3 impulse = c.n * c.jn + c.t1 * c.jt1 + c.t2 * c.jt2;
      if (math::length2(impulse) <= math::kEps) {
        continue;
      }
      A.applyImpulse(-impulse, c.ra);
      B.applyImpulse(impulse, c.rb);
    }
  }

  for (int it = 0; it < iterations; ++it) {
    for (Contact& c : contacts) {
      if (c.a < 0 || c.b < 0 || c.a >= static_cast<int>(bodies.size()) ||
          c.b >= static_cast<int>(bodies.size())) {
        continue;
      }

      RigidBody& A = bodies[c.a];
      RigidBody& B = bodies[c.b];

      const Vec3 va = A.v + math::cross(A.w, c.ra);
      const Vec3 vb = B.v + math::cross(B.w, c.rb);
      const Vec3 v_rel = vb - va;
      const double v_rel_n = math::dot(c.n, v_rel);

      double target = c.bias;
      if (v_rel_n < 0.0) {
        target += -c.e * v_rel_n;
      }

      const double delta_jn = (target - v_rel_n) / c.k_n;
      const double jn_old = c.jn;
      c.jn = std::max(0.0, c.jn + delta_jn);
      const double applied_n = c.jn - jn_old;
      if (std::fabs(applied_n) > math::kEps) {
        const Vec3 impulse_n = applied_n * c.n;
        A.applyImpulse(-impulse_n, c.ra);
        B.applyImpulse(impulse_n, c.rb);
      }

      const Vec3 va_f = A.v + math::cross(A.w, c.ra);
      const Vec3 vb_f = B.v + math::cross(B.w, c.rb);
      const Vec3 v_rel_f = vb_f - va_f;
      const double v_rel_t1 = math::dot(c.t1, v_rel_f);
      const double v_rel_t2 = math::dot(c.t2, v_rel_f);

      double jt1_candidate = c.jt1;
      if (c.k_t1 > math::kEps) {
        jt1_candidate += (-v_rel_t1) / c.k_t1;
      }

      double jt2_candidate = c.jt2;
      if (c.k_t2 > math::kEps) {
        jt2_candidate += (-v_rel_t2) / c.k_t2;
      }

      const double friction_max = c.mu * std::max(c.jn, 0.0);
      const double jt_mag = std::sqrt(jt1_candidate * jt1_candidate +
                                      jt2_candidate * jt2_candidate);
      double scale = 1.0;
      if (jt_mag > friction_max && jt_mag > math::kEps) {
        scale = (friction_max > 0.0) ? (friction_max / jt_mag) : 0.0;
      }

      jt1_candidate *= scale;
      jt2_candidate *= scale;

      const double delta_jt1 = jt1_candidate - c.jt1;
      const double delta_jt2 = jt2_candidate - c.jt2;

      c.jt1 = jt1_candidate;
      c.jt2 = jt2_candidate;

      if (std::fabs(delta_jt1) > math::kEps || std::fabs(delta_jt2) > math::kEps) {
        const Vec3 impulse_t = delta_jt1 * c.t1 + delta_jt2 * c.t2;
        A.applyImpulse(-impulse_t, c.ra);
        B.applyImpulse(impulse_t, c.rb);
      }
    }
  }

  for (RigidBody& body : bodies) {
    body.integrate(params.dt);
  }
}
