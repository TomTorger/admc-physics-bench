#include "solver_scalar_cached.hpp"

#include "contact_gen.hpp"

#include <algorithm>
#include <cmath>

namespace {
using math::Vec3;

Vec3 relative_velocity(const RigidBody& A, const RigidBody& B, const Contact& c) {
  const Vec3 va = A.v + math::cross(A.w, c.ra);
  const Vec3 vb = B.v + math::cross(B.w, c.rb);
  return vb - va;
}

void apply_impulse_pair(RigidBody& A, RigidBody& B, const Contact& c, const Vec3& impulse) {
  A.applyImpulse(-impulse, c.ra);
  B.applyImpulse(impulse, c.rb);
}
}  // namespace

void solve_scalar_cached(std::vector<RigidBody>& bodies,
                         std::vector<Contact>& contacts,
                         const ScalarParams& params) {
  const int iterations = std::max(1, params.iterations);
  preprocess_contacts(bodies, contacts, params.beta, params.slop, params.dt);

  if (params.warm_start) {
    for (Contact& c : contacts) {
      if (c.a < 0 || c.b < 0 || c.a >= static_cast<int>(bodies.size()) ||
          c.b >= static_cast<int>(bodies.size())) {
        continue;
      }
      const Vec3 impulse = c.jn * c.n + c.jt1 * c.t1 + c.jt2 * c.t2;
      if (math::length2(impulse) <= math::kEps) {
        continue;
      }
      apply_impulse_pair(bodies[c.a], bodies[c.b], c, impulse);
    }
  } else {
    for (Contact& c : contacts) {
      c.jn = 0.0;
      c.jt1 = 0.0;
      c.jt2 = 0.0;
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

      Vec3 v_rel = relative_velocity(A, B, c);
      const double restitution = std::max(c.e, params.restitution);

      if (c.k_n > math::kEps) {
        const double v_rel_n = math::dot(c.n, v_rel);
        double target = c.bias;
        if (v_rel_n < 0.0) {
          target += -restitution * v_rel_n;
        }
        double delta_jn = (target - v_rel_n) / c.k_n;
        const double new_jn = std::max(0.0, c.jn + delta_jn);
        delta_jn = new_jn - c.jn;
        c.jn = new_jn;
        if (!math::nearly_zero(delta_jn)) {
          const Vec3 impulse = delta_jn * c.n;
          apply_impulse_pair(A, B, c, impulse);
          v_rel = relative_velocity(A, B, c);
        }
      }

      if (c.k_t1 > math::kEps) {
        const double v_rel_t1 = math::dot(c.t1, v_rel);
        double delta_jt1 = -v_rel_t1 / c.k_t1;
        c.jt1 += delta_jt1;
        if (!math::nearly_zero(delta_jt1)) {
          const Vec3 impulse = delta_jt1 * c.t1;
          apply_impulse_pair(A, B, c, impulse);
          v_rel = relative_velocity(A, B, c);
        }
      }

      if (c.k_t2 > math::kEps) {
        const double v_rel_t2 = math::dot(c.t2, v_rel);
        double delta_jt2 = -v_rel_t2 / c.k_t2;
        c.jt2 += delta_jt2;
        if (!math::nearly_zero(delta_jt2)) {
          const Vec3 impulse = delta_jt2 * c.t2;
          apply_impulse_pair(A, B, c, impulse);
        }
      }

      const double jt_mag = std::sqrt(c.jt1 * c.jt1 + c.jt2 * c.jt2);
      const double max_friction = c.mu * c.jn;
      if (jt_mag > max_friction + math::kEps && jt_mag > math::kEps) {
        const double scale = (max_friction > math::kEps) ? (max_friction / jt_mag) : 0.0;
        const double new_jt1 = c.jt1 * scale;
        const double new_jt2 = c.jt2 * scale;
        const Vec3 correction = (new_jt1 - c.jt1) * c.t1 + (new_jt2 - c.jt2) * c.t2;
        c.jt1 = new_jt1;
        c.jt2 = new_jt2;
        if (math::length2(correction) > math::kEps) {
          apply_impulse_pair(A, B, c, correction);
        }
      }
    }
  }

  for (RigidBody& body : bodies) {
    body.integrate(params.dt);
  }
}
