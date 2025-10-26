#include "solver_baseline_vec.hpp"

#include <algorithm>
#include <cmath>

using math::Vec3;

void solve_baseline(std::vector<RigidBody>& bodies,
                    std::vector<Contact>& contacts,
                    const BaselineParams& params) {
  const int iterations = std::max(1, params.iterations);

  for (RigidBody& body : bodies) {
    body.syncDerived();
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

      const double restitution = std::max(c.e, params.restitution);
      const double bias = c.bias;
      if (v_rel_n > 0.0 && restitution <= math::kEps && std::fabs(bias) <= math::kEps) {
        continue;
      }

      const Vec3 ra_cross_n = math::cross(c.ra, c.n);
      const Vec3 rb_cross_n = math::cross(c.rb, c.n);
      const Vec3 Iwa = A.invInertiaWorld * ra_cross_n;
      const Vec3 Iwb = B.invInertiaWorld * rb_cross_n;

      double k_n = A.invMass + B.invMass;
      k_n += math::dot(ra_cross_n, Iwa) + math::dot(rb_cross_n, Iwb);
      if (k_n <= math::kEps) {
        continue;
      }

      double target = bias;
      if (v_rel_n < 0.0) {
        target += -restitution * v_rel_n;
      }

      const double delta_j = (target - v_rel_n) / k_n;
      const double j = std::max(0.0, delta_j);
      if (math::nearly_zero(j)) {
        continue;
      }

      const Vec3 impulse = j * c.n;
      A.applyImpulse(-impulse, c.ra);
      B.applyImpulse(impulse, c.rb);
    }
  }

  for (RigidBody& body : bodies) {
    body.integrate(params.dt);
  }
}

