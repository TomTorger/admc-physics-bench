#include "contact_gen.hpp"

#include <algorithm>

void build_contact_offsets_and_bias(std::vector<RigidBody>& bodies,
                                    std::vector<Contact>& contacts,
                                    const BaselineParams& params) {
  const double beta_dt = (params.dt > math::kEps) ? (params.beta / params.dt) : 0.0;

  for (Contact& c : contacts) {
    if (c.a < 0 || c.b < 0 || c.a >= static_cast<int>(bodies.size()) ||
        c.b >= static_cast<int>(bodies.size())) {
      continue;
    }

    RigidBody& A = bodies[c.a];
    RigidBody& B = bodies[c.b];

    c.n = math::normalize_safe(c.n);
    c.ra = c.p - A.x;
    c.rb = c.p - B.x;

    const double C = c.penetration;
    double depth = 0.0;
    if (C < -params.slop) {
      depth = -C - params.slop;
    }
    c.bias = -beta_dt * depth;
    c.e = std::max(c.e, params.restitution);
  }
}

