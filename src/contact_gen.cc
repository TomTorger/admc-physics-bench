#include "contact_gen.hpp"

#include <algorithm>
#include <cmath>

namespace {

struct ContactBuildParams {
  double beta_dt = 0.0;
  double slop = 0.0;
  double restitution = 0.0;
  double mu = 0.0;
};

math::Vec3 fallback_tangent(const math::Vec3& n) {
  if (std::fabs(n.x) < 0.57735026919) {
    return math::normalize_safe(math::cross(math::Vec3(1.0, 0.0, 0.0), n));
  }
  return math::normalize_safe(math::cross(math::Vec3(0.0, 1.0, 0.0), n));
}

math::Vec3 orthonormal_tangent(const math::Vec3& n, const math::Vec3& t) {
  math::Vec3 tangent = math::normalize_safe(t);
  if (math::length2(tangent) <= math::kEps * math::kEps) {
    tangent = fallback_tangent(n);
  }
  return tangent;
}

void build_impl(std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
                const ContactBuildParams& params) {
  for (Contact& c : contacts) {
    if (c.a < 0 || c.b < 0 || c.a >= static_cast<int>(bodies.size()) ||
        c.b >= static_cast<int>(bodies.size())) {
      continue;
    }

    RigidBody& A = bodies[c.a];
    RigidBody& B = bodies[c.b];

    c.n = math::normalize_safe(c.n);
    c.t1 = orthonormal_tangent(c.n, c.t1);
    c.t2 = math::normalize_safe(math::cross(c.n, c.t1));
    if (math::length2(c.t2) <= math::kEps * math::kEps) {
      c.t1 = fallback_tangent(c.n);
      c.t2 = math::normalize_safe(math::cross(c.n, c.t1));
    }

    c.ra = c.p - A.x;
    c.rb = c.p - B.x;

    if (std::fabs(c.penetration) <= math::kEps) {
      c.penetration = 0.0;
    }

    double depth = 0.0;
    if (c.penetration < -params.slop) {
      depth = -c.penetration - params.slop;
    }
    c.bias = -params.beta_dt * depth;
    c.e = std::max(c.e, params.restitution);
    c.mu = std::max(c.mu, params.mu);

    c.ra_cross_n = math::cross(c.ra, c.n);
    c.rb_cross_n = math::cross(c.rb, c.n);
    c.ra_cross_t1 = math::cross(c.ra, c.t1);
    c.rb_cross_t1 = math::cross(c.rb, c.t1);
    c.ra_cross_t2 = math::cross(c.ra, c.t2);
    c.rb_cross_t2 = math::cross(c.rb, c.t2);

    const math::Vec3 Iwa_n = A.invInertiaWorld * c.ra_cross_n;
    const math::Vec3 Iwb_n = B.invInertiaWorld * c.rb_cross_n;
    double k_n = A.invMass + B.invMass;
    k_n += math::dot(c.ra_cross_n, Iwa_n) + math::dot(c.rb_cross_n, Iwb_n);
    c.k_n = (k_n > math::kEps) ? k_n : 1.0;

    const math::Vec3 Iwa_t1 = A.invInertiaWorld * c.ra_cross_t1;
    const math::Vec3 Iwb_t1 = B.invInertiaWorld * c.rb_cross_t1;
    double k_t1 = A.invMass + B.invMass;
    k_t1 += math::dot(c.ra_cross_t1, Iwa_t1) + math::dot(c.rb_cross_t1, Iwb_t1);
    c.k_t1 = (k_t1 > math::kEps) ? k_t1 : 1.0;

    const math::Vec3 Iwa_t2 = A.invInertiaWorld * c.ra_cross_t2;
    const math::Vec3 Iwb_t2 = B.invInertiaWorld * c.rb_cross_t2;
    double k_t2 = A.invMass + B.invMass;
    k_t2 += math::dot(c.ra_cross_t2, Iwa_t2) + math::dot(c.rb_cross_t2, Iwb_t2);
    c.k_t2 = (k_t2 > math::kEps) ? k_t2 : 1.0;
  }
}

}  // namespace

void build_contact_offsets_and_bias(std::vector<RigidBody>& bodies,
                                    std::vector<Contact>& contacts,
                                    const BaselineParams& params) {
  ContactBuildParams p;
  p.beta_dt = (params.dt > math::kEps) ? (params.beta / params.dt) : 0.0;
  p.slop = params.slop;
  p.restitution = params.restitution;
  p.mu = 0.0;
  build_impl(bodies, contacts, p);
}

void build_contact_offsets_and_bias(std::vector<RigidBody>& bodies,
                                    std::vector<Contact>& contacts,
                                    const SolverParams& params) {
  ContactBuildParams p;
  p.beta_dt = (params.dt > math::kEps) ? (params.beta / params.dt) : 0.0;
  p.slop = params.slop;
  p.restitution = params.restitution;
  p.mu = params.mu;
  build_impl(bodies, contacts, p);
}

