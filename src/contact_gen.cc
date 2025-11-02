#include "contact_gen.hpp"

#include "config/runtime_env.hpp"
#include "mt/thread_pool.hpp"

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

bool refresh_contact_from_state_impl(const std::vector<RigidBody>& bodies,
                                     Contact& c) {
  if (c.a < 0 || c.b < 0 || c.a >= static_cast<int>(bodies.size()) ||
      c.b >= static_cast<int>(bodies.size())) {
    return false;
  }

  const RigidBody& A = bodies[c.a];
  const RigidBody& B = bodies[c.b];

  const bool a_static = A.invMass <= math::kEps;
  const bool b_static = B.invMass <= math::kEps;

  if (c.type == Contact::Type::kUnknown) {
    if (a_static != b_static) {
      c.type = Contact::Type::kSpherePlane;
    } else {
      c.type = Contact::Type::kSphereSphere;
    }
  }

  if (c.type == Contact::Type::kSpherePlane || a_static || b_static) {
    const bool sphere_is_b = a_static && !b_static;
    const bool sphere_is_a = !a_static && b_static;
    const math::Vec3 plane_normal = math::normalize_safe(
        sphere_is_b ? c.n : (sphere_is_a ? -c.n : c.n));

    double plane_offset = c.plane_offset;
    if (!std::isfinite(plane_offset)) {
      plane_offset = math::dot(plane_normal, c.p);
    }
    c.plane_offset = plane_offset;

    const RigidBody& sphere = sphere_is_b ? B : A;
    double& radius = sphere_is_b ? c.radius_b : c.radius_a;
    if (radius <= math::kEps) {
      radius = math::length(c.p - sphere.x);
    }

    const double dist = math::dot(plane_normal, sphere.x) - plane_offset;
    c.C = dist - radius;

    const double correction = radius - dist;
    const math::Vec3 contact_point = sphere.x - correction * plane_normal;

    if (sphere_is_b) {
      c.n = plane_normal;
    } else {
      c.n = -plane_normal;
    }
    c.p = contact_point;
    c.ra = c.p - A.x;
    c.rb = c.p - B.x;
  } else {
    math::Vec3 delta = B.x - A.x;
    double distance = math::length(delta);
    if (distance <= 1e-12) {
      distance = 1e-12;
      if (math::length2(c.n) <= math::kEps * math::kEps) {
        c.n = math::Vec3(1.0, 0.0, 0.0);
      }
      delta = c.n * distance;
    }
    const math::Vec3 normal = math::normalize_safe(delta);

    double& radius_a = c.radius_a;
    double& radius_b = c.radius_b;
    if (radius_a <= math::kEps) {
      radius_a = math::length(c.p - A.x);
    }
    if (radius_b <= math::kEps) {
      radius_b = math::length(c.p - B.x);
    }
    if (radius_a <= math::kEps) {
      radius_a = radius_b;
    }
    if (radius_b <= math::kEps) {
      radius_b = radius_a;
    }

    c.C = distance - (radius_a + radius_b);

    const double correction =
        radius_a - 0.5 * (radius_a + radius_b - distance);
    const math::Vec3 contact_point =
        0.5 * (A.x + B.x) + correction * normal;

    c.n = normal;
    c.p = contact_point;
    c.ra = c.p - A.x;
    c.rb = c.p - B.x;
  }

  if (std::fabs(c.C) <= math::kEps) {
    c.C = 0.0;
  }

  return true;
}

void build_impl(std::vector<RigidBody>& bodies, std::vector<Contact>& contacts,
                const ContactBuildParams& params) {
  const std::size_t count = contacts.size();
  if (count == 0) {
    return;
  }

  auto per_contact = [&](std::size_t index) {
    Contact& c = contacts[index];
    if (!refresh_contact_from_state_impl(bodies, c)) {
      return;
    }

    RigidBody& A = bodies[c.a];
    RigidBody& B = bodies[c.b];

    c.t1 = orthonormal_tangent(c.n, c.t1);
    c.t2 = math::normalize_safe(math::cross(c.n, c.t1));
    if (math::length2(c.t2) <= math::kEps * math::kEps) {
      c.t1 = fallback_tangent(c.n);
      c.t2 = math::normalize_safe(math::cross(c.n, c.t1));
    }

    c.bias = -params.beta_dt * std::max(0.0, -c.C - params.slop);
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
    c.TWn_a = Iwa_n;
    c.TWn_b = Iwb_n;

    const math::Vec3 Iwa_t1 = A.invInertiaWorld * c.ra_cross_t1;
    const math::Vec3 Iwb_t1 = B.invInertiaWorld * c.rb_cross_t1;
    double k_t1 = A.invMass + B.invMass;
    k_t1 += math::dot(c.ra_cross_t1, Iwa_t1) + math::dot(c.rb_cross_t1, Iwb_t1);
    c.k_t1 = (k_t1 > math::kEps) ? k_t1 : 1.0;
    c.TWt1_a = Iwa_t1;
    c.TWt1_b = Iwb_t1;

    const math::Vec3 Iwa_t2 = A.invInertiaWorld * c.ra_cross_t2;
    const math::Vec3 Iwb_t2 = B.invInertiaWorld * c.rb_cross_t2;
    double k_t2 = A.invMass + B.invMass;
    k_t2 += math::dot(c.ra_cross_t2, Iwa_t2) + math::dot(c.rb_cross_t2, Iwb_t2);
    c.k_t2 = (k_t2 > math::kEps) ? k_t2 : 1.0;
    c.TWt2_a = Iwa_t2;
    c.TWt2_b = Iwb_t2;
  };

  auto& pool = admc::mt::ThreadPool::instance();
  const std::size_t chunk = admc::config::chunk_size();
  if (pool.size() <= 1 || count <= chunk) {
    for (std::size_t i = 0; i < count; ++i) {
      per_contact(i);
    }
    return;
  }

  pool.parallel_for(count, per_contact, chunk);
}

}  // namespace

void refresh_contacts_from_state(const std::vector<RigidBody>& bodies,
                                 std::vector<Contact>& contacts) {
  const std::size_t count = contacts.size();
  if (count == 0) {
    return;
  }

  auto per_contact = [&](std::size_t index) {
    refresh_contact_from_state_impl(bodies, contacts[index]);
  };

  auto& pool = admc::mt::ThreadPool::instance();
  const std::size_t chunk = admc::config::chunk_size();
  if (pool.size() <= 1 || count <= chunk) {
    for (std::size_t i = 0; i < count; ++i) {
      per_contact(i);
    }
    return;
  }

  pool.parallel_for(count, per_contact, chunk);
}

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

