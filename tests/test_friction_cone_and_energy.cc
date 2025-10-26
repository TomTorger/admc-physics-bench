#include "contact_gen.hpp"
#include "metrics.hpp"
#include "solver_scalar_cached.hpp"

#include "math.hpp"

#include <cassert>
#include <cmath>
#include <vector>

namespace {
RigidBody make_static_body() {
  RigidBody b;
  b.invMass = 0.0;
  b.invInertiaLocal = math::Mat3();
  b.syncDerived();
  return b;
}

RigidBody make_dynamic_body(const math::Vec3& pos, const math::Vec3& vel) {
  RigidBody b;
  b.x = pos;
  b.v = vel;
  b.invMass = 1.0;
  b.invInertiaLocal = math::Mat3::identity();
  b.syncDerived();
  return b;
}

void simulate_steps(std::vector<RigidBody>& bodies,
                    std::vector<Contact>& contacts,
                    const ScalarParams& params,
                    int steps) {
  for (int i = 0; i < steps; ++i) {
    std::vector<RigidBody> pre = bodies;
    solve_scalar_cached(bodies, contacts, params);
    const Energy e = kinetic_energy_delta(pre, bodies);
    assert(e.delta <= 1e-8);
    for (const Contact& c : contacts) {
      const double jt_mag = std::sqrt(c.jt1 * c.jt1 + c.jt2 * c.jt2);
      const double cone = c.mu * c.jn + 1e-12;
      assert(jt_mag <= cone + 1e-12);
    }
  }
}
}

int main() {
  ScalarParams params;
  params.iterations = 40;
  params.beta = 0.0;
  params.slop = 0.0;
  params.restitution = 0.0;
  params.dt = 1.0 / 120.0;
  params.warm_start = true;

  {
    std::vector<RigidBody> bodies;
    bodies.push_back(make_static_body());
    bodies.push_back(make_dynamic_body(math::Vec3(0.0, 1.0, 0.0), math::Vec3(2.0, 0.0, 0.0)));

    Contact c;
    c.a = 0;
    c.b = 1;
    c.p = math::Vec3(0.0, 0.0, 0.0);
    c.n = math::Vec3(0.0, 1.0, 0.0);
    c.mu = 0.6;
    c.e = 0.0;
    c.C = 0.0;
    std::vector<Contact> contacts = {c};

    const double initial_speed = math::length(bodies[1].v);
    simulate_steps(bodies, contacts, params, 30);
    const double final_speed = math::length(bodies[1].v);
    assert(final_speed < initial_speed);
  }

  {
    const math::Vec3 normal = math::normalize_safe(math::Vec3(0.0, 1.0, 0.5));
    math::Vec3 t1, t2;
    build_tangent_frame(normal, t1, t2);

    std::vector<RigidBody> bodies;
    bodies.push_back(make_static_body());
    bodies.push_back(make_dynamic_body(math::Vec3(0.0, 1.0, 0.0), t1 * 3.0));

    Contact c;
    c.a = 0;
    c.b = 1;
    c.p = math::Vec3(0.0, 0.0, 0.0);
    c.n = normal;
    c.e = 0.0;
    c.C = 0.0;

    std::vector<Contact> contacts_low = {c};
    contacts_low[0].mu = 0.2;
    std::vector<RigidBody> sliding_bodies = bodies;
    simulate_steps(sliding_bodies, contacts_low, params, 10);
    const math::Vec3 v_slide = sliding_bodies[1].v;
    const math::Vec3 tangential_slide = v_slide - math::dot(v_slide, normal) * normal;
    assert(math::length(tangential_slide) > 1e-3);

    std::vector<Contact> contacts_high = {c};
    contacts_high[0].mu = 1.0;
    std::vector<RigidBody> sticking_bodies = bodies;
    simulate_steps(sticking_bodies, contacts_high, params, 10);
    const math::Vec3 v_stick = sticking_bodies[1].v;
    const math::Vec3 tangential_stick = v_stick - math::dot(v_stick, normal) * normal;
    assert(math::length(tangential_stick) < 1e-6);
  }

  return 0;
}
