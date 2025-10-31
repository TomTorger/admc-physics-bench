#include "contact_gen.hpp"
#include "metrics.hpp"
#include "scenes.hpp"
#include "solver_baseline_vec.hpp"
#include "solver_scalar_cached.hpp"
#include "solver_scalar_soa.hpp"
#include "solver_scalar_soa_native.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

void test_frictionless_parity() {
  Scene base = make_two_spheres_head_on();

  BaselineParams baseline_params;
  baseline_params.iterations = 32;
  baseline_params.beta = 0.0;
  baseline_params.slop = 0.0;
  baseline_params.restitution = 1.0;
  baseline_params.dt = 1.0 / 60.0;

  SolverParams solver_params;
  solver_params.iterations = 32;
  solver_params.beta = 0.0;
  solver_params.slop = 0.0;
  solver_params.restitution = 1.0;
  solver_params.mu = 0.0;
  solver_params.dt = 1.0 / 60.0;
  solver_params.warm_start = true;

  Scene baseline_scene = base;
  const std::vector<RigidBody> pre = baseline_scene.bodies;
  build_contact_offsets_and_bias(baseline_scene.bodies, baseline_scene.contacts,
                                 baseline_params);
  solve_baseline(baseline_scene.bodies, baseline_scene.contacts, baseline_params);

  Scene cached_scene = base;
  build_contact_offsets_and_bias(cached_scene.bodies, cached_scene.contacts,
                                 solver_params);
  solve_scalar_cached(cached_scene.bodies, cached_scene.contacts, solver_params);

  Scene soa_scene = base;
  build_contact_offsets_and_bias(soa_scene.bodies, soa_scene.contacts, solver_params);
  SoaParams soa_params;
  static_cast<SolverParams&>(soa_params) = solver_params;
  RowSOA rows;
  build_soa(soa_scene.bodies, soa_scene.contacts, soa_params, rows);
  solve_scalar_soa(soa_scene.bodies, soa_scene.contacts, rows, soa_params);

  Scene native_scene = base;
  build_contact_offsets_and_bias(native_scene.bodies, native_scene.contacts,
                                 solver_params);
  RowSOA native_rows;
  build_soa(native_scene.bodies, native_scene.contacts, soa_params, native_rows);
  solve_scalar_soa_native(native_scene.bodies, native_scene.contacts, native_rows,
                          soa_params);

  const Drift drift = directional_momentum_drift(pre, cached_scene.bodies);
  assert(drift.max_abs < 1e-10);
  const Drift native_drift = directional_momentum_drift(pre, native_scene.bodies);
  assert(native_drift.max_abs < 1e-10);

  for (std::size_t i = 0; i < baseline_scene.bodies.size(); ++i) {
    const RigidBody& b0 = baseline_scene.bodies[i];
    const RigidBody& b1 = cached_scene.bodies[i];
    const RigidBody& b2 = soa_scene.bodies[i];
    const RigidBody& b3 = native_scene.bodies[i];
    assert(std::fabs(b0.v.x - b1.v.x) < 1e-6);
    assert(std::fabs(b0.v.y - b1.v.y) < 1e-6);
    assert(std::fabs(b0.v.z - b1.v.z) < 1e-6);
    assert(std::fabs(b0.v.x - b2.v.x) < 1e-6);
    assert(std::fabs(b0.v.y - b2.v.y) < 1e-6);
    assert(std::fabs(b0.v.z - b2.v.z) < 1e-6);
    assert(std::fabs(b0.v.x - b3.v.x) < 1e-6);
    assert(std::fabs(b0.v.y - b3.v.y) < 1e-6);
    assert(std::fabs(b0.v.z - b3.v.z) < 1e-6);
  }

  const double speed = pre[0].v.x;
  assert(std::fabs(baseline_scene.bodies[0].v.x + speed) < 1e-6);
  assert(std::fabs(baseline_scene.bodies[1].v.x - speed) < 1e-6);
}

Scene make_single_box_scene(double lateral_velocity, double mu) {
  Scene scene;
  scene.bodies.reserve(2);

  RigidBody ground;
  ground.invMass = 0.0;
  ground.invInertiaLocal = math::Mat3();
  ground.syncDerived();
  scene.bodies.push_back(ground);

  RigidBody box;
  box.x = math::Vec3(0.0, 1.0, 0.0);
  box.invMass = 1.0;
  box.invInertiaLocal = math::Mat3::identity();
  box.syncDerived();
  box.v = math::Vec3(lateral_velocity, 0.0, 0.0);
  scene.bodies.push_back(box);

  Contact c;
  c.a = 0;
  c.b = 1;
  c.p = math::Vec3(0.0, 0.0, 0.0);
  c.n = math::Vec3(0.0, 1.0, 0.0);
  c.mu = mu;
  scene.contacts.push_back(c);

  return scene;
}

void check_friction_case(double lateral_velocity, bool expect_stick) {
  const double mu = 0.8;
  Scene scene = make_single_box_scene(lateral_velocity, mu);

  SolverParams params;
  params.iterations = 40;
  params.beta = 0.0;
  params.slop = 0.0;
  params.restitution = 0.0;
  params.mu = mu;
  params.dt = 1.0 / 120.0;
  params.warm_start = true;

  build_contact_offsets_and_bias(scene.bodies, scene.contacts, params);
  solve_scalar_cached(scene.bodies, scene.contacts, params);

  assert(std::fabs(cone_consistency(scene.contacts) - 1.0) < 1e-12);

  const Contact& c = scene.contacts.front();
  const RigidBody& A = scene.bodies[c.a];
  const RigidBody& B = scene.bodies[c.b];
  const math::Vec3 va = A.v + math::cross(A.w, c.ra);
  const math::Vec3 vb = B.v + math::cross(B.w, c.rb);
  const math::Vec3 v_rel = vb - va;
  const double vt1 = math::dot(c.t1, v_rel);
  const double vt2 = math::dot(c.t2, v_rel);
  const double vt_mag = std::sqrt(vt1 * vt1 + vt2 * vt2);
  const double jt_mag = std::sqrt(c.jt1 * c.jt1 + c.jt2 * c.jt2);
  const double limit = c.mu * std::max(c.jn, 0.0);

  if (expect_stick) {
    assert(vt_mag < 1e-5);
    assert(jt_mag <= limit + 1e-6);
  } else {
    assert(std::fabs(jt_mag - limit) < 1e-6);
    assert(vt_mag > 1e-6);
  }
}

void test_friction_threshold() {
  check_friction_case(0.05, true);
  check_friction_case(5.0, false);
}

std::uint64_t hash_state(const std::vector<RigidBody>& bodies) {
  std::uint64_t hash = 1469598103934665603ull;
  const std::uint64_t prime = 1099511628211ull;
  for (const RigidBody& b : bodies) {
    const double values[] = {b.x.x, b.x.y, b.x.z, b.v.x, b.v.y, b.v.z,
                             b.w.x, b.w.y, b.w.z};
    for (double v : values) {
      std::uint64_t bits = 0;
      std::memcpy(&bits, &v, sizeof(double));
      hash ^= bits;
      hash *= prime;
    }
  }
  return hash;
}

void test_determinism() {
  SolverParams params;
  params.iterations = 20;
  params.beta = 0.1;
  params.slop = 0.001;
  params.restitution = 0.0;
  params.mu = 0.5;
  params.dt = 1.0 / 60.0;
  params.warm_start = true;

  const int steps = 20;

  Scene scene_a = make_box_stack(4);
  Scene scene_b = make_box_stack(4);

  for (int i = 0; i < steps; ++i) {
    build_contact_offsets_and_bias(scene_a.bodies, scene_a.contacts, params);
    solve_scalar_cached(scene_a.bodies, scene_a.contacts, params);
  }

  for (int i = 0; i < steps; ++i) {
    build_contact_offsets_and_bias(scene_b.bodies, scene_b.contacts, params);
    solve_scalar_cached(scene_b.bodies, scene_b.contacts, params);
  }

  const std::uint64_t hash_a = hash_state(scene_a.bodies);
  const std::uint64_t hash_b = hash_state(scene_b.bodies);
  assert(hash_a == hash_b);
}

}  // namespace

int main() {
  test_frictionless_parity();
  test_friction_threshold();
  test_determinism();
  return 0;
}
