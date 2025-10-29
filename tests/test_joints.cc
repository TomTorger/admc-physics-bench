#include "joints.hpp"
#include "metrics.hpp"
#include "scenes.hpp"
#include "solver_scalar_cached.hpp"
#include "solver_scalar_soa.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

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

void run_cached_step(Scene& scene, const SolverParams& params) {
  build_distance_joint_rows(scene.bodies, scene.joints, params.dt);
  solve_scalar_cached(scene.bodies, scene.contacts, scene.joints, params);
}

void run_soa_step(Scene& scene, const SolverParams& params) {
  build_distance_joint_rows(scene.bodies, scene.joints, params.dt);
  RowSOA rows;
  build_soa(scene.bodies, scene.contacts, params, rows);
  JointSOA joint_rows;
  build_joint_soa(scene.bodies, scene.joints, params.dt, joint_rows);
  solve_scalar_soa(scene.bodies, scene.contacts, rows, joint_rows, params);
  scatter_joint_impulses(joint_rows, scene.joints);
}

void test_pendulum_length_preservation() {
  Scene scene = make_pendulum(1);

  SolverParams params;
  params.iterations = 60;
  params.beta = 0.2;
  params.slop = 0.0;
  params.restitution = 0.0;
  params.mu = 0.0;
  params.dt = 1.0 / 240.0;
  params.warm_start = true;

  const int steps = 80;
  for (int i = 0; i < steps; ++i) {
    run_cached_step(scene, params);
  }
  build_distance_joint_rows(scene.bodies, scene.joints, params.dt);
  const double joint_err = joint_error_Linf(scene.joints);
  assert(joint_err < 1e-5);

  Scene momentum_scene = make_pendulum(1);
  for (DistanceJoint& j : momentum_scene.joints) {
    j.beta = 0.0;
  }
  SolverParams momentum_params = params;
  momentum_params.beta = 0.0;
  const std::vector<RigidBody> pre = momentum_scene.bodies;
  for (int i = 0; i < steps; ++i) {
    run_cached_step(momentum_scene, momentum_params);
  }
  const Drift drift = directional_momentum_drift(pre, momentum_scene.bodies);
  assert(drift.max_abs < 1e-9);
}

void test_chain_stability() {
  Scene cached_scene = make_chain_64();
  Scene soa_scene = cached_scene;

  SolverParams params;
  params.iterations = 30;
  params.beta = 0.2;
  params.slop = 0.0;
  params.restitution = 0.0;
  params.mu = 0.0;
  params.dt = 1.0 / 180.0;
  params.warm_start = true;

  const int steps = 300;
  for (int i = 0; i < steps; ++i) {
    run_cached_step(cached_scene, params);
    run_soa_step(soa_scene, params);
  }
  build_distance_joint_rows(cached_scene.bodies, cached_scene.joints, params.dt);
  build_distance_joint_rows(soa_scene.bodies, soa_scene.joints, params.dt);

  assert(joint_error_Linf(cached_scene.joints) < 5e-3);
  assert(joint_error_Linf(soa_scene.joints) < 5e-3);

  for (std::size_t i = 0; i < cached_scene.bodies.size(); ++i) {
    const RigidBody& a = cached_scene.bodies[i];
    const RigidBody& b = soa_scene.bodies[i];
    assert(std::isfinite(a.v.x) && std::isfinite(a.v.y) && std::isfinite(a.v.z));
    assert(std::isfinite(b.v.x) && std::isfinite(b.v.y) && std::isfinite(b.v.z));
    assert(std::fabs(a.v.x - b.v.x) < 1e-6);
    assert(std::fabs(a.v.y - b.v.y) < 1e-6);
    assert(std::fabs(a.v.z - b.v.z) < 1e-6);
  }
}

void test_rope_tension_clamp() {
  Scene scene;
  scene.bodies.reserve(2);

  RigidBody a;
  a.invMass = 1.0;
  a.invInertiaLocal = math::Mat3::identity();
  a.syncDerived();
  scene.bodies.push_back(a);

  RigidBody b;
  b.x = math::Vec3(0.5, 0.0, 0.0);
  b.invMass = 1.0;
  b.invInertiaLocal = math::Mat3::identity();
  b.syncDerived();
  scene.bodies.push_back(b);

  DistanceJoint joint;
  joint.a = 0;
  joint.b = 1;
  joint.rest = 1.0;
  joint.compliance = 0.0;
  joint.beta = 0.2;
  joint.rope = true;
  scene.joints.push_back(joint);

  SolverParams params;
  params.iterations = 20;
  params.beta = 0.2;
  params.slop = 0.0;
  params.restitution = 0.0;
  params.mu = 0.0;
  params.dt = 1.0 / 120.0;
  params.warm_start = false;

  run_cached_step(scene, params);
  build_distance_joint_rows(scene.bodies, scene.joints, params.dt);
  const DistanceJoint& solved = scene.joints.front();
  assert(solved.jd >= -1e-12);

  const RigidBody& A = scene.bodies[solved.a];
  const RigidBody& B = scene.bodies[solved.b];
  const math::Vec3 va = A.v + math::cross(A.w, solved.ra);
  const math::Vec3 vb = B.v + math::cross(B.w, solved.rb);
  const double v_rel_d = math::dot(solved.d_hat, vb - va);
  assert(v_rel_d >= -1e-9);
}

void test_joint_determinism() {
  Scene scene_a = make_chain_64();
  Scene scene_b = make_chain_64();

  SolverParams params;
  params.iterations = 40;
  params.beta = 0.2;
  params.slop = 0.0;
  params.restitution = 0.0;
  params.mu = 0.0;
  params.dt = 1.0 / 180.0;
  params.warm_start = true;

  const int steps = 120;
  for (int i = 0; i < steps; ++i) {
    run_cached_step(scene_a, params);
    run_cached_step(scene_b, params);
  }

  const std::uint64_t hash_a = hash_state(scene_a.bodies);
  const std::uint64_t hash_b = hash_state(scene_b.bodies);
  assert(hash_a == hash_b);
}

}  // namespace

int main() {
  test_pendulum_length_preservation();
  test_chain_stability();
  test_rope_tension_clamp();
  test_joint_determinism();
  return 0;
}
