#include "contact_gen.hpp"
#include "metrics.hpp"
#include "scenes.hpp"
#include "solver_baseline_vec.hpp"
#include "solver_scalar_cached.hpp"

#include <cassert>
#include <cmath>

namespace {
void compare_two_spheres() {
  BaselineParams base_params;
  base_params.iterations = 32;
  base_params.beta = 0.0;
  base_params.slop = 0.0;
  base_params.restitution = 1.0;
  base_params.dt = 1.0 / 60.0;

  ScalarParams scalar_params;
  scalar_params.iterations = base_params.iterations;
  scalar_params.beta = base_params.beta;
  scalar_params.slop = base_params.slop;
  scalar_params.restitution = base_params.restitution;
  scalar_params.dt = base_params.dt;
  scalar_params.warm_start = false;

  Scene base_scene = make_two_spheres_head_on();

  std::vector<RigidBody> baseline_bodies = base_scene.bodies;
  std::vector<Contact> baseline_contacts = base_scene.contacts;
  build_contact_offsets_and_bias(baseline_bodies, baseline_contacts, base_params);
  solve_baseline(baseline_bodies, baseline_contacts, base_params);

  std::vector<RigidBody> scalar_bodies = base_scene.bodies;
  std::vector<Contact> scalar_contacts = base_scene.contacts;
  solve_scalar_cached(scalar_bodies, scalar_contacts, scalar_params);

  for (std::size_t i = 0; i < baseline_bodies.size(); ++i) {
    const RigidBody& b = baseline_bodies[i];
    const RigidBody& s = scalar_bodies[i];
    assert(std::fabs(b.v.x - s.v.x) < 1e-6);
    assert(std::fabs(b.v.y - s.v.y) < 1e-6);
    assert(std::fabs(b.v.z - s.v.z) < 1e-6);
  }

  const Drift drift = directional_momentum_drift(base_scene.bodies, scalar_bodies);
  assert(drift.max_abs < 1e-10);
}

void compare_cloud_hash() {
  const int N = 256;
  BaselineParams base_params;
  base_params.iterations = 20;
  base_params.beta = 0.2;
  base_params.slop = 0.005;
  base_params.restitution = 0.0;
  base_params.dt = 1.0 / 60.0;

  ScalarParams scalar_params;
  scalar_params.iterations = base_params.iterations;
  scalar_params.beta = base_params.beta;
  scalar_params.slop = base_params.slop;
  scalar_params.restitution = base_params.restitution;
  scalar_params.dt = base_params.dt;
  scalar_params.warm_start = false;

  Scene base_scene = make_spheres_box_cloud(N);
  for (Contact& c : base_scene.contacts) {
    c.mu = 0.0;
    c.e = 0.0;
  }

  std::vector<RigidBody> baseline_bodies = base_scene.bodies;
  std::vector<Contact> baseline_contacts = base_scene.contacts;
  build_contact_offsets_and_bias(baseline_bodies, baseline_contacts, base_params);
  solve_baseline(baseline_bodies, baseline_contacts, base_params);

  std::vector<RigidBody> scalar_bodies = base_scene.bodies;
  std::vector<Contact> scalar_contacts = base_scene.contacts;
  solve_scalar_cached(scalar_bodies, scalar_contacts, scalar_params);

  for (std::size_t i = 0; i < baseline_bodies.size(); ++i) {
    const RigidBody& b = baseline_bodies[i];
    const RigidBody& s = scalar_bodies[i];
    assert(std::fabs(b.v.x - s.v.x) < 1e-5);
    assert(std::fabs(b.v.y - s.v.y) < 1e-5);
    assert(std::fabs(b.v.z - s.v.z) < 1e-5);
  }

  const std::uint64_t hash_baseline = state_hash64(baseline_bodies);
  const std::uint64_t hash_scalar = state_hash64(scalar_bodies);
  assert(hash_baseline == hash_scalar);
}
}  // namespace

int main() {
  compare_two_spheres();
  compare_cloud_hash();
  return 0;
}
