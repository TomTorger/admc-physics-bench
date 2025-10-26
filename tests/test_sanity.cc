#include "contact_gen.hpp"
#include "metrics.hpp"
#include "scenes.hpp"
#include "solver_baseline_vec.hpp"

#include <cassert>
#include <cmath>

int main() {
  BaselineParams params;
  params.iterations = 32;
  params.beta = 0.0;
  params.slop = 0.0;
  params.restitution = 1.0;
  params.dt = 1.0 / 60.0;

  Scene scene = make_two_spheres_head_on();
  const std::vector<RigidBody> pre = scene.bodies;

  build_contact_offsets_and_bias(scene.bodies, scene.contacts, params);
  solve_baseline(scene.bodies, scene.contacts, params);

  const Drift drift = directional_momentum_drift(pre, scene.bodies);
  assert(drift.max_abs < 1e-10);

  const double u = pre[0].v.x;
  const double v1 = scene.bodies[0].v.x;
  const double v2 = scene.bodies[1].v.x;

  assert(std::fabs(v1 + u) < 1e-6);
  assert(std::fabs(v2 - u) < 1e-6);
  assert(std::fabs(scene.bodies[0].v.y) < 1e-12);
  assert(std::fabs(scene.bodies[0].v.z) < 1e-12);
  assert(std::fabs(scene.bodies[1].v.y) < 1e-12);
  assert(std::fabs(scene.bodies[1].v.z) < 1e-12);

  return 0;
}

