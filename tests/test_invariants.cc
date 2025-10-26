#include "contact_gen.hpp"
#include "metrics.hpp"
#include "scenes.hpp"
#include "solver_baseline_vec.hpp"

#include "math.hpp"

#include <cassert>
#include <cmath>
#include <random>

int main() {
  BaselineParams params;
  params.iterations = 20;
  params.beta = 0.0;
  params.slop = 0.0;
  params.restitution = 0.0;
  params.dt = 1.0 / 60.0;

  Scene scene = make_spheres_box_cloud(128);
  const std::vector<RigidBody> pre = scene.bodies;

  build_contact_offsets_and_bias(scene.bodies, scene.contacts, params);
  solve_baseline(scene.bodies, scene.contacts, params);

  const Drift drift = directional_momentum_drift(pre, scene.bodies);
  assert(drift.max_abs < 1e-10);

  std::mt19937 rng(1337);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  int generated = 0;
  while (generated < 4) {
    math::Vec3 dir(dist(rng), dist(rng), dist(rng));
    dir = math::normalize_safe(dir);
    if (math::length2(dir) <= math::kEps) {
      continue;
    }

    double pre_sum = 0.0;
    double post_sum = 0.0;
    for (std::size_t i = 0; i < scene.bodies.size(); ++i) {
      const RigidBody& pre_body = pre[i];
      const RigidBody& post_body = scene.bodies[i];
      if (pre_body.invMass <= math::kEps) {
        continue;
      }
      const double mass = 1.0 / pre_body.invMass;
      pre_sum += math::dot(pre_body.v * mass, dir);
      post_sum += math::dot(post_body.v * mass, dir);
    }
    assert(std::fabs(post_sum - pre_sum) < 1e-10);
    ++generated;
  }

  return 0;
}

