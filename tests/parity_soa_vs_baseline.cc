#include "contact_gen.hpp"
#include "metrics.hpp"
#include "scenes.hpp"
#include "solver_baseline_vec.hpp"
#include "solver_scalar_cached.hpp"
#include "solver_scalar_soa.hpp"

#include <cassert>
#include <cmath>

namespace {

void run_two_sphere_parity() {
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
  solver_params.spheres_only = true;
  solver_params.frictionless = true;

  Scene baseline_scene = base;
  build_contact_offsets_and_bias(baseline_scene.bodies, baseline_scene.contacts,
                                 baseline_params);
  solve_baseline(baseline_scene.bodies, baseline_scene.contacts, baseline_params);

  Scene cached_scene = base;
  build_contact_offsets_and_bias(cached_scene.bodies, cached_scene.contacts,
                                 solver_params);
  solve_scalar_cached(cached_scene.bodies, cached_scene.contacts, solver_params);

  Scene soa_scene = base;
  build_contact_offsets_and_bias(soa_scene.bodies, soa_scene.contacts,
                                 solver_params);
  soa::World world(soa_scene.bodies);
  soa::ContactManifold manifold(soa_scene.contacts);
  soa::solve_soa(world, manifold, solver_params);

  for (std::size_t i = 0; i < baseline_scene.bodies.size(); ++i) {
    [[maybe_unused]] const RigidBody& ref = baseline_scene.bodies[i];
    [[maybe_unused]] const RigidBody& cached = cached_scene.bodies[i];
    [[maybe_unused]] const RigidBody& soa_body = soa_scene.bodies[i];
    assert(std::fabs(ref.v.x - cached.v.x) < 1e-6);
    assert(std::fabs(ref.v.y - cached.v.y) < 1e-6);
    assert(std::fabs(ref.v.z - cached.v.z) < 1e-6);
    assert(std::fabs(ref.v.x - soa_body.v.x) < 1e-6);
    assert(std::fabs(ref.v.y - soa_body.v.y) < 1e-6);
    assert(std::fabs(ref.v.z - soa_body.v.z) < 1e-6);
  }
}

void run_cloud_metrics() {
  Scene base = make_spheres_box_cloud(1024);

  BaselineParams baseline_params;
  baseline_params.iterations = 12;
  baseline_params.beta = 0.2;
  baseline_params.slop = 0.005;
  baseline_params.restitution = 0.0;
  baseline_params.dt = 1.0 / 60.0;

  SolverParams solver_params;
  solver_params.iterations = baseline_params.iterations;
  solver_params.beta = baseline_params.beta;
  solver_params.slop = baseline_params.slop;
  solver_params.restitution = baseline_params.restitution;
  solver_params.dt = baseline_params.dt;
  solver_params.mu = 0.0;
  solver_params.warm_start = true;
  solver_params.spheres_only = true;
  solver_params.frictionless = true;

  Scene baseline_scene = base;
  const std::vector<RigidBody> initial_bodies = baseline_scene.bodies;
  build_contact_offsets_and_bias(baseline_scene.bodies, baseline_scene.contacts,
                                 baseline_params);
  solve_baseline(baseline_scene.bodies, baseline_scene.contacts, baseline_params);

  Scene soa_scene = base;
  build_contact_offsets_and_bias(soa_scene.bodies, soa_scene.contacts,
                                 solver_params);
  soa::World world(soa_scene.bodies);
  soa::ContactManifold manifold(soa_scene.contacts);
  soa::solve_soa(world, manifold, solver_params);

  [[maybe_unused]] const double baseline_drift =
      directional_momentum_drift(initial_bodies, baseline_scene.bodies).max_abs;
  [[maybe_unused]] const double soa_drift =
      directional_momentum_drift(initial_bodies, soa_scene.bodies).max_abs;
  assert(std::fabs(baseline_drift - soa_drift) < 1e-6);

  [[maybe_unused]] const double baseline_penetration =
      constraint_penetration_Linf(baseline_scene.contacts);
  [[maybe_unused]] const double soa_penetration = constraint_penetration_Linf(soa_scene.contacts);
  assert(std::fabs(baseline_penetration - soa_penetration) < 1e-6);

  [[maybe_unused]] const double baseline_energy = energy_drift(initial_bodies, baseline_scene.bodies);
  [[maybe_unused]] const double soa_energy = energy_drift(initial_bodies, soa_scene.bodies);
  assert(std::fabs(baseline_energy - soa_energy) < 1e-6);
}

}  // namespace

int main() {
  run_two_sphere_parity();
  run_cloud_metrics();
  return 0;
}

