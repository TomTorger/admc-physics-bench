#include "contact_gen.hpp"
#include "metrics.hpp"
#include "scenes.hpp"
#include "solver_baseline_vec.hpp"
#include "solver_scalar_cached.hpp"
#include "solver_scalar_soa.hpp"

#include <cassert>
#include <cmath>

namespace {

void test_elastic_energy_parity() {
  Scene base = make_two_spheres_head_on();

  BaselineParams baseline_params;
  baseline_params.iterations = 10;
  baseline_params.beta = 0.0;
  baseline_params.slop = 0.0;
  baseline_params.restitution = 1.0;
  baseline_params.dt = 1.0 / 60.0;

  SolverParams solver_params;
  solver_params.iterations = 10;
  solver_params.beta = 0.0;
  solver_params.slop = 0.0;
  solver_params.restitution = 1.0;
  solver_params.mu = 0.0;
  solver_params.dt = 1.0 / 60.0;
  solver_params.warm_start = true;

  Scene baseline_scene = base;
  const std::vector<RigidBody> baseline_pre = baseline_scene.bodies;
  build_contact_offsets_and_bias(baseline_scene.bodies, baseline_scene.contacts,
                                 baseline_params);
  solve_baseline(baseline_scene.bodies, baseline_scene.contacts, baseline_params);
  assert(std::fabs(energy_drift(baseline_pre, baseline_scene.bodies)) < 1e-10);

  Scene cached_scene = base;
  const std::vector<RigidBody> cached_pre = cached_scene.bodies;
  build_contact_offsets_and_bias(cached_scene.bodies, cached_scene.contacts,
                                 solver_params);
  solve_scalar_cached(cached_scene.bodies, cached_scene.contacts, solver_params);
  assert(std::fabs(energy_drift(cached_pre, cached_scene.bodies)) < 1e-10);

  Scene soa_scene = base;
  const std::vector<RigidBody> soa_pre = soa_scene.bodies;
  build_contact_offsets_and_bias(soa_scene.bodies, soa_scene.contacts,
                                 solver_params);
  RowSOA rows = build_soa(soa_scene.bodies, soa_scene.contacts, solver_params);
  solve_scalar_soa(soa_scene.bodies, soa_scene.contacts, rows, solver_params);
  assert(std::fabs(energy_drift(soa_pre, soa_scene.bodies)) < 1e-10);
}

void test_penetration_with_zero_erp() {
  Scene scene = make_spheres_box_cloud(256);

  SolverParams params;
  params.iterations = 10;
  params.beta = 0.0;
  params.slop = 0.0;
  params.restitution = 0.0;
  params.mu = 0.0;
  params.dt = 1.0 / 60.0;
  params.warm_start = true;

  build_contact_offsets_and_bias(scene.bodies, scene.contacts, params);
  const double linf_pre = constraint_penetration_Linf(scene.contacts);
  assert(linf_pre > 0.0);

  solve_scalar_cached(scene.bodies, scene.contacts, params);
  build_contact_offsets_and_bias(scene.bodies, scene.contacts, params);
  const double linf_post = constraint_penetration_Linf(scene.contacts);
  assert(std::isfinite(linf_post));
}

void test_penetration_reduced_with_erp() {
  Scene scene = make_spheres_box_cloud(256);

  SolverParams params;
  params.iterations = 10;
  params.beta = 0.2;
  params.slop = 0.0;
  params.restitution = 0.0;
  params.mu = 0.0;
  params.dt = 1.0 / 60.0;
  params.warm_start = true;

  build_contact_offsets_and_bias(scene.bodies, scene.contacts, params);
  const double linf_pre = constraint_penetration_Linf(scene.contacts);
  assert(linf_pre > 0.0);

  solve_scalar_cached(scene.bodies, scene.contacts, params);
  build_contact_offsets_and_bias(scene.bodies, scene.contacts, params);
  const double linf_post = constraint_penetration_Linf(scene.contacts);
  assert(linf_post < linf_pre);
}

}  // namespace

int main() {
  test_elastic_energy_parity();
  test_penetration_with_zero_erp();
  test_penetration_reduced_with_erp();
  return 0;
}

