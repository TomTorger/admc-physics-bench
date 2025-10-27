#include "contact_gen.hpp"
#include "metrics.hpp"
#include "scenes.hpp"
#include "solver_baseline_vec.hpp"
#include "solver_scalar_cached.hpp"
#include "solver_scalar_soa.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace {

double simulate_penetration(Scene scene, double beta, int steps) {
  SolverParams params;
  params.iterations = 10;
  params.beta = beta;
  params.slop = 0.0;
  params.restitution = 0.0;
  params.mu = 0.0;
  params.dt = 1.0 / 60.0;
  params.warm_start = true;

  refresh_contacts_from_state(scene.bodies, scene.contacts);
  for (int i = 0; i < steps; ++i) {
    build_contact_offsets_and_bias(scene.bodies, scene.contacts, params);
    solve_scalar_cached(scene.bodies, scene.contacts, params);
    refresh_contacts_from_state(scene.bodies, scene.contacts);
  }
  return constraint_penetration_Linf(scene.contacts);
}

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
  refresh_contacts_from_state(baseline_scene.bodies, baseline_scene.contacts);
  build_contact_offsets_and_bias(baseline_scene.bodies, baseline_scene.contacts,
                                 baseline_params);
  solve_baseline(baseline_scene.bodies, baseline_scene.contacts, baseline_params);
  const double baseline_drift =
      energy_drift(baseline_pre, baseline_scene.bodies);
  if (std::fabs(baseline_drift) >= 1e-10) {
    std::cerr << "Baseline energy drift too large: " << baseline_drift << "\n";
    std::exit(1);
  }

  Scene cached_scene = base;
  const std::vector<RigidBody> cached_pre = cached_scene.bodies;
  refresh_contacts_from_state(cached_scene.bodies, cached_scene.contacts);
  build_contact_offsets_and_bias(cached_scene.bodies, cached_scene.contacts,
                                 solver_params);
  solve_scalar_cached(cached_scene.bodies, cached_scene.contacts, solver_params);
  const double cached_drift = energy_drift(cached_pre, cached_scene.bodies);
  if (std::fabs(cached_drift) >= 1e-10) {
    std::cerr << "Cached solver energy drift too large: " << cached_drift << "\n";
    std::exit(1);
  }

  Scene soa_scene = base;
  const std::vector<RigidBody> soa_pre = soa_scene.bodies;
  refresh_contacts_from_state(soa_scene.bodies, soa_scene.contacts);
  build_contact_offsets_and_bias(soa_scene.bodies, soa_scene.contacts,
                                 solver_params);
  RowSOA rows = build_soa(soa_scene.bodies, soa_scene.contacts, solver_params);
  solve_scalar_soa(soa_scene.bodies, soa_scene.contacts, rows, solver_params);
  const double soa_drift = energy_drift(soa_pre, soa_scene.bodies);
  if (std::fabs(soa_drift) >= 1e-10) {
    std::cerr << "SoA solver energy drift too large: " << soa_drift << "\n";
    std::exit(1);
  }
}

void test_penetration_with_zero_erp() {
  const double linf_post = simulate_penetration(make_spheres_box_cloud(256), 0.0,
                                               5);
  if (!(linf_post > 0.0 && std::isfinite(linf_post))) {
    std::cerr << "Penetration depth expected > 0 and finite with beta=0, got "
              << linf_post << "\n";
    std::exit(1);
  }
}

void test_penetration_reduced_with_erp() {
  const double linf_zero = simulate_penetration(make_spheres_box_cloud(256), 0.0,
                                               5);
  if (!(linf_zero > 0.0)) {
    std::cerr << "Expected positive penetration with beta=0, got " << linf_zero
              << "\n";
    std::exit(1);
  }

  const double linf_post = simulate_penetration(make_spheres_box_cloud(256), 0.2,
                                               5);
  if (!(linf_post < linf_zero)) {
    std::cerr << "Expected ERP to reduce penetration: zero=" << linf_zero
              << " beta=0.2=" << linf_post << "\n";
    std::exit(1);
  }
}

}  // namespace

int main() {
  test_elastic_energy_parity();
  test_penetration_with_zero_erp();
  test_penetration_reduced_with_erp();
  return 0;
}

