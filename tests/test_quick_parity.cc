#include "bench_csv_schema.hpp"
#include "contact_gen.hpp"
#include "metrics.hpp"
#include "scenes.hpp"
#include "solver_baseline_vec.hpp"
#include "solver_scalar_cached.hpp"
#include "solver_scalar_soa.hpp"
#include "types.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <string>
#include <vector>

namespace {
double max_abs_diff(const std::vector<RigidBody>& A,
                    const std::vector<RigidBody>& B) {
  double m = 0.0;
  const std::size_t n = std::min(A.size(), B.size());
  for (std::size_t i = 0; i < n; ++i) {
    const double ad = std::fabs(A[i].v.x - B[i].v.x) +
                      std::fabs(A[i].v.y - B[i].v.y) +
                      std::fabs(A[i].v.z - B[i].v.z);
    const double wd = std::fabs(A[i].w.x - B[i].w.x) +
                      std::fabs(A[i].w.y - B[i].w.y) +
                      std::fabs(A[i].w.z - B[i].w.z);
    m = std::max(m, std::max(ad, wd));
  }
  return m;
}
} // namespace

int main() {
  // CSV header stability (bench & tests must agree)
  const std::string expected(benchcsv::kHeader);
  assert(expected.find("scene,solver") == 0 &&
         "CSV header must start with scene,solver");
  assert(expected.find("ms_per_step") != std::string::npos &&
         "CSV header missing ms_per_step");

  // Frictionless parity on tiny scenes (fast)
  {
    Scene scene = make_two_spheres_head_on();
    const double expected_v0 = -scene.bodies[0].v.x;
    const double expected_v1 = -scene.bodies[1].v.x;

    BaselineParams pb;
    pb.iterations = 10;
    pb.beta = 0.0;
    pb.slop = 0.0;
    pb.restitution = 1.0;
    pb.dt = 1.0 / 60.0;

    SolverParams ps;
    ps.iterations = 10;
    ps.beta = 0.0;
    ps.slop = 0.0;
    ps.restitution = 1.0;
    ps.mu = 0.0;
    ps.dt = pb.dt;

    auto baseline_bodies = scene.bodies;
    auto cached_bodies = scene.bodies;
    auto soa_bodies = scene.bodies;
    auto baseline_contacts = scene.contacts;
    auto cached_contacts = scene.contacts;
    auto soa_contacts = scene.contacts;

    build_contact_offsets_and_bias(baseline_bodies, baseline_contacts, pb);
    solve_baseline(baseline_bodies, baseline_contacts, pb);
    assert(std::fabs(baseline_bodies[0].v.x - expected_v0) <= 1e-6);
    assert(std::fabs(baseline_bodies[1].v.x - expected_v1) <= 1e-6);

    build_contact_offsets_and_bias(cached_bodies, cached_contacts, ps);
    solve_scalar_cached(cached_bodies, cached_contacts, ps);
    assert(std::fabs(cached_bodies[0].v.x - expected_v0) <= 1e-6);
    assert(std::fabs(cached_bodies[1].v.x - expected_v1) <= 1e-6);

    build_contact_offsets_and_bias(soa_bodies, soa_contacts, ps);
    RowSOA rows;
    build_soa(soa_bodies, soa_contacts, ps, rows);
    solve_scalar_soa(soa_bodies, soa_contacts, rows, ps);
    assert(std::fabs(soa_bodies[0].v.x - expected_v0) <= 1e-6);
    assert(std::fabs(soa_bodies[1].v.x - expected_v1) <= 1e-6);

    const double d12 = max_abs_diff(baseline_bodies, cached_bodies);
    const double d13 = max_abs_diff(baseline_bodies, soa_bodies);
    assert(d12 <= 1e-6 && "Parity: baseline vs scalar must match on two_spheres");
    assert(d13 <= 1e-6 && "Parity: baseline vs SoA must match on two_spheres");
  }

  {
    Scene scene = make_spheres_box_cloud(256);

    BaselineParams pb;
    pb.iterations = 5;
    pb.beta = 0.2;
    pb.slop = 0.005;
    pb.restitution = 0.0;
    pb.dt = 1.0 / 60.0;

    SolverParams ps;
    ps.iterations = 5;
    ps.beta = 0.2;
    ps.slop = 0.005;
    ps.restitution = 0.0;
    ps.mu = 0.0;
    ps.dt = pb.dt;

    auto baseline_bodies = scene.bodies;
    auto soa_bodies = scene.bodies;
    auto baseline_contacts = scene.contacts;
    auto soa_contacts = scene.contacts;

    const int steps = 5;
    RowSOA rows;
    for (int step = 0; step < steps; ++step) {
      build_contact_offsets_and_bias(baseline_bodies, baseline_contacts, pb);
      solve_baseline(baseline_bodies, baseline_contacts, pb);

      build_contact_offsets_and_bias(soa_bodies, soa_contacts, ps);
      build_soa(soa_bodies, soa_contacts, ps, rows);
      solve_scalar_soa(soa_bodies, soa_contacts, rows, ps);
    }

    const double diff = max_abs_diff(baseline_bodies, soa_bodies);
    assert(diff <= 5e-4 && "Parity (approx): SoA within tolerance after multiple steps");
    const double cone = cone_consistency(soa_contacts);
    assert(cone >= 0.999 && "SoA friction cone violations detected");
    const double energy = kinetic_energy(soa_bodies);
    assert(std::isfinite(energy) && "Energy should remain finite");
  }

  return 0;
}
