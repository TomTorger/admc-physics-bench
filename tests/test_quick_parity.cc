#include "bench_csv_schema.hpp"
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

    auto B1 = scene.bodies;
    auto B2 = scene.bodies;
    auto B3 = scene.bodies;
    auto C1 = scene.contacts;
    auto C2 = scene.contacts;
    auto C3 = scene.contacts;

    solve_baseline(B1, C1, pb);
    solve_scalar_cached(B2, C2, ps);

    RowSOA rows = build_soa(B3, C3, ps);
    solve_scalar_soa(B3, C3, rows, ps);

    const double d12 = max_abs_diff(B1, B2);
    const double d13 = max_abs_diff(B1, B3);
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

    auto B1 = scene.bodies;
    auto B2 = scene.bodies;
    auto C1 = scene.contacts;
    auto C2 = scene.contacts;

    solve_baseline(B1, C1, pb);
    solve_scalar_cached(B2, C2, ps);

    const double d = max_abs_diff(B1, B2);
    assert(d <= 1e-4 && "Parity (approx): cached within tolerance on spheres_cloud_256");
  }

  return 0;
}
