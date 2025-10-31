#include "contact_gen.hpp"
#include "joints.hpp"
#include "scenes.hpp"
#include "solver_scalar_soa.hpp"
#include "solver_scalar_soa_native.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>

namespace {

double max_velocity_delta(const std::vector<RigidBody>& a,
                          const std::vector<RigidBody>& b) {
  double max_delta = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    max_delta = std::max(max_delta, std::fabs(a[i].v.x - b[i].v.x));
    max_delta = std::max(max_delta, std::fabs(a[i].v.y - b[i].v.y));
    max_delta = std::max(max_delta, std::fabs(a[i].v.z - b[i].v.z));
    max_delta = std::max(max_delta, std::fabs(a[i].w.x - b[i].w.x));
    max_delta = std::max(max_delta, std::fabs(a[i].w.y - b[i].w.y));
    max_delta = std::max(max_delta, std::fabs(a[i].w.z - b[i].w.z));
  }
  return max_delta;
}

}  // namespace

int main() {
  Scene scene = make_two_spheres_head_on();

  SoaParams params;
  params.iterations = 8;
  params.dt = 1.0 / 60.0;
  params.warm_start = true;

  std::vector<RigidBody> bodies_scalar = scene.bodies;
  std::vector<Contact> contacts_scalar = scene.contacts;
  std::vector<DistanceJoint> joints_scalar = scene.joints;

  std::vector<RigidBody> bodies_simd = scene.bodies;
  std::vector<Contact> contacts_simd = scene.contacts;
  std::vector<DistanceJoint> joints_simd = scene.joints;
  std::vector<RigidBody> bodies_native = scene.bodies;
  std::vector<Contact> contacts_native = scene.contacts;
  std::vector<DistanceJoint> joints_native = scene.joints;

  build_contact_offsets_and_bias(bodies_scalar, contacts_scalar, params);
  RowSOA rows_scalar;
  build_soa(bodies_scalar, contacts_scalar, params, rows_scalar);
  build_distance_joint_rows(bodies_scalar, joints_scalar, params.dt);
  JointSOA joint_rows_scalar;
  build_joint_soa(bodies_scalar, joints_scalar, params.dt, joint_rows_scalar);

  build_contact_offsets_and_bias(bodies_simd, contacts_simd, params);
  RowSOA rows_simd;
  build_soa(bodies_simd, contacts_simd, params, rows_simd);
  build_distance_joint_rows(bodies_simd, joints_simd, params.dt);
  JointSOA joint_rows_simd;
  build_joint_soa(bodies_simd, joints_simd, params.dt, joint_rows_simd);

  RowSOA rows_native;
  build_soa(bodies_native, contacts_native, params, rows_native);
  build_distance_joint_rows(bodies_native, joints_native, params.dt);
  JointSOA joint_rows_native;
  build_joint_soa(bodies_native, joints_native, params.dt, joint_rows_native);

  SoaParams scalar_params = params;
  scalar_params.use_simd = false;
  scalar_params.use_threads = false;
  SoaParams simd_params = params;
  simd_params.use_simd = true;
  simd_params.use_threads = false;

  solve_scalar_soa(bodies_scalar, contacts_scalar, rows_scalar, joint_rows_scalar,
                   scalar_params);
  solve_scalar_soa(bodies_simd, contacts_simd, rows_simd, joint_rows_simd,
                   simd_params);
  solve_scalar_soa_native(bodies_native, contacts_native, rows_native,
                          joint_rows_native, simd_params);

  const double v_delta = max_velocity_delta(bodies_scalar, bodies_simd);
  assert(v_delta < 1e-9);
  const double v_delta_native = max_velocity_delta(bodies_scalar, bodies_native);
  assert(v_delta_native < 1e-9);

  for (std::size_t i = 0; i < rows_scalar.size(); ++i) {
    const double diff_n = std::fabs(rows_scalar.jn[i] - rows_simd.jn[i]);
    const double diff_t1 = std::fabs(rows_scalar.jt1[i] - rows_simd.jt1[i]);
    const double diff_t2 = std::fabs(rows_scalar.jt2[i] - rows_simd.jt2[i]);
    assert(diff_n < 1e-10);
    assert(diff_t1 < 1e-10);
    assert(diff_t2 < 1e-10);
    const double diff_n_native =
        std::fabs(rows_scalar.jn[i] - rows_native.jn[i]);
    const double diff_t1_native =
        std::fabs(rows_scalar.jt1[i] - rows_native.jt1[i]);
    const double diff_t2_native =
        std::fabs(rows_scalar.jt2[i] - rows_native.jt2[i]);
    assert(diff_n_native < 1e-10);
    assert(diff_t1_native < 1e-10);
    assert(diff_t2_native < 1e-10);
  }

  return 0;
}
