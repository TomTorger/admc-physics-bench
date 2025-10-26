#include "joints.hpp"
#include "scenes.hpp"
#include "solver_scalar_soa.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <functional>

namespace {

std::uint64_t hash_bodies(const std::vector<RigidBody>& bodies) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (const RigidBody& body : bodies) {
    const double components[] = {body.x.x, body.x.y, body.x.z, body.v.x, body.v.y,
                                 body.v.z, body.w.x, body.w.y, body.w.z};
    for (double v : components) {
      const std::uint64_t bits = std::hash<double>{}(v);
      hash ^= bits;
      hash *= 1099511628211ULL;
    }
  }
  return hash;
}

void run_solver(const Scene& base_scene,
                const SoaParams& params,
                std::vector<RigidBody>* out_bodies) {
  std::vector<RigidBody> bodies = base_scene.bodies;
  std::vector<Contact> contacts = base_scene.contacts;
  std::vector<DistanceJoint> joints = base_scene.joints;

  build_contact_offsets_and_bias(bodies, contacts, params);
  RowSOA rows = build_soa(bodies, contacts, params);
  build_distance_joint_rows(bodies, joints, params.dt);
  JointSOA joint_rows = build_joint_soa(bodies, joints, params.dt);
  solve_scalar_soa(bodies, contacts, rows, joint_rows, params);

  *out_bodies = bodies;
}

}  // namespace

int main() {
  Scene scene = make_box_stack(2);

  SoaParams threaded;
  threaded.iterations = 6;
  threaded.dt = 1.0 / 60.0;
  threaded.use_simd = true;
  threaded.use_threads = true;
  threaded.thread_count = 4;

  SoaParams single = threaded;
  single.use_threads = true;
  single.thread_count = 1;

  std::vector<RigidBody> bodies_first;
  std::vector<RigidBody> bodies_second;
  run_solver(scene, threaded, &bodies_first);
  run_solver(scene, threaded, &bodies_second);
  assert(hash_bodies(bodies_first) == hash_bodies(bodies_second));

  std::vector<RigidBody> bodies_single;
  run_solver(scene, single, &bodies_single);
  assert(hash_bodies(bodies_first) == hash_bodies(bodies_single));

  return 0;
}
