#pragma once

#include "joints.hpp"
#include "solver_scalar_soa.hpp"

#include <span>
#include <vector>

namespace admc {

struct SceneView {
  std::vector<RigidBody>* bodies = nullptr;
  std::vector<Contact>* contacts = nullptr;
  std::vector<DistanceJoint>* joints = nullptr;
  JointSOA* joint_rows = nullptr;
  RowSOA* rows = nullptr;
};

struct Island {
  std::span<int> bodies;
  std::span<int> contacts;
  std::span<int> joints;
  std::span<int> rows;
};

std::vector<Island> build_islands(const SceneView& view);

}  // namespace admc

