#pragma once
#include "math.hpp"
#include <vector>
struct RigidBody {
  double invMass;  // 0 for static
  // TODO(Codex): add inertia, orientation (quaternion), linear/ang vel.
};
struct Contact {
  int a, b;   // body indices
  Vec3 p;     // world contact point
  Vec3 n;     // unit normal
  Vec3 ra, rb;// r = p - x
  // material: restitution, friction...
};
struct RowSOA {
  // TODO(Codex): Structure-of-Arrays buffers for batched scalar rows
};
