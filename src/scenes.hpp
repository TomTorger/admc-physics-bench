#pragma once

#include "joints.hpp"
#include "types.hpp"

#include <vector>

struct Scene {
  std::vector<RigidBody> bodies;
  std::vector<Contact> contacts;
  std::vector<DistanceJoint> joints;
};

Scene make_two_spheres_head_on();
Scene make_spheres_box_cloud(int N = 2048);
Scene make_box_stack(int layers = 8);
Scene make_pendulum(int links = 1);
Scene make_chain_64();
Scene make_rope_256();

