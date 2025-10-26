#pragma once

#include "types.hpp"

#include <vector>

struct Scene {
  std::vector<RigidBody> bodies;
  std::vector<Contact> contacts;
};

Scene make_two_spheres_head_on();
Scene make_spheres_box_cloud(int N = 2048);
Scene make_box_stack(int layers = 8);

