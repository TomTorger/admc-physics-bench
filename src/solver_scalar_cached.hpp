#pragma once

#include "joints.hpp"
#include "types.hpp"

#include <vector>

struct SolverParams {
  int iterations = 10;
  double beta = 0.2;
  double slop = 0.005;
  double restitution = 0.0;
  double mu = 0.5;
  double dt = 1.0 / 60.0;
  bool warm_start = true;
  int tile_size = 128;
  int max_contacts_per_tile = 128;
  int tile_rows = 128;
  bool spheres_only = false;
  bool frictionless = false;
};

void solve_scalar_cached(std::vector<RigidBody>& bodies,
                         std::vector<Contact>& contacts,
                         std::vector<DistanceJoint>& joints,
                         const SolverParams& params);

void solve_scalar_cached(std::vector<RigidBody>& bodies,
                         std::vector<Contact>& contacts,
                         const SolverParams& params);
