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
};

void solve_scalar_cached(std::vector<RigidBody>& bodies,
                         std::vector<Contact>& contacts,
                         std::vector<DistanceJoint>& joints,
                         const SolverParams& params);

void solve_scalar_cached(std::vector<RigidBody>& bodies,
                         std::vector<Contact>& contacts,
                         const SolverParams& params);
