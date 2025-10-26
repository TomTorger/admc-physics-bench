#pragma once

#include "types.hpp"

#include <vector>

//! Parameters for the baseline sequential impulse solver.
struct BaselineParams {
  int iterations = 10;      //!< Number of Gauss-Seidel iterations.
  double beta = 0.2;        //!< Baumgarte/ERP factor [0, 1].
  double slop = 0.005;      //!< Penetration slop (meters).
  double restitution = 0.0; //!< Restitution clamp [0, 1].
  double dt = 1.0 / 60.0;   //!< Timestep in seconds.
};

//! Solves frictionless contact constraints in-place (vector-per-row PGS).
void solve_baseline(std::vector<RigidBody>& bodies,
                    std::vector<Contact>& contacts,
                    const BaselineParams& params);

