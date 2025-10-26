#pragma once

#include "types.hpp"

#include <vector>

struct Drift {
  double max_abs = 0.0;
};

Drift directional_momentum_drift(const std::vector<RigidBody>& pre,
                                 const std::vector<RigidBody>& post);

double constraint_penetration_Linf(const std::vector<Contact>& contacts);

