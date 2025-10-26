#pragma once

#include "types.hpp"

#include <cstdint>
#include <vector>

struct Drift {
  double max_abs = 0.0;
};

Drift directional_momentum_drift(const std::vector<RigidBody>& pre,
                                 const std::vector<RigidBody>& post);

double constraint_penetration_Linf(const std::vector<Contact>& contacts);

struct Energy {
  double kinetic_before = 0.0;
  double kinetic_after = 0.0;
  double delta = 0.0;
};

Energy kinetic_energy_delta(const std::vector<RigidBody>& before,
                            const std::vector<RigidBody>& after);

std::uint64_t state_hash64(const std::vector<RigidBody>& bodies);

