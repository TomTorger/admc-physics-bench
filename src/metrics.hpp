#pragma once

#include "joints.hpp"
#include "types.hpp"

#include <vector>

struct Drift {
  double max_abs = 0.0;
};

Drift directional_momentum_drift(const std::vector<RigidBody>& pre,
                                 const std::vector<RigidBody>& post);

double constraint_penetration_Linf(const std::vector<Contact>& contacts);

double energy_drift(const std::vector<RigidBody>& pre,
                    const std::vector<RigidBody>& post);

double cone_consistency(const std::vector<Contact>& contacts);

double joint_error_Linf(const std::vector<DistanceJoint>& joints);

