#pragma once

#include "solver_baseline_vec.hpp"

#include <vector>

void build_contact_offsets_and_bias(std::vector<RigidBody>& bodies,
                                    std::vector<Contact>& contacts,
                                    const BaselineParams& params);

