#pragma once

#include "solver_baseline_vec.hpp"
#include "solver_scalar_cached.hpp"

#include <vector>

void build_contact_offsets_and_bias(std::vector<RigidBody>& bodies,
                                    std::vector<Contact>& contacts,
                                    const BaselineParams& params);

void build_contact_offsets_and_bias(std::vector<RigidBody>& bodies,
                                    std::vector<Contact>& contacts,
                                    const SolverParams& params);

void refresh_contacts_from_state(const std::vector<RigidBody>& bodies,
                                 std::vector<Contact>& contacts);

