#pragma once

#include "solver_baseline_vec.hpp"

#include <vector>

void build_tangent_frame(const math::Vec3& n, math::Vec3& t1, math::Vec3& t2);

void preprocess_contacts(std::vector<RigidBody>& bodies,
                         std::vector<Contact>& contacts,
                         double beta,
                         double slop,
                         double dt);

inline void build_contact_offsets_and_bias(std::vector<RigidBody>& bodies,
                                           std::vector<Contact>& contacts,
                                           const BaselineParams& params) {
  preprocess_contacts(bodies, contacts, params.beta, params.slop, params.dt);
}

