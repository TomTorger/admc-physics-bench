#pragma once

#include "solver_scalar_cached.hpp"

RowSOA build_soa(const std::vector<RigidBody>& bodies,
                 const std::vector<Contact>& contacts,
                 const SolverParams& params);

void solve_scalar_soa(std::vector<RigidBody>& bodies,
                      std::vector<Contact>& contacts,
                      RowSOA& rows,
                      const SolverParams& params);
