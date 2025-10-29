#pragma once

#include "solver_scalar_soa.hpp"

//! Experimental SIMD-friendly SoA solver wrapper introduced as the vectorized solver path.
void solve_scalar_soa_vectorized(std::vector<RigidBody>& bodies,
                                 std::vector<Contact>& contacts,
                                 RowSOA& rows,
                                 JointSOA& joints,
                                 const SoaParams& params,
                                 SolverDebugInfo* debug_info = nullptr);

void solve_scalar_soa_vectorized(std::vector<RigidBody>& bodies,
                                 std::vector<Contact>& contacts,
                                 RowSOA& rows,
                                 const SoaParams& params,
                                 SolverDebugInfo* debug_info = nullptr);

void solve_scalar_soa_vectorized(std::vector<RigidBody>& bodies,
                                 std::vector<Contact>& contacts,
                                 RowSOA& rows,
                                 JointSOA& joints,
                                 const SolverParams& params,
                                 SolverDebugInfo* debug_info = nullptr);

void solve_scalar_soa_vectorized(std::vector<RigidBody>& bodies,
                                 std::vector<Contact>& contacts,
                                 RowSOA& rows,
                                 const SolverParams& params,
                                 SolverDebugInfo* debug_info = nullptr);
