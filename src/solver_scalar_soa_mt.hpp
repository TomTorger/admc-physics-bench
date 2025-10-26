#pragma once

#include <vector>

struct RigidBody;
struct Contact;
struct RowSOA;
struct JointSOA;
struct SolverDebugInfo;
struct SoaParams;

void solve_scalar_soa_mt(std::vector<RigidBody>& bodies,
                         std::vector<Contact>& contacts,
                         RowSOA& rows,
                         JointSOA& joints,
                         const SoaParams& params,
                         SolverDebugInfo* debug_info = nullptr);

void solve_scalar_soa_mt(std::vector<RigidBody>& bodies,
                         std::vector<Contact>& contacts,
                         RowSOA& rows,
                         const SoaParams& params,
                         SolverDebugInfo* debug_info = nullptr);
