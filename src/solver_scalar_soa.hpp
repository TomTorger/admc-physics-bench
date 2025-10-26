#pragma once

#include "joints.hpp"
#include "solver_scalar_cached.hpp"

#include <string>
#include <thread>

struct SolverDebugInfo {
  int invalid_contact_indices = 0;
  int invalid_joint_indices = 0;
  int warmstart_contact_impulses = 0;
  int warmstart_joint_impulses = 0;
  int normal_impulse_clamps = 0;
  int tangent_projections = 0;
  int rope_clamps = 0;
  int singular_joint_denominators = 0;

  void reset() { *this = SolverDebugInfo{}; }

  void accumulate(const SolverDebugInfo& other) {
    invalid_contact_indices += other.invalid_contact_indices;
    invalid_joint_indices += other.invalid_joint_indices;
    warmstart_contact_impulses += other.warmstart_contact_impulses;
    warmstart_joint_impulses += other.warmstart_joint_impulses;
    normal_impulse_clamps += other.normal_impulse_clamps;
    tangent_projections += other.tangent_projections;
    rope_clamps += other.rope_clamps;
    singular_joint_denominators += other.singular_joint_denominators;
  }
};

std::string solver_debug_summary(const SolverDebugInfo& info);

struct SoaParams : SolverParams {
  bool use_simd = true;
  bool use_threads = true;
  int thread_count = std::thread::hardware_concurrency();
  int block_size = 128;
};

RowSOA build_soa(const std::vector<RigidBody>& bodies,
                 const std::vector<Contact>& contacts,
                 const SoaParams& params);

RowSOA build_soa(const std::vector<RigidBody>& bodies,
                 const std::vector<Contact>& contacts,
                 const SolverParams& params);

void solve_scalar_soa_scalar(std::vector<RigidBody>& bodies,
                             std::vector<Contact>& contacts,
                             RowSOA& rows,
                             JointSOA& joints,
                             const SoaParams& params,
                             SolverDebugInfo* debug_info = nullptr);

void solve_scalar_soa(std::vector<RigidBody>& bodies,
                      std::vector<Contact>& contacts,
                      RowSOA& rows,
                      JointSOA& joints,
                      const SoaParams& params,
                      SolverDebugInfo* debug_info = nullptr);

void solve_scalar_soa(std::vector<RigidBody>& bodies,
                      std::vector<Contact>& contacts,
                      RowSOA& rows,
                      JointSOA& joints,
                      const SolverParams& params,
                      SolverDebugInfo* debug_info = nullptr);

void solve_scalar_soa_scalar(std::vector<RigidBody>& bodies,
                             std::vector<Contact>& contacts,
                             RowSOA& rows,
                             const SoaParams& params,
                             SolverDebugInfo* debug_info = nullptr);

void solve_scalar_soa(std::vector<RigidBody>& bodies,
                      std::vector<Contact>& contacts,
                      RowSOA& rows,
                      const SoaParams& params,
                      SolverDebugInfo* debug_info = nullptr);

void solve_scalar_soa(std::vector<RigidBody>& bodies,
                      std::vector<Contact>& contacts,
                      RowSOA& rows,
                      const SolverParams& params,
                      SolverDebugInfo* debug_info = nullptr);
