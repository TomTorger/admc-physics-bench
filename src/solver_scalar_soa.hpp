#pragma once

#include "joints.hpp"
#include "soa/world.hpp"
#include "solver_scalar_cached.hpp"

#include <string>
#include <thread>

struct SoaTimingBreakdown {
  double contact_prep_ms = 0.0;
  double row_build_ms = 0.0;
  double joint_distance_build_ms = 0.0;
  double joint_pack_ms = 0.0;
  double solver_total_ms = 0.0;
  double solver_warmstart_ms = 0.0;
  double solver_iterations_ms = 0.0;
  double solver_integrate_ms = 0.0;
  double scatter_ms = 0.0;
  double total_step_ms = 0.0;

  void reset() { *this = SoaTimingBreakdown{}; }

  void accumulate(const SoaTimingBreakdown& other) {
    contact_prep_ms += other.contact_prep_ms;
    row_build_ms += other.row_build_ms;
    joint_distance_build_ms += other.joint_distance_build_ms;
    joint_pack_ms += other.joint_pack_ms;
    solver_total_ms += other.solver_total_ms;
    solver_warmstart_ms += other.solver_warmstart_ms;
    solver_iterations_ms += other.solver_iterations_ms;
    solver_integrate_ms += other.solver_integrate_ms;
    scatter_ms += other.scatter_ms;
    total_step_ms += other.total_step_ms;
  }

  void scale(double factor) {
    contact_prep_ms *= factor;
    row_build_ms *= factor;
    joint_distance_build_ms *= factor;
    joint_pack_ms *= factor;
    solver_total_ms *= factor;
    solver_warmstart_ms *= factor;
    solver_iterations_ms *= factor;
    solver_integrate_ms *= factor;
    scatter_ms *= factor;
    total_step_ms *= factor;
  }
};

struct SolverDebugInfo {
  int invalid_contact_indices = 0;
  int invalid_joint_indices = 0;
  int warmstart_contact_impulses = 0;
  int warmstart_joint_impulses = 0;
  int normal_impulse_clamps = 0;
  int tangent_projections = 0;
  int rope_clamps = 0;
  int singular_joint_denominators = 0;
  double parallel_stage_ms = 0.0;
  double parallel_scatter_ms = 0.0;
  SoaTimingBreakdown timings;

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
    parallel_stage_ms += other.parallel_stage_ms;
    parallel_scatter_ms += other.parallel_scatter_ms;
    timings.accumulate(other.timings);
  }
};

std::string solver_debug_summary(const SolverDebugInfo& info);

struct SoaParams : SolverParams {
  bool use_simd = true;
  bool use_threads = true;
  int thread_count = std::thread::hardware_concurrency();
  int block_size = 128;
  double convergence_threshold = 1e-4;
};

RowSOA build_soa(const std::vector<RigidBody>& bodies,
                 const std::vector<Contact>& contacts,
                 const SoaParams& params);

void build_soa(const std::vector<RigidBody>& bodies,
               const std::vector<Contact>& contacts,
               const SoaParams& params,
               RowSOA& rows);

RowSOA build_soa(const std::vector<RigidBody>& bodies,
                 const std::vector<Contact>& contacts,
                 const SolverParams& params);

void build_soa(const std::vector<RigidBody>& bodies,
               const std::vector<Contact>& contacts,
               const SolverParams& params,
               RowSOA& rows);

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

namespace soa {

void solve_soa(World& world, ContactManifold& cm, const SolverParams& params);

}
