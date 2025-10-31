#include "solver_scalar_soa_vectorized.hpp"

#include "solver_scalar_soa.hpp"
#include "solver_scalar_soa_simd.hpp"
#include "soa_pack.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>

namespace {

using Clock = std::chrono::steady_clock;
using DurationMs = std::chrono::duration<double, std::milli>;

constexpr bool kHasNativeSimd =
#if defined(ADMC_USE_AVX2) || defined(ADMC_USE_NEON)
    true;
#else
    false;
#endif

bool should_use_vectorized_kernel(const SoaParams& params, const RowSOA& rows) {
  if (!params.use_simd) {
    return false;
  }
  if (!kHasNativeSimd) {
    return false;
  }
  if (rows.N <= 0) {
    return false;
  }
  if (rows.N < soa::kLane) {
    return false;
  }
  return true;
}

double elapsed_ms(const Clock::time_point& begin, const Clock::time_point& end) {
  return DurationMs(end - begin).count();
}

void sanitize_solver_timings(SolverDebugInfo* info,
                             double measured_solver_ms) {
  if (!info) {
    return;
  }
  SoaTimingBreakdown& timings = info->timings;
  auto sanitize = [](double value) {
    if (!std::isfinite(value)) {
      return 0.0;
    }
    return std::max(0.0, value);
  };
  timings.solver_warmstart_ms = sanitize(timings.solver_warmstart_ms);
  timings.solver_iterations_ms = sanitize(timings.solver_iterations_ms);
  timings.solver_integrate_ms = sanitize(timings.solver_integrate_ms);
  timings.solver_total_ms = sanitize(timings.solver_total_ms);

  if (timings.solver_total_ms <= 0.0) {
    timings.solver_total_ms = sanitize(measured_solver_ms);
  }

  const double accounted = timings.solver_warmstart_ms +
                           timings.solver_iterations_ms +
                           timings.solver_integrate_ms;
  if (accounted <= 0.0) {
    timings.solver_iterations_ms = timings.solver_total_ms;
  } else if (timings.solver_iterations_ms <= 0.0 &&
             timings.solver_total_ms > timings.solver_warmstart_ms +
                                          timings.solver_integrate_ms) {
    timings.solver_iterations_ms =
        timings.solver_total_ms -
        (timings.solver_warmstart_ms + timings.solver_integrate_ms);
  }

  if (timings.total_step_ms <= 0.0) {
    timings.total_step_ms = timings.solver_total_ms;
  }
}

void run_scalar_fallback(std::vector<RigidBody>& bodies,
                         std::vector<Contact>& contacts,
                         RowSOA& rows,
                         JointSOA& joints,
                         const SoaParams& params,
                         SolverDebugInfo* debug_info) {
  const auto solver_begin = Clock::now();
  solve_scalar_soa_scalar(bodies, contacts, rows, joints, params, debug_info);
  const auto solver_end = Clock::now();
  sanitize_solver_timings(debug_info, elapsed_ms(solver_begin, solver_end));
}

void run_vectorized_kernel(std::vector<RigidBody>& bodies,
                           std::vector<Contact>& contacts,
                           RowSOA& rows,
                           JointSOA& joints,
                           const SoaParams& params,
                           SolverDebugInfo* debug_info) {
  const auto solver_begin = Clock::now();
  solve_scalar_soa_simd(bodies, contacts, rows, joints, params, debug_info);
  const auto solver_end = Clock::now();
  sanitize_solver_timings(debug_info, elapsed_ms(solver_begin, solver_end));
}

}  // namespace

void solve_scalar_soa_vectorized(std::vector<RigidBody>& bodies,
                                 std::vector<Contact>& contacts,
                                 RowSOA& rows,
                                 JointSOA& joints,
                                 const SoaParams& params,
                                 SolverDebugInfo* debug_info) {
  SolverDebugInfo local_info;
  SolverDebugInfo* info = debug_info ? debug_info : &local_info;

  if (should_use_vectorized_kernel(params, rows)) {
    run_vectorized_kernel(bodies, contacts, rows, joints, params, info);
  } else {
    run_scalar_fallback(bodies, contacts, rows, joints, params, info);
  }
}

void solve_scalar_soa_vectorized(std::vector<RigidBody>& bodies,
                                 std::vector<Contact>& contacts,
                                 RowSOA& rows,
                                 const SoaParams& params,
                                 SolverDebugInfo* debug_info) {
  static JointSOA empty_joints;
  empty_joints.clear();

  SolverDebugInfo local_info;
  SolverDebugInfo* info = debug_info ? debug_info : &local_info;

  if (should_use_vectorized_kernel(params, rows)) {
    run_vectorized_kernel(bodies, contacts, rows, empty_joints, params, info);
  } else {
    run_scalar_fallback(bodies, contacts, rows, empty_joints, params, info);
  }
}

void solve_scalar_soa_vectorized(std::vector<RigidBody>& bodies,
                                 std::vector<Contact>& contacts,
                                 RowSOA& rows,
                                 JointSOA& joints,
                                 const SolverParams& params,
                                 SolverDebugInfo* debug_info) {
  SoaParams derived;
  static_cast<SolverParams&>(derived) = params;

  SolverDebugInfo local_info;
  SolverDebugInfo* info = debug_info ? debug_info : &local_info;

  if (should_use_vectorized_kernel(derived, rows)) {
    run_vectorized_kernel(bodies, contacts, rows, joints, derived, info);
  } else {
    run_scalar_fallback(bodies, contacts, rows, joints, derived, info);
  }
}

void solve_scalar_soa_vectorized(std::vector<RigidBody>& bodies,
                                 std::vector<Contact>& contacts,
                                 RowSOA& rows,
                                 const SolverParams& params,
                                 SolverDebugInfo* debug_info) {
  SoaParams derived;
  static_cast<SolverParams&>(derived) = params;
  solve_scalar_soa_vectorized(bodies, contacts, rows, derived, debug_info);
}
