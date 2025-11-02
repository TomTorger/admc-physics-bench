// bench/bench_runner.hpp
#pragma once
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "bench_cli.hpp"
#include "bench_scenes.hpp"
#include "bench_csv_schema.hpp"

#include "contact_gen.hpp"
#include "joints.hpp"
#include "metrics.hpp"
#include "metrics_micro.hpp"
#include "scenes.hpp"
#include "solver_baseline_vec.hpp"
#include "solver_scalar_cached.hpp"
#include "solver_scalar_soa.hpp"
#include "solver_scalar_soa_native.hpp"
#include "solver/solver_scalar_soa_native_par.hpp"
#include "solver_scalar_soa_vectorized.hpp"
#include "solver_scalar_soa_mt.hpp"
#include "solver_scalar_soa_simd.hpp"
#include "soa_pack.hpp"

namespace bench {

using Clock      = std::chrono::steady_clock;
using DurationMs = std::chrono::duration<double, std::milli>;
inline double elapsed_ms(const Clock::time_point& a, const Clock::time_point& b) {
  return DurationMs(b - a).count();
}

struct BenchResult {
  std::string scene;
  std::string solver;
  int iterations = 0;
  int steps      = 0;
  double dt      = 0.0;
  std::size_t bodies  = 0;
  std::size_t contacts= 0;
  std::size_t joints  = 0;
  int tile_size       = 0;
  double ms_per_step  = 0.0;
  double drift_max    = 0.0;
  double penetration_linf = 0.0;
  double energy_drift     = 0.0;
  double cone_consistency = 0.0;
  double joint_Linf       = 0.0;
  bool   simd     = false;
  int    threads  = 1;
  bool   has_soa_timings = false;
  double soa_parallel_stage_ms = 0.0;
  double soa_parallel_scatter_ms = 0.0;
  SoaTimingBreakdown soa_timings{};
  std::string soa_debug_summary;
  std::string commit_sha; // not resolved here
};

inline void configure_baseline_params(const std::string& scene, BaselineParams& p) {
  if (scene == "two_spheres") {
    p.beta = 0.0; p.slop = 0.0; p.restitution = 1.0;
  }
}
inline void configure_solver_params(const std::string& scene, SolverParams& p) {
  if (scene == "two_spheres") {
    p.beta = 0.0; p.slop = 0.0; p.restitution = 1.0;
    p.mu = 0.0; p.spheres_only = true; p.frictionless = true;
  } else if (scene == "spheres_cloud_10k" || scene == "spheres_cloud_50k" ||
             scene == "spheres_cloud_1024" || scene == "spheres_cloud_4096" ||
             scene == "spheres_cloud_8192") {
    p.mu = 0.0; p.spheres_only = true; p.frictionless = true;
  } else if (scene == "spheres_cloud_10k_fric") {
    if (p.mu <= 0.0) p.mu = 0.5;
    p.spheres_only = true; p.frictionless = false;
  }
}

inline int steps_for_scene_default(const std::string& scene, int fallback_steps) {
  if (scene == "two_spheres") return 1;
  return std::max(1, fallback_steps);
}

enum class SoaVariant { Legacy, Vectorized, Native, Parallel };

using SoaSolveFn = void (*)(std::vector<RigidBody>&,
                            std::vector<Contact>&,
                            RowSOA&,
                            JointSOA&,
                            const SoaParams&,
                            SolverDebugInfo*);

inline SoaSolveFn select_soa_solver(SoaVariant v) {
  switch (v) {
    case SoaVariant::Legacy:
      return [](auto& bodies, auto& contacts, auto& rows, auto& joints, const SoaParams& params, SolverDebugInfo* dbg) {
        solve_scalar_soa(bodies, contacts, rows, joints, params, dbg);
      };
    case SoaVariant::Vectorized:
      return [](auto& bodies, auto& contacts, auto& rows, auto& joints, const SoaParams& params, SolverDebugInfo* dbg) {
        solve_scalar_soa_vectorized(bodies, contacts, rows, joints, params, dbg);
      };
    case SoaVariant::Native:
      return [](auto& bodies, auto& contacts, auto& rows, auto& joints, const SoaParams& params, SolverDebugInfo* dbg) {
        bool used_parallel = false;
#if defined(ADMC_ENABLE_PARALLEL) && !defined(ADMC_DETERMINISTIC)
        if (params.use_threads && params.thread_count > 1) {
          used_parallel = admc::solve_scalar_soa_native_parallel(bodies, contacts, rows, joints, params, dbg);
        }
#endif
        if (!used_parallel) {
          solve_scalar_soa_native(bodies, contacts, rows, joints, params, dbg);
        }
      };
    case SoaVariant::Parallel:
      return [](auto& bodies, auto& contacts, auto& rows, auto& joints, const SoaParams& params, SolverDebugInfo* dbg) {
#if defined(ADMC_ENABLE_PARALLEL) && !defined(ADMC_DETERMINISTIC)
        if (params.use_threads && params.thread_count > 1) {
          if (admc::solve_scalar_soa_parallel(bodies, contacts, rows, joints, params, dbg)) {
            return;
          }
        }
#endif
        solve_scalar_soa_native(bodies, contacts, rows, joints, params, dbg);
      };
  }
  return nullptr;
}

inline const char* solver_label(SoaVariant v) {
  switch (v) {
    case SoaVariant::Legacy:     return "scalar_soa";
    case SoaVariant::Vectorized: return "vec_soa";
    case SoaVariant::Native:     return "scalar_soa_native";
    case SoaVariant::Parallel:   return "scalar_soa_parallel";
  }
  return "scalar_soa";
}

inline BenchResult run_soa_variant(const std::string& scene_name,
                                   const Scene& base_scene,
                                   const SoaParams& params_in,
                                   int steps,
                                   double ms_per_step_hint,
                                   SoaVariant variant) {
  std::vector<RigidBody> bodies = base_scene.bodies;
  std::vector<Contact>   contacts = base_scene.contacts;
  std::vector<DistanceJoint> joints = base_scene.joints;
  const std::vector<RigidBody> pre = bodies;

  RowSOA rows; JointSOA joint_rows;
  SolverDebugInfo agg_dbg; agg_dbg.reset();
  SoaTimingBreakdown total{};
  auto solver = select_soa_solver(variant);

  for (int s = 0; s < steps; ++s) {
    SolverDebugInfo step_dbg; SoaTimingBreakdown t{};
    const auto step_begin = Clock::now();

    const auto c0 = Clock::now();
    build_contact_offsets_and_bias(bodies, contacts, params_in);
    const auto c1 = Clock::now(); t.contact_prep_ms += elapsed_ms(c0, c1);

    const auto r0 = Clock::now();
    build_soa(bodies, contacts, params_in, rows);
    const auto r1 = Clock::now(); t.row_build_ms += elapsed_ms(r0, r1);

    const auto jd0 = Clock::now();
    build_distance_joint_rows(bodies, joints, params_in.dt);
    const auto jd1 = Clock::now(); t.joint_distance_build_ms += elapsed_ms(jd0, jd1);

    const auto jp0 = Clock::now();
    build_joint_soa(bodies, joints, params_in.dt, joint_rows);
    const auto jp1 = Clock::now(); t.joint_pack_ms += elapsed_ms(jp0, jp1);

    const auto sv0 = Clock::now();
    solver(bodies, contacts, rows, joint_rows, params_in, &step_dbg);
    const auto sv1 = Clock::now();
    double solver_ms = step_dbg.timings.solver_total_ms;
    if (solver_ms <= 0.0) solver_ms = elapsed_ms(sv0, sv1);
    t.solver_total_ms        += solver_ms;
    t.solver_warmstart_ms    += step_dbg.timings.solver_warmstart_ms;
    t.solver_iterations_ms   += step_dbg.timings.solver_iterations_ms;
    t.solver_integrate_ms    += step_dbg.timings.solver_integrate_ms;

    const auto sc0 = Clock::now();
    scatter_joint_impulses(joint_rows, joints);
    const auto sc1 = Clock::now(); t.scatter_ms += elapsed_ms(sc0, sc1);

    const auto step_end = Clock::now();
    t.total_step_ms += elapsed_ms(step_begin, step_end);

    agg_dbg.accumulate(step_dbg);
    total.accumulate(t);
  }
  build_distance_joint_rows(bodies, joints, params_in.dt);

  BenchResult r;
  r.scene = scene_name;
  r.solver = solver_label(variant);
  r.iterations = params_in.iterations;
  r.steps = steps;
  r.dt = params_in.dt;
  r.bodies = base_scene.bodies.size();
  r.contacts = base_scene.contacts.size();
  r.joints = base_scene.joints.size();
  r.drift_max = directional_momentum_drift(pre, bodies).max_abs;
  r.penetration_linf = constraint_penetration_Linf(contacts);
  r.energy_drift = energy_drift(pre, bodies);
  r.cone_consistency = cone_consistency(contacts);
  r.joint_Linf = joint_error_Linf(joints);
  r.simd = params_in.use_simd;
  r.threads = params_in.use_threads ? std::max(1, params_in.thread_count) : 1;
  r.tile_size = params_in.tile_size;
  r.commit_sha = "";

  const double derived_ms = (steps > 0) ? (total.total_step_ms / double(steps)) : 0.0;
  r.ms_per_step = (ms_per_step_hint >= 0.0) ? ms_per_step_hint : derived_ms;

  if (steps > 0) {
    r.has_soa_timings = true;
    r.soa_timings = total;
    r.soa_timings.scale(1.0 / steps);
    if (agg_dbg.parallel_stage_ms > 0.0) {
      r.soa_parallel_stage_ms = agg_dbg.parallel_stage_ms / steps;
    }
    if (agg_dbg.parallel_scatter_ms > 0.0) {
      r.soa_parallel_scatter_ms = agg_dbg.parallel_scatter_ms / steps;
    }
    SolverDebugInfo dbg_avg = agg_dbg;
    dbg_avg.timings = r.soa_timings;
    if (steps > 0) {
      dbg_avg.parallel_stage_ms = r.soa_parallel_stage_ms;
      dbg_avg.parallel_scatter_ms = r.soa_parallel_scatter_ms;
    }
    r.soa_debug_summary = solver_debug_summary(dbg_avg);
  }
  return r;
}

inline BenchResult run_baseline(const std::string& scene_name,
                                const Scene& base_scene,
                                const BaselineParams& p,
                                int steps, double ms_hint) {
  std::vector<RigidBody> bodies = base_scene.bodies;
  std::vector<Contact> contacts = base_scene.contacts;
  const std::vector<RigidBody> pre = bodies;

  for (int i = 0; i < steps; ++i) {
    build_contact_offsets_and_bias(bodies, contacts, p);
    solve_baseline(bodies, contacts, p);
  }

  BenchResult r{};
  r.scene = scene_name; r.solver = "baseline";
  r.iterations = p.iterations; r.steps = steps; r.dt = p.dt;
  r.bodies = base_scene.bodies.size(); r.contacts = base_scene.contacts.size();
  r.joints = base_scene.joints.size();
  r.ms_per_step = ms_hint;
  r.drift_max = directional_momentum_drift(pre, bodies).max_abs;
  r.penetration_linf = constraint_penetration_Linf(contacts);
  r.energy_drift = energy_drift(pre, bodies);
  r.cone_consistency = cone_consistency(contacts);
  r.joint_Linf = 0.0;
  r.simd = false; r.threads = 1; r.tile_size = 0; r.commit_sha = "";
  return r;
}

inline BenchResult run_cached(const std::string& scene_name,
                              const Scene& base_scene,
                              const SolverParams& p,
                              int steps, double ms_hint) {
  std::vector<RigidBody> bodies = base_scene.bodies;
  std::vector<Contact> contacts = base_scene.contacts;
  std::vector<DistanceJoint> joints = base_scene.joints;
  const std::vector<RigidBody> pre = bodies;

  for (int i = 0; i < steps; ++i) {
    build_contact_offsets_and_bias(bodies, contacts, p);
    build_distance_joint_rows(bodies, joints, p.dt);
    solve_scalar_cached(bodies, contacts, joints, p);
  }
  build_distance_joint_rows(bodies, joints, p.dt);

  BenchResult r{};
  r.scene = scene_name; r.solver = "cached";
  r.iterations = p.iterations; r.steps = steps; r.dt = p.dt;
  r.bodies = base_scene.bodies.size(); r.contacts = base_scene.contacts.size();
  r.joints = base_scene.joints.size();
  r.ms_per_step = ms_hint;
  r.drift_max = directional_momentum_drift(pre, bodies).max_abs;
  r.penetration_linf = constraint_penetration_Linf(contacts);
  r.energy_drift = energy_drift(pre, bodies);
  r.cone_consistency = cone_consistency(contacts);
  r.joint_Linf = joint_error_Linf(joints);
  r.simd = false; r.threads = 1; r.tile_size = 0; r.commit_sha = "";
  return r;
}

// --------------------------- Public suite runner -----------------------------
inline std::vector<BenchResult>
run_suite_for_scene(const std::string& scene_name,
                    const Scene& scene,
                    const BenchConfig& cfg) {
  const int iterations = std::max(1, cfg.iterations);
  const int steps      = steps_for_scene_default(scene_name, cfg.steps);
  const double dt      = cfg.dt;

  // Tile sizes: only used by certain solvers; keep -1 sentinel as “no override”
  std::vector<int> tile_sizes = cfg.tile_sizes;
  if (tile_sizes.empty()) tile_sizes.push_back(-1);

  // Thread counts to sweep
  std::vector<int> thread_counts = cfg.threads_list;
  if (thread_counts.empty()) thread_counts.push_back(std::max(1, cfg.threads));

  std::vector<BenchResult> out;

  auto do_soa_variant = [&](SoaVariant v, bool use_simd) {
    for (int tc : thread_counts) {
      SoaParams p{};
      p.iterations   = iterations;
      p.dt           = dt;
#if defined(ADMC_DETERMINISTIC)
      const bool deterministic_mode = true;
#else
      const bool deterministic_mode = cfg.deterministic;
#endif
      p.use_threads  = (!deterministic_mode && tc > 1);
      p.thread_count = std::max(1, tc);
      p.use_simd     = use_simd;
      if (cfg.tile_rows > 0) p.tile_rows = cfg.tile_rows;
      p.spheres_only = cfg.spheres_only;
      p.frictionless = cfg.frictionless;
      if (cfg.convergence_threshold >= 0.0) p.convergence_threshold = cfg.convergence_threshold;
      configure_solver_params(scene_name, p);

      for (int ts : tile_sizes) {
        SoaParams p2 = p;
        if (ts > 0) { p2.tile_size = ts; p2.max_contacts_per_tile = ts; }
        auto r = run_soa_variant(scene_name, scene, p2, steps, -1.0, v);
        out.push_back(std::move(r));
      }
    }
  };

  for (const auto& solver : cfg.solvers) {
    if (solver == "baseline") {
      BaselineParams p{};
      p.iterations = iterations; p.dt = dt;
      configure_baseline_params(scene_name, p);
      const auto t0 = Clock::now();
      auto r = run_baseline(scene_name, scene, p, steps, 0.0);
      const auto t1 = Clock::now();
      r.ms_per_step = (steps > 0) ? (std::chrono::duration<double>(t1 - t0).count() * 1e3 / steps) : 0.0;
      out.push_back(std::move(r));
    } else if (solver == "cached") {
      SolverParams p{};
      p.iterations = iterations; p.dt = dt;
      configure_solver_params(scene_name, p);
      if (cfg.tile_rows > 0) p.tile_rows = cfg.tile_rows;
      p.spheres_only = cfg.spheres_only;
      p.frictionless = cfg.frictionless;
      const auto t0 = Clock::now();
      auto r = run_cached(scene_name, scene, p, steps, 0.0);
      const auto t1 = Clock::now();
      r.ms_per_step = (steps > 0) ? (std::chrono::duration<double>(t1 - t0).count() * 1e3 / steps) : 0.0;
      out.push_back(std::move(r));
    } else if (solver == "soa") {
      do_soa_variant(SoaVariant::Legacy, /*use_simd*/ false);
    } else if (solver == "vec_soa") {
      do_soa_variant(SoaVariant::Vectorized, /*use_simd*/ false);
    } else if (solver == "soa_native") {
      do_soa_variant(SoaVariant::Native, /*use_simd*/ true);
    } else if (solver == "soa_parallel") {
      do_soa_variant(SoaVariant::Parallel, /*use_simd*/ true);
    } else {
      std::cerr << "[bench] Unknown solver: " << solver << "\n";
    }
  }
  return out;
}

} // namespace bench
