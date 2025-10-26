#include "benchmark/benchmark.h"

#include "contact_gen.hpp"
#include "metrics.hpp"
#include "scenes.hpp"
#include "solver_baseline_vec.hpp"
#include "solver_scalar_cached.hpp"
#include "solver_scalar_soa.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {
struct BenchmarkResult {
  std::string scene;
  std::string solver;
  int iterations = 0;
  std::size_t bodies = 0;
  std::size_t contacts = 0;
  double ms_per_step = 0.0;
  double drift_max = 0.0;
  double penetration_linf = 0.0;
  double energy_drift = 0.0;
  double cone_consistency = 0.0;
};

std::vector<BenchmarkResult> g_results;

void record_result(const BenchmarkResult& result) {
  g_results.push_back(result);
}

constexpr int kStepsPerRun = 60;

BenchmarkResult run_baseline_result(const std::string& scene_name,
                                    const Scene& base_scene,
                                    const BaselineParams& params,
                                    int steps,
                                    double ms_per_step) {
  std::vector<RigidBody> bodies = base_scene.bodies;
  std::vector<Contact> contacts = base_scene.contacts;
  const std::vector<RigidBody> pre = bodies;
  for (int i = 0; i < steps; ++i) {
    build_contact_offsets_and_bias(bodies, contacts, params);
    solve_baseline(bodies, contacts, params);
  }
  BenchmarkResult result;
  result.scene = scene_name;
  result.solver = "baseline_vec";
  result.iterations = params.iterations;
  result.bodies = base_scene.bodies.size();
  result.contacts = base_scene.contacts.size();
  result.ms_per_step = ms_per_step;
  result.drift_max = directional_momentum_drift(pre, bodies).max_abs;
  result.penetration_linf = constraint_penetration_Linf(contacts);
  result.energy_drift = energy_drift(pre, bodies);
  result.cone_consistency = cone_consistency(contacts);
  return result;
}

BenchmarkResult run_cached_result(const std::string& scene_name,
                                  const Scene& base_scene,
                                  const SolverParams& params,
                                  int steps,
                                  double ms_per_step) {
  std::vector<RigidBody> bodies = base_scene.bodies;
  std::vector<Contact> contacts = base_scene.contacts;
  const std::vector<RigidBody> pre = bodies;
  for (int i = 0; i < steps; ++i) {
    build_contact_offsets_and_bias(bodies, contacts, params);
    solve_scalar_cached(bodies, contacts, params);
  }
  BenchmarkResult result;
  result.scene = scene_name;
  result.solver = "scalar_cached";
  result.iterations = params.iterations;
  result.bodies = base_scene.bodies.size();
  result.contacts = base_scene.contacts.size();
  result.ms_per_step = ms_per_step;
  result.drift_max = directional_momentum_drift(pre, bodies).max_abs;
  result.penetration_linf = constraint_penetration_Linf(contacts);
  result.energy_drift = energy_drift(pre, bodies);
  result.cone_consistency = cone_consistency(contacts);
  return result;
}

BenchmarkResult run_soa_result(const std::string& scene_name,
                               const Scene& base_scene,
                               const SolverParams& params,
                               int steps,
                               double ms_per_step) {
  std::vector<RigidBody> bodies = base_scene.bodies;
  std::vector<Contact> contacts = base_scene.contacts;
  const std::vector<RigidBody> pre = bodies;
  for (int i = 0; i < steps; ++i) {
    build_contact_offsets_and_bias(bodies, contacts, params);
    RowSOA rows = build_soa(bodies, contacts, params);
    solve_scalar_soa(bodies, contacts, rows, params);
  }
  BenchmarkResult result;
  result.scene = scene_name;
  result.solver = "scalar_soa";
  result.iterations = params.iterations;
  result.bodies = base_scene.bodies.size();
  result.contacts = base_scene.contacts.size();
  result.ms_per_step = ms_per_step;
  result.drift_max = directional_momentum_drift(pre, bodies).max_abs;
  result.penetration_linf = constraint_penetration_Linf(contacts);
  result.energy_drift = energy_drift(pre, bodies);
  result.cone_consistency = cone_consistency(contacts);
  return result;
}

template <typename StepFn>
double run_benchmark_loop(benchmark::State& state,
                          const Scene& base_scene,
                          const StepFn& step_fn) {
  double total_elapsed = 0.0;
  int iterations = 0;
  for (auto _ : state) {
    std::vector<RigidBody> bodies = base_scene.bodies;
    std::vector<Contact> contacts = base_scene.contacts;
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < kStepsPerRun; ++step) {
      step_fn(bodies, contacts);
    }
    const auto end = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(end - start).count();
    total_elapsed += elapsed;
    ++iterations;
    state.SetIterationTime(elapsed);
  }
  return iterations > 0 ? (total_elapsed / iterations / static_cast<double>(kStepsPerRun)) * 1e3
                        : 0.0;
}

void update_counters(benchmark::State& state,
                     const Scene& scene,
                     double ms_per_step) {
  state.counters["bodies"] = benchmark::Counter(
      static_cast<double>(scene.bodies.size()), benchmark::Counter::kAvgThreads);
  state.counters["contacts"] = benchmark::Counter(
      static_cast<double>(scene.contacts.size()), benchmark::Counter::kAvgThreads);
  state.counters["ms_per_step"] =
      benchmark::Counter(ms_per_step, benchmark::Counter::kAvgThreads);
}

void BenchSpheresCloudBaseline4096(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_spheres_box_cloud(4096);
  BaselineParams params;
  params.beta = 0.2;
  params.slop = 0.005;
  params.restitution = 0.0;
  params.iterations = 10;
  params.dt = 1.0 / 60.0;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene, [&](std::vector<RigidBody>& bodies, std::vector<Contact>& contacts) {
        build_contact_offsets_and_bias(bodies, contacts, params);
        solve_baseline(bodies, contacts, params);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_baseline_result("spheres_box_cloud_4096", base_scene, params,
                                      kStepsPerRun, ms_per_step));
    recorded = true;
  }
}

void BenchSpheresCloudCached4096(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_spheres_box_cloud(4096);
  SolverParams params;
  params.beta = 0.2;
  params.slop = 0.005;
  params.restitution = 0.0;
  params.iterations = 10;
  params.dt = 1.0 / 60.0;
  params.mu = 0.5;
  params.warm_start = true;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene, [&](std::vector<RigidBody>& bodies, std::vector<Contact>& contacts) {
        build_contact_offsets_and_bias(bodies, contacts, params);
        solve_scalar_cached(bodies, contacts, params);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_cached_result("spheres_box_cloud_4096", base_scene, params,
                                    kStepsPerRun, ms_per_step));
    recorded = true;
  }
}

void BenchSpheresCloudSoA4096(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_spheres_box_cloud(4096);
  SolverParams params;
  params.beta = 0.2;
  params.slop = 0.005;
  params.restitution = 0.0;
  params.iterations = 10;
  params.dt = 1.0 / 60.0;
  params.mu = 0.5;
  params.warm_start = true;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene, [&](std::vector<RigidBody>& bodies, std::vector<Contact>& contacts) {
        build_contact_offsets_and_bias(bodies, contacts, params);
        RowSOA rows = build_soa(bodies, contacts, params);
        solve_scalar_soa(bodies, contacts, rows, params);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_soa_result("spheres_box_cloud_4096", base_scene, params,
                                 kStepsPerRun, ms_per_step));
    recorded = true;
  }
}

void BenchSpheresCloudBaseline8192(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_spheres_box_cloud(8192);
  BaselineParams params;
  params.beta = 0.2;
  params.slop = 0.005;
  params.restitution = 0.0;
  params.iterations = 10;
  params.dt = 1.0 / 60.0;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene, [&](std::vector<RigidBody>& bodies, std::vector<Contact>& contacts) {
        build_contact_offsets_and_bias(bodies, contacts, params);
        solve_baseline(bodies, contacts, params);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_baseline_result("spheres_box_cloud_8192", base_scene, params,
                                      kStepsPerRun, ms_per_step));
    recorded = true;
  }
}

void BenchSpheresCloudCached8192(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_spheres_box_cloud(8192);
  SolverParams params;
  params.beta = 0.2;
  params.slop = 0.005;
  params.restitution = 0.0;
  params.iterations = 10;
  params.dt = 1.0 / 60.0;
  params.mu = 0.5;
  params.warm_start = true;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene, [&](std::vector<RigidBody>& bodies, std::vector<Contact>& contacts) {
        build_contact_offsets_and_bias(bodies, contacts, params);
        solve_scalar_cached(bodies, contacts, params);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_cached_result("spheres_box_cloud_8192", base_scene, params,
                                    kStepsPerRun, ms_per_step));
    recorded = true;
  }
}

void BenchSpheresCloudSoA8192(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_spheres_box_cloud(8192);
  SolverParams params;
  params.beta = 0.2;
  params.slop = 0.005;
  params.restitution = 0.0;
  params.iterations = 10;
  params.dt = 1.0 / 60.0;
  params.mu = 0.5;
  params.warm_start = true;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene, [&](std::vector<RigidBody>& bodies, std::vector<Contact>& contacts) {
        build_contact_offsets_and_bias(bodies, contacts, params);
        RowSOA rows = build_soa(bodies, contacts, params);
        solve_scalar_soa(bodies, contacts, rows, params);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_soa_result("spheres_box_cloud_8192", base_scene, params,
                                 kStepsPerRun, ms_per_step));
    recorded = true;
  }
}

void BenchBoxStackBaseline(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_box_stack(8);
  BaselineParams params;
  params.beta = 0.2;
  params.slop = 0.005;
  params.restitution = 0.0;
  params.iterations = 10;
  params.dt = 1.0 / 60.0;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene, [&](std::vector<RigidBody>& bodies, std::vector<Contact>& contacts) {
        build_contact_offsets_and_bias(bodies, contacts, params);
        solve_baseline(bodies, contacts, params);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_baseline_result("box_stack_8", base_scene, params, kStepsPerRun,
                                      ms_per_step));
    recorded = true;
  }
}

void BenchBoxStackCached(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_box_stack(8);
  SolverParams params;
  params.beta = 0.2;
  params.slop = 0.005;
  params.restitution = 0.0;
  params.iterations = 10;
  params.dt = 1.0 / 60.0;
  params.mu = 0.5;
  params.warm_start = true;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene, [&](std::vector<RigidBody>& bodies, std::vector<Contact>& contacts) {
        build_contact_offsets_and_bias(bodies, contacts, params);
        solve_scalar_cached(bodies, contacts, params);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_cached_result("box_stack_8", base_scene, params, kStepsPerRun,
                                    ms_per_step));
    recorded = true;
  }
}

void BenchBoxStackSoA(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_box_stack(8);
  SolverParams params;
  params.beta = 0.2;
  params.slop = 0.005;
  params.restitution = 0.0;
  params.iterations = 10;
  params.dt = 1.0 / 60.0;
  params.mu = 0.5;
  params.warm_start = true;

  const double ms_per_step = run_benchmark_loop(
      state, base_scene, [&](std::vector<RigidBody>& bodies, std::vector<Contact>& contacts) {
        build_contact_offsets_and_bias(bodies, contacts, params);
        RowSOA rows = build_soa(bodies, contacts, params);
        solve_scalar_soa(bodies, contacts, rows, params);
      });

  update_counters(state, base_scene, ms_per_step);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_soa_result("box_stack_8", base_scene, params, kStepsPerRun,
                                 ms_per_step));
    recorded = true;
  }
}

void write_results_csv() {
  if (g_results.empty()) {
    return;
  }
  std::filesystem::create_directories("results");
  const std::string path = "results/results.csv";
  const bool exists = std::filesystem::exists(path);
  std::ofstream out(path, std::ios::app);
  if (!exists || out.tellp() == 0) {
    out << "scene,solver,iterations,N_bodies,N_contacts,ms_per_step,drift_max,Linf_penetration,energy_drift,cone_consistency\n";
  }
  for (const BenchmarkResult& r : g_results) {
    out << r.scene << ',' << r.solver << ',' << r.iterations << ',' << r.bodies << ','
        << r.contacts << ',' << r.ms_per_step << ',' << r.drift_max << ','
        << r.penetration_linf << ',' << r.energy_drift << ',' << r.cone_consistency
        << '\n';
  }
}

void print_results_table() {
  if (g_results.empty()) {
    return;
  }
  std::cout << "\nBenchmark Summary\n";
  std::cout << std::left << std::setw(24) << "scene" << std::setw(16) << "solver"
            << std::setw(12) << "ms/step" << std::setw(14) << "drift" << std::setw(16)
            << "energy" << std::setw(18) << "cone" << '\n';
  for (const BenchmarkResult& r : g_results) {
    std::cout << std::left << std::setw(24) << r.scene << std::setw(16) << r.solver
              << std::setw(12) << std::fixed << std::setprecision(3) << r.ms_per_step
              << std::setw(14) << std::scientific << std::setprecision(3)
              << r.drift_max << std::setw(16) << r.energy_drift << std::setw(18)
              << r.cone_consistency << '\n';
  }
  std::cout << std::defaultfloat;
}

}  // namespace

BENCHMARK(BenchSpheresCloudBaseline4096)->UseManualTime();
BENCHMARK(BenchSpheresCloudCached4096)->UseManualTime();
BENCHMARK(BenchSpheresCloudSoA4096)->UseManualTime();
BENCHMARK(BenchSpheresCloudBaseline8192)->UseManualTime();
BENCHMARK(BenchSpheresCloudCached8192)->UseManualTime();
BENCHMARK(BenchSpheresCloudSoA8192)->UseManualTime();
BENCHMARK(BenchBoxStackBaseline)->UseManualTime();
BENCHMARK(BenchBoxStackCached)->UseManualTime();
BENCHMARK(BenchBoxStackSoA)->UseManualTime();

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  write_results_csv();
  print_results_table();
  benchmark::Shutdown();
  return 0;
}
