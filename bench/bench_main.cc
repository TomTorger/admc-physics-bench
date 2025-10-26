#include "benchmark/benchmark.h"

#include "contact_gen.hpp"
#include "metrics.hpp"
#include "scenes.hpp"
#include "solver_baseline_vec.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
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
};

std::vector<BenchmarkResult> g_results;

void record_result(const BenchmarkResult& result) {
  g_results.push_back(result);
}

BenchmarkResult run_steps(const std::string& scene_name,
                          const Scene& base_scene,
                          const BaselineParams& params,
                          int steps, double avg_ms_per_step) {
  std::vector<RigidBody> bodies = base_scene.bodies;
  std::vector<Contact> contacts = base_scene.contacts;
  const std::vector<RigidBody> pre = bodies;
  for (int i = 0; i < steps; ++i) {
    build_contact_offsets_and_bias(bodies, contacts, params);
    solve_baseline(bodies, contacts, params);
  }
  const Drift drift = directional_momentum_drift(pre, bodies);
  const double linf = constraint_penetration_Linf(contacts);

  BenchmarkResult result;
  result.scene = scene_name;
  result.solver = "baseline_vec";
  result.iterations = params.iterations;
  result.bodies = base_scene.bodies.size();
  result.contacts = base_scene.contacts.size();
  result.ms_per_step = avg_ms_per_step;
  result.drift_max = drift.max_abs;
  result.penetration_linf = linf;
  return result;
}

void write_results_csv() {
  if (g_results.empty()) {
    return;
  }
  std::filesystem::create_directories("results");
  std::ofstream out("results/results.csv", std::ios::trunc);
  out << "scene,solver,iterations,N_bodies,N_contacts,ms_per_step,drift_max,Linf_penetration\n";
  for (const BenchmarkResult& r : g_results) {
    out << r.scene << ',' << r.solver << ',' << r.iterations << ',' << r.bodies << ','
        << r.contacts << ',' << r.ms_per_step << ',' << r.drift_max << ','
        << r.penetration_linf << '\n';
  }
}

constexpr int kStepsPerRun = 60;

void BenchSpheresCloud(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_spheres_box_cloud(4096);
  BaselineParams params;
  params.beta = 0.2;
  params.restitution = 0.0;
  params.iterations = 10;
  params.dt = 1.0 / 60.0;

  double total_elapsed = 0.0;
  int iterations = 0;
  for (auto _ : state) {
    std::vector<RigidBody> bodies = base_scene.bodies;
    std::vector<Contact> contacts = base_scene.contacts;
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < kStepsPerRun; ++step) {
      build_contact_offsets_and_bias(bodies, contacts, params);
      solve_baseline(bodies, contacts, params);
    }
    const auto end = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(end - start).count();
    total_elapsed += elapsed;
    ++iterations;
    state.SetIterationTime(elapsed);
  }

  const double ms_per_step =
      iterations > 0 ? (total_elapsed / iterations / kStepsPerRun) * 1e3 : 0.0;

  state.counters["bodies"] = benchmark::Counter(static_cast<double>(base_scene.bodies.size()),
                                                 benchmark::Counter::kAvgThreads);
  state.counters["contacts"] = benchmark::Counter(static_cast<double>(base_scene.contacts.size()),
                                                   benchmark::Counter::kAvgThreads);
  state.counters["ms_per_step"] =
      benchmark::Counter(ms_per_step, benchmark::Counter::kAvgThreads);

  if (state.thread_index == 0 && !recorded) {
    record_result(
        run_steps("spheres_box_cloud", base_scene, params, kStepsPerRun, ms_per_step));
    recorded = true;
  }
}

void BenchBoxStack(benchmark::State& state) {
  static bool recorded = false;
  const Scene base_scene = make_box_stack(8);
  BaselineParams params;
  params.beta = 0.2;
  params.restitution = 0.0;
  params.iterations = 10;
  params.dt = 1.0 / 60.0;

  double total_elapsed = 0.0;
  int iterations = 0;
  for (auto _ : state) {
    std::vector<RigidBody> bodies = base_scene.bodies;
    std::vector<Contact> contacts = base_scene.contacts;
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < kStepsPerRun; ++step) {
      build_contact_offsets_and_bias(bodies, contacts, params);
      solve_baseline(bodies, contacts, params);
    }
    const auto end = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(end - start).count();
    total_elapsed += elapsed;
    ++iterations;
    state.SetIterationTime(elapsed);
  }

  const double ms_per_step =
      iterations > 0 ? (total_elapsed / iterations / kStepsPerRun) * 1e3 : 0.0;

  state.counters["bodies"] = benchmark::Counter(static_cast<double>(base_scene.bodies.size()),
                                                 benchmark::Counter::kAvgThreads);
  state.counters["contacts"] = benchmark::Counter(static_cast<double>(base_scene.contacts.size()),
                                                   benchmark::Counter::kAvgThreads);
  state.counters["ms_per_step"] =
      benchmark::Counter(ms_per_step, benchmark::Counter::kAvgThreads);

  if (state.thread_index == 0 && !recorded) {
    record_result(run_steps("box_stack", base_scene, params, kStepsPerRun, ms_per_step));
    recorded = true;
  }
}
}  // namespace

BENCHMARK(BenchSpheresCloud)->UseManualTime();
BENCHMARK(BenchBoxStack)->UseManualTime();

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  benchmark::RunSpecifiedBenchmarks();
  write_results_csv();
  benchmark::Shutdown();
  return 0;
}

