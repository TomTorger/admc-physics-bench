#include <benchmark/benchmark.h>
// TODO(Codex): Bench scenes: build once, run fixed iterations, write CSV.
static void BM_Spheres10k_Baseline(benchmark::State& st){ /* ... */ }
BENCHMARK(BM_Spheres10k_Baseline);
BENCHMARK_MAIN();
