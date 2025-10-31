# Implementation Brief: Fully SoA-Native SIMD Gauss–Seidel Solver

This brief is for Codex. It lays out the coding tasks required to implement the "fully SoA-native, SIMD-batched Gauss–Seidel" solver that the design notes identify as the sure-bet win for ADMC. Follow each step precisely and keep physics guardrails intact (drift, penetration, cone metrics).

## 1. High-Level Outcomes
1. Introduce a new solver variant that keeps bodies and contacts in structure-of-arrays buffers for the entire Gauss–Seidel loop, using the ADMC scalars (`p_{k^\pm}`) already defined in the project.
2. Wire the solver into the benchmark harness so it can be selected via CLI flags and included in CSV outputs alongside existing solvers.
3. Document the solver in the public docs so maintainers understand its pipeline, configuration knobs, and benchmark expectations.

## 2. Required Code Changes

### 2.1 Create the Solver Module
- Add new source files: `src/solver_scalar_soa_native.hpp` and `src/solver_scalar_soa_native.cc`.
- Expose the entry point `BenchmarkResult solve_scalar_soa_native(...)` mirroring the signatures of the other solver front-ends (see `solver_scalar_soa_vectorized.hpp`). Keep function names consistent with the existing `solve_scalar_soa_*` family so bench integration is straightforward.
- In the header, declare a `SoaNativeStats` struct carrying timing counters similar to `SoaTimingBreakdown` but focused on: staging, warm start, normal iteration, friction iteration, writeback. Provide a helper to convert to `SoaTimingBreakdown` so telemetry remains uniform.

### 2.2 Structure-of-Arrays Staging
- Reuse the tiler utilities in `src/soa/tiler.{hpp,cpp}` to harvest contacts into SIMD-friendly batches. Extend the tiler if necessary to produce SoA body velocity slices; any additions stay in `src/soa/`.
- Ensure body state is mirrored into SoA buffers (`linVelX[]`, `linVelY[]`, etc.). If these helpers do not exist, add them alongside the solver implementation but keep them local to the new solver module.
- Persist warm-start impulses in parallel SoA arrays so the solver can warm start without converting back to AoS. Hook into the existing warm-start storage types defined in `solver_scalar_soa.hpp`.

### 2.3 SIMD Gauss–Seidel Loop
- Implement lane-wide kernels that:
  1. Load SoA body velocities for both bodies in the batch.
  2. Compute relative velocities against pre-staged Jacobian axes (normals and two tangents) using FMA operations when available.
  3. Update normal impulses with Baumgarte biasing and clamp via the ADMC invariants.
  4. Solve the two tangent constraints jointly using friction disk projection that respects `params.mu` and cached `|lambda_n|`.
  5. Write delta impulses back through SoA body arrays immediately (Gauss–Seidel order) so later constraints see updated velocities.
- Keep SIMD width configurable via compile-time constants (e.g., 4 for AVX2, 8 for AVX-512) and provide masked fallbacks for tail lanes.
- Measure per-phase timings with `ScopedTimer` utilities already used elsewhere (`metrics_micro.hpp`). Accumulate into the `SoaNativeStats` struct.

### 2.4 Integration with Existing Interfaces
- Update `src/CMakeLists.txt` inside the root `CMakeLists.txt` `add_library(core ...)` list to include the new `.cc` file.
- Create `src/solver_scalar_soa_native.hpp` declarations and include them where needed.
- Add the solver to `bench/bench_main.cc`:
  - Include the new header near the other solver includes.
  - Extend the solver-name parsing logic (search for `solver == "scalar_soa_vectorized"`) to accept aliases: `scalar_soa_native`, `soa_native`, and `native_soa`.
  - Implement a `run_soa_native_result(...)` helper modeled after `run_soa_vectorized_result`. Ensure CSV rows set `solver` to `scalar_soa_native`, fill `soa_timings` using the converted `SoaTimingBreakdown`, and mark `simd=true`.
  - Register the solver in the `--solvers=` CLI expansion so it is available for automated sweeps.

### 2.5 Testing Hooks
- Add parity tests in `tests/test_solvers.cc` or create a new test file if cleaner. Confirm the new solver matches baseline impulse outcomes within existing tolerances on at least the default regression scenes.
- Update `tests/parity_soa_vs_baseline.cc` (or create a sibling) to exercise the new solver.
- Ensure the solver participates in `test_simd_parity` so SIMD and scalar code paths stay in sync.

## 3. Documentation Requirements
- Add a new subsection to `docs/soa_solver_design.md` titled "Fully SoA-Native SIMD Solver" summarizing the algorithm, data layout, and benchmarks it targets. Cross-link to this brief if helpful.
- Update or create a doc under `docs/` (e.g., `docs/soa_native_solver_usage.md`) explaining:
  - CLI flags to run the solver via `bench`.
  - Any tunable parameters (tile size, SIMD width toggles).
  - Expected performance characteristics compared to `scalar_soa_vectorized` based on the design intent.
- Mention the requirement to keep ADMC invariants (`p_{k^\pm}`) intact and how the solver enforces them.

## 4. Benchmark Checklist
- After implementation, run the standard large-scene sweeps used in prior reports, e.g.:
  - `./build/bench/bench --scene=spheres_cloud_10k --solvers=baseline,scalar_soa_native,scalar_soa_vectorized --steps=30 --iters=10`
  - `./build/bench/bench --scene=spheres_cloud_50k --solvers=baseline,scalar_soa_native --steps=30 --iters=10`
- Capture CSV output paths and summarize relative `ms_per_step`, `drift_max`, and `cone_consistency`.
- Record any microbench timings if the new solver introduces dedicated kernels.

## 5. Engineering Guardrails
- Preserve determinism across runs—verify repeatability with identical seeds.
- Do not change shared solver parameters defaults unless necessary; if you must, document the rationale in the code comments and docs.
- Keep the new solver optional: existing pipelines must continue to build even if `scalar_soa_native` is not selected at runtime.
- Maintain cross-platform considerations (AVX2, AVX-512, NEON). Provide scalar fallbacks when SIMD is unavailable.

Following this checklist will deliver a production-ready, ADMC-aligned SoA solver integrated with the benchmarking and documentation infrastructure.
