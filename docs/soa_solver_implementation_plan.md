# Plan to Surpass Classic AoS/PGS Solvers

This plan sequences the work required for the Structure-of-Arrays (SoA) solver family to outperform the classic array-of-structures (AoS) Projected Gauss–Seidel (PGS) algorithms while preserving ADMC-aligned physics quality. Each stage builds on existing SIMD batches and benchmarking infrastructure.

## Stage 0 — Prerequisites (Done / maintain)
- Keep benchmark harness and CSV schema stable so regressions surface quickly.
- Preserve the ADMC scalar-row invariants documented in `docs/alg_*` notes.
- Maintain the SoA improvement diary (`docs/soa_improvement_potentials.md`) after every material change.

## Stage 1 — End-to-end SoA body pipeline
**Goal:** Remove residual AoS dependencies so contacts, joints, and body state stay in SoA across the solve.

Tasks
- Introduce dual-SoA body storage (`types.hpp`, `solver_scalar_soa*.cc`) with indirection tables mapping scene bodies to lane slots.
- Update warm-start and iteration entry points to load/write `vx[]`, `vy[]`, `vz[]`, `wx[]`, `wy[]`, `wz[]` arrays without struct reconstruction.
- Move scatter helpers in `solver_scalar_soa_vectorized.cc` to operate on the SoA body buffers and flush results back to AoS only after the solve finishes.

Quality Gates
- Benchmarks must show identical momentum drift and constraint error to the current reference at 10 iterations.
- Deterministic regression tests (`tests/`) stay bitwise identical.

## Stage 2 — SIMD warm start and iteration fusion
**Goal:** Keep SIMD lanes hot from warm start through friction solves.

Tasks
- Vectorize the warm-start kernel so the SIMD batcher in `solver_scalar_soa_vectorized.cc` applies cached impulses with masked FMA updates.
- Fuse normal and tangential solves into a single lane loop that reuses relative velocity terms produced for the normal row.
- Replace per-lane Coulomb disk projection with a vectorized clamp that consumes squared magnitudes to defer square roots.

Quality Gates
- Benchmark `spheres_cloud_4096` must show ≥10% reduction in solver iteration time versus the baseline vectorized build.
- Validate warm-start determinism by running `./build/tests/solver_warmstart_test` twice and diffing outputs (script in `scripts/` to be added if missing).

## Stage 3 — Vector-friendly scatter/accumulate
**Goal:** Eliminate scalar gather/scatter bottlenecks now dominating solver cost.

Tasks
- Introduce lane-local scratchpads per body pair so impulse deltas accumulate before a single SoA write (files: `solver_scalar_soa_simd.cc`, `soa_pack.hpp`).
- Experiment with `std::span`-backed micro-tiles sorted by body index to guarantee contiguous stores for high-degree bodies.
- Add AVX2/NEON gather/scatter helpers with fallback scalar paths for tail masks.

Quality Gates
- Solver iteration share should fall below 45% of frame time on `spheres_cloud_4096` at 10 iterations (see benchmark CSV).
- Profiling (`scripts/profile_solver.py`) must confirm L1 miss rate drops relative to current SIMD build.

## Stage 4 — Parallel row assembly and reuse
**Goal:** Close the remaining gap by shrinking row construction costs.

Tasks
- Thread the row builder (`solver_scalar_soa.cc`, `solver_scalar_soa_mt.cc`) over coarse batches using the existing job system (or introduce a simple thread pool under `bench/`).
- Cache invariant geometric terms (world inertia, tangent axes) between frames keyed by contact ID to avoid recompute on persistent manifolds.
- Extend SIMD batching infrastructure to row assembly so dot-product chains operate on vector lanes before the iteration loop begins.

Quality Gates
- Row build share drops below 15% on dense cloud scenes (check `results/results.csv`).
- Multi-threaded builder preserves determinism: re-run benchmark twice and diff CSV outputs.

## Stage 5 — Adaptive iteration strategy and cleanup
**Goal:** Ensure the solver scales gracefully and remains maintainable.

Tasks
- Implement convergence-based early exit in `solver_scalar_soa_vectorized.cc` with tunable thresholds exposed via `SolverParams`.
- Add instrumentation to `metrics_micro.*` to record per-iteration residual norms so adaptive logic can be tuned offline.
- Refresh documentation (README, SoA design note) and publish a benchmark appendix demonstrating wins over classic PGS.

Quality Gates
- Adaptive iterations should not degrade `cone_consistency` or `drift_max`; guard with automated checks in `tests/`.
- Final release notes include before/after charts for `spheres_cloud_4096` and `box_stack_layers` showing ≥20% wall-clock improvement over AoS baseline while staying within guardrails.

## Tracking & Reporting
- After each stage, append findings to `docs/soa_improvement_potentials.md` with timing tables and takeaways.
- Use `results/` CSV snapshots (stored with dated filenames) to anchor performance claims.
- Summarize milestone progress in PR descriptions and link back to this plan.

This staged approach keeps the solver correct, measurable, and incrementally faster, making it feasible to surpass the classic AoS/PGS implementations without sacrificing the ADMC theoretical guarantees.
