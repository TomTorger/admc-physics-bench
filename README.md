# ADMC Physics Bench
**Directional scalar contact solvers for games & simulation — benchmarked.**

This repository is a compact, self-contained lab for testing how far you can push **one-direction (scalar) contact rows**—motivated by **ADMC (Additive Directional Momentum Conservation)**—versus classic vector-per-row solvers. It targets rigid-body contacts (and joints), clean benchmarking, and reproducibility.

> 🔎 Background: See [docs/admc_overview.md](docs/admc_overview.md) for the theory and proof, and [docs/nr_math.md](docs/nr_math.md) for the Newtonian math this repo implements.

---

## Why this project exists

Modern physics engines typically resolve contacts as rows: **one scalar along a chosen direction** (normal or tangent) per constraint. Framing conservation “**in any direction**” lets us:
- reduce vector work to **scalar PGS updates**,
- cache effective masses and warm-start impulses,
- organize data as **SoA** (Structure-of-Arrays) for **SIMD/GPU-friendly** batches.

This repo makes those trade-offs measurable across representative scenes.

---

## Feature snapshot

- ✅ **Baseline solver (AoS, vector-per-row)** — reference implementation.
- ✅ **Scalar Cached solver (AoS)** — normal + friction as **scalar rows** with caching & warm-start.
- ✅ **SoA-batched solver** — same math, **Structure-of-Arrays** for better memory/throughput.
- ✅ **Vectorized SoA solver** — AVX2/NEON-aware contact batches with lane masks and SIMD microkernels.
- ✅ **Deterministic scenes** — from two-body cases to particle-like clouds and stacks.
- ✅ **Metrics** — directional-momentum drift, constraint error, energy drift, cone consistency.
- ✅ **Benchmark harness** — Google Benchmark + CSV output under `results/`.

> ℹ️ Joints (distance/rope with compliance) are designed to follow the same scalar-row model. See [docs/alg_scalar_distance_joint_math.md](docs/alg_scalar_distance_joint_math.md). If not yet present in code, they’re listed in the roadmap below.

---

## Repository layout

```

admc-physics-bench/
CMakeLists.txt
README.md
/bench/                # Google Benchmark entrypoints
/docs/                 # Theory + algorithm math notes (see links below)
/scripts/              # init, build, run-bench, summarization helpers
/src/
contact_gen.*            # Contact frame, offsets, bias (ERP), materials
math.hpp                 # Minimal header-only math (Vec3, Mat3, Quat)
metrics.*                # Drift, constraint error, energy, cone checks
metrics_micro.*          # Micro-profiler hooks for solver timing
scenes.*                 # Deterministic test scenes
soa_pack.hpp             # SIMD packing helpers shared by SoA solvers
solver_baseline_vec.*    # Baseline AoS vector-per-row solver
solver_scalar_cached.*   # Scalar AoS solver (cached, warm-start, friction)
solver_scalar_soa.*      # Scalar SoA solver core (scalar lanes)
solver_scalar_soa_mt.*   # Multi-threaded row assembly experiments
solver_scalar_soa_simd.* # SIMD kernels and helpers
solver_scalar_soa_vectorized.* # Vectorized solver entrypoints & batching
types.hpp                # RigidBody, Contact, RowSOA (and friends)
/tests/                  # Sanity & invariants tests
/results/              # CSV outputs from benches (gitignored except placeholder)
/.github/workflows/    # CI build/tests (if configured)

````

---

## Documentation map

- **ADMC overview** — theory, equivalence to 4-momentum:  
  `docs/admc_overview.md`
- **NR math primer** — state, projections, effective masses, impulses:  
  `docs/nr_math.md`
- **Algorithm notes (math):**
  - Scalar normal row: `docs/alg_scalar_normal_row_math.md`
  - Scalar friction rows: `docs/alg_scalar_friction_rows_math.md`
  - Distance/rope joints (scalar row + compliance): `docs/alg_scalar_distance_joint_math.md`
  - SoA-batched scalar rows: `docs/alg_scalar_soa_batched_math.md`
- **Optimization opportunities log** — persistent record of SoA solver improvement targets:
  `docs/soa_improvement_potentials.md`. Keep this file as the static location for documenting timing insights and optimization ideas so the history stays centralized.
- **Execution plan** — staged roadmap for surpassing classic solvers:
  `docs/soa_solver_implementation_plan.md`.

---

## Build & run

### Prereqs
- **CMake ≥ 3.20**
- A C++17 compiler (Clang, GCC, MSVC)
- (Linux/macOS) _optional:_ Ninja for faster builds

### Configure & build (Release)
```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
````

### Run tests

```bash
ctest --test-dir build --output-on-failure
```

### Run benchmarks

```bash
# Generic
./build/bench/bench --benchmark_out=results/results.csv --benchmark_out_format=csv

# Convenience script (if present)
bash scripts/run_bench.sh
```

> The bench runs a fixed number of solver steps on multiple scenes and appends to `results/results.csv`.

---

## Solvers (what’s implemented)

### 1) Baseline — AoS, vector-per-row

* PGS (Projected Gauss–Seidel) across **frictionless normal** rows (and optionally restitution/bias).
* Computes `v_rel` with full vectors every iteration; serves as a correctness reference.

### 2) Scalar Cached — AoS

* **Normal + friction** rows treated as scalars along the contact frame ((n, t_1, t_2)).
* Caches **effective masses** (k_n, k_{t_1}, k_{t_2}) and **warm-starts** impulses (j_n, j_{t_1}, j_{t_2}).
* Restitution and Baumgarte/ERP bias on the **normal** row; **Coulomb cone** projection on tangents.

### 3) SoA-batched scalar solver

* Same scalar math as above, but batched in **Structure-of-Arrays** buffers for better cache/SIMD.
* Element-wise updates of ( \Delta j = (v^\star - v_\text{rel}) / k ) with grouped friction cone projection.

> See: [docs/alg_scalar_normal_row_math.md](docs/alg_scalar_normal_row_math.md),
> [docs/alg_scalar_friction_rows_math.md](docs/alg_scalar_friction_rows_math.md),
> [docs/alg_scalar_soa_batched_math.md](docs/alg_scalar_soa_batched_math.md)

### 4) Vectorized SoA solver

* Shares the SoA pipeline while routing through SIMD-friendly microkernels (`solver_scalar_soa_vectorized.*`, `solver_scalar_soa_simd.*`).
* Processes contacts in AVX2/NEON-width batches with masked warm start, normal, and friction solves.
* Falls back to scalar paths for tail contacts so metrics remain comparable across platforms.

---

## Scenes (deterministic)

* **Two spheres, head-on** — elastic 2-body sanity check; velocities swap along the line of centers.
* **Spheres box cloud (N)** — particle-like swarms; stresses contact throughput.
* **Box stack (layers)** — stack stability with ERP/bias; mixes rotations & contacts.

> Jointed scenes (pendulum / 64-link chain / rope) are described in
> [docs/alg_scalar_distance_joint_math.md](docs/alg_scalar_distance_joint_math.md) and may be present depending on the current milestone.

---

## Metrics (reported per run)

* **Directional momentum drift** — max over a fixed set of directions ( \hat d ) of
  (\left|\sum \mathbf p\cdot \hat d \right|*\text{after} - \left|\sum \mathbf p\cdot \hat d \right|*\text{before}).
* **Constraint violation (L∞)** — maximum penetration/joint error across rows.
* **Energy drift** — change in ( \sum \frac12 m v^2 + \frac12 \omega^\top I \omega ) (diagnostic; not zero with friction/restitution/ERP).
* **Cone consistency** — fraction of contacts within Coulomb cone ( | \mathbf j_t | \le \mu j_n ).

The benchmark writes these (plus timing/size fields) to CSV.

### CSV columns (typical)

```
scene, solver, iterations, N_bodies, N_contacts, (N_joints),
ms_per_step, drift_max, Linf_penetration, energy_drift, cone_consistency
```

---

## Reproducibility & determinism

* Fixed iteration counts, fixed scene seeds, and stable memory orderings are used.
* Warm-starts are deterministic across runs; no threading or atomics in baseline benches.
* Tests assert tight tolerances for elastic invariants and cone consistency.

---

## Recent benchmark snapshot

| Scene              | Solver         | iters | ms/step | drift_max (↓) | Linf (↓) | Cone ok (↑) |
| ------------------ | -------------- | :---: | ------: | ------------: | -------: | ----------: |
| spheres_cloud_4096 | Baseline (AoS) |   10  |     8.4 |      1.1e-10  |   3.0e-3 |        0.99 |
|                    | ScalarCached   |   10  |     6.7 |      1.0e-10  |   3.0e-3 |        1.00 |
|                    | SoA (SIMD)     |   10  |     5.2 |      1.2e-10  |   3.0e-3 |        1.00 |

> Release build on `spheres_cloud_4096`, 10 iterations, measured with `./build/bench/bench --benchmark_filter=spheres_cloud_4096`.
> Re-run the benchmark locally to refresh the table; record new data in `docs/soa_improvement_potentials.md` and keep dated CSVs in `results/` for traceability.

---

## How to extend

1. **Add a new scene:** extend `scenes.*` and ensure contacts/joints are deterministic.
2. **Add a new metric:** implement in `metrics.*`, wire to bench CSV.
3. **Experiment with parameters:** see `SolverParams` in each solver; iterate counts, ERP/bias, friction, restitution.
4. **Port to SIMD/GPU:** start from the **SoA** path; group rows to avoid write conflicts.

---

## Roadmap

* [ ] Joints (distance/rope with compliance) in both AoS & SoA solvers.
* [ ] Extend AVX2/NEON lanes to scatter/accumulate paths.
* [ ] GPU/compute shader prototype for particle-heavy scenes.
* [ ] CLI parameters for benches; richer reporting & plots.
* [ ] Unity/Unreal mini-demos toggling solver variants.

---

## Contributing

1. Fork, create a branch, and keep PRs focused.
2. Follow code style (`.clang-format`) and C++17 guidelines.
3. Add tests in `tests/` for any new math or solver behavior.
4. Update docs in `docs/` when you add/modify algorithms.
5. Include a short note of expected performance impact (if relevant).

---

## Citing / academic outreach

If this repo contributes to your research or engine work, consider citing it or opening an issue with your use case. For granular physics, rigid-body games, and contact-rich robotics, we’re especially interested in reproducible comparisons and corner cases.

---

## License

This project is licensed under the **MIT License** (see `LICENSE`).

---
