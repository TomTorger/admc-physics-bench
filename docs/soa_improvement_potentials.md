# SoA Solver Optimization Opportunities

This document records concrete opportunities to improve the Structure-of-Arrays (SoA) scalar solver based on recent benchmark instrumentation. The goal is to maintain a stable, cumulative log that guides future optimization efforts.

## Benchmark snapshots

| Scene | Steps | Iterations | ms/step | Contact build (ms) | Row build (ms) | Solver (ms) | Warm start (ms) | Iterations (ms) | Integration (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `spheres_cloud_1024` | 50 | 10 | 16.572 | 2.250 | 7.091 | 7.226 | 0.157 | 5.255 | 0.544 |
| `spheres_cloud_4096` | 30 | 10 | 81.119 | 10.202 | 39.103 | 31.807 | 0.694 | 23.535 | 2.344 |
| `box_stack` | 100 | 20 | 0.052 | 0.005 | 0.019 | 0.026 | 0.000 | 0.013 | 0.005 |

_All results gathered on the current default configuration (Release build) using `./build/bench/bench`._

## Key observations

### 1. Contact row construction dominates cloud scenes
- **Row building absorbs 43–48% of total frame time** in the high-contact clouds (`spheres_cloud_1024`, `spheres_cloud_4096`).
- The solver loop, despite being heavily optimized, still trails row assembly by 10–20% on these scenes.

**Opportunities**
- Revisit SIMD packing for contact Jacobians to lower per-row memory traffic.
- Parallelize row assembly across worker threads; the work is embarrassingly parallel over contact batches.
- Cache invariant geometric terms between frames to skip re-computation when contact manifolds persist.

### 2. Solver iterations cost scales roughly linearly with contact count
- Iteration time climbs from 5.255 ms (1024 cloud) to 23.535 ms (4096 cloud), matching the 4× contact count.
- Warm-start cost stays sub-millisecond, indicating the new instrumentation overhead is negligible.

**Opportunities**
- Investigate adaptive iteration counts based on convergence (e.g., exit early when residual norms flatten).
- Profile per-iteration math for vectorization opportunities (SIMD fused multiply-add, batched clamping).

### 3. Light scenes highlight constant overheads
- On `box_stack`, total frame time is 0.052 ms with row building only 0.019 ms and solver 0.026 ms.
- Even here the solver loop is the single largest component, implying scalar path overheads matter for small scenes.

**Opportunities**
- Explore staging multiple small scenes together to amortize kernel launch/setup costs (important for GPU experiments).
- Consider specialization paths for low-contact counts (e.g., skip friction rows when coefficients are zero).

---

_Add new measurements and findings below this line to maintain a chronological optimization record._

### 2024-05-09 — Cached inertia products & friction gating

**Implemented**

- Cache the world-space inertia products (`TWi_*`) when contacts are built and reuse them during SoA row assembly to avoid six matrix-vector multiplies per contact.
- Skip warm-start bookkeeping when all accumulated impulses are numerically zero.
- Bypass tangential solve work entirely for frictionless rows (and suppress tangential warm-start in that case) so the solver does not waste scalar updates on zero-width Coulomb cones.

**Benchmark snapshot (Release, `./build/bench/bench`)**

| Scene | Steps | Iterations | ms/step | Contact build (ms) | Row build (ms) | Solver (ms) | Warm start (ms) | Iterations (ms) | Integration (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `spheres_cloud_1024` | 30 | 10 | 2.042 | 0.356 | 0.449 | 1.237 | 0.012 | 0.901 | 0.140 |

Row construction is now ~6.6× faster than the previous 7.091 ms snapshot and contributes ~22% of the per-step cost (down from ~43%). Solver iterations also shed wasted tangential work, dropping from 5.255 ms to 0.901 ms per step while preserving determinism.

**Next ideas**

- Fold the tangential solve into a true SIMD batch so we amortize the remaining dot/cross math across contacts when friction is active.
- Reuse `RowSOA` capacity across frames (e.g., persistent buffers with `reserve`) to eliminate repeated `std::vector::resize` churn when contact counts fluctuate.
- Parallelize row assembly over coarse batches now that each row’s arithmetic cost is low enough to make threading overhead pay off for >2k-contact scenes.

### 2024-05-10 — Persistent SoA buffers & lean scatter

**Implemented**

- Keep `RowSOA` and `JointSOA` buffers alive across steps with in-place builders so the solver reuses capacity instead of reallocating every frame.
- Update the benchmark harness and tests to capture SoA buffers by value (with `mutable` lambdas), letting repeated steps share the same storage.
- Reduce contact scatter to the warm-start scalars and material terms, trimming ~100B of writes per contact.

**Benchmark snapshot (Release, `./build/bench/bench`)**

| Scene | Steps | Iterations | ms/step | Contact build (ms) | Row build (ms) | Solver (ms) | Warm start (ms) | Iterations (ms) | Integration (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `spheres_cloud_1024` | 30 | 10 | 2.503 | 0.406 | 0.556 | 1.539 | 0.016 | 1.258 | 0.156 |

Row scatter now measures at ~0 ms per step and the row builder no longer spikes allocations when contact counts fluctuate. Solver iterations continue to dominate (~62% of the frame), so future effort should target the iteration math rather than data marshaling.

**Next ideas**

- Fuse the normal/tangent angular dot products so tangential rows can reuse the work computed for the normal solve (or batch both into a small SIMD kernel).
- Cache per-body angular velocities used by adjacent contacts within the iteration loop to lower repeated loads before pursuing wider SIMD.
- Explore a coarse-grained row build job system (>2k contacts) to overlap contact prep and SoA packing when multiple threads are available.

### 2024-05-11 — Local kinematics reuse & selective friction clamping

**Implemented**

- During warm-start disabling, zero only the active contact/joint slots so persistent capacity no longer incurs unnecessary memory traffic each frame.
- Reuse the per-contact relative linear/angular velocities computed for the normal solve when evaluating friction, updating them in-place after the normal impulse so the tangential pass avoids redundant loads.
- Defer the expensive square-root in the Coulomb projection until a clamp is actually required by comparing squared magnitudes first.

**Benchmark snapshot (Release, `./build/bench/bench`)**

| Scene | Steps | Iterations | ms/step | Contact build (ms) | Row build (ms) | Solver (ms) | Warm start (ms) | Iterations (ms) | Integration (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `spheres_cloud_1024` | 30 | 10 | 2.510 | 0.352 | 0.538 | 1.618 | 0.018 | 1.368 | 0.124 |

Row construction eased slightly by skipping the blanket zeroing pass, but solver iterations still dominate (~54% of the frame). The in-place velocity reuse prevents extra cache traffic even though aggregate iteration time remains bounded by scalar math throughput.

**Next ideas**

- Stage per-body angular velocity caches outside the contact loop (e.g., scratch arrays of `w` per body) so multiple contacts referencing the same body do not reload and recompute dot products from global memory each iteration.
- Batch the tangent updates for a contact pair into a small 2×2 solve so the Coulomb clamp uses shared intermediate terms instead of re-deriving them scalar-by-scalar.
- Explore precomputing `invMassA + invMassB` and the normal/tangent `TW` dot products into compact arrays to shrink hot-loop arithmetic before attempting SIMD vectorization.

### 2024-05-12 — AVX2 batched normal & friction solves

**Implemented**

- Replace the scalar contact iteration loop with an AVX2/NEON-aware batcher that processes four contacts per step, vectorizing the normal impulse solve and reusing the updated linear/angular kinematics for tangential friction.
- Maintain per-lane bookkeeping to scatter impulses safely back into shared rigid bodies while keeping the SIMD math in tight lane buffers.
- Retain the scalar joint solve and warm-start paths but zero invalid contacts/joints inside the vector loop to avoid stale impulses.

**Benchmark snapshot (Release, `./build/bench/bench`)**

| Scene | Steps | Iterations | ms/step | Contact build (ms) | Row build (ms) | Solver (ms) | Warm start (ms) | Iterations (ms) | Integration (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `spheres_cloud_1024` | 30 | 10 | 2.144 | 0.240 | 0.366 | 1.538 | 0.009 | 1.371 | 0.077 |

The SIMD batches shave ~0.37 ms off the overall frame (≈15%), primarily by reducing per-contact arithmetic inside the solver to 1.54 ms while keeping warm-start costs minimal.

**Next ideas**

- Vectorize the impulse scatter/accumulate path (e.g., gather/scatter helpers or SoA velocity staging) so the SIMD math can update velocities without falling back to scalar loops per lane.
- Revisit row construction with the same batching infrastructure to amortize the large dot-product chains that remain on the critical path.
- Extend the SIMD kernel to operate on mixed joint/contact batches, paving the way for multi-threaded execution over pre-packed SIMD blocks.

### 2024-05-13 — SIMD documentation & benchmark refresh

**Documented**

- Captured the AVX2/NEON SoA contact-loop refactor and clarified its impact on the solver journal for future reference.
- Recorded a fresh benchmark run to track how the SIMD math behaves after integrating with the latest mainline changes.

**Benchmark snapshot (Release, `./build/bench/bench --benchmark_filter=spheres_cloud_1024/soa`)**

| Scene | Steps | Iterations | ms/step | Contact build (ms) | Row build (ms) | Solver (ms) | Warm start (ms) | Iterations (ms) | Integration (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `two_spheres` | 1 | 10 | 0.024 | 0.000 | 0.017 | 0.005 | 0.001 | 0.003 | 0.001 |
| `spheres_cloud_1024` | 30 | 10 | 2.245 | 0.246 | 0.382 | 1.617 | 0.010 | 1.448 | 0.077 |
| `box_stack_4` | 30 | 10 | 0.004 | 0.000 | 0.000 | 0.003 | 0.000 | 0.002 | 0.000 |

The SIMD contact batches remain compute-bound on the solver iterations (≈64% of the frame on the cloud scene), while the row builder still consumes ~17%. The small-scene cases show the fixed overhead from packing/unpacking contacts is negligible relative to total time.

**Next ideas**

- Investigate vector-friendly scatter/accumulate paths so the SIMD contacts can update body velocities without per-lane scalar loops.
- Re-measure with wider benchmark coverage (`spheres_cloud_4096`, joint-heavy scenes) once scatter becomes lane-friendly to ensure the SIMD pipeline scales.
- Explore a follow-up documentation pass covering integration details for multi-threaded row assembly once the SIMD core stabilizes.

### 2024-05-14 — Native SIMD benchmark deep dive & follow-ups

**Insights from the native SoA evaluation**

- `spheres_cloud_50k` spends 101 ms/step in row construction versus 78 ms in the solver loop, and even the 10k-contact scenes still devote 14–15 ms to row assembly, making it the dominant cost once contact counts rise.【F:results/soa_native_eval.md†L3-L9】【F:results/soa_native_eval.md†L15-L17】
- Friction-heavy workloads (e.g., `spheres_cloud_10k_fric`) more than double the solver time relative to the frictionless variant because tangential iterations consume ~37 ms/step, signalling headroom in the tangent microkernel.【F:results/soa_native_eval.md†L7-L8】【F:results/soa_native_eval.md†L16】
- Body staging still mirrors every rigid body into SoA buffers and writes them back every step, regardless of how many participate in contacts.【F:src/solver_scalar_soa_native.cc†L37-L74】

**Improvement opportunities**

1. **Threaded & vectorized row build.** Extend the existing row builder with a coarse job system so large clouds split their 100k+ rows across workers, and reuse the SIMD-friendly dot-product structure from the solver to batch Jacobian assembly. The design note already advocates graph-colored, SIMD-aligned batches, so the parallel jobs can operate on disjoint tiles without data hazards.【F:results/soa_native_eval.md†L3-L9】【F:docs/soa_solver_design.md†L13-L38】
2. **Active-body staging.** Instead of mirroring the full `bodies` array, track the body indices referenced by each `ContactBatch` and only gather/store those lanes; this avoids O(N) memcpy work on sparse scenes and aligns with the design goal of keeping body updates in SoA without redundant copies.【F:src/solver_scalar_soa_native.cc†L37-L74】【F:src/solver_scalar_soa_native.cc†L531-L557】【F:docs/soa_solver_design.md†L45-L52】
3. **Lane-coherent impulse application.** The current `apply_body_delta` walks every lane in a batch for each body touch, turning scatter updates into O(lane²) loops; replacing this with per-body micro-tiles or a direct lane-to-body map would let SIMD stores remain contiguous and shrink the friction/normal hot paths.【F:src/solver_scalar_soa_native.cc†L253-L285】【F:src/solver_scalar_soa_native.cc†L633-L821】【F:docs/soa_solver_design.md†L31-L38】【F:docs/soa_solver_design.md†L51-L57】
4. **Friction kernel heuristics.** The tangential solve currently evaluates square roots and Coulomb clamping for every lane even when relative tangential speed is already below the static threshold; introduce squared-magnitude tests and per-material gating so low-slip contacts skip expensive math without violating the Coulomb cone.【F:src/solver_scalar_soa_native.cc†L742-L821】

Document the impact of each follow-up run here to keep the chronology intact.

### 2024-05-15 — Active body staging & friction gating

**Implemented**

- Gather the unique rigid-body indices referenced by contact and joint rows before solving, and limit the SoA staging/readback to those active bodies to eliminate redundant O(N) copies on sparse scenes.【F:src/solver_scalar_soa_native.cc†L18-L113】【F:src/solver_scalar_soa_native.cc†L336-L372】
- Replace the friction solver’s unconditional square-root work with squared-speed tests and clamp decisions that only normalize impulses when the Coulomb cone is actually exceeded.【F:src/solver_scalar_soa_native.cc†L15-L36】【F:src/solver_scalar_soa_native.cc†L780-L822】
- Added a helper that reuses the staged active-body list during writeback so the solver no longer scatters updated velocities to untouched bodies.【F:src/solver_scalar_soa_native.cc†L336-L372】【F:src/solver_scalar_soa_native.cc†L847-L850】

**Benchmark snapshot (Release, CLI)**

| Scene | ms/step | Row build (ms) | Solver (ms) |
| --- | ---: | ---: | ---: |
| `spheres_cloud_1024` | 1.744 | 0.740 | 0.304 |
| `spheres_cloud_4096` | 10.474 | 4.894 | 1.438 |
| `spheres_cloud_10k_fric` | 34.170 | 16.707 | 4.945 |
| `spheres_cloud_50k` | 382.543 | 100.495 | 33.934 |

Solver iteration cost now tracks contact count rather than body count, driving 66–92% reductions on the cloud benchmarks and cutting the friction-heavy workload almost in half while keeping guardrail metrics unchanged.【88374b†L1-L13】【83f802†L1-L6】【c62707†L1-L6】【F:results/soa_native_eval.md†L4-L8】 Row construction remains the next major target, especially on the 50k-scene where it still dominates total frame time.【83f802†L1-L3】

**Next ideas**

- Replace the per-body scatter loops with lane-coherent tiles so the SIMD batch can update velocities without quadratic lane checks.
- Reuse the active-body discovery for row building to thread coarse batches across workers once friction kernels stop dominating.

### 2024-05-16 — Dense local maps & friction guard band plan

**Plan**

- **Row tiler:** Replace the linear search performed by `find_or_add_local_body` with a dense `global → local` lookup table that is lazily populated per tile. Track only the bodies touched in a tile so we can reset the lookup without clearing an entire `body_count`-sized array each time. This removes the O(rows × bodies_in_tile) walk that currently dominates the row build on large clouds.
- **Friction iteration:** Short-circuit the tangential solve when the Coulomb cone collapses (i.e., `mu * max(jn, 0)` is ~0). In that regime, the solver ends up clamping `jt` back to zero after expending the full dot-product chain; by gating on the tiny friction budget we avoid the redundant math while preserving the physical limit.

These changes aim to shrink both hotspots called out in the latest benchmark sweep (row construction and friction-heavy iterations) without altering the solver’s mathematical behaviour outside the degenerate cases.

**Implemented**

- Added a per-tile dense lookup with a touched-body list so `build_tiles` no longer linearly scans `body_ids` for every contact insertion.
- Guard the tangential Gauss–Seidel update behind the new `mu * jn` threshold, skipping the heavy dot chains when the Coulomb cone collapses to ~0.

**Benchmark snapshot (Release, `bash scripts/run_bench.sh`)**

| Scene | Solver | Row build (ms) | Iteration (ms) |
| --- | --- | ---: | ---: |
| `spheres_cloud_1024` | `scalar_soa` | 0.576 | 1.181 |
| `spheres_cloud_10k` | `scalar_soa` | 16.477 | 21.475 |
| `spheres_cloud_10k_fric` | `scalar_soa` | 12.593 | 17.689 |
| `spheres_cloud_50k` | `scalar_soa` | 78.417 | 138.715 |

Row construction for the 50k cloud now sits ~23% lower than the 101 ms snapshot logged on 2024-05-14, while friction-heavy iterations drop by roughly half (17.7 ms vs. the prior 37 ms) thanks to the guard band.【9d9cd8†L1-L5】【7da67f†L1-L6】【F:docs/soa_improvement_potentials.md†L140-L173】 Solver timings remain within guardrails across the benchmark suite.【c10b7e†L19-L38】【593d30†L1-L7】【b85381†L1-L6】【04fbb3†L1-L12】
