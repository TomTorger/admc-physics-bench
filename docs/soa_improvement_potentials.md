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
