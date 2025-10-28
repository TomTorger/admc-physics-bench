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
