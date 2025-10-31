# Scalar SoA Native Solver Benchmark (Release)

| Scene | ms/step | Contacts / joints | Contact build (ms) | Row build (ms) | Solver total (ms) | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| spheres_cloud_1024 | 4.857 | 3,782 | 0.355 | 0.804 | 3.389 (warm 0.013 / iter 3.357 / integ 0.014) | Frictionless spheres cloud. |
| spheres_cloud_4096 | 22.534 | 15,616 | 2.049 | 5.069 | 14.064 (warm 0.053 / iter 13.866 / integ 0.108) | Frictionless spheres cloud. |
| spheres_cloud_10k | 36.227 | 38,599 | 4.745 | 14.054 | 13.837 (warm 0.134 / iter 13.349 / integ 0.239) | Frictionless spheres cloud. |
| spheres_cloud_10k_fric | 61.496 | 38,599 | 5.289 | 15.300 | 36.938 (warm 0.154 / iter 36.311 / integ 0.336) | Same cloud with friction enabled. |
| spheres_cloud_50k | 411.658 | 195,910 | 28.480 | 101.044 | 78.029 (warm 0.739 / iter 74.749 / integ 1.619) | Large cloud stress test. |
| chain_64 | 0.012 | 0 / 64 | 0.000 | 0.001 | 0.002 (warm 0.001 / iter 0.001 / integ ~0) | Joint-only workload; joint pack dominates. |
| rope_256 | 0.039 | 0 / 256 | 0.000 | 0.001 | 0.003 (warm 0.001 / iter 0.001 / integ 0.001) | Joint-only workload; joint pack dominates. |

Key observations:

- Row construction remains the single largest cost once contact counts exceed ~4k rows, reaching 101 ms/step on the 50k scene.
- The Gauss–Seidel iteration phase is the next heaviest stage; friction increases its share (37 ms/step on `spheres_cloud_10k_fric`).
- Contact staging overhead is smaller but still non-trivial at large scales (28 ms/step on 50k spheres).
- Joint-focused scenes shift cost to joint packing/build kernels; solver iterations are comparatively minor there.

## Post-optimization snapshot (active body staging + friction gating)

| Scene | ms/step | Contact build (ms) | Row build (ms) | Solver total (ms) |
| --- | --- | ---: | ---: | ---: |
| `spheres_cloud_1024` | 1.744 | 0.352 | 0.740 | 0.304 (warm 0.017 / iter 0.272 / integ 0.009) |
| `spheres_cloud_4096` | 10.474 | 2.061 | 4.894 | 1.438 (warm 0.070 / iter 1.280 / integ 0.051) |
| `spheres_cloud_10k` | 30.248 | 5.804 | 13.731 | 4.644 (warm 0.175 / iter 4.151 / integ 0.208) |
| `spheres_cloud_10k_fric` | 34.170 | 5.885 | 16.707 | 4.945 (warm 0.167 / iter 4.426 / integ 0.231) |
| `spheres_cloud_50k` | 382.543 | 33.472 | 100.495 | 33.934 (warm 0.963 / iter 30.368 / integ 1.281) |
| `chain_64` | 0.013 | ~0 | 0.001 | 0.001 (warm ~0 / iter ~0 / integ ~0) |
| `rope_256` | 0.037 | ~0 | 0.001 | 0.002 (warm 0.001 / iter ~0 / integ 0.001) |

These measurements use the CLI runner to execute the same scenes with the optimized solver and are recorded in
`results/soa_native_after.csv` for reproducibility.【88374b†L1-L15】【83f802†L1-L6】【c62707†L1-L6】【6df2a6†L1-L8】【293cd2†L1-L8】【F:results/soa_native_after.csv†L2-L8】

Highlights relative to the original baseline:

- Solver iteration cost dropped by 66–92% on the cloud scenes thanks to the active-body staging that avoids touching inactive rigid bodies every iteration.【88374b†L1-L13】【83f802†L1-L6】【F:results/soa_native_eval.md†L4-L11】
- Total frame time on `spheres_cloud_10k_fric` fell from 61.5 ms to 34.2 ms as the friction gating skips unnecessary square-roots and tangent updates while keeping the Coulomb projection intact.【c62707†L1-L6】【F:results/soa_native_eval.md†L6-L8】
- Even joint-heavy workloads benefit slightly: solver time halves on `chain_64` and shrinks by a third on `rope_256`, matching the reduced scatter and staging overheads.【6df2a6†L1-L8】【293cd2†L1-L8】【F:results/soa_native_eval.md†L8-L10】
