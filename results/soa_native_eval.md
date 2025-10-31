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
