## Background
- [NR math primer](docs/nr_math.md)
- [Scalar normal row](docs/alg_scalar_normal_row_math.md)
- [Scalar friction rows](docs/alg_scalar_friction_rows_math.md)
- [Scalar distance joint](docs/alg_scalar_distance_joint_math.md)
- [SoA-batched scalar rows](docs/alg_scalar_soa_batched_math.md)

# Scalar Impulse Lab (directional contact rows)
Goal: benchmark vector-per-row vs scalar directional rows (cached & SoA) for contact solvers.
## Build
```

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

```
## Run
```

./build/bench/bench --benchmark_out=results.csv --benchmark_out_format=csv

```
