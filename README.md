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
