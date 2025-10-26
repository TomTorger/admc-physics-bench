#!/usr/bin/env bash
set -euo pipefail
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/bench/bench --benchmark_out=results/results.csv --benchmark_out_format=csv || true
