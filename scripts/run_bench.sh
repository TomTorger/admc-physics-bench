#!/usr/bin/env bash
set -euo pipefail
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
STAMP=$(date +%Y%m%d)
OUT_DIR="results/${STAMP}"
mkdir -p "${OUT_DIR}"
./build/bench/bench --benchmark_out="${OUT_DIR}/results.csv" --benchmark_out_format=csv || true
