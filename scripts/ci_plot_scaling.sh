#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}" 
RESULTS_DIR="results"
ASSETS_DIR="docs/assets"
CSV_PATH="${RESULTS_DIR}/perf_scaling.csv"

mkdir -p "${BUILD_DIR}" "${RESULTS_DIR}" "${ASSETS_DIR}"

cmake -S . -B "${BUILD_DIR}" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF
cmake --build "${BUILD_DIR}" -j

BENCH="${BUILD_DIR}/bench/bench"
if [[ ! -x "${BENCH}" ]]; then
  echo "Bench binary not found at ${BENCH}" >&2
  exit 1
fi

rm -f "${CSV_PATH}"

SCENE="spheres_cloud"
SIZES="512,1024,2048,4096,8192"
SOLVERS="baseline,cached,scalar_soa,scalar_soa_vectorized"
ITERS="10"
STEPS="60"

"${BENCH}" \
  --scene="${SCENE}" \
  --sizes="${SIZES}" \
  --solvers="${SOLVERS}" \
  --iters="${ITERS}" \
  --steps="${STEPS}" \
  --csv="${CSV_PATH}" || true

if [[ ! -s "${CSV_PATH}" ]]; then
  echo "Scaling bench did not produce ${CSV_PATH}" >&2
  exit 1
fi

python3 tools/plot_perf.py --inputs "${CSV_PATH}" --out "${ASSETS_DIR}/perf_scaling.svg"
