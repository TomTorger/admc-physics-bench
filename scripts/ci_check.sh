#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="${BUILD_DIR:-build}"
RESULTS_DIR="results"
CSV="${RESULTS_DIR}/ci_quick.csv"

mkdir -p "${BUILD_DIR}" "${RESULTS_DIR}"

echo "== Configure (Release) =="
cmake -S . -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release

echo "== Build =="
cmake --build "${BUILD_DIR}" -j

echo "== Tests =="
ctest --test-dir "${BUILD_DIR}" --output-on-failure

echo "== Quick Bench Suite =="
# Try a small, consistent menu; skip gracefully if solver/scene unsupported.
BENCH="${BUILD_DIR}/bench/bench"
if [[ ! -x "${BENCH}" ]]; then
  echo "Bench binary not found at ${BENCH}; exiting without quick bench."
  exit 0
fi

# Ensure CSV header if file is new/empty
if [[ ! -s "${CSV}" ]]; then
  echo "scene,solver,iterations,steps,N_bodies,N_contacts,N_joints,ms_per_step,drift_max,Linf_penetration,energy_drift,cone_consistency,simd,threads" > "${CSV}"
fi

run_case () {
  local scene="$1" solver="$2" iters="${3:-10}" steps="${4:-30}"
  "${BENCH}" --scene="${scene}" --solver="${solver}" --iters="${iters}" --steps="${steps}" --csv="${CSV}" || true
}

# Small, quick cases (keep runtime low):
run_case two_spheres baseline 10 1
run_case two_spheres cached   10 1
run_case two_spheres soa      10 1

run_case spheres_cloud_1024 baseline 10 30
run_case spheres_cloud_1024 cached   10 30
run_case spheres_cloud_1024 soa      10 30

run_case box_stack_4 baseline 10 30
run_case box_stack_4 cached   10 30
run_case box_stack_4 soa      10 30

if [[ "${RUN_LARGE:-0}" == "1" ]]; then
  run_case spheres_cloud_8192 cached 10 30
  run_case spheres_cloud_8192 soa    10 30
fi

echo "== Done. CSV at ${CSV} =="
