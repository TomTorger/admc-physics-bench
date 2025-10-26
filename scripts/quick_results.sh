#!/usr/bin/env bash
set -euo pipefail
export BUILD_DIR="build"
bash scripts/ci_check.sh
echo
echo "Top of results:"
head -n 5 results/ci_quick.csv || true
echo
echo "Last few rows:"
tail -n 10 results/ci_quick.csv || true
