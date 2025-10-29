#!/usr/bin/env python3
"""Format the latest quick bench CSV results as a Markdown summary."""
from __future__ import annotations

import csv
import math
import sys
from collections import OrderedDict
from pathlib import Path


def _parse_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> int:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/ci_quick.csv")
    if not csv_path.exists():
        print(f"No benchmark results found at {csv_path}.")
        return 0

    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        print("Benchmark CSV exists but contains no data.")
        return 0

    scenes: "OrderedDict[str, OrderedDict[str, dict[str, str]]]" = OrderedDict()
    for row in rows:
        scene = row.get("scene", "unknown")
        solver = row.get("solver", "unknown")
        scene_map = scenes.setdefault(scene, OrderedDict())
        scene_map[solver] = row  # last occurrence wins

    print("## Quick Bench Performance Summary\n")
    print("Performance numbers are the latest samples written to `results/ci_quick.csv`."\
          " Speedups are relative to the baseline solver within the same scene.\n")
    print("| Scene | Solver | ms/step | Speedup vs baseline | drift_max | Linf_penetration | energy_drift | cone_consistency |")
    print("|:------|:-------|-------:|--------------------:|----------:|-----------------:|-------------:|-----------------:|")

    for scene, solvers in scenes.items():
        baseline = solvers.get("baseline")
        baseline_ms = _parse_float(baseline.get("ms_per_step")) if baseline else None
        for solver, row in solvers.items():
            ms_per_step = _parse_float(row.get("ms_per_step", ""))
            speedup = None
            if baseline_ms and ms_per_step and ms_per_step > 0:
                speedup = baseline_ms / ms_per_step
            drift = _parse_float(row.get("drift_max", ""))
            penetration = _parse_float(row.get("Linf_penetration", ""))
            energy = _parse_float(row.get("energy_drift", ""))
            cone = _parse_float(row.get("cone_consistency", ""))

            def fmt(value: float | None, precision: int = 3, fallback: str = "—") -> str:
                if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                    return fallback
                fmt_str = f"{{:.{precision}f}}"
                return fmt_str.format(value)

            ms_display = fmt(ms_per_step)
            speed_display = f"{fmt(speedup, 2)}×" if speedup is not None else "baseline"
            drift_display = fmt(drift)
            pen_display = fmt(penetration)
            energy_display = fmt(energy)
            cone_display = fmt(cone)

            print(f"| {scene} | {solver} | {ms_display} | {speed_display} | {drift_display} | {pen_display} | {energy_display} | {cone_display} |")

    print("\n_All timings are milliseconds per simulation step (lower is better)._\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
