#!/usr/bin/env python3
import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_csv(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            rows.extend(reader)
    return rows


def to_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene", default="spheres_cloud_1024_like")
    args = ap.parse_args()

    rows = read_csv(args.inputs)
    rows = [row for row in rows if "spheres_cloud" in row.get("scene", "")]

    data = defaultdict(lambda: defaultdict(list))
    for row in rows:
        try:
            nbodies = int(row.get("N_bodies", 0))
        except Exception:
            continue
        solver = row.get("solver", "unknown")
        ms = to_float(row.get("ms_per_step", 0.0))
        data[solver][nbodies].append(ms)

    if not rows:
        raise SystemExit("No matching rows found in CSV inputs")

    import statistics

    x_values = sorted({int(row["N_bodies"]) for row in rows})

    plt.figure(figsize=(9, 5))
    for solver, series in sorted(data.items()):
        y_values = []
        for nb in x_values:
            samples = series.get(nb, [])
            if samples:
                y_values.append(statistics.median(samples))
            else:
                y_values.append(float("nan"))
        plt.plot(x_values, y_values, marker="o", label=solver)
    plt.xlabel("Bodies (spheres_cloud)")
    plt.ylabel("ms / step (median)")
    plt.title("Solver Performance Scaling (lower is better)")
    plt.grid(True, which="both", axis="both")
    plt.xscale("log", base=2)
    plt.yscale("log")
    plt.legend()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, format="svg")

    plt.figure(figsize=(9, 5))
    baseline = data.get("baseline", {})
    for solver, series in sorted(data.items()):
        if solver == "baseline":
            continue
        y_values = []
        for nb in x_values:
            base_samples = baseline.get(nb, [])
            solver_samples = series.get(nb, [])
            if base_samples and solver_samples:
                import statistics

                y_values.append(
                    statistics.median(base_samples) / statistics.median(solver_samples)
                )
            else:
                y_values.append(float("nan"))
        plt.plot(x_values, y_values, marker="o", label=f"{solver} speedup vs baseline")
    plt.xlabel("Bodies (spheres_cloud)")
    plt.ylabel("Speedup vs baseline (×)")
    plt.title("Scaling Speedup")
    plt.grid(True, which="both", axis="both")
    plt.xscale("log", base=2)
    plt.tight_layout()
    speedup_path = args.out.replace(".svg", "_speedup.svg")
    plt.savefig(speedup_path, format="svg")


if __name__ == "__main__":
    main()
