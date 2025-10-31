#!/usr/bin/env python3
import argparse
import csv
import math
import os
import re
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


def normalize_solver(name):
    label = (name or "unknown").strip()
    return label, label.lower()


def parse_scene_size(scene):
    if not scene:
        return None
    match = re.search(r"(\d+)(?!.*\d)", scene)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def parse_int(value):
    if value is None:
        return None
    try:
        return int(str(value).replace(",", ""))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scene", default="spheres_cloud_1024_like")
    args = ap.parse_args()

    rows = read_csv(args.inputs)
    rows = [row for row in rows if "spheres_cloud" in row.get("scene", "")]

    data = defaultdict(lambda: defaultdict(list))
    labels = {}
    for row in rows:
        scene = row.get("scene", "")
        size = parse_scene_size(scene)
        if size is None:
            size = parse_int(row.get("N_bodies"))
        if size is None:
            continue
        solver_label, solver_key = normalize_solver(row.get("solver", "unknown"))
        labels[solver_key] = solver_label
        ms = to_float(row.get("ms_per_step", 0.0))
        data[solver_key][size].append(ms)

    if not rows:
        raise SystemExit("No matching rows found in CSV inputs")

    import statistics

    x_values = sorted({size for series in data.values() for size in series})

    plt.figure(figsize=(9, 5))
    for solver_key in sorted(labels, key=lambda key: labels[key].lower()):
        series = data.get(solver_key, {})
        y_values = []
        for nb in x_values:
            samples = series.get(nb, [])
            if samples:
                y_values.append(statistics.median(samples))
            else:
                y_values.append(float("nan"))
        plt.plot(x_values, y_values, marker="o", label=labels[solver_key])
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
    baseline_key = next((key for key, label in labels.items() if label.lower() == "baseline"), None)
    baseline = data.get(baseline_key, {}) if baseline_key else {}
    for solver_key in sorted(labels, key=lambda key: labels[key].lower()):
        if solver_key == baseline_key:
            continue
        series = data.get(solver_key, {})
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
        plt.plot(
            x_values,
            y_values,
            marker="o",
            label=f"{labels[solver_key]} speedup vs baseline",
        )
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
