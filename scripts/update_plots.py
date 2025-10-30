#!/usr/bin/env python3
"""Generate performance plots for sphere cloud benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd


def _latest_results_dir(root: Path) -> Path:
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No result directories found under {root}")
    return max(candidates, key=lambda path: path.name)


def load_latest_results(root: Optional[Path] = None) -> pd.DataFrame:
    base = root or Path("results")
    latest_dir = _latest_results_dir(base)
    csv_files = sorted(latest_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {latest_dir}")
    return pd.read_csv(csv_files[0])


def _solver_order() -> list[str]:
    return ["baseline", "cached", "soa", "vec_soa"]


def plot_speedup(df: pd.DataFrame, out_path: Path, scenes: Iterable[str]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    order = _solver_order()
    for scene in scenes:
        rows = df[df["scene"] == scene]
        if rows.empty:
            continue
        base = rows[rows["solver"] == "baseline"]["ms_per_step"].mean()
        speeds = []
        for solver in order:
            solver_rows = rows[rows["solver"] == solver]
            value = solver_rows["ms_per_step"].mean()
            speeds.append(base / value if value and value > 0 else 0.0)
        ax.plot(order, speeds, marker="o", label=scene)
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1.0)
    ax.set_ylabel("Speedup vs baseline (↑ better)")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def plot_ms(df: pd.DataFrame, out_path: Path, scenes: Iterable[str]) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    order = _solver_order()
    for scene in scenes:
        rows = df[df["scene"] == scene]
        if rows.empty:
            continue
        ms = []
        for solver in order:
            solver_rows = rows[rows["solver"] == solver]
            value = solver_rows["ms_per_step"].mean()
            ms.append(value if value and value > 0 else 0.0)
        ax.plot(order, ms, marker="o", label=scene)
    ax.set_ylabel("ms per step (↓ better)")
    ax.set_ylim(bottom=0)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, format="svg")
    plt.close(fig)


def main() -> None:
    df = load_latest_results()
    scenes = ["spheres_cloud_10k", "spheres_cloud_50k"]
    docs_img = Path("docs/img")
    plot_speedup(df, docs_img / "perf_speedup_spheres_cloud.svg", scenes)
    plot_ms(df, docs_img / "ms_per_step_spheres_cloud.svg", scenes)


if __name__ == "__main__":
    main()
