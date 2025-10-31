#!/usr/bin/env python3
"""Format the latest quick bench CSV results as a Markdown summary."""
from __future__ import annotations

import csv
import math
import sys
from collections import OrderedDict
from pathlib import Path


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def _fmt_number(value: float | None, *, unit: str | None = None) -> str:
    if value is None or math.isnan(value) or math.isinf(value):
        return "—"
    abs_value = abs(value)
    if abs_value and (abs_value >= 1e4 or abs_value < 1e-3):
        formatted = f"{value:.3e}"
    elif abs_value >= 1:
        formatted = f"{value:.3f}"
    else:
        formatted = f"{value:.4f}"
    if unit:
        return f"{formatted} {unit}"
    return formatted


def _fmt_time(ms: float | None) -> str:
    if ms is None or math.isnan(ms) or math.isinf(ms):
        return "—"
    if ms >= 100:
        return f"{ms:.1f}"
    if ms >= 10:
        return f"{ms:.2f}"
    if ms >= 1:
        return f"{ms:.3f}"
    return f"{ms:.4f}"


def _solver_order_key(name: str) -> tuple[int, str]:
    normalized = name.lower()
    priority = {
        "baseline": 0,
        "baseline_vec": 0,
        "cached": 1,
        "scalar": 2,
        "soa": 2,
        "scalar_soa": 2,
        "soa_native": 3,
        "scalar_soa_native": 3,
        "native_soa": 3,
        "soa_vectorized": 4,
        "scalar_soa_vectorized": 4,
    }
    for key, value in priority.items():
        if normalized == key or normalized.startswith(f"{key}_"):
            return value, normalized
    return 10, normalized


def _clean_row(row: dict[str | None, str]) -> dict[str, str]:
    cleaned: dict[str, str] = {}
    for key, value in row.items():
        if key is None:
            continue
        cleaned[key.strip()] = (value or "").strip()
    return cleaned


def _simd_label(flag: str | None) -> str:
    if flag is None:
        return "—"
    normalized = flag.strip().lower()
    if normalized in {"1", "true", "yes", "simd"}:
        return "SIMD"
    if normalized in {"0", "false", "no", "scalar"}:
        return "Scalar"
    return normalized or "—"


def main() -> int:
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/ci_quick.csv")
    if not csv_path.exists():
        print(f"No benchmark results found at {csv_path}.")
        return 0

    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        cleaned_rows = [_clean_row(row) for row in reader]

    if not cleaned_rows:
        print("Benchmark CSV exists but contains no data.")
        return 0

    scenes: "OrderedDict[str, OrderedDict[str, dict[str, str]]]" = OrderedDict()
    scene_order: list[str] = []
    for row in reversed(cleaned_rows):
        scene = row.get("scene", "unknown") or "unknown"
        solver = row.get("solver", "unknown") or "unknown"
        if scene not in scenes:
            scenes[scene] = OrderedDict()
            scene_order.append(scene)
        if solver not in scenes[scene]:
            scenes[scene][solver] = row

    if not scene_order:
        print("No benchmark rows found in CSV.")
        return 0

    ordered_scenes = list(reversed(scene_order))
    unique_solvers = sorted(
        {solver for scene in ordered_scenes for solver in scenes[scene]},
        key=_solver_order_key,
    )
    run_count = sum(len(scenes[scene]) for scene in ordered_scenes)

    print("## Quick Bench Performance Summary\n")
    print(
        f"Ran {len(ordered_scenes)} scene{'s' if len(ordered_scenes) != 1 else ''} × "
        f"{len(unique_solvers)} solver{'s' if len(unique_solvers) != 1 else ''} "
        f"({run_count} total runs). Lower ms/step is better.\n"
    )
    print(f"Solvers covered: {', '.join(unique_solvers)}.\n")

    print("### Scene configuration\n")
    print("| Scene | Iterations × steps | Bodies | Contacts | Joints |")
    print("|:------|:-------------------|-------:|---------:|-------:|")
    for scene in ordered_scenes:
        solver_rows = scenes[scene]
        config_row = None
        for name in ("baseline", "baseline_vec"):
            if name in solver_rows:
                config_row = solver_rows[name]
                break
        if config_row is None and solver_rows:
            config_row = next(iter(solver_rows.values()))
        iterations = _parse_int(config_row.get("iterations")) if config_row else None
        steps = _parse_int(config_row.get("steps")) if config_row else None
        bodies = _parse_int(config_row.get("N_bodies")) if config_row else None
        contacts = _parse_int(config_row.get("N_contacts")) if config_row else None
        joints = _parse_int(config_row.get("N_joints")) if config_row else None
        iter_steps = "—"
        if iterations is not None and steps is not None:
            iter_steps = f"{iterations} × {steps}"
        print(
            f"| {scene} | {iter_steps} | {bodies if bodies is not None else '—'} | "
            f"{contacts if contacts is not None else '—'} | {joints if joints is not None else '—'} |"
        )

    print("\n### Solver performance\n")
    print(
        "| Scene | Solver | ms/step | vs baseline | Δms | drift_max | Linf_penetration | "
        "energy_drift | cone_consistency | Threads | SIMD |"
    )
    print(
        "|:------|:-------|-------:|:-----------|-----:|---------:|-----------------:|"
        "-------------:|-----------------:|-------:|:-----|"
    )

    for scene in ordered_scenes:
        solver_rows = scenes[scene]
        baseline_row = None
        for name in ("baseline", "baseline_vec"):
            if name in solver_rows:
                baseline_row = solver_rows[name]
                break
        baseline_ms = _parse_float(baseline_row.get("ms_per_step")) if baseline_row else None
        baseline_solver_name = baseline_row.get("solver") if baseline_row else None

        for solver, row in sorted(solver_rows.items(), key=lambda item: _solver_order_key(item[0])):
            ms_per_step = _parse_float(row.get("ms_per_step"))
            drift = _parse_float(row.get("drift_max"))
            penetration = _parse_float(row.get("Linf_penetration"))
            energy = _parse_float(row.get("energy_drift"))
            cone = _parse_float(row.get("cone_consistency"))
            threads = _parse_int(row.get("threads"))
            simd_flag = row.get("simd")
            speed_note = "—"
            delta_display = "—"

            if baseline_ms is not None and ms_per_step is not None and ms_per_step > 0:
                speed_ratio = baseline_ms / ms_per_step
                if solver == baseline_solver_name:
                    speed_note = "baseline"
                    delta_display = "0.000"
                else:
                    if speed_ratio >= 1:
                        speed_note = f"{speed_ratio:.2f}× faster"
                    else:
                        slower = 1 / speed_ratio if speed_ratio > 0 else float("inf")
                        speed_note = f"{slower:.2f}× slower"
                    delta = ms_per_step - baseline_ms
                    delta_display = f"{delta:+.3f}"

            print(
                f"| {scene} | {solver} | {_fmt_time(ms_per_step)} | {speed_note} | {delta_display} | "
                f"{_fmt_number(drift)} | {_fmt_number(penetration)} | {_fmt_number(energy)} | {_fmt_number(cone)} | "
                f"{threads if threads is not None else '—'} | {_simd_label(simd_flag)} |"
            )

    print(
        "\nSpeed comparisons use the baseline solver within each scene. Metrics come from the latest rows in "
        f"`{csv_path}`. Only the most recent result for each scene/solver pairing is shown. SIMD indicates whether the solver "
        "used the vectorized SoA path.\n"
    )
    print("_All timings are milliseconds per simulation step (lower is better)._\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
