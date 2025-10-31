# Using the Fully SoA-Native SIMD Solver

The `scalar_soa_native` solver keeps body and contact state in structure-of-arrays form throughout the Gauss–Seidel loop. This section explains how to run it from the benchmark harness, expose useful parameters, and interpret its telemetry.

## Command-line invocation

The solver is available through several aliases:

```
--solvers=scalar_soa_native
--solvers=soa_native
--solvers=native_soa
```

You can mix it with other solvers:

```
./build/bench/bench \
  --scene=spheres_cloud_10k \
  --solvers=baseline,soa_native,vec_soa \
  --steps=30 --iters=10 --csv=results/native_vs_baseline.csv
```

The default benchmark suite now includes `soa_native`, so `./build/bench/bench --benchmark` will record its results automatically.

## Key parameters

`scalar_soa_native` honours the standard `SoaParams` fields. The most relevant knobs are:

- `iterations`: Gauss–Seidel iterations (defaults to the same value used by the other solvers).
- `tile_size`: Maximum contacts staged per SIMD tile. Override with `--tile-rows` if you need larger batches for extreme contact sets.
- `spheres_only` / `frictionless`: Shortcut flags forwarded from the CLI (`--spheres-only`, `--frictionless`).
- `warm_start`: Enabled by default; disable via `--no-warm-start` to gauge cold-start performance.
- `convergence_threshold`: Early-exit impulse tolerance (default `1e-4`). Override with `--convergence-threshold=VALUE`; set `0` to disable and run the full iteration budget.

The solver always runs its SIMD kernels (`SoaParams::use_simd` is forced to `true` inside the harness). Tail contacts rely on masked vector lanes, so no scalar slow path is required.

## Telemetry

`SoaNativeStats` records the following milliseconds-per-step timing breakdown and exports it to `SoaTimingBreakdown`:

- `staging_ms`: Mirroring body velocities into SoA buffers.
- `warmstart_ms`: Applying cached impulses.
- `normal_ms`: Normal constraint updates per iteration.
- `friction_ms`: Tangent constraint projection.
- `writeback_ms`: Storing SoA body state back to the scene.

These numbers appear in the CSV output and the console summary under the `solver` column `scalar_soa_native`.

When the convergence guard trips early you will still see the requested `iterations` in the CSV, but the debug summary will note the observed impulse threshold. For profiling runs, drop `--convergence-threshold` to zero to force the full iteration budget and collect apples-to-apples timings.

## Performance expectations

On the current benchmark suite the SIMD solver roughly halves the solver-phase time relative to the earlier vectorized path (e.g. `spheres_cloud_10k` drops from ~14.7 ms to ~10.8 ms). End-to-end frame time still trails the AoS baseline because SoA row construction remains the dominant cost (≈9–10 ms per step). Expect further reductions as row-build optimizations land; use the CSV output to compare `ms_per_step` against `baseline` and `vec_soa` runs.

## ADMC invariants

The solver maintains the additive ADMC scalars (`p_{k^\pm}`) during every lane update. Impulses are clamped through the same invariants as the scalar SoA solver, so conservation and determinism remain within existing guardrails. Any parity test failure should be investigated as a potential regression.
