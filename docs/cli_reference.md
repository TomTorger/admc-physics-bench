# ADMC Bench CLI Reference

The benchmark harness is built as `build/bench/bench`. It drives every solver/scene combination, records timing and physics metrics, and can optionally forward to Google Benchmark. This page summarizes the supported flags, defaults, and enumerations.

```bash
cmake -S . -B build -G Ninja -DADMC_BUILD_BENCH=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j
build/bench/bench \
  --scenes spheres_cloud_1024,box_stack_4 \
  --solvers baseline,cached,soa,soa_native,soa_parallel \
  --iters 10 --steps 30 --tile-sizes 64,128,256 \
  --csv results/$(date +%Y%m%d)/results.csv
```

## Flag index

### Scene and solver selection

- `--scene=KEY` — run a single scene (see [Scenes](#scene-keys)).
- `--scenes=CSV` — comma-separated list of scene keys.
- `--sizes=CSV` — optional list of integer sizes applied to the **first** scene (e.g. `--scene=spheres_cloud --sizes=1024,8192` expands to `spheres_cloud_1024` and `spheres_cloud_8192`). Ignored for non-parametric scenes.
- `--solvers=CSV` — comma-separated solver keys (see [Solvers](#solver-keys)). Aliases are normalized (e.g. `scalar_cached` → `cached`).
- Environment overrides: `BENCH_PRESET=full` appends larger clouds, `RUN_LARGE=1` mirrors the legacy “large” sweep.

### Run control

- `--iters=N` — solver iterations per step (default `10`).
- `--steps=N` — integration steps per run (default `30`, overridden to `1` for `two_spheres`).
- `--dt=SECONDS` — timestep size (default `1/60`).
- `--tile-sizes=CSV` — SoA tile sweep; each value overrides `tile_size` and `max_contacts_per_tile`. Default inherits solver defaults (`128`).
- `--tile_rows=N` — limit the maximum rows per tile (default `128` via solver defaults when unset).
- `--threads=N` — legacy single value used when `--threads-list` is omitted (default `1`).
- `--threads-list=CSV` — explicit thread counts to sweep. Absent values fall back to `{1, hardware_concurrency}` unless `--deterministic` is set.
- `--deterministic` — forces single-thread mode (equivalent to `threads_list={1}`) even if the build enables threading.
- `--spheres-only` — hint for SoA solvers to assume sphere contacts (skips box/contact code paths).
- `--frictionless` — treat tangential rows as disabled regardless of scene defaults.
- `--convergence-threshold=VALUE` — stop SoA iterations early when the RMS residual drops below `VALUE` (disabled when negative).

### Output, CSV, and Google Benchmark

- `--csv=PATH` — write results to `PATH`. Directories are created as needed. Without this flag, the harness appends to `results/YYYYMMDD/results.csv`.
- `--no-csv` — suppress CSV output.
- `--human=compact|legacy` — pick the human-readable format. `compact` (default) prints the headline table + solver breakdowns; `legacy` restores the original verbose blocks.
- `--timings=min|wide|json|off` — emit machine-parsable lines after each solver row (CSV-like `min`/`wide`, or JSON Lines). Default `off`.
- `--columns=N` — override the terminal width used for alignment/truncation (defaults to `$COLUMNS` env or `100`).
- `--benchmark` — enable Google Benchmark mode; any unknown flags are forwarded (e.g. `--benchmark_filter`, `--benchmark_out_format`). In this mode you should still add `--csv=...` if you want the harness CSV alongside the Google Benchmark output.
- `--preset=NAME` — select a pre-defined scene bundle (currently `full` mirrors the large preset). This mirrors the `BENCH_PRESET` environment variable.
- Unknown `--flag` values are passed straight through to Google Benchmark when `--benchmark` is specified.

## Scene keys

| Key | Description | Notes |
| --- | --- | --- |
| `two_spheres` | Two elastic spheres colliding head-on. | Restitution `1.0`, friction disabled, `steps=1`. |
| `spheres_cloud_1024`, `spheres_cloud_4096`, `spheres_cloud_8192`, `spheres_cloud_10k`, `spheres_cloud_50k` | Dense sphere pile above a static floor. | Frictionless by default (`mu=0`, tangents skipped for cached/SoA). |
| `spheres_cloud_10k_fric` | Same geometry as `10k` with tangential friction enabled. | Solver `mu` forced to `0.5`. |
| `box_stack_4`, `box_stack` | Vertical stack of unit boxes on a ground plane. | ERP/Baumgarte bias active; defaults to `mu=0.5`. |
| `pendulum` | One-link pendulum attached to a static pivot. | Distance joint with `beta=0.2`. |
| `chain_64` | 64-link compliant chain. | Distance joints with `compliance=1e-8`. |
| `rope_256` | Long rope with rope joints. | Zero-compliance rope constraints. |
| `spheres_cloud` | Parametric alias requiring `--sizes`. | Expands to `spheres_cloud_<N>`. |
| `box_stack_<layers>` | Explicit layer count (e.g. `box_stack_8`). | Parsed via the generic `_N` suffix helper. |

Full material, ERP, and timestep settings are tabulated in `docs/scenes.md`.

## Solver keys

| Canonical key | Description | Accepted aliases |
| --- | --- | --- |
| `baseline` | AoS, vector-per-row sequential impulse. | `baseline_vec` |
| `cached` | Scalar AoS solver with cached effective mass and warm-start. | `scalar_cached` |
| `soa` | Scalar Structure-of-Arrays batcher (legacy path). | `scalar_soa`, `soa_mt`, `soa_simd` |
| `soa_parallel` | Island-parallel native solver with work-stealing scheduler. | `scalar_soa_parallel`, `parallel_soa` |
| `vec_soa` | SIMD-friendly SoA path that currently forwards to scalar kernels. | `scalar_soa_vectorized`, `soa_vec`, `soa_vectorized` |
| `soa_native` | Fully SoA-native solver with lane-specialized kernels. | `scalar_soa_native`, `native_soa` |

The harness normalizes aliases before dispatch, so CSV rows always use the canonical keys above.

## Plotting helper

Use the included plotting script to regenerate the CI charts locally:

```bash
python3 tools/plot_perf.py --inputs results/*.csv --out docs/assets/perf_scaling.svg
```

The script expects the harness CSV schema documented in `docs/metrics.md`. It filters `spheres_cloud_*` rows, computes per-solver medians by size, and emits both `perf_scaling.svg` and a `_speedup` companion.

## Tips

- Run the harness multiple times with different `--csv` files (or append to one file) and post-process medians/percentiles for stability analysis.
- Combine `--deterministic` with a single `--threads-list` entry when comparing absolute timings across machines.
- When exploring SoA tiling, sweep both `--tile-sizes` and `--threads-list` so the CSV captures the throughput surface for later plotting.
