# Benchmark Metrics and CSV Schema

Every row written by `build/bench/bench` captures one solver/scene configuration after it has executed `steps` integration steps. Timing values are averaged per-step, physics metrics compare the initial and final state, and the CSV columns are stable so downstream tooling (e.g. `tools/plot_perf.py`) can rely on them.

## Sampling cadence

1. Clone and store the **pre-step** body state for the selected scene.
2. Run the solver for `steps` iterations of the outer loop (each with `iterations` Gauss–Seidel passes unless the solver exits early by threshold).
3. Measure wall-clock timings across the loop and normalize by `steps`.
4. Re-compute invariant metrics by comparing the final state against the stored pre-state.

## Timing column

- **`ms_per_step`** — mean milliseconds per integration step over the `steps` executed. For SoA solvers, additional breakdowns (contact prep, row build, solver, scatter) are present in the log output but not the CSV.

## Physics quality columns

Unless noted, the units are SI (meters, seconds, kilograms) stemming from the simulation state.

- **`drift_max`** — maximum directional momentum drift across a fixed set of 10 unit directions \(\{\hat{\mathbf d}_\ell\}\). For each direction,
  \[
  \Delta_\ell = \sum_i \mathbf p_i^\text{after}\cdot\hat{\mathbf d}_\ell - \sum_i \mathbf p_i^\text{before}\cdot\hat{\mathbf d}_\ell,\qquad
  \text{drift\_max} = \max_\ell |\Delta_\ell|.
  \]
  The direction set includes Cartesian axes and four additional well-conditioned vectors:
  \((\pm1,0,0)\), \((0,\pm1,0)\), \((0,0,\pm1)\),
  \((0.531176,-0.847845,0.013421)\),
  \((-0.271321,0.349121,0.897932)\),
  \((0.713512,0.204913,-0.669324)\),
  \((-0.402156,-0.551249,-0.730441)\).
- **`Linf_penetration`** — maximum constraint violation across all contacts:
  \[
  \text{Linf\_penetration} = \max_{c \in \text{contacts}} \max(0, -C_c).
  \]
  Contact `C` values are the positional error accumulated during ERP/Baumgarte correction; the units are meters.
- **`energy_drift`** — change in total kinetic energy between pre- and post-state:
  \[
  \Delta E = \sum_i \tfrac12 m_i \|\mathbf v_i^\text{after}\|^2 + \tfrac12 \boldsymbol\omega_i^{\text{after}\top} \mathbf I_i \boldsymbol\omega_i^\text{after}
           - \sum_i \tfrac12 m_i \|\mathbf v_i^\text{before}\|^2 - \tfrac12 \boldsymbol\omega_i^{\text{before}\top} \mathbf I_i \boldsymbol\omega_i^\text{before}.
  \]
  The CSV stores \(\Delta E\) (joules). To report a percentage, compute \(\Delta E / \max(|E_0|, \epsilon)\) with \(\epsilon \approx 10^{-12}\) when post-processing.
- **`cone_consistency`** — fraction of evaluated contacts that satisfy the Coulomb cone inequality:
  \[
  \text{cone\_consistency} = \frac{\#\{c : \|\mathbf j_{t,c}\| \le \mu_c \max(j_{n,c},0)\}}{\#\{c : c \text{ active}\}},
  \]
  where \(\mathbf j_{t,c} = (j_{t1}, j_{t2})\) are tangential impulses. Contacts attached to static bodies or marked inactive are skipped; if no contacts participate the ratio is reported as `1.0`.
- **`joint_Linf`** — maximum absolute constraint violation for distance/rope joints after the final solve:
  \[
  \text{joint\_Linf} = \max_j |C_j|.
  \]

## Scene sizing columns

- **`iterations`** — Gauss–Seidel iterations requested for the solver.
- **`steps`** — outer integration steps executed for the scene.
- **`N_bodies`**, **`N_contacts`**, **`N_joints`** — counts taken from the initial scene (before warm-start).
- **`tile_size`** — SoA tile size used for the step (`0`/`-1` when not applicable).
- **`threads`** — number of worker threads active for the solve (`1` for serial runs).
- **`simd`** — boolean flag (`1` or `0`) indicating whether the solver executed SIMD lanes.
- **`commit_sha`** — optional commit identifier. CI jobs populate this field; local runs leave it blank unless you fill it in downstream.

## CSV schema

Header order (see `bench/bench_csv_schema.hpp`):
```
scene,solver,iterations,steps,N_bodies,N_contacts,N_joints,tile_size,
ms_per_step,drift_max,Linf_penetration,energy_drift,cone_consistency,
simd,threads,commit_sha
```

Additional quantities printed to stdout (SoA timing breakdowns, thread speedups) are not currently written to the CSV.
