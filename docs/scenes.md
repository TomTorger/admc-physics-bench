# Scene Parameter Reference

All benchmark scenes are constructed deterministically from analytic layouts. This table captures the default physical parameters and solver tweaks so that reproductions match the values plotted in CI.

## Contact-focused scenes

| Scene key | N (dynamic bodies) | Geometry & distribution | μ (solver default) | Restitution | dt (s) | Iterations (default) | Seed | Goal / stressor | Determinism notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `two_spheres` | 2 | Two spheres of radius 0.5 placed at ±0.5 m on X, fired toward each other at 2 m/s. | 0.0 (`--frictionless` forced) | 1.0 | 1/60 | 10 (steps forced to 1) | Analytic | Elastic sanity check for drift & restitution. | No randomness; solver restricted to sphere-only fast path. |
| `spheres_cloud_1024` | 1024 + static ground | Uniform cubic grid of spheres (radius 0.5) resting over a plane; nearest-neighbour contacts pre-populated. | 0.0 | 0.0 | 1/60 | 10 | Analytic | Throughput benchmark for dense contacts (frictionless). | Occupancy derived from integer lattice; deterministic contact ordering. |
| `spheres_cloud_4096` | 4096 + static ground | Same generator as above with `N=4096`. | 0.0 | 0.0 | 1/60 | 10 | Analytic | Larger frictionless cloud for scaling. | Deterministic lattice. |
| `spheres_cloud_8192` | 8192 + static ground | Same as above with `N=8192`. | 0.0 | 0.0 | 1/60 | 10 | Analytic | Upper mid-size scaling point. | Deterministic lattice. |
| `spheres_cloud_10k` | 10000 + static ground | Same as above with `N=10000`. | 0.0 | 0.0 | 1/60 | 10 | Analytic | Pushes cache and SoA batching; frictionless. | Deterministic lattice. |
| `spheres_cloud_50k` | 50000 + static ground | Same generator with `N=50000`. | 0.0 | 0.0 | 1/60 | 10 | Analytic | Stress test for multi-thread SoA. | Deterministic lattice (requires large memory footprint). |
| `spheres_cloud_10k_fric` | 10000 + static ground | Identical placement to `spheres_cloud_10k`. | 0.5 (tangential rows enabled) | 0.0 | 1/60 | 10 | Analytic | Frictional workload for cached/SoA solvers. | Deterministic lattice; solver overrides `frictionless=false`. |
| `box_stack_4` | 4 + static ground | Stack of four 1 m tall boxes aligned on Y. | 0.5 | 0.0 | 1/60 | 10 | Analytic | Penetration/Baumgarte stability check. | Contacts built deterministically; bias applied on normal row. |
| `box_stack` (`box_stack_8`) | 8 + static ground | Same as `box_stack_4` with eight layers. | 0.5 | 0.0 | 1/60 | 10 | Analytic | Taller stack for drift & ERP behavior. | Deterministic layout. |

> Use `--scene=spheres_cloud --sizes=...` or `--scene=box_stack --sizes=8` to generate the same variants programmatically. Solver defaults come from `bench::configure_solver_params`.

## Jointed scenes

| Scene key | N (dynamic bodies) | Joint configuration | μ (solver default) | Restitution | dt (s) | Iterations (default) | Seed | Goal / stressor | Determinism notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `pendulum` | 1 + static pivot | Single distance joint (`beta=0.2`, `compliance=0.0`) with slight initial velocity. | 0.5 | 0.0 | 1/60 | 10 | Analytic | Distance joint parity with cached impulses. | Joint rows built deterministically; no randomness. |
| `chain_64` | 64 + static pivot | 64-link chain (distance joints) with `compliance=1e-8`, `beta=0.2`. | 0.5 | 0.0 | 1/60 | 10 | Analytic | Tests compliant joints and warm-start reuse. | Link order is deterministic; solver may enable SoA joint batching. |
| `rope_256` | 256 + static anchor | Rope-style joints with zero compliance (`rope=true`, `beta=0`). | 0.5 | 0.0 | 1/60 | 10 | Analytic | Large joint system for SoA/native joint scatter. | Deterministic spacing along X axis. |

All joint scenes share the same deterministic layout and do not require seeds. When run with `--deterministic`, multi-threaded scattering is disabled to keep joint order identical.
