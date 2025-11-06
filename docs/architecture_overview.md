# ADMC Next-Gen Solver Architecture

## 1. Motivation & Goals

We want a **fresh repository** that:

- Keeps the existing **baseline AoS solver** as a clean reference.
- Encodes what we’ve learned from `admc-physics-bench` about performance, data layout, and ADMC invariants.
- Provides a **modular, extensible architecture** for future solver research:
  - SIMD & SoA variants.
  - Block and Krylov-accelerated solvers.
  - Island/tile-based parallelization.
  - GPU-ready backends.
- Makes it easy to **compare solver variants** on shared scenes with consistent metrics (ADMC invariants, penetration, energy & momentum drift, etc.).

This document:

1. Summarizes **current learnings** from the existing repo.
2. States **design principles** for the next-gen solver.
3. Outlines **ideas & improvements** we want the architecture to support.
4. Proposes **solver architectures** and their strengths/weaknesses.
5. Describes a **modular file structure** for the new repository.
6. Lists **key literature** for reference.


---

## 2. Learnings from the Current Repo

### 2.1 Baseline AoS Solver

**Characteristics**

- AoS storage: `std::vector<RigidBody>`, `std::vector<Contact>`.
- Sequential impulses / PGS:
  - Normal contact rows.
  - ERP/Baumgarte position correction.
  - Basic restitution.
- Minimal caching; recomputes many quantities per iteration.

**What we’ve learned**

- **Clarity & correctness**: the baseline is easy to reason about, debug, and validate.
- **Low constant cost**: excellent for tiny scenes (a handful of contacts).
- **Performance ceiling**: for thousands of contacts, cache behaviour and lack of SIMD become limiting.
- Ideal to keep as a **reference and regression baseline**, but not as the main performance path.

---

### 2.2 Scalar Cached Solver (AoS + Cached Rows)

**Characteristics**

- Still AoS for bodies/contacts.
- Builds per-contact scalar rows:
  - One normal row.
  - Two tangent rows for friction.
- Caches:
  - Effective masses for rows.
  - Accumulated impulses for warm-starting.

**What we’ve learned**

- **Warm-starting** helps convergence: fewer effective iterations when configurations are slowly changing.
- Cached denominators reduce some per-iteration overhead.
- However, the extra scalar bookkeeping can **offset gains** in some scenes, especially when contact graphs change rapidly.
- It’s a good example of “classic” game-physics optimizations, but not the end-state for performance.

---

### 2.3 Scalar SoA Solver

**Characteristics**

- Per-step conversion from AoS → SoA (`RowSOA`):
  - Scalar arrays for normals, tangents, effective mass, impulses, etc.
- Scalar PGS/Sequential impulses over SoA arrays.
- Good timing decomposition:
  - Contact prep, row build, solver (warm/iter/integrate), scatter.

**What we’ve learned**

- SoA **improves locality** and reduces cache misses in the solver loop.
- However, **row build cost becomes dominant**:
  - For large scenes, AoS→SoA construction and contact prep are often >50% of frame time.
- Helps us see that:
  - Solver math is relatively cheap.
  - Constraint assembly and data movement are the real hotspots.

---

### 2.4 SoA-Native SIMD Solver

**Characteristics**

- World and constraints represented in SoA form.
- Tiles/groupings:
  - Each tile has a local body table and batched rows.
- SIMD-friendly kernels for:
  - Normal rows (especially spheres).
  - Optionally friction rows.

**What we’ve learned**

- For non-trivial scenes (10k–50k contacts), **SoA-Native is the fastest CPU solver**.
- Significant speedups over baseline in the solver phase.
- Still constrained by:
  - Contact prep.
  - Row build / tiling.
- Confirms that **data layout + tiling** are critical for high performance.

---

### 2.5 SoA-Parallel Solver

**Characteristics**

- Island-based + tile-based parallel wrapper around SoA-Native.
- Target: run independent islands or tiles on multiple threads.

**What we’ve learned**

- Conceptually right: islands/tiles are the correct parallelization granules.
- In practice, current implementation reveals:
  - Parallel overhead and scheduling can kill gains.
  - We must design a clean separation between:
    - **Solver kernels** on well-defined local data.
    - **Execution backends** (single-thread, multi-thread, GPU).

---

### 2.6 Vec-SoA Path

**Characteristics**

- SoA-friendly layout, currently forwarding largely to scalar kernels.
- Intended as a home for heavier vectorization / experimentation.

**What we’ve learned**

- Data layout alone is not enough; **kernels must be genuinely SIMD-aware**.
- Useful as a staging area for future vectorized implementations, but not competitive yet.

---

### 2.7 High-Level Takeaways

Across all solvers and scenes:

1. **Constraint assembly dominates** large scenes.  
   - Contact prep and AoS→SoA row build are the main costs.
2. **SoA-Native wins at scale**.  
   - Vector-friendly layouts and tiling unlock most of the solver speed.
3. **AoS baseline still best for tiny scenes**.  
   - Constant overhead matters when contacts are few.
4. **Parallel scaling requires design**, not just “add threads”.
   - Need explicit islands, tiles, and conflict-free batches.
5. **Instrumentation is invaluable**.  
   - Timing breakdowns + ADMC invariants are critical to guide optimization.


---

## 3. Design Principles for the Next-Gen Solver

The new architecture should follow these principles:

1. **Layered architecture**
   - Separate responsibilities:
     - Physics model & invariants (ADMC / NR).
     - Constraint assembly.
     - Solver algorithms.
     - Data layouts.
     - Execution backends (CPU, GPU).

2. **Row-centric API, layout-agnostic interface**
   - Public APIs speak in terms of bodies, contacts, and **constraint rows**.
   - Implementations can use AoS, SoA, AoSoA, or GPU buffers under the hood.

3. **Tile/Island-centric performance**
   - Internally, constraints and bodies are handled in **tiles** and **islands** for locality and parallelism.
   - Tiles are small, cache-resident chunks suitable for SIMD.

4. **ADMC-aware but not ADMC-locked**
   - ADMC invariants are part of metrics and checks.
   - Core solver stays standard enough for people who know PGS / sequential impulses to follow.

5. **Metrics-first**
   - Every solver run can emit:
     - Timing breakdowns.
     - ADMC directional drifts.
     - Penetration & joint errors.
     - Energy & momentum drift.
   - Benchmarking is a first-class use case, not an afterthought.

6. **Experiment-friendly**
   - Easy to introduce new solver variants:
     - Block solvers.
     - Over-relaxed PGS.
     - Krylov-accelerated or ADMM-inspired schemes.
   - Config-driven selection of solver + layout + backend.

7. **Preserve the baseline**
   - Keep the simple AoS baseline as a **canonical reference implementation**.
   - Use it in tests and as a sanity check for new solvers.


---

## 4. Ideas & Improvement Opportunities

### 4.1 Data Layout & Row Build

- **SoA-all-the-way where it matters**
  - Avoid repeated AoS→SoA conversions.
  - For high-performance paths, treat SoA/AoSoA as canonical.

- **AoSoA / tile-centric layout**
  - Represent the world as:
    - A set of islands.
    - Each island holding a vector of SoA tiles.
  - Each tile contains:
    - Local body table.
    - Compact arrays of constraint rows (normals, tangents, joints).

- **Cross-frame caching**
  - Persist:
    - Contact frames (normal & tangents).
    - Relative offsets.
    - Effective masses.
  - Recompute only when:
    - Orientations change beyond a threshold.
    - Contact topology changes.

- **Contact archetypes**
  - Special-case simple geometries:
    - Sphere–sphere.
    - Sphere–plane.
  - Use closed-form contact and effective-mass formulas to minimize per-frame work.

---

### 4.2 Solver Algorithms

- **Residual-based adaptive iterations**
  - Track per-tile residual norms.
  - Early exit tiles that are converged while continuing on “hard” tiles.

- **Block solvers per manifold**
  - Solve small coupled systems:
    - 3 DOF (1 normal + 2 tangents).
    - Or 6/12 DOF for multi-point contacts.
  - Use these as preconditioners inside PGS or as standalone solvers for tightly coupled patches.

- **Over-relaxed PGS (SOR)**
  - Add a relaxation parameter ω to accelerate convergence.
  - Tune globally or per-tile; keep it configurable.

- **Krylov-accelerated PGS**
  - Use residual estimates to apply light-weight acceleration (e.g. Anderson acceleration or limited-memory techniques).
  - Still clamp impulses and respect inequality constraints.

- **ADMM-inspired variants (long-term)**
  - Investigate ADMM-style splitting methods for contact problems, especially for:
    - Better friction cone handling.
    - Improved propagation in large stacks/clusters.

---

### 4.3 Parallelism & Scheduling

- **Island decomposition**
  - Identify independent constraint subgraphs.
  - Schedule islands across threads with minimal synchronization.

- **Graph coloring within islands**
  - Color the constraint graph so each color has non-overlapping bodies.
  - Process colors sequentially, rows within a color in parallel.

- **Tile-based task system**
  - Use tiles as the basic units of work for the executor.
  - Each task operates on a local world copy or on disjoint subsets of bodies.

- **Configurable execution backends**
  - Single-thread backend for simplicity and debugging.
  - Multi-thread backend for production performance.
  - GPU backends in the future, sharing the same constraint and tile abstractions.

---

### 4.4 Diagnostics & Tooling

- **Central metrics module**
  - Timing, ADMC invariants, penetration/joint error, and energy drift.
  - Hooks that any solver can call with minimal coupling.

- **Bench harness**
  - CLI for:
    - Selecting scenes.
    - Selecting solver + layout + backend.
    - Controlling iterations, tolerances, and other parameters.
  - Output results as human-readable tables and CSV/JSON for analysis.

- **Regression tests**
  - Compare new solvers against baseline on:
    - Simple scenes (two spheres, small stacks).
    - Moderate scenes (small clouds).
  - Use tolerance-based comparisons for positions, velocities, and invariants.


---

## 5. Candidate Architectures & Trade-offs

### 5.1 Architecture A — Row-Centric Pipeline

**Concept**

- Treat **constraint rows** as the central abstraction.
- Pipeline:
  1. World state → contact/joint generation.
  2. Contacts/joints → constraint rows.
  3. Rows → solver.
  4. Solver impulses → velocity & position updates.

**Strengths**

- Very **transparent** and easy to instrument.
- Naturally supports different solver algorithms on the same row set.
- Good mental model for ADMC and scalar-row reasoning.

**Weaknesses**

- Naive implementation leads to **large global arrays**.
- Without tiling, row build could stay the main bottleneck.
- Needs an additional layer for SIMD/parallel tile-level processing.

---

### 5.2 Architecture B — Tile/Island-Centric SoA Engine

**Concept**

- Organize the universe as **islands** of **tiles**:
  - Each tile:
    - Has a local body table.
    - Owns SoA constraint rows.
- Solvers operate tile-by-tile, with tile-local SoA kernels.

**Strengths**

- Great **cache locality** and clear SIMD story.
- Natural units for **task-based parallelism**.
- Row build can be fused with tiling:
  - For each contact, directly fill a tile slot instead of a global array.

**Weaknesses**

- Higher conceptual complexity (tiles + islands + local/global mappings).
- Needs careful API design to remain approachable.

---

### 5.3 Architecture C — Hybrid (Row API, Tile Engine, Multi-Backend)

**Concept**

- Public interface is row-centric (Architecture A).
- High-performance implementation is tile/island-centric (Architecture B).
- Multiple execution backends share common abstractions:
  - AoS Baseline.
  - SoA tile CPU solver.
  - Future GPU solver.

**Strengths**

- Best of both worlds:
  - Clean, solver-independent API.
  - High-performance internal implementation.
- Easier to add new solver variants or backends without breaking user code.

**Weaknesses**

- More layers → more boilerplate if not carefully managed.
- Needs discipline to avoid duplication across backends.

---

### 5.4 Recommended Architecture

We propose to adopt **Architecture C**:

- **Externally**:
  - Row-centric APIs that talk about:
    - Worlds.
    - Contacts & joints.
    - Constraint rows and solver configurations.

- **Internally**:
  - Tile- and island-centric SoA/AoSoA world:
    - Efficient SIMD and cache-friendly operations.
    - Natural units for multi-thread scheduling.

- **Backends**:
  - `baseline` AoS solver for reference & regression.
  - `pgs_soa` tile-based solver for performance.
  - Additional `block`, `krylov`, and GPU solvers as extensions.

This gives us a stable high-level model while allowing aggressive optimization under the hood.


---

## 6. Modularization & Components

### 6.1 Core Modules

1. **`core`**
   - Vector/matrix/quaternion types.
   - Generic math utilities.
   - Aligned allocators and small fixed-size containers.

2. **`admc`**
   - ADMC scalar invariants (`p_{k±}`).
   - Helpers for mapping between ADMC and NR quantities.
   - Metrics for directional momentum conservation.

3. **`world`**
   - Rigid body definitions (mass, inertia, pose, velocities).
   - Materials and other physical properties.
   - World/scene containers and island structure.

4. **`constraints`**
   - Contact manifolds and joint descriptions.
   - Constraint row representations (normal, tangents, joints).
   - Builders from contacts/joints to rows.
   - Constraint graph utilities (islands, graph coloring).

5. **`layout`**
   - AoS and SoA/AoSoA views over bodies and constraints.
   - Tile and island data structures.
   - Conversion & staging utilities.

6. **`solver`**
   - Common solver configuration and statistics types.
   - Abstract solver interfaces.
   - Concrete implementations:
     - `baseline` (AoS reference).
     - `pgs_scalar` (row-centric PGS).
     - `pgs_soa` (tile-based SoA PGS).
     - `block` (block-based solvers).
     - `krylov` (accelerated solvers).

7. **`backend`**
   - Execution backends:
     - Single-thread executor.
     - Multi-thread executor (task-based).
     - GPU executor stubs (for future use).

8. **`metrics`**
   - Timing utilities.
   - Physics quality metrics (penetration, drift, invariants).
   - Logging & export.

9. **`scenes`**
   - Scene generation and configuration.
   - Reusable benchmark scenes.

10. **`bench` & `tools`**
    - CLI benchmarking harness.
    - Result export/processing tools.

11. **`legacy`**
    - Original baseline AoS solver preserved for reference and tests.


---

## 7. Proposed Directory Layout

A concrete layout for the new repo:

```text
admc-next/
├─ CMakeLists.txt
├─ README.md
├─ docs/
│  ├─ architecture_overview.md        # High-level description (this document or successor)
│  ├─ solver_design.md                # Detailed solver algorithms & math
│  ├─ layout_and_tiling.md            # SoA/AoSoA, tiles, islands
│  ├─ admc_invariants.md              # ADMC theory & metrics
│  └─ benchmarks.md                   # Bench setup, methodology, and results
├─ include/
│  └─ admc/
│     ├─ core/
│     │  ├─ math_types.hpp            # Vec3, Mat3, Quat...
│     │  ├─ math_utils.hpp
│     │  ├─ aligned_allocator.hpp
│     │  └─ small_array.hpp
│     ├─ admc/
│     │  ├─ invariants.hpp            # p_{k±} and directional momentum helpers
│     │  ├─ metrics.hpp               # ADMC-based drift checks
│     │  └─ nr_bridge.hpp             # Non-relativistic mappings
│     ├─ world/
│     │  ├─ body.hpp                  # RigidBody, BodyId, etc.
│     │  ├─ material.hpp              # Material/friction/restition
│     │  ├─ world_state.hpp           # World, islands, global IDs
│     │  └─ transforms.hpp
│     ├─ constraints/
│     │  ├─ contact.hpp               # ContactPoint, ContactManifold
│     │  ├─ joint.hpp                 # Joint definitions
│     │  ├─ constraint_row.hpp        # Row representation
│     │  ├─ constraint_builder.hpp    # Builds rows from contacts/joints
│     │  └─ constraint_graph.hpp      # Islands, graph coloring, etc.
│     ├─ layout/
│     │  ├─ aos_world_view.hpp        # Views over AoS body/contact arrays
│     │  ├─ soa_world.hpp             # SoA representation of bodies
│     │  ├─ tile.hpp                  # Tile struct + local body tables
│     │  ├─ island.hpp                # Island containers
│     │  └─ layout_utils.hpp          # Conversion & staging helpers
│     ├─ solver/
│     │  ├─ solver_common.hpp         # SolverParams, SolverStats
│     │  ├─ solver_interface.hpp      # Abstract solver API
│     │  ├─ baseline/
│     │  │  ├─ baseline_solver.hpp    # AoS reference solver
│     │  ├─ pgs_scalar/
│     │  │  ├─ scalar_solver.hpp      # Row-centric scalar PGS
│     │  ├─ pgs_soa/
│     │  │  ├─ soa_solver.hpp         # Tile-based SoA PGS
│     │  │  ├─ soa_kernels.hpp        # SIMD kernels (normals, friction)
│     │  ├─ block/
│     │  │  ├─ block_solver.hpp       # Block-manifold solvers
│     │  ├─ krylov/
│     │  │  ├─ krylov_solver.hpp      # Accelerated variants
│     │  └─ config/
│     │     ├─ solver_presets.hpp     # “Fast”, “Stable”, etc.
│     ├─ backend/
│     │  ├─ executor.hpp              # Executor abstraction
│     │  ├─ single_thread_executor.hpp
│     │  ├─ multi_thread_executor.hpp # Task-based scheduler
│     │  └─ gpu_executor_stub.hpp     # Placeholder for future GPU backend
│     ├─ metrics/
│     │  ├─ timing.hpp                # Scoped timers, breakdown structs
│     │  ├─ physics_metrics.hpp       # Penetration, drift, etc.
│     │  └─ logging.hpp               # Logging hooks/utilities
│     └─ scenes/
│        ├─ scene_factory.hpp         # Registration/creation of scenes
│        ├─ spheres_cloud.hpp
│        ├─ box_stacks.hpp
│        ├─ joint_scenes.hpp
│        └─ randomized_scenes.hpp
├─ src/
│  ├─ core/
│  │  ├─ math_types.cpp               # Non-inline math (if any)
│  │  └─ math_utils.cpp
│  ├─ world/
│  │  ├─ world_state.cpp
│  │  └─ transforms.cpp
│  ├─ constraints/
│  │  ├─ constraint_builder.cpp
│  │  └─ constraint_graph.cpp
│  ├─ layout/
│  │  ├─ soa_world.cpp
│  │  ├─ tile.cpp
│  │  └─ island.cpp
│  ├─ solver/
│  │  ├─ baseline/
│  │  │  └─ baseline_solver.cpp
│  │  ├─ pgs_scalar/
│  │  │  └─ scalar_solver.cpp
│  │  ├─ pgs_soa/
│  │  │  ├─ soa_solver.cpp
│  │  │  └─ soa_kernels.cpp
│  │  ├─ block/
│  │  │  └─ block_solver.cpp
│  │  └─ krylov/
│  │     └─ krylov_solver.cpp
│  ├─ backend/
│  │  ├─ single_thread_executor.cpp
│  │  ├─ multi_thread_executor.cpp
│  │  └─ gpu_executor_stub.cpp
│  ├─ metrics/
│  │  ├─ timing.cpp
│  │  └─ physics_metrics.cpp
│  └─ scenes/
│     ├─ spheres_cloud.cpp
│     ├─ box_stacks.cpp
│     ├─ joint_scenes.cpp
│     └─ randomized_scenes.cpp
├─ apps/
│  ├─ bench/
│  │  ├─ CMakeLists.txt
│  │  └─ main_bench.cpp               # CLI bench driver
│  ├─ viewer/
│  │  ├─ CMakeLists.txt
│  │  └─ main_viewer.cpp              # Optional interactive viewer
│  └─ tools/
│     ├─ dump_metrics.cpp
│     └─ convert_results_to_csv.cpp
├─ tests/
│  ├─ CMakeLists.txt
│  ├─ test_math.cpp
│  ├─ test_admc_invariants.cpp
│  ├─ test_baseline_vs_soa.cpp
│  ├─ test_constraint_builder.cpp
│  ├─ test_solvers_regression.cpp
│  └─ test_scenes.cpp
└─ legacy/
   ├─ README.md                        # Notes on the legacy baseline
   ├─ baseline_aos_solver.hpp
   └─ baseline_aos_solver.cpp
````

Notes:

* `include/admc` is the public API surface.
* `src/` holds implementations; performance-critical pieces can stay header-only if needed.
* `legacy/` keeps a near-verbatim copy of the old baseline AoS solver.

---

## 8. Using the Baseline in the New Repo

To preserve the baseline:

* Place it in `legacy/` with minimal changes.
* Provide a small adapter so it can be invoked through the same bench harness:

  * Example: `run_legacy_baseline(World&, Contacts&, SolverParams&)`.
* In tests, compare:

  * Baseline vs `pgs_soa` and other solvers on selected scenes.
  * Check:

    * Positions and velocities (within tolerances).
    * ADMC invariants.
    * Penetration levels.

This ensures that optimization and architectural changes do not silently degrade physical behavior.

---

## 9. Key Literature & Links

A non-exhaustive list of references that inform this architecture:

* Erin Catto, **“Iterative Dynamics with Temporal Coherence”**, GDC 2005.
  [https://box2d.org/files/ErinCatto_IterativeDynamics_GDC2005.pdf](https://box2d.org/files/ErinCatto_IterativeDynamics_GDC2005.pdf)

* Erin Catto, **“Understanding Constraints”**, GDC 2014.
  [https://box2d.org/files/ErinCatto_UnderstandingConstraints_GDC2014.pdf](https://box2d.org/files/ErinCatto_UnderstandingConstraints_GDC2014.pdf)

* Da Wang & Martin Servin, **“Warm starting the projected Gauss-Seidel algorithm for nonsmooth discrete element simulation”**, *Computational Particle Mechanics*, 2016.
  Preprint: [https://arxiv.org/abs/1509.04042](https://arxiv.org/abs/1509.04042)
  Journal PDF: [https://umit.cs.umu.se/modsimcomplmech/docs/warm_start.pdf](https://umit.cs.umu.se/modsimcomplmech/docs/warm_start.pdf)

* Y.-L. Chen, M. Ly, C. Wojtan, **“Unified treatment of contact, friction and shock propagation in rigid body animation”**, 2023 (ADMM-based contact solver).
  Project page: [https://visualcomputing.ist.ac.at/publications/2023/UTCFS/](https://visualcomputing.ist.ac.at/publications/2023/UTCFS/)

* “From Compliant to Rigid Contact Simulation: a Unified and Robust ADMM Framework”, 2024.
  arXiv: [https://arxiv.org/abs/2405.17020](https://arxiv.org/abs/2405.17020)

* Alp Aydinoglu & Michael Posa, **“Real-Time Multi-Contact Model Predictive Control via ADMM”**, 2022.
  [https://dair.seas.upenn.edu/assets/pdf/Aydinoglu2022.pdf](https://dair.seas.upenn.edu/assets/pdf/Aydinoglu2022.pdf)

These works provide context and inspiration for:

* Iterative PGS / sequential impulsive solvers.
* Warm-starting and convergence improvements.
* ADMM-style methods for contact and friction.
* General design trade-offs in modern rigid body simulation.

---

## 10. Roadmap (High-Level)

1. **Bootstrap**

   * Implement core math, world, constraints.
   * Port the **legacy baseline** into `legacy/`.
   * Add a minimal bench app and a simple scene or two.

2. **SoA Tile Backend**

   * Implement `layout::tile` and `layout::soa_world`.
   * Port the current SoA-Native solver into `solver::pgs_soa`.
   * Reproduce current benchmark performance.

3. **Executor Layer**

   * Single-thread executor.
   * Multi-thread executor using tile-level tasks and islands.

4. **Metrics & Tests**

   * Implement ADMC and physics metrics.
   * Add regression tests comparing baseline vs new solvers.

5. **Advanced Solvers**

   * Block solvers for manifolds.
   * Over-relaxed PGS and residual-based adaptive iterations.
   * Prototype Krylov/ADMM-inspired variants using existing row data.

6. **Optional Extras**

   * GPU-backed solver using the same tile/row abstractions.
   * Simple visual viewer to inspect scenes and check stability.

This roadmap should give a clear path from “clean baseline + fresh architecture” to a robust, research-friendly solver stack that’s ready for further experimentation.
