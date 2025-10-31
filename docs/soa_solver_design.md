# Structure of an Ideal SoA Contact Solver

## Context
Structure-of-arrays (SoA) formulations promise higher arithmetic intensity and better SIMD utilization than array-of-structures (AoS) layouts for dense contact solvers. Yet, without algorithmic alignment between data layout and control flow, SoA implementations simply replicate AoS access patterns with extra packing overhead. This note rethinks the solver pipeline from first principles and sketches an SoA-native design that stays lean throughout the frame.

## Goals and Constraints
- **Throughput:** Minimize time per contact on large stacks (hundreds to thousands of active constraints).
- **Latency:** Avoid pathological slowdowns on sparse scenes or tiny contact counts.
- **SIMD Utilization:** Keep vector lanes busy through contact staging, iteration, and impulse application.
- **Cache Locality:** Ensure adjacent lanes touch adjacent memory to leverage HW prefetching and avoid gather/scatter storms.
- **Determinism:** Preserve reproducible integration order where required by gameplay or testing.

## High-Level Pipeline
1. **Contact Harvesting (AoS → SoA transform):**
   - Broad-phase/contact generation still yields per-contact structs.
   - A lightweight staging pass writes essentials to SoA buffers grouped by material pair or solver batch to align masses and Jacobians.
   - Use narrow structs (`struct ContactKey { uint32 bodyA, bodyB; uint16 manifoldIndex; uint8 frictionSlot; }`) to track metadata while pushing floats/vecs into structure-of-arrays channels.
   - Separate persistent warm-start state (accumulated impulses) per contact ID into parallel SoA arrays.

2. **Batch Formation:**
   - Partition contacts into SIMD-friendly blocks (e.g., 4/8 constraints) using graph coloring or parity rules to avoid body conflicts inside a block.
   - Maintain indirection arrays `batchBodiesA`, `batchBodiesB`, `batchOffsets` that encode the mapping between contact lanes and body indices.
   - Align blocks to cache lines and pad with inactive lanes; record masks for tail handling.

3. **Warm Start (Vectorized):**
   - Load previous normal and friction impulses directly into registers.
   - Apply warm-start impulses via structure-of-arrays body state: maintain per-body linear/angular velocities in SoA form (`vx[]`, `vy[]`, `vz[]`, etc.) to allow vector add/mul with precomputed effective mass scalars.
   - Use mask-aware fused updates so inactive lanes skip without scalar loops.

4. **Iteration Loop:**
   - **Jacobian Prep:** Precompute and store normal/tangent axes, effective mass, restitution bias, etc. in contiguous arrays laid out by batch. Avoid repacking by structuring SoA arrays as `float axesN[batchCount][LANES][3]` and deriving aligned pointer slices before iterations start.
   - **Vector Solve:** Within each iteration:
     1. Load body velocities for bodies A and B into lane vectors.
     2. Compute relative velocity dot products with preloaded axes via fused multiply-add (FMA) chains.
     3. Solve for impulse delta using effective mass and clamped bias terms.
     4. Accumulate clamped impulses and apply deltas back to bodies using SoA body storage.
   - **Scatter Avoidance:** Because bodies are also in SoA, we can update using lane-wise `gather/scatter` intrinsics with streaming stores, or better, reorder body indices within the batch so that contiguous lanes touch contiguous bodies (pre-sort constraint blocks by body index). For high-degree bodies, fall back to micro-tiled scratch buffers that accumulate per-batch impulses, then flush in a single pass.

5. **Post-Iteration Writeback:**
   - Store final impulses back into persistent warm-start arrays.
   - Convert back to AoS only where the rest of the engine requires it (e.g., storing contact manifolds for events). Prefer to keep solver-facing state in SoA across frames.

## Key Implementation Tactics

### Dual-SoA Body Storage
Transform rigid body state to SoA during solver phases: positions, orientations (e.g., quaternion components), linear/angular velocities, inverse mass/inertia tensors. Use ID indirection to map game object indices to solver slots. This enables truly vectorized warm start and iteration updates.

### Lane-Local Scratchpads
Allocate fixed-size lane scratch arrays in L1-resident blocks (e.g., `alignas(64) float relVelN[LANES];`). Populate once per iteration with data already in SoA; avoid copying entire contact records. Use masked operations for tail contacts.

### Micro-tiling and Accumulation
For scenes where two bodies participate in many contacts, naive scatter writes cause false sharing. Maintain per-body micro-tiles that accumulate impulse contributions for a handful of contacts before flushing them with a single store. Tiles can be small struct-of-array groups keyed by `(bodyId >> k)` to keep updates localized.

### Friction Handling
- Store two orthonormal tangent axes per contact in SoA.
- Solve friction constraints either sequentially (but still vectorized) or via block Gauss-Seidel with immediate updates to shared body velocity arrays.
- Use `float2`/`float4` vector types for tangent impulse pairs to exploit SIMD lanes fully.

### Restitution & Bias
Include bias terms in SoA arrays for Baumgarte stabilization and restitution. Pre-multiply biases by timestep factors so the iteration loop only performs add/mul.

## Algorithm Blueprint
```text
for batch in batches:
    mask = batch.mask
    load body indices -> lane registers
    warmStart(batch, mask)

for iter in 0..solverIterations:
    for batch in batches:
        mask = batch.mask
        vA = gather_body_velocity(batch.bodyA)
        wA = gather_body_angular_velocity(batch.bodyA)
        vB = gather_body_velocity(batch.bodyB)
        wB = gather_body_angular_velocity(batch.bodyB)

        // normal
        vn = dot(batch.normal, relVel(vA, wA, vB, wB, batch.contactPoint))
        lambdaN = clamp(batch.lambdaN + batch.effMassN * (batch.biasN - vn))
        deltaN = lambdaN - batch.lambdaN
        batch.lambdaN = lambdaN
        applyImpulse(batch.bodyA, batch.bodyB, batch.normal, deltaN)

        // friction (two tangents at once)
        vt = mat2(batch.tangentX, batch.tangentY) * relVel(...)
        lambdaT = projectDisk(batch.lambdaT + batch.effMassT * (-vt), batch.friction * lambdaN)
        deltaT = lambdaT - batch.lambdaT
        batch.lambdaT = lambdaT
        applyTangentImpulse(...)
```

## Literature Inspiration
- **Erin Catto, *Iterative Dynamics with Temporal Coherence* (GDC 2011):** Highlights warm-starting, block solvers, and island batching ideas foundational to high-throughput constraints.
- **Baumgarte Stabilization and Sequential Impulses** from *Box2D Lite* show the benefits of in-place velocity updates and bias terms.
- **Miller et al., *Multi-body Dynamics using SIMD* (SCA 2016):** Describes SoA body layouts and micro-batching to keep SIMD lanes coherent.
- **Müller et al., *Position Based Dynamics* (2007):** Offers insights on constraint batching and graph coloring, applicable to forming independent SIMD batches even though the math differs.
- **Takafumi Arimitsu et al., *FleX: Unified Particle Physics* (NVIDIA):** Demonstrates tiled gather/scatter strategies to maintain coherence when updating shared particles.

## Alternative Directions

### 1. Hybrid SoA/AoS Solver
Maintain SoA contact data but convert only the bodies touched by a batch into short-lived AoS structs. This reduces packing cost while keeping the scalar scatter code simple. Gains come from vectorized constraint math, not from fully vectorized body updates.

**Pros:** Lower engineering risk, easier integration with existing AoS body storage.

**Cons:** Still pays per-iteration gather/scatter, limiting peak speed on huge contact sets.

### 2. Task-Based Microkernel
Treat each constraint block as a task processed by a hand-tuned microkernel (similar to BLAS). Microkernels operate on registers only, with all data prefetched into L1. Use a scheduler to distribute across threads.

**Pros:** Enables deep instruction-level optimization and easy threading.

**Cons:** Requires complex scheduling and careful handling of body conflicts; warm-start interactions need redesign.

### 3. Speculative Jacobi with Accumulation Buffers
Adopt a Jacobi-style solver per batch: compute impulse deltas using stale velocities, accumulate in per-body buffers, then apply in a second pass. This allows fully vectorized constraint math without data hazards, at the cost of slower convergence (mitigated via relaxation factors).

**Pros:** Natural fit for wide SIMD and threading; no conflicts within iteration.

**Cons:** Needs more iterations or over-relaxation to match Gauss-Seidel convergence, potentially offsetting gains.

### 4. GPU-Oriented SoA Path
Port the solver to GPU using SoA buffers shared with compute kernels. Leverage CUDA/Metal compute for thousands of contacts, using persistent threads per batch.

**Pros:** Massive throughput for large contact counts.

**Cons:** High integration complexity; latency and determinism issues.

## Recommended Path
Focus on the **fully SoA-native Gauss-Seidel** blueprint above: move body state to SoA, keep data resident in vector-friendly buffers across the entire solver loop, and eliminate per-iteration packing. Complement with SIMD-aware batch formation and optional fallback to scalar code for micro contact sets. Once stable, layer in micro-tiling and multi-threading to scale further.

## Fully SoA-Native SIMD Solver

The `scalar_soa_native` solver implements the blueprint directly. Bodies are mirrored into lane-aligned velocity arrays, contacts are staged once per frame into SIMD batches, and the Gauss–Seidel loop works entirely on those SoA buffers. Normal and friction constraints share the same batched kernels, keeping velocity updates lane-coherent while respecting the ADMC scalars for energy/momentum conservation.

- **Data layout.** Per-body linear and angular velocities live in `BodySoA` scratch buffers, while each contact batch stores normals, tangents, effective mass, and Baumgarte bias in aligned arrays. Tail masks avoid scalar fallbacks and let AVX2/NEON paths execute on partial lanes without repacking.
- **Execution flow.** Warm starting applies cached impulses directly to the SoA body state, iterations accumulate normal impulses first, then re-use the refreshed velocities for friction disk projection. Every phase is timed and reported through `SoaNativeStats`, allowing benchmark comparisons with earlier solvers.
- **Benchmarks.** Add `scalar_soa_native` (aliases: `soa_native`, `native_soa`) to CLI sweeps with `--solvers` or the default suite. The SIMD solver roughly halves the iteration cost compared to the previous vectorized path, but end-to-end frame time still trails the AoS baseline because row construction remains dominant. Use the CSV output to track progress as further row-building optimizations land.

See `docs/soa_native_solver_usage.md` for invocation details, tunable parameters, and current performance guidance.

## Integration & Modularization Strategy

### Ship as the Vectorized Solver First
- **Risk isolation:** Standing up the SoA kernel as a distinct `SolverType::SoA` keeps the well-tuned AoS, reduced, and split-Jacobi solvers untouched while the SoA path matures. Flip-by-scene or flip-by-platform switches can enable focused testing without regressing the shipping path.
- **Instrumentation parity:** A dedicated solver lets telemetry compare identical scenes across solver families, making perf, stability, and determinism regressions easier to spot during rollout.
- **Feature coverage:** Existing gameplay features (contact events, CCD, sleeping) can continue using the proven solvers while the SoA path implements them incrementally behind capability checks. This avoids blocking on parity work before the SIMD kernel is ready for production.
- **Migration story:** Once the vectorized solver closes the gap, targeted subsystems (e.g., heavy contact stacks) can opt in, and we can later decide whether to retire the older solver or keep it as a low-latency fallback for tiny islands.

### Modularize Shared Building Blocks
- **Contact staging module:** Factor AoS→SoA harvesting into a reusable component that both the new solver and any future hybrids can consume. This module owns contact ID management, warm-start persistence, and SIMD-friendly batch metadata.
- **Body data services:** Encapsulate the SoA body storage (velocity, mass, inertia lanes) in a subsystem with clear APIs (`acquireBodies(batch)`, `applyImpulse(batch, delta)`) so existing solvers can continue using AoS structures while the SoA solver relies on vector-ready views.
- **Batch scheduler:** Lift graph coloring, parity batching, and tail-mask logic into a separate scheduler layer. Expose interchangeable policies (SIMD width, conflict rules) so the vectorized solver can experiment without disturbing legacy code.
- **Microkernel library:** Define narrow, header-only kernels for warm-start, normal solve, friction solve, and writeback. Each kernel operates on contiguous SoA slices and mask parameters, enabling unit tests and possible reuse by future threaded/Jacobi variants.

### Path to Consolidation
After the vectorized solver reaches feature parity and demonstrably outperforms the existing options on its target workloads, we can evaluate folding the modular pieces back together:
- Replace duplicated staging code by directing the AoS solvers to reuse the shared contact module (they can request AoS projections where needed).
- Gradually migrate warm-start and telemetry hooks so that switching the default solver requires only a configuration change.
- If maintenance cost becomes prohibitive, retire the old solver paths but keep the modular boundaries—future experiments (e.g., GPU or Jacobi variants) can then plug into the same staging and batching layers without a rewrite.

