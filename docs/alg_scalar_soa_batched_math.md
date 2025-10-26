<!-- File: docs/alg_scalar_soa_batched_math.md -->

# Algorithm Math: SoA-Batched Scalar Rows (Normal, Friction, Joints)

This file formalizes the **batched** scalar formulation used by `solver_scalar_soa`, enabling SIMD/GPU-friendly updates.

## 1) Batched row model
Consider \(N\) scalar rows with unit directions \(\{\hat{\mathbf d}_i\}\), indices \((a_i,b_i)\), offsets \((\mathbf r_{a_i},\mathbf r_{b_i})\), and effective masses \(\{k_i\}\). Stack scalars into arrays:
\[
\mathbf k = [k_i]_{i=1}^N,\quad \mathbf j = [j_i],\quad
\mathbf v_\text{rel} = [\hat{\mathbf d}_i \cdot \mathbf v_{\text{rel},i}],\quad
\mathbf v^\star = [v_{\text{rel},i}^\star].
\]

The **PGS step** in batch form:
\[
\Delta \mathbf j = \frac{\mathbf v^\star - \mathbf v_\text{rel}}{\mathbf k} \quad (\text{element-wise divide}),
\quad \mathbf j \leftarrow \Pi(\mathbf j + \Delta \mathbf j),
\]
where \(\Pi\) denotes per-row projection:
- normal: \(j_i \leftarrow \max(0, j_i)\),
- friction: cone projection with \(j_i\) grouped per contact into pairs (or triplets with normal).

## 2) Velocity updates (scatter-add)
For each row \(i\), the impulse is \( \mathbf P_i = \Delta j_i \hat{\mathbf d}_i\). Apply:
\[
\mathbf v_{a_i} \mathrel{-}= m_{a_i}^{-1}\,\mathbf P_i,\qquad
\boldsymbol\omega_{a_i} \mathrel{-}= \mathbf I_{a_i}^{-1} (\mathbf r_{a_i}\times \mathbf P_i),
\]
\[
\mathbf v_{b_i} \mathrel{+}= m_{b_i}^{-1}\,\mathbf P_i,\qquad
\boldsymbol\omega_{b_i} \mathrel{+}= \mathbf I_{b_i}^{-1} (\mathbf r_{b_i}\times \mathbf P_i).
\]
These are **independent per row** and can be performed in SIMD lanes or GPU threads with conflict-free scatter (or via coloring/integration strategies if bodies are shared).

## 3) Precomputation (once per step)
For each row:
\[
k_i = m_{a_i}^{-1} + m_{b_i}^{-1}
    + (\mathbf r_{a_i} \times \hat{\mathbf d}_i)^\top \mathbf I_{a_i}^{-1} (\mathbf r_{a_i} \times \hat{\mathbf d}_i)
    + (\mathbf r_{b_i} \times \hat{\mathbf d}_i)^\top \mathbf I_{b_i}^{-1} (\mathbf r_{b_i} \times \hat{\mathbf d}_i),
\]
store \(k_i\), \(j_i\) (warm start), \(\hat{\mathbf d}_i\), \(\mathbf r_{a_i}\), \(\mathbf r_{b_i}\), and indices \(a_i,b_i\) in **Structure-of-Arrays** buffers.

## 4) Cone projection (friction) in batch
Group rows \((i_{n}, i_{t1}, i_{t2})\) per contact. After updating the two tangents, perform:
\[
\mathbf j_t = \begin{bmatrix} j_{t1}\\ j_{t2} \end{bmatrix},\qquad
\mathbf j_t \leftarrow \min\!\left(1, \frac{\mu\, j_n}{\|\mathbf j_t\|+\epsilon}\right)\, \mathbf j_t.
\]
Compute \(\Delta \mathbf j_t\) and apply the difference; this remains per-contact local and vectorizes over contacts.

## 5) Convergence and stability
- With warm-starting, the batched PGS typically converges in a few iterations for contact-dense scenes.
- Determinism benefits from fixed iteration counts and SoA memory order.
- All **math is identical** to the scalar single-row case; speedups come from data layout and batched operations.

## 6) Why this accelerates
- Element-wise arithmetics on \((\mathbf v^\star - \mathbf v_\text{rel})/\mathbf k\) map cleanly to wide SIMD.
- Reduced register pressure vs vector-heavy per-row math.
- Better cache behavior: contiguous loads of \(\hat{\mathbf d}_i, k_i, j_i\).

