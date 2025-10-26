<!-- File: docs/alg_scalar_distance_joint_math.md -->

# Algorithm Math: Scalar Distance (Linear) Joint

Many joints reduce to a single scalar constraint along a direction \(\hat{\mathbf d}\). Solving them as scalar rows mirrors the contact normal row and benefits from the same caching and batching.

## 1) Constraint definition
Given two anchor points in world:
\[
\mathbf p_a = \mathbf x_a + \mathbf r_a,\qquad \mathbf p_b = \mathbf x_b + \mathbf r_b,
\]
desired distance \(L\) and current vector \( \mathbf \Delta = \mathbf p_b - \mathbf p_a\), with
\[
C \equiv \|\mathbf \Delta\| - L \approx \hat{\mathbf d}\cdot (\mathbf p_b - \mathbf p_a) - L,\qquad \hat{\mathbf d} \equiv
\begin{cases}
\frac{\mathbf \Delta}{\|\mathbf \Delta\|}, & \|\mathbf \Delta\|> \epsilon\\
\text{previous } \hat{\mathbf d}, & \text{otherwise}
\end{cases}
\]
Linearized **velocity** constraint:
\[
\dot C = \hat{\mathbf d}\cdot\left[(\mathbf v_b + \boldsymbol\omega_b \times \mathbf r_b) - (\mathbf v_a + \boldsymbol\omega_a \times \mathbf r_a)\right].
\]

## 2) Row Jacobian and effective mass
Use the same \(J_d\) as in the NR primer with \(\hat{\mathbf d}\) instead of \(\hat{\mathbf n}\). The scalar effective mass is
\[
k_d = m_a^{-1} + m_b^{-1}
    + (\mathbf r_a \times \hat{\mathbf d})^\top \mathbf I_a^{-1} (\mathbf r_a \times \hat{\mathbf d})
    + (\mathbf r_b \times \hat{\mathbf d})^\top \mathbf I_b^{-1} (\mathbf r_b \times \hat{\mathbf d}).
\]

## 3) Bias and update rule
With Baumgarte/ERP stabilization (\(\beta\in[0,1]\), time step \(h\)):
\[
b = -\frac{\beta}{h} C,
\qquad
\Delta j_d = \frac{-\dot C + b}{k_d}.
\]
Apply the impulse \( \mathbf P_d = \Delta j_d\, \hat{\mathbf d}\) as usual.

**Projection:** If the joint is **inequality-type** (e.g., rope), enforce \(j_d \ge 0\); for equality joints (rod), no projection (or two-sided clamp).

## 4) Conservation remarks
For **rigid rod** (equality) without bias, the row preserves total directional momentum (equal/opposite impulses). Bias terms inject constraint forces that may alter energy; this is expected (stabilization).

## 5) Notes for speed
- Cache \(k_d\) and \( \hat{\mathbf d}\) per joint per step.
- Warm-start \(\,j_d\) across iterations/steps for faster convergence.
- Joint rows batch identically to contact rows in the SoA solver.

