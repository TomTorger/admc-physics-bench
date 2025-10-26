# Algorithm Math: Scalar Normal Contact Row (Frictionless)

This file derives the scalar normal row used by `solver_scalar_cached` (and also matches the normal row in `solver_scalar_soa`). The point is: solve **one scalar** along \(\hat{\mathbf n}\).

## 1) Constraint and Jacobian
Normal non-penetration velocity constraint (one row):
\[
\phi \equiv v_{\text{rel},n} + b \ge 0, \quad v_{\text{rel},n} \equiv \hat{\mathbf n}\cdot\left[(\mathbf v_b + \boldsymbol\omega_b\times\mathbf r_b)-(\mathbf v_a + \boldsymbol\omega_a\times\mathbf r_a)\right].
\]
Row Jacobian \(J_n\) (see NR primer) gives \(k_n = (J_n \mathbf M^{-1} J_n^\top)^{-1}\).

## 2) Impulse solution (projected Gauss–Seidel step)
We seek \(j_n \ge 0\) s.t.
\[
v_{\text{rel},n}^{\text{new}} = v_{\text{rel},n} + k_n\,\Delta j_n \approx v_{\text{rel},n}^\star.
\]
Thus \(\Delta j_n = (v_{\text{rel},n}^\star - v_{\text{rel},n})/k_n\), with **projection** \(j_n \leftarrow \max(0, j_n+\Delta j_n)\) and \(\Delta j_n\) re-computed from the clamped value.

**Targets.**
- Purely elastic: \(v_{\text{rel},n}^\star = - e\, v_{\text{rel},n}\) if \(v_{\text{rel},n}<0\); else \(v_{\text{rel},n}^\star=0\).
- With bias: \(v_{\text{rel},n}^\star \leftarrow v_{\text{rel},n}^\star + b\).

## 3) Updates
Apply \( \mathbf P_n = \Delta j_n\,\hat{\mathbf n}\):
\[
\begin{aligned}
\mathbf v_a &\leftarrow \mathbf v_a - m_a^{-1}\,\mathbf P_n, &
\boldsymbol\omega_a &\leftarrow \boldsymbol\omega_a - \mathbf I_a^{-1}(\mathbf r_a \times \mathbf P_n), \\
\mathbf v_b &\leftarrow \mathbf v_b + m_b^{-1}\,\mathbf P_n, &
\boldsymbol\omega_b &\leftarrow \boldsymbol\omega_b + \mathbf I_b^{-1}(\mathbf r_b \times \mathbf P_n).
\end{aligned}
\]

## 4) Equivalence to “momentum along \(\hat{\mathbf n}\)”
The two bodies receive equal and opposite impulses \(\pm \mathbf P_n\), so the **sum** of directional momenta \(\sum_i \mathbf p_i\cdot\hat{\mathbf n}\) is unchanged by the row itself. Any change in \(v_{\text{rel},n}\) is strictly the prescribed target (restitution/bias). This is the NR expression of “conservation of momentum in *any* direction,” modulo non-elastic targets.

## 5) Implementation hooks for speed
- Cache \(k_n\) and warm-start \(j_n\) (both scalars) per contact.
- During iterations recompute only \(v_{\text{rel},n}\) and apply scalar \(\Delta j_n\).
- Lift back to vectors only when applying \( \Delta j_n \hat{\mathbf n}\).

