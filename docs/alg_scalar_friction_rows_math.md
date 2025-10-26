# Algorithm Math: Scalar Tangent (Friction) Rows

We model Coulomb friction with two orthonormal tangents \( \hat{\mathbf t}_1, \hat{\mathbf t}_2 \perp \hat{\mathbf n}\), solved as two **independent scalar** rows and then clamped inside the friction cone.

## 1) Tangent kinematics and effective masses
For \(j\in\{1,2\}\):
\[
v_{\text{rel},t_j} = \hat{\mathbf t}_j \cdot \big[(\mathbf v_b + \boldsymbol\omega_b \times \mathbf r_b) - (\mathbf v_a + \boldsymbol\omega_a \times \mathbf r_a)\big],
\]
\[
k_{t_j} = m_a^{-1} + m_b^{-1}
         + (\mathbf r_a \times \hat{\mathbf t}_j)^\top \mathbf I_a^{-1} (\mathbf r_a \times \hat{\mathbf t}_j)
         + (\mathbf r_b \times \hat{\mathbf t}_j)^\top \mathbf I_b^{-1} (\mathbf r_b \times \hat{\mathbf t}_j).
\]

## 2) Target and update
A common discrete target is **stick** (drive tangent relative velocity toward zero):
\[
v_{\text{rel},t_j}^\star = 0 \quad\Rightarrow\quad
\Delta j_{t_j} = -\frac{v_{\text{rel},t_j}}{k_{t_j}}.
\]
Apply \( \mathbf P_{t_j} = \Delta j_{t_j}\,\hat{\mathbf t}_j \) to bodies \(a,b\) (equal/opposite) exactly as for the normal row.

## 3) Coulomb cone clamp
Let \( \mathbf j_t = (j_{t_1}, j_{t_2})\) be the **accumulated** tangent impulses and \(j_n \ge 0\) the **accumulated** normal impulse.
\[
\|\mathbf j_t\| \le \mu\, j_n.
\]
If violated after updates, project back:
\[
\mathbf j_t \leftarrow \mu\, j_n \, \frac{\mathbf j_t}{\|\mathbf j_t\|}.
\]
Recompute \( \Delta \mathbf j_t = \mathbf j_t - \mathbf j_t^{\text{prev}}\) and apply the corresponding impulse correction along \( \hat{\mathbf t}_1, \hat{\mathbf t}_2\).

## 4) Energy and momentum notes
- Friction rows dissipate kinetic energy (work done opposite motion).
- Directional momentum along any axis is altered only by **non-conservative targets** (here, tangent rows drive \(v_{\text{rel},t}\to 0\)); the **row math remains scalar**, enabling speedups identical to the normal row.

## 5) Special cases
- **Spheres (no rotation):** \(k_{t_j} = m_a^{-1} + m_b^{-1}\).
- **Anisotropic inertia:** formulas above hold; precompute \((\mathbf r \times \hat{\mathbf t}_j)\) once per contact.

