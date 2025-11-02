# Non-Relativistic Math Primer (for Directional Contact Solvers)

This note collects the Newtonian (NR) equations we rely on in the repo. It ties the “directional” view to standard constraint math and gives closed forms used in the solvers and tests.

## 1) State, momentum, and projections
For a rigid body \(i\):
- Position \( \mathbf x_i\), orientation \(q_i\) (unit quaternion).
- Linear velocity \( \mathbf v_i\), angular velocity \( \boldsymbol\omega_i\).
- Inverse mass \( m_i^{-1}\), world inverse inertia \( \mathbf I_i^{-1}\) (3×3).

**Linear and angular momentum** (NR):
\[
\mathbf p_i = m_i \mathbf v_i,\qquad \mathbf L_i = \mathbf I_i \boldsymbol\omega_i.
\]

**Projection along a unit direction** \( \hat{\mathbf n} \):
\[
p_{i,n} \equiv \mathbf p_i \cdot \hat{\mathbf n} = m_i\,(\mathbf v_i\cdot \hat{\mathbf n}).
\]
Total directional momentum is additive: \(P_n \equiv \sum_i p_{i,n}\).

## 2) Contact kinematics
For a world contact point \(\mathbf p\) with offsets \(\mathbf r_a=\mathbf p-\mathbf x_a\), \(\mathbf r_b=\mathbf p-\mathbf x_b\), the **relative velocity at the point** is
\[
\mathbf v_\text{rel} = \big(\mathbf v_b + \boldsymbol\omega_b \times \mathbf r_b\big)
                     - \big(\mathbf v_a + \boldsymbol\omega_a \times \mathbf r_a\big).
\]
Normal component: \( v_{\text{rel},n} = \hat{\mathbf n}\cdot \mathbf v_\text{rel}\).
Pick any orthonormal tangents \( \hat{\mathbf t}_1, \hat{\mathbf t}_2 \perp \hat{\mathbf n} \) for friction rows:
\[
v_{\text{rel},t_j} = \hat{\mathbf t}_j \cdot \mathbf v_\text{rel},\quad j\in\{1,2\}.
\]

## 3) Effective mass (scalar rows)
For a unit direction \(\hat{\mathbf d}\) (normal or a tangent), the **scalar effective mass** is
\[
k_d \equiv m_a^{-1} + m_b^{-1}
       + (\mathbf r_a \times \hat{\mathbf d})^\top \mathbf I_a^{-1} (\mathbf r_a \times \hat{\mathbf d})
       + (\mathbf r_b \times \hat{\mathbf d})^\top \mathbf I_b^{-1} (\mathbf r_b \times \hat{\mathbf d}).
\]
This is \(k_d = (J_d \mathbf M^{-1} J_d^\top)^{-1}\) in Jacobian form with
\[
J_d=\begin{bmatrix}
- \hat{\mathbf d}^\top & -(\mathbf r_a \times \hat{\mathbf d})^\top & \ \hat{\mathbf d}^\top & (\mathbf r_b \times \hat{\mathbf d})^\top
\end{bmatrix}.
\]

## 4) Impulse update (frictionless normal row)
A scalar impulse increment \(\Delta j_n\) along \(\hat{\mathbf n}\) changes velocities as
\[
\Delta \mathbf v_a = - m_a^{-1}\, \Delta j_n\, \hat{\mathbf n},\quad
\Delta \boldsymbol\omega_a = - \mathbf I_a^{-1}(\mathbf r_a \times (\Delta j_n\,\hat{\mathbf n})),
\]
\[
\Delta \mathbf v_b = + m_b^{-1}\, \Delta j_n\, \hat{\mathbf n},\quad
\Delta \boldsymbol\omega_b = + \mathbf I_b^{-1}(\mathbf r_b \times (\Delta j_n\,\hat{\mathbf n})).
\]
The induced change in relative normal speed is
\[
\Delta v_{\text{rel},n} = k_n\, \Delta j_n.
\]
Given target \(v_{\text{rel},n}^\star\) (e.g., restitution and/or bias),
\[
\boxed{\ \Delta j_n = \dfrac{v_{\text{rel},n}^\star - v_{\text{rel},n}}{k_n}\ }.
\]

## 5) Restitution and position correction (bias)
- **Restitution** \(e\in[0,1]\): if \(v_{\text{rel},n}<0\) (approaching),
  \(v_{\text{rel},n}^\star = -e\, v_{\text{rel},n}\).
- **Baumgarte/ERP bias** (penetration \(C\le 0\), slop \(s\ge0\), factor \(\beta\in[0,1]\), time step \(h\)):
  \[
  b = -\frac{\beta}{h}\,\max(0, -C - s),\qquad
  v_{\text{rel},n}^\star \mathrel{+}= b.
  \]

## 6) Coulomb friction (two scalar rows)
For each tangent \( \hat{\mathbf t}_j\):
\[
\Delta j_{t_j} = \frac{v_{\text{rel},t_j}^\star - v_{\text{rel},t_j}}{k_{t_j}},\qquad
v_{\text{rel},t_j}^\star \approx 0\ \text{(viscous/XPBD variants may differ)}.
\]
Clamp **after** updates:
\[
\sqrt{j_{t_1}^2 + j_{t_2}^2} \le \mu\, j_n.
\]

## 7) Directional momentum conservation (NR)
Impulse along \(\hat{\mathbf n}\) changes total directional momentum by
\[
\Delta P_n = \Delta j_n - \Delta j_n = 0\quad\text{(equal and opposite on \(a,b\))},
\]
so momentum **along any direction** is conserved in **isolated elastic** rows; bias/restitution prescribe target changes (energy dissipation is expected with \(e<1\), friction).

## 8) Metrics used in this repo
- **Directional momentum drift** for \(\{\hat{\mathbf d}_\ell\}\):
  \(\max_\ell \left| \sum_i \mathbf p_i^\text{after}\cdot\hat{\mathbf d}_\ell - \sum_i \mathbf p_i^\text{before}\cdot\hat{\mathbf d}_\ell \right|\).
- **Constraint violation (penetration)**: \( \|C\|_\infty\).
- **Energy drift** and **friction cone consistency** are tracked for validation.

Full derivations, units, and CSV column mapping are provided in `docs/metrics.md`.

