# ADMC: Additive Directional Momentum Conservation — Overview & Proof

> This note summarizes the “ADMC” formulation used in this library and sketches the proof that it is **exactly equivalent** to standard special-relativistic conservation of energy and momentum. It also records the non-relativistic limit and the uniqueness result behind the (p_{k^\pm}) representation. For Newtonian contact math and solver formulas, see `docs/nr_math.md`. Bench metric definitions live in `docs/metrics.md`.

## 1) Setup and notation

We work in **momentum units**:
[
P^\mu=(M,\mathbf p), \qquad E=c,M,\qquad p_f\equiv m_0 c .
]
The dispersion relation is
[
M=\sqrt{p_f^2+\mathbf p^2}.
]
This is just standard SR, expressed with (M) (energy divided by (c)) as the time-like component. 

Fix a Cartesian axis (k\in{x,y,z}). Define the **directional scalars**
[
p_{k^\pm};\equiv; M ;\pm; \tfrac12,p_k .
]
These are additive over constituents of an isolated system (sums over particles), and are scalars under **spatial** rotations of the chosen frame (not Lorentz scalars). Under parity, (p_k\mapsto -p_k) swaps (p_{k^+}\leftrightarrow p_{k^-}). 

**ADMC Postulate.** In any isolated process, for each axis (k), the two totals (\sum_i p_{k^+}^{(i)}) and (\sum_i p_{k^-}^{(i)}) are separately conserved. 

## 2) Linear relations you’ll use in code

The change of variables is linear and invertible:
[
\begin{pmatrix} p_{k^+} \ p_{k^-}\end{pmatrix}
==============================================

\begin{pmatrix}
1 & \tfrac12\
1 & -\tfrac12
\end{pmatrix}
\begin{pmatrix} M\ p_k\end{pmatrix},
\qquad
\begin{pmatrix} M\ p_k\end{pmatrix}
===================================

\begin{pmatrix}
\tfrac12 & \tfrac12\
1 & -1
\end{pmatrix}
\begin{pmatrix} p_{k^+}\ p_{k^-}\end{pmatrix}.
]
Therefore, for any collection of particles,
[
\sum_i (p_{k^+}^{(i)}+p_{k^-}^{(i)}) = 2\sum_i M^{(i)},\qquad
\sum_i (p_{k^+}^{(i)}-p_{k^-}^{(i)}) = \sum_i p_k^{(i)} .
]
These two identities are the algebraic “add/subtract” backbone used throughout. 

## 3) Equivalence theorem (ADMC ⇔ energy & momentum conservation)

**Theorem.** Assuming the standard dispersion (M=\sqrt{p_f^2+\mathbf p^2}), the ADMC postulate is **equivalent** to the usual conservation of energy (E=cM) and three-momentum (\mathbf p) for arbitrary isolated processes (i.e., to four-momentum conservation). 

**Proof sketch.**
• *(ADMC ⇒ energy & momentum).* Add and subtract the two conserved sums for a fixed axis (k):
[
\sum_i(p_{k^+}^{(i)}+p_{k^-}^{(i)})=2\sum_i M^{(i)},\quad
\sum_i(p_{k^+}^{(i)}-p_{k^-}^{(i)})=\sum_i p_k^{(i)}.
]
Both right-hand sides are constants; do this for (k=x,y,z) to get conservation of (\sum_i \mathbf p^{(i)}) and (\sum_i M^{(i)}); with (E=cM), energy is conserved. 

• *(Energy & momentum ⇒ ADMC).* If (\sum_i M^{(i)}) and (\sum_i p_k^{(i)}) are conserved, then by the inverse linear map, each (\sum_i p_{k^\pm}^{(i)}=\sum_i M^{(i)}\pm \tfrac12\sum_i p_k^{(i)}) is conserved, for every (k) and each sign. 

Thus ADMC is just a re-parameterization of the standard conservation statement—nothing more, nothing less. 

**Corollary (Any direction).** Since vector momentum is conserved, (\sum \mathbf p\cdot \hat{\mathbf n}) is conserved for **every** unit vector (\hat{\mathbf n}). The ADMC bookkeeping lets you read that conservation directly from the (k)-axis you choose (or after rotating the frame). 

## 4) Uniqueness of the (p_{k^\pm}) form

Under four mild structural assumptions—**additivity, isotropy, parity,** and **local invertibility**—the only possible functional form is
[
p_{k^\pm}=\alpha,M \pm \beta,p_k,
]
i.e., an affine form fixed **up to a common rescaling**. Introducing
[
S_k\equiv p_{k^+}+p_{k^-},\qquad D_k\equiv p_{k^+}-p_{k^-},
]
one shows from additivity/parity/isotropy that (S_k=u,M) and (D_k=v,p_k) with constants (u,v), then recombines to the affine form above. A natural calibration sets (D_k=p_k) and (S_k=2M), giving the canonical
[
p_{k^\pm}=M\pm\tfrac12 p_k .
]
This fixes both constants without loss of generality.  

## 5) Non-relativistic (NR) limit and the kinetic-energy piece

Expand (M) for (p\ll p_f):
[
M=p_f\sqrt{1+\frac{\mathbf p^2}{p_f^2}}
= p_f+\frac{\mathbf p^2}{2p_f}-\frac{\mathbf p^4}{8p_f^3}+\cdots,
\quad E=cM=m_0 c^2+\frac{\mathbf p^2}{2m_0}+\cdots .
]
Subtracting the rest term and multiplying the even-in-(\mathbf p) part by (c) exposes the Newtonian kinetic energy (\tfrac12 m_0 v^2=\mathbf p^2/(2m_0)). In this limit, the **sum** (\sum(p_{k^+}+p_{k^-})) captures total kinetic energy (up to the common factor), while the **difference** (\sum(p_{k^+}-p_{k^-})) is the directional momentum—exactly the classical invariants in elastic processes.  

## 6) Computational reading (why engines & sims care)

* **Scalar rows.** (p_{k^\pm}) turn “energy + momentum projection” into two **scalars** per direction (k). In a constraint or contact along normal (\hat n), you can equivalently solve a **single scalar** (the normal row), with transverse components passing through—this is precisely the structure modern sequential-impulse solvers exploit.
* **Cheap invariants.** Precompute incoming (\sum p_{k^\pm}) once; iterate candidates and check add/sub invariants without juggling full vectors.
* **NR usage.** In Newtonian engines, the “even” piece is the kinetic term while the “odd” piece is the momentum projection—handy for fast acceptance/rejection, warm-starting, and SoA batching.

(These are engineering corollaries; the core physics is unchanged.) 

## 7) What ADMC is **not**

* **Not** a new dynamics or modification of SR; it’s a bookkeeping re-expression within any inertial frame, equivalent to four-momentum conservation. 
* (p_{k^\pm}) are not Lorentz invariants; they are frame scalars under spatial rotations only. 

## 8) Quick checklist (implementation/tests)

* Verify the linear identities in §2 for random test systems; they must hold to machine precision. 
* For elastic toy problems (2-body head-on), confirm constancy of each (\sum p_{k^\pm}) over the step; differences equal directional momentum; sums equal (2\sum M). 
* In NR scenes, track the even piece vs (\sum \tfrac12 m v^2) and the odd piece vs (\sum p_k). 

## 9) Prior art and related formulations

* **Sequential impulses / projected Gauss–Seidel.** Erin Catto’s GDC 2006 talk and subsequent Box2D notes formalize scalar contact rows with warm-start, ERP, and restitution. This is the practical lineage that ADMC-compatible solvers follow. [[box2d.org](https://box2d.org/files/ErinCatto_GDC2006_Tutorial.pdf)]
* **Warm-started PGS convergence studies.** Empirical analyses (e.g., the warm-starting report on arXiv:1305.3903) quantify the convergence improvements from cached impulses—mirroring the scalar-cached solver in this repo. [[arXiv:1305.3903](https://arxiv.org/abs/1305.3903)]
* **Complementarity-based time stepping.** Stewart–Trinkle and Anitescu–Potra formulations treat contacts as LCP/DVI problems. They provide a contrast when situating directional scalar rows within broader nonsmooth dynamics literature. [[RPI Foswiki](https://foswiki.cs.rpi.edu/foswiki/bin/view/Cad/TextStewartTrinkleNotes)]

---

### References to the internal note

All statements above are summarized from the accompanying technical note *“Equivalence of Absolute Directional Momentum Conservation with Special Relativistic Kinematics”*, especially Sections 2–3 (definitions, equivalence), uniqueness (affine form), and the NR expansion.     
