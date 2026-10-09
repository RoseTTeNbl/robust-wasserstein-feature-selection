# Certified Feature Selection for Wasserstein Robust Hypothesis Testing

9 October 2026.

## Abstract

We study how to select a limited number of features while designing a hypothesis test that remains reliable under class-conditional Wasserstein uncertainty. The objective is the sum of the two worst-case classwise losses. For nonnegative, nondecreasing convex losses that are finite on the real line, we obtain an exact finite optimization problem by constructing a whole-space test from its feasible solutions. The construction uses only convexity of the loss and the triangle inequality of the transport distance. We then use the risk itself to limit the coordinatewise transport prices. Eliminating the continuous variables gives a family of lower bounds: each is a residual loss divided by a transport correction. Their maximum is a convex function that agrees with the original risk at every binary feature selection. Tangents to these bounds generate a global search over feature subsets, with feasible tests providing upper bounds. Exact continuous and discrete solves terminate after finitely many subsets. The resulting certificates concern optimization within the specified uncertainty sets; population error control requires their separate calibration.

## 1. The selection problem

We can observe at most $d$ of $D$ features. The choice of features and the test using them must be optimized together: a good subset is one on which a test can keep both classwise losses small when the distributions move within their specified uncertainty sets.

Pool the observations from the two classes into $x_1,\ldots,x_N\in\mathbb R^D$. Observation $i$ has class $c_i\in\{0,1\}$, sign $y_i=2c_i-1$, and weight $q_i>0$. Write $I_k=\{i:c_i=k\}$. The empirical distributions and uncertainty sets are

$$
\widehat P_k=\sum_{i\in I_k}q_i\delta_{x_i},\qquad
\sum_{i\in I_k}q_i=1,\qquad
\mathcal U_k=\{P:W_1(P,\widehat P_k)\le\theta_k\},\quad k=0,1.
$$

Here $\delta_x$ denotes unit mass at $x$. The distance $W_1$ is the minimum expected $L^1$ distance needed to transport one distribution to the other; the distributions have finite first moments. The radii $\theta_k>0$ and feature scales are fixed before selection. Each class has total weight one, so the pooled weights sum to two.

For an integer budget $0\le d\le D$, a binary vector $z$ selects $S(z)=\{j:z_j=1\}$, with $\mathcal Z_d=\{z\in\{0,1\}^D:\mathbf1^\top z\le d\}$. A measurable score $g$ depends only on the selected coordinates. A positive score favors class zero; the losses are therefore $\Phi(-g)$ for class zero and $\Phi(g)$ for class one. The problem is

$$
\begin{aligned}
V(z)&=\inf_{g=g(x_{S(z)})}
\left\{\sup_{P\in\mathcal U_0}\mathbb E_P\Phi(-g)
       +\sup_{P\in\mathcal U_1}\mathbb E_P\Phi(g)\right\},\\
\mathrm{OPT}&=\min_{z\in\mathcal Z_d}V(z).
\end{aligned}
\tag{1}
$$

Thus $V(z)$ already includes the best possible test for the chosen features; selection does not hold a previously fitted test fixed.

The function $\Phi:\mathbb R\to[0,\infty)$ is finite, convex and nondecreasing. Nondecreasing means that a larger score against the true class cannot reduce its loss. We consider nonconstant losses with $\Phi(0)>0$. If $\Phi(0)=0$ or the loss is constant, the constant test already solves the problem. Specific formulas for $\Phi$ are not needed for the construction below; examples are given in Section 6. The constant score $g=0$ provides the common benchmark

$$
0\le V(z)\le B:=2\Phi(0).
$$

## 2. A finite problem that still designs the whole test

The same pooled observations serve as candidate target locations. To compare source $i$ with target $\ell$, use one vector of coordinatewise distances:

$$
\mathbf d_{i\ell}=|x_i-x_\ell|\in\mathbb R_+^D,\qquad
d_z(x_i,x_\ell)=\mathbf d_{i\ell}^{\top}z.
$$

Absolute value is taken coordinatewise. The vector stores the geometry once; its inner product with $z$ is the distance using the selected features.

Let $u_\ell$ be the score assigned to target $\ell$. Let $a_i$ be a loss allowance at source $i$, and let $\lambda_k\ge0$ be the price per unit of transport for class $k$. Write $q=(q_i)$, $\theta=(\theta_0,\theta_1)^\top$, $\lambda=(\lambda_0,\lambda_1)^\top$, and $J(a,\lambda)=q^\top a+\theta^\top\lambda$.

**Theorem 1 (Exact finite formulation).**

For every binary $z$,

$$
\begin{aligned}
V(z)=\min_{a,u,\lambda}\quad &J(a,\lambda)\\
\text{subject to}\quad
&\Phi(y_i u_\ell)\le a_i+
  \lambda_{c_i}\mathbf d_{i\ell}^{\top}z,
  \qquad i,\ell=1,\ldots,N,\\
&\lambda\ge0,\qquad a,u\in\mathbb R^N.
\end{aligned}
\tag{2}
$$

The minimum is attained. Every feasible solution yields a measurable test on the entire observation space, using only the selected coordinates, with robust risk at most $J(a,\lambda)$.

The inequality in (2) has a direct meaning: after moving from source $i$ to target $\ell$, the loss is at most its source allowance plus the transport price. The class sign is $y_i$, not the original label of target $\ell$; either class can send mass to any target location.

The whole-space conclusion is essential. It allows a finite feasible objective to be used as an upper bound for the original test problem. Its proof blends two source scores, one from each class. Convexity controls the blended losses, and the triangle inequality controls the required transport allowances. The construction and both directions of the equality are given in Appendix A; no particular inverse or closed form of $\Phi$ is used.

## 3. Use the risk to control which coordinates are available

Taking $\ell=i$ in (2) gives $a_i\ge\Phi(y_i u_i)\ge0$. Consequently,

$$
\theta^\top\lambda\le J(a,\lambda).
\tag{3}
$$

The transport-price contribution is part of the total risk, so it cannot exceed that risk. This inequality supplies the bound needed to connect feature selection with continuous test design.

Introduce two vectors $w_0,w_1\in\mathbb R_+^D$ of coordinate prices. On a binary selection we can take $w_{kj}=\lambda_k z_j$: an unobserved coordinate has price zero. To obtain a useful continuous extension, use the risk bound $\eta$ to constrain these prices:

$$
\begin{aligned}
v(z)=\min_{\eta,a,u,\lambda,w_0,w_1}\quad &\eta\\
\text{subject to}\quad
&\Phi(y_i u_\ell)\le a_i+\mathbf d_{i\ell}^{\top}w_{c_i},
  \qquad \forall i,\ell,\\
&J(a,\lambda)\le\eta,\qquad \lambda\ge0,\\
&0\le w_k\le\lambda_k\mathbf1,\qquad k=0,1,\\
&\theta_0w_0+\theta_1w_1\le\eta z.
\end{aligned}
\tag{4}
$$

Vector inequalities hold coordinatewise; $a,u,\eta$ are real variables. The first line designs the test using the coordinate prices. The next lines bound its risk and keep each coordinate price below the class price. The last line makes an unselected coordinate unavailable and ties the allowed prices to the risk being minimized.

For binary $z$, an optimizer of (2) enters (4) with $\eta=J$ and $w_k=\lambda_k z$, by (3). Conversely, a feasible point of (4) has zero prices on unselected coordinates and $w_k\le\lambda_k z$. The same $a,u,\lambda$ therefore satisfy (2), with objective at most $\eta$. Hence $v(z)=V(z)$ at every binary selection. For fractional $z\in[0,1]^D$, (4) defines the extension used to construct lower bounds.

Fixing $z$ makes (4) a convex problem. This alone does not prove that its value $v(z)$ is convex in $z$, since the last line contains the product $\eta z$. The next step establishes that property and supplies information that remains useful when the feature subset changes.

## 4. One solved subset gives lower bounds for all subsets

Consider a nonnegative matrix $\Pi$ with $\Pi_{i\ell}$ units of mass sent from source $i$ to target $\ell$. Its rows retain the source weights, $\Pi\mathbf1=q$. The masses received from the two classes and their coordinatewise movements are

$$
p_{k\ell}=\sum_{i\in I_k}\Pi_{i\ell},\qquad
A_k(\Pi)=\sum_{i\in I_k,\ell}\Pi_{i\ell}\mathbf d_{i\ell}
\in\mathbb R_+^D.
$$

At a target receiving both classes, one score must serve both. The least weighted loss that this common score can achieve is

$$
\Psi(p_0,p_1)=\inf_s\{p_0\Phi(-s)+p_1\Phi(s)\},
\qquad H(\Pi)=\sum_\ell\Psi(p_{0\ell},p_{1\ell}).
\tag{5}
$$

The infimum permits limiting scores when one class has zero mass. $H$ measures the loss left after the best score is chosen at every target. It is nonnegative. Scaling all masses by $t\ge0$ scales $H$ by $t$; this property concerns the masses and does not require $\Phi$ itself to be homogeneous.

To use a matrix that may move more mass than the original budget allows, introduce a nonnegative correction vector $\mu\in\mathbb R_+^D$. The pair $(\Pi,\mu)$ must satisfy

$$
\Pi\ge0,\qquad \Pi\mathbf1=q,\qquad \mu\ge0,\qquad
\mathbf1^\top\bigl(A_k(\Pi)-\theta_k\mu\bigr)_+\le\theta_k,
\quad k=0,1.
\tag{6}
$$

Here $t_+=\max(t,0)$, applied coordinatewise to a vector. After $\theta_k\mu_j$ is deducted from movement in coordinate $j$, the remaining positive movements must fit inside the class budget $\theta_k$. These conditions contain no $z$: the same pair can be used at any selection. The matrix need not be a transport plan inside the original uncertainty balls. The correction is what makes it useful as a risk lower bound.

The reason for the correction in the denominator can be seen directly. For $0\le z\le1$, (6) implies $z^\top A_k(\Pi)\le\theta_k(1+\mu^\top z)$. For any feasible point of (2) on a binary selection,

$$
\begin{aligned}
H(\Pi)
&\le\sum_{i,\ell}\Pi_{i\ell}\Phi(y_i u_\ell)\\
&\le q^\top a+\sum_k\lambda_k z^\top A_k(\Pi)\\
&\le J(a,\lambda)+(\mu^\top z)\theta^\top\lambda\\
&\le(1+\mu^\top z)J(a,\lambda).
\end{aligned}
$$

The four steps use, respectively, best-score loss, finite loss constraints, the corrected transport budgets, and (3). Thus $H(\Pi)/(1+\mu^\top z)$ is a lower bound on $V(z)$.

**Theorem 2 (Exact representation and convexity).**

For every $z\in[0,1]^D$,

$$
v(z)=\max_{\Pi,\mu\ \text{satisfying }(6)}
\frac{H(\Pi)}{1+\mu^\top z}.
\tag{7}
$$

The maximum is attained, $v$ is convex, and $v(z)=V(z)$ on binary selections.

The preceding inequalities explain why each fraction is a valid lower bound. Equality with $v$ requires more: minimizing the Lagrangian of (4) gives the row masses, transport inequalities and one normalization equation. Rescaling those multipliers gives exactly (7). Appendix C derives the objective and every constraint by eliminating one group of variables at a time. For fixed $(\Pi,\mu)$, the fraction is convex because its numerator is nonnegative and its denominator is a positive affine function. The maximum of these convex lower functions is therefore convex.

## 5. From lower bounds to a global algorithm

At a selection $z^t$, solving (4) and its dual gives a pair $(\Pi^t,\mu^t)$. Write $h_t=H(\Pi^t)$ for its remaining target loss and $r_t=1+(\mu^t)^\top z^t$ for the correction at the current selection. The function $f_t(z)=h_t/[1+(\mu^t)^\top z]$ is a lower bound everywhere. Its gradient at $z^t$ is $-h_t\mu^t/r_t^2$, so its tangent is

$$
L_t(z)=\frac{h_t}{r_t}
-\frac{h_t}{r_t^2}(\mu^t)^\top(z-z^t)\le v(z).
\tag{8}
$$

This affine inequality is the global cut. Convexity puts the tangent below $f_t$, and (7) puts $f_t$ below $v$. An optimal pair makes $L_t(z^t)=v(z^t)$. A merely feasible pair also gives a valid cut, using its own $h_t$. This argument does not assume that $v$ is differentiable.

Given the accumulated cuts, solve the discrete master problem

$$
\min_{z\in\mathcal Z_d,\eta}\ \eta
\qquad\text{subject to}\qquad
\eta\ge0,\quad \eta\ge L_t(z)\ \text{for every stored cut}.
\tag{9}
$$

It searches the entire set of allowed feature subsets. Each new cut changes the lower model for all subsets, while a continuous solve refits the entire test at the next candidate. The search is not confined to a neighborhood of the current subset or to a fixed combination of earlier tests.

**Algorithm 1: Global feature selection**

**Input:** Initial $z^0\in\mathcal Z_d$; tolerance $\varepsilon\ge0$.

1. Set $t=0$, $\mathrm{LB}=0$, $\mathrm{UB}=B$; retain $z^0$ and $g=0$ as the best feasible test.
2. **While** $\mathrm{UB}-\mathrm{LB}>\varepsilon$:
   1. At the current binary selection $z^t$, solve the full problem (4).
   2. Construct an original feasible test. If its risk bound is smaller than $\mathrm{UB}$, update $\mathrm{UB}$ and retain this selection and test.
   3. Obtain a feasible dual pair and add the cut (8) using its own numerator $h_t$.
   4. Solve (9); update $\mathrm{LB}$ with its globally valid solver bound, take its binary candidate as $z^{t+1}$ and set $t\gets t+1$.
   5. If the time or accuracy budget is exhausted, return the best test and the achieved bounds.
3. **Return:** Best feature subset and test, with interval $[\mathrm{LB},\mathrm{UB}]$.

**Theorem 3 (Bounds and exact finite termination).**

Every collection of valid cuts and every feasible original integer test give

$$
\mathrm{LB}\le\mathrm{OPT}\le\mathrm{UB}.
\tag{10}
$$

With exact continuous solutions, optimal dual pairs and globally solved masters, the algorithm terminates after finitely many different subsets. Termination with $\mathrm{UB}-\mathrm{LB}\le\varepsilon$ guarantees $\mathrm{UB}-\mathrm{OPT}\le\varepsilon$.

To see why, a previously solved subset has both a touching cut and an equally good feasible test. If an exact master selects it again, the global gap closes. Otherwise a new subset is evaluated, and there are only finitely many subsets. Appendix D proves the bound and termination statements in detail. Valid but nonoptimal certificates retain the bound guarantee, not the exact finite-termination argument. An interrupted master's global solver bound is usable; its current feasible objective is not a lower bound.

Adding available coordinates cannot worsen $V$, because a test can ignore them. The search may therefore use $\mathbf1^\top z=d$ when features have no additional charge. Cuts from fractional selections may strengthen the master, but only binary feasible tests update the original upper bound.

## 6. Loss choices and the scope of the result

The loss enters through $\Phi$ in the continuous problem and through $\Psi$ in the dual objective. Examples covered by the same derivation are shown below; boundary values use continuous limits.

| Loss | $\Phi(s)$ | $\Psi(p_0,p_1)$ |
| --- | --- | --- |
| Half hinge | $\tfrac12(1+s)_+$ | $\min(p_0,p_1)$ |
| Logistic | $\log(1+e^s)$ | $(p_0+p_1)\log(p_0+p_1)-p_0\log p_0-p_1\log p_1$ |
| Exponential | $e^s$ | $2\sqrt{p_0p_1}$ |
| Squared hinge | $(1+s)_+^2$ | $4p_0p_1/(p_0+p_1)$ |

For example, half-hinge masses $p_0=0.2$ and $p_1=0.5$ leave loss $0.2$: the best common score favors the larger mass but cannot distinguish the smaller mass at the same location. Appendix F.1 derives all four formulas. For half hinge, a score clipped to $[-1,1]$ gives randomized class-zero probability $(1+g)/2$, so $V/2$ is the equal-prior worst-case error. Other losses define surrogate objectives. The current implementation uses half hinge and logistic; the other rows illustrate the theoretical scope.

The loss formulas need not be fixed before deriving the algorithm. The transport geometry still matters: the whole-space construction uses the triangle inequality, and the coordinate-price formulation uses the $L^1$ decomposition. Appendix E also compares fixed price allowances with the risk-dependent allowance in (4). Stronger continuous lower bounds do not guarantee faster finite master solves. The certified interval concerns the stated robust optimization problem; whether its uncertainty sets cover an unknown population is a separate statistical question.

## Appendix A. Projection and the exact finite formulation

We use one pool $x_1,\ldots,x_N$ for both sources and targets. Source $i$ has class $c_i\in\{0,1\}$, sign $y_i=2c_i-1$, and weight $q_i>0$, with $\sum_{i\in I_k}q_i=1$ and $I_k=\{i:c_i=k\}$. Set $\mathbf d_{i\ell}=|x_i-x_\ell|$ componentwise and $d_z(x_i,x_\ell)=\mathbf d_{i\ell}^{\top}z$. The loss is finite, convex, nonnegative, and nondecreasing, with $\Phi(0)>0$, and is nonconstant. Both radii are positive. The argument below establishes the finite representation in (2) for this whole loss class, without interchanging an infinite-dimensional infimum and supremum.

### A.1 Selected-coordinate uncertainty

For a binary $z$, let $S=\{j:z_j=1\}$ and let $\pi_S$ be coordinate projection. Projecting any full-space coupling gives

$$
W_1((\pi_S)_\#P,(\pi_S)_\#\widehat P_k)
\le W_1(P,\widehat P_k).
\tag{A.1}
$$

Conversely, take a coupling from the projected empirical distribution to a candidate projected distribution. If several sources have the same projection, split their common marginal according to their original weights and retain their source identities. For mass originating at $x_i$, keep $x_{i,S^c}$ and move only the selected coordinates. This defines a full-space distribution and coupling with exactly the projected cost. Finite first moments are preserved. Hence the distributions seen by a selected-coordinate test are exactly the projected Wasserstein ball of the same radius. This lifting uses the full ambient space; for $S=\varnothing$, the projected space consists of one point and the same conclusion holds.

### A.2 A finite adversary gives one value inequality

For finite scores $u_1,\ldots,u_N$, restrict class $k$'s adversary to targets in the projected pool. The source identities are retained, and its largest loss is

$$
\begin{aligned}
Q_k(u)=\max_{\Pi_{i\ell}\ge0}\quad&
\sum_{i\in I_k,\ell}\Pi_{i\ell}\Phi(y_i u_\ell)\\
\text{subject to}\quad&\sum_\ell\Pi_{i\ell}=q_i\quad(i\in I_k),\\
&\sum_{i\in I_k,\ell}\Pi_{i\ell}\mathbf d_{i\ell}^{\top}z\le\theta_k.
\end{aligned}
\tag{A.2}
$$

Staying at the source is feasible. The row constraints bound every mass variable, so the optimum is finite and attained. Linear programming duality gives

$$
Q_k(u)=\min_{a_i\in\mathbb R,\lambda_k\ge0}
\left\{\sum_{i\in I_k}q_i a_i+\theta_k\lambda_k:
\Phi(y_i u_\ell)\le a_i+\lambda_k\mathbf d_{i\ell}^{\top}z\right\}.
\tag{A.3}
$$

The intercepts are unrestricted multipliers of mass conservation, whereas $\lambda_k$ is a nonnegative budget multiplier.

Let $T(z)$ denote the infimum of the finite program (2). Joint minimization of (A.3) over the scores and both classes gives $T(z)=\inf_u[Q_0(u)+Q_1(u)]$. Every original measurable test induces pool scores. For mass $\Pi_{i\ell}$, use the full-space target $(x_{\ell,S},x_{i,S^c})$, retaining the unselected source coordinates. Its distance from $x_i$ is exactly $d_z(x_i,x_\ell)$, and $g(x_{\ell,S},x_{i,S^c})=g(x_\ell)$. Thus every feasible projected transport has an admissible full-space lift with the same test loss. Therefore

$$
T(z)\le V(z).
\tag{A.4}
$$

Different targets may coincide after projection. Allowing them separate finite score variables can only further relax this inequality. The following construction proves the converse without requiring those scores to interpolate a single function.

### A.3 Constructing a bounded test on the whole space

Take any feasible $(a,u,\lambda)$ in (2). Define

$$
E_k(x)=\min_{i\in I_k}\{a_i+\lambda_k d_z(x_i,x)\},
\qquad d_z(x_i,x)=\sum_jz_j|x_{i,j}-x_j|.
\tag{A.5}
$$

For each $x$, choose $i_k(x)$ as the smallest index attaining this minimum. Write $i_0=i_0(x)$, $i_1=i_1(x)$, and

$$
c=d_z(x_{i_0},x_{i_1}),\qquad
r_0=d_z(x_{i_0},x),\qquad r_1=d_z(x_{i_1},x).
\tag{A.6}
$$

The triangle inequality gives $c\le r_0+r_1$. If $c>0$, set

$$
\alpha=\frac{\min(c,r_0)}{c},\qquad
g(x)=(1-\alpha)u_{i_0}+\alpha u_{i_1}.
\tag{A.7}
$$

Then $0\le\alpha\le1$, $\alpha c\le r_0$, and $(1-\alpha)c\le r_1$. The finite constraints at the two endpoint targets give

$$
\begin{aligned}
\Phi(-u_{i_0})&\le a_{i_0},&
\Phi(-u_{i_1})&\le a_{i_0}+\lambda_0c,\\
\Phi(u_{i_0})&\le a_{i_1}+\lambda_1c,&
\Phi(u_{i_1})&\le a_{i_1}.
\end{aligned}
\tag{A.8}
$$

Convexity of $\Phi$ now yields both required loss bounds:

$$
\begin{aligned}
\Phi(-g(x))
&\le(1-\alpha)\Phi(-u_{i_0})+\alpha\Phi(-u_{i_1})\\
&\le a_{i_0}+\lambda_0\alpha c
\le a_{i_0}+\lambda_0r_0=E_0(x),\\
\Phi(g(x))
&\le(1-\alpha)\Phi(u_{i_0})+\alpha\Phi(u_{i_1})\\
&\le a_{i_1}+\lambda_1(1-\alpha)c
\le a_{i_1}+\lambda_1r_1=E_1(x).
\end{aligned}
\tag{A.9}
$$

If $c=0$, take $g(x)=u_{i_0}$. The own-source constraint gives $\Phi(-g(x))\le a_{i_0}\le E_0(x)$; the class-1 constraint at target $i_0$ gives $\Phi(g(x))\le a_{i_1}\le E_1(x)$. Thus no division by zero is needed.

Each envelope is a finite minimum of continuous selected-coordinate functions. The smallest-index minimizers are Borel measurable because their selection regions are defined by finitely many comparisons of such functions. On each selected source pair, (A.7) is a Borel function, with a separate Borel branch for $c=0$. It depends only on the selected coordinates and satisfies

$$
\min_\ell u_\ell\le g(x)\le\max_\ell u_\ell\qquad\text{for every }x.
\tag{A.10}
$$

The construction extends the feasible loss bounds; it need not satisfy $g(x_\ell)=u_\ell$.

For a coupling from class $k$'s empirical sources to an admissible $P$, integrate the sourcewise envelope inequality:

$$
\mathbb E_P\Phi((2k-1)g)
\le\sum_{i\in I_k}q_i a_i+\lambda_k\int d_z(x_i,x)\,d\gamma_k
\le\sum_{i\in I_k}q_i a_i+\theta_k\lambda_k.
\tag{A.11}
$$

An optimal full-space coupling exists for finite-first-moment measures; near-optimal couplings and a limit give the same inequality. The projected cost is no larger than the full $L^1$ cost. Taking the supremum, summing over classes, and minimizing over finite feasible solutions gives $V(z)\le T(z)$. Together with (A.4), this proves the exact finite representation and its feasible-test upper-bound guarantee.

## Appendix B. Attainment and finite convex strong duality

### B.1 Growth of the loss and compact sublevels

A finite convex function on $\mathbb R$ is continuous. Nonconstancy and monotonicity give $s_1<s_2$ with $\Phi(s_2)>\Phi(s_1)$. If $m_1=[\Phi(s_2)-\Phi(s_1)]/(s_2-s_1)>0$, convexity gives

$$
\Phi(t)\ge\Phi(s_2)+m_1(t-s_2),\qquad t\ge s_2.
\tag{B.1}
$$

Thus $\Phi(t)\to\infty$ as $t\to\infty$; this growth is a consequence of the assumptions, rather than an additional condition.

The constant finite test $u=0$, $a_i=\Phi(0)$, and $\lambda=0$ has cost $B=2\Phi(0)$. It is feasible in the risk-scaled model with $w=0$ and $\eta=B$, and in the fixed price bounds defined in Appendix E. Own-source constraints imply $a_i\ge\Phi(y_i u_i)\ge0$. On an objective sublevel at most $B$,

$$
0\le a_i\le B/q_i,\qquad
0\le\lambda_k\le B/\theta_k,\qquad
0\le w_{kj}\le B/\theta_k.
\tag{B.2}
$$

For the risk-scaled model, $0\le J\le\eta\le B$. At every target $\ell$, a source from each class bounds $\Phi(-u_\ell)$ and $\Phi(u_\ell)$ above by finite uniform constants, using (B.2) and the finite pool distances. Equation (B.1) then bounds both tails of $u_\ell$. All variables on these sublevels are bounded, and continuity makes the sublevels closed. They are nonempty, so each finite primal minimum is attained.

### B.2 Strict feasibility in the appropriate linear face

For the ordinary finite program, take finite scores, positive $\lambda$, and sufficiently large intercepts to make every loss inequality strict. For a program with coordinate-price constraints at fixed $z$, coordinates with $z_j=0$ force $w_{0j}=w_{1j}=0$. Remove those fixed variables, or work in the corresponding face of the linear domain. Choose small positive $w_{kj}$ at every remaining coordinate and positive $\lambda_k$ strictly larger than all its entries. Then choose the intercepts large enough for strict loss inequalities. The separate coordinate bounds or shared-budget inequalities can be made strict by making these active prices sufficiently small. For the risk-dependent price constraint, choose $\eta$ sufficiently large that

$$
J<\eta,\qquad \theta_0w_{0j}+\theta_1w_{1j}<\eta z_j
\quad\text{whenever }z_j>0.
\tag{B.3}
$$

This is a strict feasible point in the relative interior of the remaining linear domain. The finite-dimensional convex strong-duality criterion therefore gives zero gap and attained multiplier optima.

The removed zero-coordinate rows do not obstruct the unreduced dual written below. Their price variables have no stationarity conditions in the reduced program. Restore their nonnegative coordinate-constraint multipliers sufficiently large to satisfy the corresponding coverage inequalities. Their penalty and normalization coefficients contain $z_j=0$, so this restoration does not change the dual objective or normalization. Ordinary strict feasibility for all unreduced rows is neither available nor necessary.

### B.3 The minimum loss at one target

For $p_0,p_1\ge0$, define $\Psi(p_0,p_1)$ as in (5), and put $m_0=\inf_t\Phi(t)\ge0$. Then

$$
m_0(p_0+p_1)\le\Psi(p_0,p_1)\le\Phi(0)(p_0+p_1).
\tag{B.4}
$$

The upper bound uses the score zero. Being an infimum of continuous linear functions of $(p_0,p_1)$, $\Psi$ is concave and upper semicontinuous. It is finite and therefore continuous when $p_0,p_1>0$. At a boundary point $(0,\bar p_1)$, choose a finite $t_\varepsilon$ with $\Phi(t_\varepsilon)\le m_0+\varepsilon$. For any approaching $(p_0,p_1)$,

$$
\Psi(p_0,p_1)\le p_0\Phi(-t_\varepsilon)+p_1\Phi(t_\varepsilon).
\tag{B.5}
$$

Taking a limit superior and then $\varepsilon\downarrow0$ gives at most $\bar p_1m_0$; the lower bound in (B.4) gives the matching limit inferior. The other axis follows by interchanging $p_0,p_1$, and continuity at the origin follows from the upper bound. Hence $\Psi$ is continuous on the nonnegative quadrant. When $p_0,p_1>0$, the score objective grows at both tails by (B.1), so its minimum is attained. On an axis it may have only an infimum; none of the dual arguments require a finite minimizing score there.

Scaling the masses, rather than the loss function, gives

$$
\Psi(\rho p_0,\rho p_1)=\rho\Psi(p_0,p_1),\qquad H(\rho\Pi)=\rho H(\Pi),\qquad \rho\ge0.
\tag{B.6}
$$

For a matrix whose row sums are $q_i$, the total mass of each class is one, so $0\le H(\Pi)\le B$. These facts apply to every loss in the stated class; no homogeneity assumption on $\Phi$ is used.

## Appendix C. Duals obtained by eliminating the primal variables

For any nonnegative matrix, write

$$
p_{k\ell}(\Pi)=\sum_{i\in I_k}\Pi_{i\ell},\qquad
A_k(\Pi)=\sum_{i\in I_k,\ell}\Pi_{i\ell}\mathbf d_{i\ell},\qquad
H(\Pi)=\sum_\ell\Psi(p_{0\ell},p_{1\ell}).
\tag{C.1}
$$

All vector inequalities below are componentwise.

### C.1 The dual of the finite problem

Assign $\Pi_{i\ell}\ge0$ to the loss inequalities in (2). Its Lagrangian is

$$
\begin{aligned}
\mathcal L={}&J+\sum_{i,\ell}\Pi_{i\ell}
[\Phi(y_i u_\ell)-a_i-\lambda_{c_i}\mathbf d_{i\ell}^{\top}z]\\
={}&\sum_i a_i\left(q_i-\sum_\ell\Pi_{i\ell}\right)
+\sum_\ell[p_{0\ell}\Phi(-u_\ell)+p_{1\ell}\Phi(u_\ell)]\\
&+\sum_k\lambda_k[\theta_k-A_k(\Pi)^{\top}z].
\end{aligned}
\tag{C.2}
$$

There are two elementary minimization rules behind the constraints:

$$
\inf_{t\in\mathbb R}bt=
\begin{cases}0,&b=0,\\-\infty,&b\ne0,\end{cases}
\qquad
\inf_{t\ge0}bt=
\begin{cases}0,&b\ge0,\\-\infty,&b<0.\end{cases}
\tag{C.3}
$$

Here $b$ is a coefficient, not an additional model variable. Apply the first rule to each free intercept: if $q_i-\sum_\ell\Pi_{i\ell}$ is nonzero, sending $a_i$ to the appropriate infinity makes the infimum $-\infty$. Finite dual values therefore require $\Pi\mathbf1=q$.

The score variables separate by target after the loss constraints have entered the Lagrangian:

$$
\begin{aligned}
&\inf_{u\in\mathbb R^N}\sum_\ell
[p_{0\ell}\Phi(-u_\ell)+p_{1\ell}\Phi(u_\ell)]\\
&\qquad=\sum_\ell\inf_{s\in\mathbb R}
[p_{0\ell}\Phi(-s)+p_{1\ell}\Phi(s)]
=H(\Pi).
\end{aligned}
\tag{C.4}
$$

For completeness, the left side is at least the sum of the separate infima. For any $\varepsilon>0$, choose one finite score at each of the $N$ targets within $\varepsilon/N$ of its infimum; this produces the reverse inequality after $\varepsilon\downarrow0$. Thus the equality remains valid even if an individual infimum is not attained.

Finally, apply the second rule in (C.3) to each nonnegative $\lambda_k$. Its coefficient must be nonnegative, leaving no extra objective term. The only surviving part of the minimized Lagrangian is $H(\Pi)$, subject to row conservation and the two transport budgets. Thus

$$
V(z)=\max_{\Pi\ge0}
\{H(\Pi):\Pi\mathbf1=q,\ A_k(\Pi)^{\top}z\le\theta_k\ (k=0,1)\},
\qquad z\text{ binary}.
\tag{C.5}
$$

These matrices describe admissible finite adversarial transports in the projected space, with full-space lifts from Appendix A. Strong duality and attainment follow from Appendix B.

### C.2 The risk-scaled program

Let $\tau\ge0$ multiply $J-\eta\le0$. To distinguish scaled masses and prices, use $\bar\Pi$, $\bar\beta_k$, and $\bar\mu$ for the remaining multipliers. Write $\bar p_{k\ell}=p_{k\ell}(\bar\Pi)$. The full Lagrangian is

$$
\begin{aligned}
\mathcal L_v={}&\eta+\tau(J-\eta)
+\sum_{i,\ell}\bar\Pi_{i\ell}
[\Phi(y_i u_\ell)-a_i-\mathbf d_{i\ell}^{\top}w_{c_i}]\\
&+\sum_k\bar\beta_k^{\top}(w_k-\lambda_k\mathbf1)
+\bar\mu^{\top}(\theta_0w_0+\theta_1w_1-\eta z)\\
={}&\eta(1-\tau-\bar\mu^{\top}z)
+\sum_i a_i(\tau q_i-\sum_\ell\bar\Pi_{i\ell})\\
&+\sum_\ell[\bar p_{0\ell}\Phi(-u_\ell)+\bar p_{1\ell}\Phi(u_\ell)]\\
&+\sum_k\lambda_k(\tau\theta_k-\mathbf1^{\top}\bar\beta_k)
+\sum_k w_k^{\top}[\bar\beta_k+\theta_k\bar\mu-A_k(\bar\Pi)].
\end{aligned}
\tag{C.6}
$$

The variable $\eta$ remains free: its coefficient must vanish, giving $\tau+\bar\mu^{\top}z=1$. The free intercepts impose $\bar\Pi\mathbf1=\tau q$, and the free scores contribute $H(\bar\Pi)$. The infima over nonnegative $\lambda_k$ and $w_k$ require respectively $\mathbf1^{\top}\bar\beta_k\le\tau\theta_k$ and $A_k(\bar\Pi)\le\bar\beta_k+\theta_k\bar\mu$. Hence the attained dual is

$$
\begin{aligned}
v(z)=\max_{\tau,\bar\Pi,\bar\beta_0,\bar\beta_1,\bar\mu\ge0}\quad&H(\bar\Pi)\\
\text{subject to}\quad&\bar\Pi\mathbf1=\tau q,\qquad
\mathbf1^{\top}\bar\beta_k\le\tau\theta_k,\\
&A_k(\bar\Pi)\le\bar\beta_k+\theta_k\bar\mu\quad(k=0,1),\\
&\tau+\bar\mu^{\top}z=1.
\end{aligned}
\tag{C.7}
$$

Although primal feasibility implies $a\ge0$ and $\eta\ge0$, neither is added as a variable-domain constraint in this derivation. This is why their coefficients give equalities rather than inequalities.

### C.3 Eliminating the allocation multipliers

For fixed $\Pi,\mu\ge0$, the inequalities $\beta_k\ge0$ and $A_k(\Pi)\le\beta_k+\theta_k\mu$ are equivalent to

$$
\beta_k\ge(A_k(\Pi)-\theta_k\mu)_+.
\tag{C.8}
$$

Every feasible allocation therefore has sum at least that of the positive part. Conversely, if that positive part has sum at most $\theta_k$, taking it as $\beta_k$ satisfies every allocation and coverage constraint. Thus existence of $\beta_k$ is equivalent to

$$
\sum_j(A_{kj}(\Pi)-\theta_k\mu_j)_+\le\theta_k,
\qquad k=0,1.
\tag{C.9}
$$

Together with $\Pi\mathbf1=q$, these are exactly the two-variable certificate conditions in (6). The normalized form similarly replaces $\theta_k$ on the right by $\tau\theta_k$ and uses $\bar\Pi,\bar\mu$ on the left:

$$
\begin{aligned}
v(z)=\max_{\tau,\bar\Pi,\bar\mu\ge0}\quad&H(\bar\Pi)\\
\text{subject to}\quad&\bar\Pi\mathbf1=\tau q,\quad
\sum_j(A_{kj}(\bar\Pi)-\theta_k\bar\mu_j)_+\le\tau\theta_k,\\
&\tau+\bar\mu^{\top}z=1.
\end{aligned}
\tag{C.10}
$$

No additional assumption is involved in removing $\beta$; it is an exact projection of the multiplier constraints.

## Appendix D. Binary exactness, normalization, cuts, and termination

### D.1 Binary value equality

For any original finite feasible point, own-source constraints give $\theta^{\top}\lambda\le J$. At a binary $z$, take $w_k=\lambda_k z$ and $\eta=J$. Then $\theta_0w_0+\theta_1w_1=z\theta^{\top}\lambda\le zJ$, so this point is feasible in (4). Conversely, any risk-scaled feasible point has both prices zero where $z_j=0$ and has $w_{kj}\le\lambda_k$ where $z_j=1$. Thus $w_k\le\lambda_kz$, and the nonnegative vectors $\mathbf d_{i\ell}$ imply

$$
\Phi(y_i u_\ell)\le a_i+\mathbf d_{i\ell}^{\top}w_{c_i}
\le a_i+\lambda_{c_i}\mathbf d_{i\ell}^{\top}z.
\tag{D.1}
$$

The same $(a,u,\lambda)$ is original feasible with $J\le\eta$. The two directions prove $v(z)=V(z)$ on binary selections.

One may also eliminate $\eta$ and minimize $J$ with price allowance $Jz$. Given a feasible point with $J<\eta$, add $(\eta-J)/2$ to every intercept. Because the weights sum to one in each class, the new $J$ is exactly $\eta$. The loss bounds relax and the price allowance is unchanged. Hence this elimination preserves the fixed-$z$ value without assuming that every epigraph optimizer already has $J=\eta$.

### D.2 Removing the normalization

When $\tau>0$, let $\Pi=\bar\Pi/\tau$ and $\mu=\bar\mu/\tau$. Linearity of $A_k$ and positive homogeneity of the positive part transform the normalized constraints into (6). Normalization and mass homogeneity give

$$
\tau=\frac1{1+\mu^{\top}z},\qquad
H(\bar\Pi)=\frac{H(\Pi)}{1+\mu^{\top}z}.
\tag{D.2}
$$

Conversely, any certificate in (6) can be multiplied by this positive $\tau$ to satisfy (C.10). If $\tau=0$, row sums and nonnegativity force $\bar\Pi=0$, so the objective is zero. Such points cannot improve the nonnegative maximum. This proves (7), and completes the dual representation in Theorem 2.

The outer maximum is attained. To see this directly, the row-constrained matrix set is compact. For any feasible $\Pi,\mu$, choose $\beta_k=(A_k-\theta_k\mu)_+$ and replace $\mu_j$ by

$$
\mu'_j=\max\left(0,
\frac{A_{0j}(\Pi)-\beta_{0j}}{\theta_0},
\frac{A_{1j}(\Pi)-\beta_{1j}}{\theta_1}\right).
\tag{D.3}
$$

Then $0\le\mu'\le\mu$, all expanded coverage constraints remain feasible, and decreasing $\mu$ cannot decrease the fraction for $z\ge0$. The matrices and allocations are bounded, and (D.3) gives a uniform bound on the needed prices. Continuity of $H$ from Appendix B then yields a maximizing certificate on a compact set.

In the nontrivial case the value is positive. Assign every source row to one target; its received class masses are both one, and convexity gives $\Psi(1,1)=2\Phi(0)=B$. Finite residual prices make this a feasible certificate with positive fraction. Consequently an optimal normalized certificate has $\tau>0$. The zero-scale discussion nevertheless explains the entire dual feasible set.

### D.3 Convexity and the affine cuts

Fix a certificate and write $h=H(\Pi)\ge0$. Its fraction has derivatives

$$
\nabla\frac{h}{1+\mu^{\top}z}
=-\frac{h\mu}{(1+\mu^{\top}z)^2},\qquad
\nabla^2\frac{h}{1+\mu^{\top}z}
=\frac{2h\mu\mu^{\top}}{(1+\mu^{\top}z)^3}\succeq0.
\tag{D.4}
$$

Thus each fixed-certificate fraction is convex, and their maximum over the fixed conditions (6) is convex. This proves convexity of $v$ even though the product $\eta z$ in its primal formulation is not jointly convex.

At $z^t$, put $r_t=1+\mu^{\top}z^t$. Convexity gives

$$
v(z)\ge\frac{h}{1+\mu^{\top}z}
\ge \frac{h}{r_t}-\frac{h\mu^{\top}(z-z^t)}{r_t^2}=L_t(z).
\tag{D.5}
$$

This is (8). If the certificate is optimal at its anchor, $h/r_t=v(z^t)$ and the cut touches the value. A feasible but nonoptimal certificate still gives a valid cut using its own $h$, but need not touch. Neither $v$ nor $\Psi$ needs to be differentiable for this argument. With normalized prices $\bar\mu=\mu/r_t$ and value $v_t=h/r_t$, the same gradient is $-v_t\bar\mu$; this is the equivalent normalized cut formula.

In numerical calculations, first ensure the row masses and movement inequalities are feasible, or repair them conservatively and recompute the loss. A valid lower fraction may use any $0\le h\le H(\Pi)$, with the same $h$ in both its value and gradient. If $\Psi$ is computed by minimizing over a trial score, that score gives an upper bound on its infimum, so it cannot be used as a lower certificate without a verified error bound. Use the closed forms below or a certified lower evaluation of the infimum. Lowering the whole affine cut by a nonnegative constant also preserves validity, but is different from reducing the numerator and recomputing its tangent. These conservative changes retain the bound guarantee; they need not retain touching or exact finite termination.

### D.4 A direct inequality for the original risk

Let $(a,u,\lambda)$ be any original feasible solution at a binary $z$, and take any certificate satisfying (6). Put $s=\mu^{\top}z$. Componentwise, $A_k\le(A_k-\theta_k\mu)_++\theta_k\mu$, and $0\le z\le\mathbf1$ implies

$$
z^{\top}A_k
\le\sum_j(A_{kj}-\theta_k\mu_j)_++\theta_k\mu^{\top}z
\le\theta_k(1+s).
\tag{D.6}
$$

Consequently,

$$
\begin{aligned}
H(\Pi)&\le\sum_{i,\ell}\Pi_{i\ell}\Phi(y_i u_\ell)\\
&\le\sum_iq_i a_i+\sum_k\lambda_k z^{\top}A_k\\
&\le J+s\theta^{\top}\lambda\le J(1+s).
\end{aligned}
\tag{D.7}
$$

The first step minimizes each target score; the second uses finite feasibility and row conservation; the last uses nonnegative intercepts. Minimizing over original feasible solutions proves the lower bound on $V$ directly. Expanded certificate masses are not required to be admissible original transports.

### D.5 Global bounds and exact finite termination

At binary selections, every cut in (D.5) is below $v=V$. An original optimal selection with epigraph value $V(z)$ therefore remains feasible in (9), so its global master bound is a lower bound on $\mathrm{OPT}$. Appendix A makes every finite feasible integer-selection test an original robust-risk upper bound. This proves (10).

Suppose exact oracle solutions and a globally solved master select an already evaluated subset $z^s$. Its own touching cut is $V(z^s)$, and no other cut exceeds that value there. The master optimum is therefore $V(z^s)$, while the stored feasible test gives an upper bound no larger than $V(z^s)$. Hence the lower bound, upper bound, and $\mathrm{OPT}$ coincide. Otherwise the next subset is new. There are finitely many binary subsets, proving the finite-termination assertion in Theorem 3. Merely feasible non-tight cuts, numerical revisits, or a master that has not been solved globally do not imply this exact conclusion; the achieved certified interval remains the valid output.

If $S\subseteq S'$, a test based on $S$ can ignore the added coordinates on $S'$, so $V(S')\le V(S)$. With no feature charge, a solution can thus be padded to exactly $d$ features. This justifies an equality cardinality constraint when used below. A global solver bound from an interrupted master is valid; the objective of its current integer incumbent is not such a bound.

## Appendix E. Fixed coordinate bounds and shared-budget comparisons

Own-source constraints give $a_i\ge0$, so the constant-test bound implies

$$
J\le B\quad\Longrightarrow\quad
\theta^{\top}\lambda\le J\le B,\qquad
\lambda_k\le U_k:=B/\theta_k.
\tag{E.1}
$$

This bounds solutions on the relevant sublevel, not every feasible $\lambda$. A coarse but correct bound preserves the optimal value.

Define $v_{\rm ind}(z)$ by minimizing $J$ under the common loss and price constraints of (4), with $a,u$ free, $\lambda,w\ge0$, and independent caps $w_k\le U_k z$, omitting $\eta$ and $J\le\eta$. Define $v_B(z)$ in the same way, replacing those caps by the shared constraint $\theta_0w_0+\theta_1w_1\le Bz$.

For a binary selection, every original optimizer enters both models with $w_k=\lambda_k z$ by (E.1). Conversely, either bounded-price model has inactive prices zero and $w_k\le\lambda_kz$, so (D.1) gives original feasibility at unchanged cost. Therefore

$$
v_{\rm ind}(z)=v_B(z)=V(z),\qquad z\text{ binary}.
\tag{E.2}
$$

This is optimal-value equivalence, not equality of all feasible sets. Even without using attainment, if $V(z)<B$, a sequence with $J<V(z)+\varepsilon<B$ enters both problems as $\varepsilon\downarrow0$; if $V(z)=B$, the constant test already gives the converse value inequality. No additional constraint $\lambda_k\le U_k$ is needed in the bounded-price programs.

For fractional selections the constraints define different continuous extensions, rather than the original model with a fractional product substituted. For instance, $w=1$, $\lambda=1$, $U=2$, $z=1/2$ satisfies the independent caps but not $w\le\lambda z$. Increasing a valid price bound enlarges the set feasible under the separate coordinate bounds, weakening its fractional value while preserving binary exactness if an optimizer is retained for every considered subset.

### E.1 A fixed shared price budget

The program $v_B$ was just defined by minimizing $J$ with a constant coordinate-price allowance $Bz$. Introduce $\beta_k\ge0$ for $w_k-\lambda_k\mathbf1\le0$ and $\mu\ge0$ for $\theta_0w_0+\theta_1w_1-Bz\le0$. Its Lagrangian groups as

$$
\begin{aligned}
\mathcal L_B={}&J+\sum_{i,\ell}\Pi_{i\ell}
[\Phi(y_i u_\ell)-a_i-\mathbf d_{i\ell}^{\top}w_{c_i}]\\
&+\sum_k\beta_k^{\top}(w_k-\lambda_k\mathbf1)
+\mu^{\top}(\theta_0w_0+\theta_1w_1-Bz)\\
={}&\sum_i a_i(q_i-\sum_\ell\Pi_{i\ell})
+\sum_\ell[p_{0\ell}\Phi(-u_\ell)+p_{1\ell}\Phi(u_\ell)]\\
&+\sum_k\lambda_k(\theta_k-\mathbf1^{\top}\beta_k)
+\sum_k w_k^{\top}[\beta_k+\theta_k\mu-A_k(\Pi)]-B\mu^{\top}z.
\end{aligned}
\tag{E.3}
$$

Free $a$ again gives row conservation, and free $u$ contributes $H$. Since $\lambda,w\ge0$, their coefficients must be nonnegative. The dual is therefore

$$
\begin{aligned}
v_B(z)=\max_{\Pi,\beta_0,\beta_1,\mu\ge0}\quad&H(\Pi)-B\mu^{\top}z\\
\text{subject to}\quad&\Pi\mathbf1=q,\quad
\mathbf1^{\top}\beta_k\le\theta_k,\quad
A_k(\Pi)\le\beta_k+\theta_k\mu\quad(k=0,1).
\end{aligned}
\tag{E.4}
$$

These expanded certificates need not satisfy the original budgets in (C.5); the residual prices and their penalty account for that difference.

### E.2 Separate coordinate bounds and comparison

For completeness, nonnegative multipliers $\xi_k$ on the independent caps replace the final term of (E.3) by $\sum_k\xi_k^{\top}(w_k-U_kz)$. The $w_k$ coefficients become $\beta_k+\xi_k-A_k$, giving

$$
\begin{aligned}
v_{\rm ind}(z)=\max_{\Pi,\beta_0,\beta_1,\xi_0,\xi_1\ge0}\quad&
H(\Pi)-\sum_kU_k\xi_k^{\top}z\\
\text{subject to}\quad&\Pi\mathbf1=q,\quad
\mathbf1^{\top}\beta_k\le\theta_k,\quad A_k(\Pi)\le\beta_k+\xi_k.
\end{aligned}
\tag{E.5}
$$

All conditions follow from the same variable-by-variable infima in Appendix C. Eliminating $\beta$ in (E.4) instead gives

$$
v_B(z)=\max\{H(\Pi)-B\mu^{\top}z:
\Pi,\mu\text{ satisfy (6)}\}.
\tag{E.6}
$$

Both expressions are maxima of affine functions over selection-independent certificate conditions and are therefore convex. Needed residual multipliers are uniformly bounded: use the smallest coverage residual for the shared constraint, and $\xi_k=(A_k-\beta_k)_+$ for the separate coordinate bounds. This uniformly bounds the affine slopes and establishes continuity, including at zero coordinates.

The shared constraint implies every independent cap, so $v_B\ge v_{\rm ind}$. For a shared certificate, put $h=H(\Pi)\le B$ and $s=\mu^{\top}z\ge0$. Then

$$
\frac h{1+s}-(h-Bs)=s\left(B-\frac h{1+s}\right)\ge0.
\tag{E.7}
$$

Taking maxima proves $v\ge v_B\ge v_{\rm ind}$ on $[0,1]^D$, while all three agree at binary selections. This is a comparison of complete value functions; it is not a statement that arbitrary finite collections of their cuts have the same ordering or that the stronger extension is always faster.

## Appendix F. Loss instances, an analytic example, and an optional master

### F.1 Four explicit minimum-loss formulas

For the half-hinge loss $\Phi(t)=(1+t)_+/2$, clipping the score to $[-1,1]$ does not increase either loss. On that interval the objective is $(p_0+p_1)/2+(p_1-p_0)t/2$, whose minimum is

$$
\Psi(p_0,p_1)=\min(p_0,p_1),\qquad B=1.
\tag{F.1}
$$

For logistic loss, $\Phi(t)=\log(1+e^t)$, positive $p_0,p_1$ give derivative $(-p_0+p_1e^t)/(1+e^t)$. The minimizing score satisfies $e^t=p_0/p_1$, and substitution gives

$$
\Psi(p_0,p_1)=(p_0+p_1)\log(p_0+p_1)-p_0\log p_0-p_1\log p_1,
\qquad B=2\log2.
\tag{F.2}
$$

For exponential loss $\Phi(t)=e^t$, stationarity in $p_0e^{-t}+p_1e^t$ gives $e^{2t}=p_0/p_1$, hence

$$
\Psi(p_0,p_1)=2\sqrt{p_0p_1},\qquad B=2.
\tag{F.3}
$$

For squared hinge $\Phi(t)=(1+t)_+^2$, clipping again permits $t\in[-1,1]$. There the objective is $p_0(1-t)^2+p_1(1+t)^2$, minimized at $t=(p_0-p_1)/(p_0+p_1)$ when $p_0+p_1>0$. Therefore

$$
\Psi(p_0,p_1)=\frac{4p_0p_1}{p_0+p_1},\qquad B=2,
\tag{F.4}
$$

with value zero at $p_0=p_1=0$. All formulas extend continuously to the axes, with $0\log0=0$. The logistic and exponential axis values can be infima attained only as the score tends to infinity. These examples illustrate mass homogeneity of $\Psi$ and $H$, not homogeneity of the underlying loss.

### F.2 One coordinate with one source per class

Let $x_1=0$, $c_1=0$, $x_2=2$, $c_2=1$, $q_1=q_2=1$, and $\theta_0=\theta_1=1/2$, with half hinge. Then $B=1$. Reflection and exchange of the classes preserve the convex fixed-$z$ programs, so averaging gives a symmetric optimizer with $w_0=w_1=r$. Choosing $\lambda_0=\lambda_1=r$ minimizes the price cost without affecting the loss constraints. At an endpoint, the two half-hinge losses sum to at least one; hence $a_1+a_2+2r\ge1$, while own-source constraints give $a_1+a_2\ge0$.

For $0\le r\le1/2$, take $a_1=a_2=1/2-r$, $u_1=2r$, and $u_2=-2r$. For $r\ge1/2$, zero intercepts and scores $u_1=1$, $u_2=-1$ are feasible. The least cost at price $r$ is therefore

$$
J(r)=(1-2r)_++r.
\tag{F.5}
$$

Independent caps give $r\le2z$ and the shared budget gives $r\le z$, so

$$
v_{\rm ind}(z)=\max(1/2,1-2z),\qquad
v_B(z)=\max(1/2,1-z).
\tag{F.6}
$$

The risk-dependent constraint gives $r\le\eta z$ and $\eta\ge J(r)$, which imply $\eta\ge1-r\ge1-\eta z$. Equality is feasible with $r=z/(1+z)$ and $\eta=1/(1+z)$; the remaining condition $\eta\ge r$ holds because $z\le1$. Thus

$$
v(z)=\frac1{1+z},\qquad 0\le z\le1.
\tag{F.7}
$$

The binary values are $1$ and $1/2$ for every extension. At $z=1/4$ their values are $1/2$, $3/4$, and $4/5$, respectively. The risk-scaled tangent at $z=1$ is $3/4-z/4$.

### F.3 Retaining full certificate fractions

Suppose $b>0$ is a certified risk of a feasible original test on one budget-feasible subset. It bounds $\mathrm{OPT}$, not necessarily the risk of every subset. Restricting the master to $0\le\eta\le b$ nevertheless retains an original optimal selection. For a certificate $s$, write $h_s=H(\Pi^s)$ and retain the full lower function $f_s(z)=h_s/(1+(\mu^s)^{\top}z)$. Its binary epigraph can be represented using shared product variables $\zeta_j=\eta z_j$:

$$
\begin{aligned}
&0\le\eta\le b,\qquad 0\le\zeta_j\le bz_j,\qquad
\zeta_j\le\eta,\qquad \zeta_j\ge\eta-b(1-z_j),\\
&\mathbf1^{\top}z=d,\qquad \mathbf1^{\top}\zeta=d\eta.
\end{aligned}
\tag{F.8}
$$

For binary $z$, these inequalities force $\zeta=\eta z$. If the feature budget has not been padded to equality, only $\mathbf1^{\top}\zeta\le d\eta$ is justified. Each retained fraction adds the linear row

$$
h_s\le\eta+(\mu^s)^{\top}\zeta.
\tag{F.9}
$$

Keeping the affine cuts (8) as well uses only $D$ shared additional variables. Every original optimal selection with $\eta=V(z)$ and $\zeta=\eta z$ remains feasible, so a global bound from this master is still valid. Exact touching certificates give the same finite-termination argument. Improve $b$ only from a feasible original integer-selection test and update all product bounds consistently; a value at a fractional selection cannot supply this cap.

For a fixed certificate with $r_t=1+\mu^{\top}z^t$, the difference between its fraction and its own tangent is

$$
\frac h{1+\mu^{\top}z}
-\left[\frac h{r_t}-\frac{h\mu^{\top}(z-z^t)}{r_t^2}\right]
=\frac{h[1+\mu^{\top}z-r_t]^2}{(1+\mu^{\top}z)r_t^2}\ge0.
\tag{F.10}
$$

This preserves information lost by the tangent at integer selections. It does not ensure dominance of the lifted fractional master, because its product variables need not be exact at fractional $z$. For example, $f(z)=(1+4z_1+5z_2+z_3)^{-1}$, $d=1$, $b=1/5$, and anchor $(1,0,0)$ admit the lifted point $z=(0,5/6,1/6)$, $\eta=1/6$, $\zeta=(0,1/6,0)$ satisfying (F.8)--(F.9), but its matched tangent is $14/75>1/6$. Retaining the affine cuts avoids this loss. Runtime effects remain empirical.

### F.4 Why the metric assumption matters

The interpolation proof uses the triangle inequality. To see why a squared transport cost cannot simply replace the metric, consider the two sources $x_1=0$, $x_2=2$ of the one-coordinate example, but impose cost budgets $1/2$ with ground cost $|x-y|^2$. In the target pool, the finite choice

$$
a_1=a_2=0,\qquad u_1=1,\quad u_2=-1,\qquad
\lambda_0=\lambda_1=1/4
\tag{F.11}
$$

is feasible and has $J=1/4$, because the cross-source squared cost is four. In full space, each class may move half its mass to the common point $1$, at cost $1/2$. At that point any score satisfies $\Phi(-g(1))+\Phi(g(1))\ge1$, so the sum of classwise risks is at least $1/2$. The finite pool therefore fails to represent the full-space problem exactly under this squared-cost budget. Extending the loss class does not justify an unmodified extension to $W_2^2$ transport costs.
