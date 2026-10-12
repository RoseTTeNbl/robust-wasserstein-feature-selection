# Certified Feature Selection for Wasserstein Robust Hypothesis Testing

10 October 2026.

## Abstract

We study budgeted feature selection for binary hypothesis testing under class-conditional Wasserstein uncertainty. For finite, nonnegative, nondecreasing convex losses, an exact finite formulation constructs a whole-space test from every feasible solution. Scaling coordinate transport prices by the risk gives dual certificate fractions that are exact on binary selections. Every chosen subset receives a full continuous solve, producing an actual feasible-test upper bound and a globally valid tangent. Retaining the safe fraction from the same dual result strengthens the lower model without another inner solve. Shared binary products give an exact linear master formulation. After each full evaluation, the global integer master minimizes all retained lower functions. Its certified bound determines whether to stop; otherwise its binary candidate receives the next full inner solve. Verified solve accuracies imply finite termination at the requested risk precision, with an evaluation bound determined by the data movement weights. Population coverage requires separate calibration.

## 1. The selection problem

We can observe at most $d$ of $D$ features. The choice of features and the test using them must be optimized together: a good subset is one on which a test can keep both classwise losses small when the distributions move within their specified uncertainty sets.

Pool the observations from the two classes into $x_1,\ldots,x_N\in\mathbb R^D$. Observation $i$ has class $c_i\in\{0,1\}$ and weight $q_i>0$. We encode its class by the sign $y_i=2c_i-1$: $y_i=-1$ for class zero and $y_i=+1$ for class one. Both $c_i$ and $y_i$ are fixed, observed labels, not optimization variables. Write $I_k=\{i:c_i=k\}$. The empirical distributions and uncertainty sets are

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

The inequality in (2) has a direct meaning: after moving from source $i$ to target $\ell$, the loss is at most its source allowance plus the transport price. The source retains its class $c_i$ and sign $y_i$, while $u_\ell$ is the score at the target. Explicitly,

$$
\Phi(y_i u_\ell)=
\begin{cases}
\Phi(-u_\ell),&c_i=0,\\
\Phi(u_\ell),&c_i=1.
\end{cases}
$$

Thus the loss uses the source class even when the target observation originally belongs to the other class; either class can send mass to any target location.

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

The preceding inequalities explain why each fraction is a valid lower bound. Equality with $v$ requires more: minimizing the Lagrangian of (4) gives the row masses, transport inequalities and one normalization equation. Rescaling those multipliers gives exactly (7). Appendix B derives the objective and every constraint by eliminating one group of variables at a time. For fixed $(\Pi,\mu)$, the fraction is convex because its numerator is nonnegative and its denominator is a positive affine function. The maximum of these convex lower functions is therefore convex.

## 5. Choosing which binary subset to evaluate

Fix a requested risk accuracy $\varepsilon>0$. Let $0\le\delta\le\varepsilon/4$
be the allowed inner risk gap, and $0\le\xi\le\varepsilon/4$ the allowed
master candidate gap. Each binary subset receives a full solve of
(4), with all continuous variables reoptimized. The global
integer master then uses the completed solves to select the next subset.

#### Solve the subset and store a global cut.

At $z^s\in\mathcal Z_d$, the inner solve returns a finite feasible solution
$(a^s,u^s,\lambda^s)$ and its whole-space test. Its verified risk bound is
$R_s=J(a^s,\lambda^s)$. Retaining the constant test ensures $R_s\le B$.
The dual solve returns a pair $(\Pi^s,\mu^s)$ satisfying (6).
Write its verified fraction as $f_s(z)=h_s/[1+(\mu^s)^\top z]$, with
$0\le h_s\le H(\Pi^s)$. The certified value $\ell_s\le f_s(z^s)$ must satisfy
$$
\ell_s\le V(z^s)\le R_s,\qquad
0\le R_s-\ell_s\le\delta\le\varepsilon/4.
$$
In exact arithmetic $\ell_s=f_s(z^s)$; any numerical downward shift is
included in this gap. The fraction's gradient gives the cut
$$
g_s=-\frac{h_s\mu^s}{[1+(\mu^s)^\top z^s]^2},\qquad
L_s(z)=\ell_s+g_s^\top(z-z^s)\le v(z).
\tag{8}
$$
Convexity puts the tangent below $f_s$, while (7) puts $f_s$
below $v$. Store this cut and the actual feasible test together.
Keep the fraction's shape as well, scaling its numerator to the certified
anchor value:
$$
F_s(z)=[\ell_s]_+\frac{1+(\mu^s)^\top z^s}{1+(\mu^s)^\top z}
\le f_s(z)\le v(z).
$$
This function uses the same completed dual result. Its numerator remains
between zero and $h_s$, as shown in Appendix C.
In exact arithmetic $F_s=f_s$. Away from its anchor, a tangent can become
negative while its fraction stays positive. Retaining $F_s$ preserves this
information at the cost of evaluating a fraction; retain $L_s$ as well
when numerical downward shifts are present.

#### Combine the completed evaluations.

After $t$ full evaluations, the actual test bounds and retained lower functions give
$$
U_t=\min\{B,R_1,\ldots,R_t\},\qquad
a_t(z)=\max\{0,L_s(z),F_s(z):s=1,\ldots,t\}.
$$
Here $U_t$ is the global upper bound and $a_t(z)$ is a lower estimate of
each subset's optimal risk. For $t=0$, set $U_0=B$ and $a_0=0$.
Every full-budget subset then has $a_0(z)=0$, so the first subset may be
fixed by a data screen, but still receives a full inner solve. Every
subsequent subset comes from the global master.
For the same completed evaluations, retaining the fractions only raises the
lower model, strengthening the global bound and excluding subsets that cannot improve it.
Adding coordinates cannot increase $V$, since a test can ignore them.
The retained lower functions are also coordinatewise nonincreasing.
Thus both minima can be attained on full-budget subsets
$\mathbf1^\top z=d$, which the implementation uses.

#### Select the next subset with the global master.

The master minimizes the lower model over all allowed subsets:
$$
\begin{aligned}
\min_{z\in\mathcal Z_d,\eta}\quad&\eta\\
\text{subject to}\quad&0\le\eta\le U_t,\qquad\mathbf1^\top z=d,\\
&\eta\ge L_s(z),\qquad \eta\ge F_s(z)\quad(s=1,\ldots,t).
\end{aligned}
\tag{9}
$$
The current best test makes this master feasible, and the bound
$\eta\le U_t$ preserves the minimum of $a_t$.
For binary $z$, use shared products $b_j=\eta z_j$. The bounds
$$
0\le b_j\le U_tz_j,\qquad b_j\le\eta,\qquad
b_j\ge\eta-U_t(1-z_j)\quad(j=1,\ldots,D).
$$
enforce these products exactly. Each constraint $\eta\ge F_s(z)$ is then
one linear row,
$$
\eta+(\mu^s)^\top b\ge[\ell_s]_+\bigl[1+(\mu^s)^\top z^s\bigr].
$$
Thus the stronger master is a mixed-integer linear problem with only $D$
additional variables shared by all stored fractions.
Appendix C proves both equivalences.
At each iteration, solve this master and let $z_{\mathrm M}$ be its binary
candidate. Update $\mathrm{LB}_t$ with the maximum of its previous value
and the master's certified global bound. Stop if
$U_t-\mathrm{LB}_t\le\varepsilon$. Otherwise require
$a_t(z_{\mathrm M})-\mathrm{LB}_t\le\xi$. Then
$$
a_t(z_{\mathrm M})\le\mathrm{LB}_t+\xi
<U_t-\varepsilon+\xi\le U_t-3\varepsilon/4<U_t-\varepsilon/2.
$$
The candidate therefore permits more improvement than the inner error
$\delta\le\varepsilon/4$; Appendix C shows that it cannot
be a previously completed subset. Set $z^{t+1}=z_{\mathrm M}$ and fully
solve the inner problem there. Only its actual feasible test updates the
upper bound, while its verified dual result supplies the new lower functions.
If either solve cannot reach the required accuracy within the budget,
return the available test and bounds with that status.

**Algorithm 1: Global master selection with full inner solves**

**Input:** $\varepsilon>0$; $0\le\delta\le\varepsilon/4$; $0\le\xi\le\varepsilon/4$.

1. Set $t=0$, $U_0=B$, $\mathrm{LB}_0=0$; retain the constant test and initialize empty histories.
2. **While** $U_t-\mathrm{LB}_t>\varepsilon$:
   1. Solve (9) and update $\mathrm{LB}_t$ with its certified global bound.
   2. **If** $U_t-\mathrm{LB}_t\le\varepsilon$:
      1. **Return** The retained best test and subset, with interval $[\mathrm{LB}_t,U_t]$.
   3. Require $a_t(z_{\mathrm M})-\mathrm{LB}_t\le\xi$.
   4. Set $s=t+1$, $z^s=z_{\mathrm M}$; fully solve (4), obtaining a feasible test and verified fraction with $R_s-\ell_s\le\delta$.
   5. Compute (8) at $z^s$; store $L_s$, $F_s$ and its actual test, retaining all previous lower functions.
   6. Set $U_{t+1}=\min\{U_t,R_s\}$, $\mathrm{LB}_{t+1}=\mathrm{LB}_t$, and $t\gets t+1$.
3. **Return** The retained best test and subset, with interval $[\mathrm{LB}_t,U_t]$.

The evaluation bound uses the coordinate movement in the data. Let
$d_{i\ell,j}$ be entry $j$ of $\mathbf d_{i\ell}$ and define
$$
M_j=\max_{k=0,1}
\frac{\sum_{i\in I_k}q_i\max_\ell d_{i\ell,j}}{\theta_k},
\qquad \rho(z,z')=\sum_j M_j|z_j-z'_j|.
$$
Thus $\rho$ measures how selections differ in transport units. Clipping a
feasible correction to $0\le\mu^s\le M$ preserves feasibility; use it before
computing the value and gradient in (8). The positive separation
scale and number of retained coordinate weights are
$$
r_\varepsilon=\frac{\varepsilon/2-\delta}{B}>0,\qquad
M_{(1)}\ge\cdots\ge M_{(D)},\qquad
s_\varepsilon=\min\left\{s\in\{0,\ldots,D\}:
\sum_{j>s}M_{(j)}<r_\varepsilon\right\}.
$$
The proof shows that evaluated subsets are at least $r_\varepsilon$ apart
in $\rho$; $s_\varepsilon$ makes the omitted weight smaller than this distance.
Let $T_\varepsilon$ count completed full binary evaluations begun before
stopping, including the first. Refinements at one selected point count as
one evaluation.

**Theorem 3 (Global optimality and evaluation bound).**

Algorithm 1, with the stated feasibility, correction and
accuracy requirements and all generated cuts and fractions retained, maintains
$$
\mathrm{LB}_t\le\mathrm{OPT}\le U_t.
\tag{10}
$$
If the required solves complete without a premature budget exit, it returns
a feasible test with robust risk at most $\mathrm{OPT}+\varepsilon$. The number
of completed evaluations satisfies
$$
T_\varepsilon\le
\min\left\{|\mathcal Z_d|,
\sum_{i=0}^{\min(d,s_\varepsilon)}\binom{s_\varepsilon}{i}\right\}
\le2^{s_\varepsilon}.
$$

Appendix C proves these statements.

## 6. Loss choices and the scope of the result

The loss enters through $\Phi$ in the continuous problem and through $\Psi$ in the dual objective. Examples covered by the same derivation are shown below; boundary values use continuous limits.

| Loss | $\Phi(s)$ | $\Psi(p_0,p_1)$ |
| --- | --- | --- |
| Half hinge | $\tfrac12(1+s)_+$ | $\min(p_0,p_1)$ |
| Logistic | $\log(1+e^s)$ | $(p_0+p_1)\log(p_0+p_1)-p_0\log p_0-p_1\log p_1$ |
| Exponential | $e^s$ | $2\sqrt{p_0p_1}$ |
| Squared hinge | $(1+s)_+^2$ | $4p_0p_1/(p_0+p_1)$ |

For example, half-hinge masses $p_0=0.2$ and $p_1=0.5$ leave loss $0.2$: the best common score favors the larger mass but cannot distinguish the smaller mass at the same location. Appendix D derives all four formulas. For half hinge, a score clipped to $[-1,1]$ gives randomized class-zero probability $(1+g)/2$, so $V/2$ is the equal-prior worst-case error. Other losses define surrogate objectives. The current implementation uses half hinge and logistic; the other rows illustrate the theoretical scope.

The loss formulas need not be fixed before deriving the algorithm. The transport geometry still matters: the whole-space construction uses the triangle inequality, and the coordinate-price formulation uses the $L^1$ decomposition. The certified interval concerns the stated robust optimization problem; whether its uncertainty sets cover an unknown population is a separate statistical question.

## Appendix A. Proof of Theorem 1: the finite formulation

Under the assumptions of Section 1, fix binary $z$.

Define

$$
T(z):=\inf\{J(a,\lambda):(a,u,\lambda)\text{ is feasible in }(2)\}.
$$

We establish

$$
T(z)\le V(z)\le T(z).
$$

### A.1 Restrict the adversary, then identify the finite objective

$$
S:=\{j:z_j=1\}.
$$

$P_S$ denotes the marginal of $P$ on $S$. Lifting a projected transport while keeping each source's unselected coordinates fixed gives

$$
\begin{aligned}
d_z(x,x')&:=\sum_{j=1}^Dz_j|x_j-x'_j|=\|x_S-x'_S\|_1\le\|x-x'\|_1,\\
W_1(P_S,\widehat P_{k,S})&\le W_1(P,\widehat P_k),\\
\|x_i-(x_{\ell,S},x_{i,S^c})\|_1&=d_z(x_i,x_\ell),\\
g(x_{\ell,S},x_{i,S^c})&=g(x_\ell).
\end{aligned}
\tag{A.1}
$$

Coincident source projections are split according to their weights. The lift preserves test loss, cost and finite first moments. For a score vector $u$, the finite class-$k$ transport problem is

$$
\begin{aligned}
Q_k(u)=\max_{\Pi\ge0}\quad&\sum_{i\in I_k,\ell}\Pi_{i\ell}\Phi(y_i u_\ell)\\
\text{subject to}\quad&\sum_\ell\Pi_{i\ell}=q_i\quad(i\in I_k),\\
&\sum_{i\in I_k,\ell}\Pi_{i\ell}d_z(x_i,x_\ell)\le\theta_k.
\end{aligned}
\tag{A.2}
$$

$$
u(g):=(g(x_1),\ldots,g(x_N)).
$$

Equation (A.1) lifts every feasible finite transport to a distribution in $\mathcal U_k$, hence

$$
Q_k(u(g))\le\sup_{P\in\mathcal U_k}\mathbb E_P\Phi((2k-1)g).
\tag{A.3}
$$

$$
\Pi_{i\ell}=q_i\mathbf1_{\{i=\ell\}}\text{ is feasible in }(A.2),\qquad0\le\Pi_{i\ell}\le q_i.
$$

The feasible set is nonempty and compact. With row multipliers $a_i\in\mathbb R$ and budget multiplier $\lambda_k\ge0$,

$$
\begin{aligned}
\mathcal L_k={}&\sum_{i\in I_k,\ell}\Pi_{i\ell}\Phi(y_i u_\ell)
+\sum_{i\in I_k}a_i\left(q_i-\sum_\ell\Pi_{i\ell}\right)\\
&+\lambda_k\left(\theta_k-\sum_{i\in I_k,\ell}\Pi_{i\ell}d_z(x_i,x_\ell)\right)\\
={}&\sum_{i\in I_k}q_i a_i+\theta_k\lambda_k
+\sum_{i\in I_k,\ell}\Pi_{i\ell}
[\Phi(y_i u_\ell)-a_i-\lambda_kd_z(x_i,x_\ell)].
\end{aligned}
\tag{A.4}
$$

At feasible transports, the row terms vanish and the budget term is nonnegative. The unrestricted nonnegative mass supremum is therefore

$$
\sup_{\Pi\ge0}\mathcal L_k=
\begin{cases}
\sum_{i\in I_k}q_i a_i+\theta_k\lambda_k,
&\Phi(y_i u_\ell)\le a_i+\lambda_kd_z(x_i,x_\ell)\quad(i\in I_k,\ \forall\ell),\\
+\infty,&\text{otherwise}.
\end{cases}
\tag{A.5}
$$

Finite linear programming strong duality gives

$$
Q_k(u)=\min_{\substack{a_i\in\mathbb R,\ \lambda_k\ge0\\
\Phi(y_i u_\ell)\le a_i+\lambda_kd_z(x_i,x_\ell)\ (i\in I_k,\ \forall\ell)}}
\left[\sum_{i\in I_k}q_i a_i+\theta_k\lambda_k\right].
\tag{A.6}
$$

Summing the classwise duals with their common score vector gives

$$
\begin{aligned}
T(z)&=\inf_u[Q_0(u)+Q_1(u)]\\
&\le\inf_g[Q_0(u(g))+Q_1(u(g))]\le V(z).
\end{aligned}
\tag{A.7}
$$

Separate scores at coincident projected targets only enlarge the finite score domain.

### A.2 Extend the finite loss bounds to the whole space

For any feasible $(a,u,\lambda)$ in (2), construct a score satisfying

$$
\Phi((2k-1)g(x))\le a_i+\lambda_kd_z(x_i,x)
\quad(i\in I_k,\ k=0,1).
$$

It suffices to control the smallest classwise allowance:

$$
\begin{aligned}
E_k(x)&=\min_{i\in I_k}\{a_i+\lambda_kd_z(x_i,x)\},\\
i_k(x)&=\min\operatorname*{arg\,min}_{i\in I_k}\{a_i+\lambda_kd_z(x_i,x)\}.
\end{aligned}
\tag{A.8}
$$

The outer minimum selects the smallest source index in a tie. Abbreviate $i_k=i_k(x)$; the triangle inequality gives

$$
c=d_z(x_{i_0},x_{i_1}),\qquad r_k=d_z(x_{i_k},x),\qquad c\le r_0+r_1.
\tag{A.9}
$$

The four source--target pairs $(i_0,i_0)$, $(i_0,i_1)$, $(i_1,i_0)$ and $(i_1,i_1)$ in (2) imply

$$
\begin{aligned}
\Phi(-u_{i_0})&\le a_{i_0},&\Phi(-u_{i_1})&\le a_{i_0}+\lambda_0c,\\
\Phi(u_{i_0})&\le a_{i_1}+\lambda_1c,&\Phi(u_{i_1})&\le a_{i_1}.
\end{aligned}
\tag{A.10}
$$

For $c>0$, choose

$$
\alpha=\frac{\min(c,r_0)}c,\qquad g(x)=(1-\alpha)u_{i_0}+\alpha u_{i_1}.
\tag{A.11}
$$

Since $c\le r_0+r_1$,

$$
\begin{aligned}
0\le\alpha\le1,\qquad \alpha c&=\min(c,r_0)\le r_0,\\
(1-\alpha)c&=c-\min(c,r_0)=\max(c-r_0,0)\le r_1.
\end{aligned}
\tag{A.12}
$$

Convexity and (A.10)--(A.12) yield

$$
\begin{aligned}
\Phi(-g(x))
&\le(1-\alpha)\Phi(-u_{i_0})+\alpha\Phi(-u_{i_1})\\
&\le(1-\alpha)a_{i_0}+\alpha(a_{i_0}+\lambda_0c)
=a_{i_0}+\lambda_0\alpha c\\
&\le a_{i_0}+\lambda_0r_0=E_0(x),\\
\Phi(g(x))
&\le(1-\alpha)\Phi(u_{i_0})+\alpha\Phi(u_{i_1})\\
&\le(1-\alpha)(a_{i_1}+\lambda_1c)+\alpha a_{i_1}
=a_{i_1}+\lambda_1(1-\alpha)c\\
&\le a_{i_1}+\lambda_1r_1=E_1(x).
\end{aligned}
\tag{A.13}
$$

$$
c=0:\qquad g(x):=u_{i_0}.
$$

Equation (A.10) then gives

$$
\Phi(-g(x))\le a_{i_0}\le E_0(x),\qquad
\Phi(g(x))\le a_{i_1}+\lambda_1c=a_{i_1}\le E_1(x).
\tag{A.14}
$$

Finite comparisons of continuous functions make $i_k$ and $g$ Borel measurable. The score depends only on $x_S$ and satisfies

$$
\min_\ell u_\ell\le g(x)\le\max_\ell u_\ell.
$$

For $P\in\mathcal U_k$, let $\gamma_k$ be a coupling on $I_k\times\mathbb R^D$, with source masses $q_i$, target marginal $P$ and cost at most $\theta_k$; coincident atoms retain their source indices. Then

$$
\gamma_k(\{i\}\times\mathbb R^D)=q_i,\qquad
\int d_z(x_i,x)\,d\gamma_k(i,x)
\le\int\|x_i-x\|_1\,d\gamma_k(i,x)\le\theta_k.
\tag{A.15}
$$

$$
\Phi((2k-1)g(x))\le E_k(x)\le a_i+\lambda_kd_z(x_i,x),\qquad i\in I_k.
$$

Integration with both coupling marginals gives

$$
\begin{aligned}
\mathbb E_P\Phi((2k-1)g)
&=\int\Phi((2k-1)g(x))\,d\gamma_k(i,x)\\
&\le\int a_i\,d\gamma_k(i,x)+\lambda_k\int d_z(x_i,x)\,d\gamma_k(i,x)\\
&=\sum_{i\in I_k}a_i\gamma_k(\{i\}\times\mathbb R^D)+\lambda_k\int d_z(x_i,x)\,d\gamma_k(i,x)\\
&=\sum_{i\in I_k}q_i a_i+\lambda_k\int d_z(x_i,x)\,d\gamma_k(i,x)\\
&\le\sum_{i\in I_k}q_i a_i+\theta_k\lambda_k.
\end{aligned}
\tag{A.16}
$$

Taking classwise suprema and adding gives

$$
\sum_{k=0}^1\sup_{P\in\mathcal U_k}\mathbb E_P\Phi((2k-1)g)
\le\sum_{k=0}^1\left[\sum_{i\in I_k}q_i a_i+\theta_k\lambda_k\right]
=J(a,\lambda).
$$

$$
V(z)\le\inf\{J(a,\lambda):(a,u,\lambda)\text{ is feasible in }(2)\}=T(z)\le V(z).
$$

This proves equality and whole-space feasibility.

### A.3 Attainment of the primal minima

$$
u=0,\quad a_i=\Phi(0),\quad\lambda=0
\quad\Longrightarrow\quad J=B;
\qquad w_0=w_1=0,\quad\eta=B.
$$

These feasible points make the objective sublevels $J\le B$ and $J\le\eta\le B$ nonempty. Own-source constraints and positive weights/radii imply

$$
\begin{aligned}
0\le q_i a_i&\le q^{\top}a\le J\le B
&&\implies0\le a_i\le B/q_i,\\
0\le\theta_k\lambda_k&\le\theta^{\top}\lambda\le J\le B
&&\implies0\le\lambda_k\le B/\theta_k,\\
0\le w_{kj}&\le\lambda_k\le B/\theta_k,
&&0\le J\le\eta\le B\quad\text{in }(4).
\end{aligned}
\tag{A.17}
$$

Fix $i_k\in I_k$. Finite pool distances give

$$
\begin{aligned}
C&=\max_{k,\ell}\left\{B/q_{i_k}+(B/\theta_k)\|\mathbf d_{i_k\ell}\|_1\right\},\\
\Phi((2k-1)u_\ell)&\le B/q_{i_k}+(B/\theta_k)\|\mathbf d_{i_k\ell}\|_1\le C.
\end{aligned}
$$

$$
s_1<s_2,\quad\Phi(s_2)>\Phi(s_1),\qquad b:=\frac{\Phi(s_2)-\Phi(s_1)}{s_2-s_1}>0.
$$

Nonconstancy supplies these points; convexity implies, for $t>s_2$,

$$
\frac{\Phi(t)-\Phi(s_2)}{t-s_2}\ge b
\quad\Longrightarrow\quad
\Phi(t)\ge\Phi(s_2)+b(t-s_2)\longrightarrow\infty.
\tag{A.18}
$$

$$
R>0,\qquad\Phi(R)>C.
$$

Monotonicity yields

$$
u_\ell\ge R\implies\Phi(u_\ell)>C,\qquad
u_\ell\le-R\implies\Phi(-u_\ell)>C;
\qquad |u_\ell|<R.
$$

The nonempty objective sublevels are bounded by the preceding estimates and closed by continuity of $\Phi$. They are compact; both continuous objectives attain their minima. Feasibility at value $B$ preserves the unrestricted minima.

## Appendix B. Proof of Theorem 2: from the risk model to global lower functions

Fix $z\in[0,1]^D$.

### B.1 Eliminate the design variables to obtain the dual

$$
\eta\in\mathbb R,\quad a,u\in\mathbb R^N,\quad\lambda,w_0,w_1\ge0.
$$

Assign nonnegative multipliers as follows.

| Constraint | Multiplier |
| --- | --- |
| $\Phi(y_i u_\ell)-a_i-\mathbf d_{i\ell}^{\top}w_{c_i}\le0$ | $\bar\Pi_{i\ell}$ |
| $J-\eta\le0$ | $\tau$ |
| $w_{kj}-\lambda_k\le0$ | $\bar\beta_{kj}$ |
| $\theta_0w_{0j}+\theta_1w_{1j}-\eta z_j\le0$ | $\bar\mu_j$ |

$$
\bar\Pi\in\mathbb R_+^{N\times N},\qquad\tau\ge0,\qquad\bar\beta_k,\bar\mu\in\mathbb R_+^D.
$$

Bars denote multipliers before row-mass normalization. The Lagrangian is

$$
\begin{aligned}
\mathcal L={}&\eta+\tau(J-\eta)
+\sum_{i,\ell}\bar\Pi_{i\ell}[\Phi(y_i u_\ell)-a_i-\mathbf d_{i\ell}^{\top}w_{c_i}]\\
&+\sum_k\bar\beta_k^{\top}(w_k-\lambda_k\mathbf1)
+\bar\mu^{\top}(\theta_0w_0+\theta_1w_1-\eta z).
\end{aligned}
\tag{B.1}
$$

Weak duality gives, for fixed nonnegative multipliers,

$$
\inf_{\substack{\eta\in\mathbb R,\ a,u\in\mathbb R^N\\\lambda,w_0,w_1\ge0}}\mathcal L
\le\mathcal L(\eta,a,u,\lambda,w_0,w_1)\le\eta.
$$

The infimum retains only the displayed variable domains, since the primal constraints have entered $\mathcal L$. Define

$$
\bar p_{k\ell}=\sum_{i\in I_k}\bar\Pi_{i\ell},\qquad
A_k(\bar\Pi)=\sum_{i\in I_k,\ell}\bar\Pi_{i\ell}\mathbf d_{i\ell}.
\tag{B.2}
$$

$$
\begin{aligned}
\sum_{i,\ell}\bar\Pi_{i\ell}a_i
&=\sum_i a_i\sum_\ell\bar\Pi_{i\ell},\\
\sum_{i,\ell}\bar\Pi_{i\ell}\Phi(y_i u_\ell)
&=\sum_\ell[\bar p_{0\ell}\Phi(-u_\ell)+\bar p_{1\ell}\Phi(u_\ell)],\\
\sum_{i,\ell}\bar\Pi_{i\ell}\mathbf d_{i\ell}^{\top}w_{c_i}
&=\sum_k w_k^{\top}A_k(\bar\Pi),\\
\tau J&=\sum_i\tau q_i a_i+\sum_k\tau\theta_k\lambda_k.
\end{aligned}
\tag{B.3}
$$

Collecting coefficients gives

$$
\begin{aligned}
\mathcal L={}&\eta(1-\tau-\bar\mu^{\top}z)
+\sum_i a_i(\tau q_i-\sum_\ell\bar\Pi_{i\ell})\\
&+\sum_\ell[\bar p_{0\ell}\Phi(-u_\ell)+\bar p_{1\ell}\Phi(u_\ell)]\\
&+\sum_k\lambda_k(\tau\theta_k-\mathbf1^{\top}\bar\beta_k)
+\sum_k w_k^{\top}[\bar\beta_k+\theta_k\bar\mu-A_k(\bar\Pi)].
\end{aligned}
\tag{B.4}
$$

For scalar $b\in\mathbb R$,

$$
\inf_{t\in\mathbb R}bt=\begin{cases}0,&b=0,\\-\infty,&b\ne0,\end{cases}
\qquad
\inf_{t\ge0}bt=\begin{cases}0,&b\ge0,\\-\infty,&b<0.\end{cases}
$$

Applying these identities to the free variables gives

$$
\begin{aligned}
\inf_{\eta\in\mathbb R}\eta(1-\tau-\bar\mu^{\top}z)>-\infty
&\iff\tau+\bar\mu^{\top}z=1,\\
\inf_{a\in\mathbb R^N}\sum_i a_i(\tau q_i-\sum_\ell\bar\Pi_{i\ell})>-\infty
&\iff\bar\Pi\mathbf1=\tau q.
\end{aligned}
\tag{B.5}
$$

The targetwise score infimum is

$$
\begin{aligned}
\inf_{u\in\mathbb R^N}\sum_\ell
[\bar p_{0\ell}\Phi(-u_\ell)+\bar p_{1\ell}\Phi(u_\ell)]
&=\sum_\ell\inf_{s\in\mathbb R}
[\bar p_{0\ell}\Phi(-s)+\bar p_{1\ell}\Phi(s)]\\
&=\sum_\ell\Psi(\bar p_{0\ell},\bar p_{1\ell})=H(\bar\Pi).
\end{aligned}
\tag{B.6}
$$

The equality holds even when a scalar minimum is not attained. For every $\epsilon>0$, choose approximate minimizers satisfying

$$
\Psi(\bar p_{0\ell},\bar p_{1\ell})
\le\bar p_{0\ell}\Phi(-u_\ell)+\bar p_{1\ell}\Phi(u_\ell)
\le\Psi(\bar p_{0\ell},\bar p_{1\ell})+\epsilon/N.
$$

Summing and letting $\epsilon\downarrow0$ proves (B.6). Score zero bounds all infima:

$$
0\le\Psi(p_0,p_1)\le\Phi(0)(p_0+p_1),\qquad
0\le H(\bar\Pi)\le\Phi(0)\sum_{i,\ell}\bar\Pi_{i\ell}=\tau B.
$$

Here $\bar\Pi\mathbf1=\tau q$ and $\sum_iq_i=2$. Eliminating nonnegative prices gives

$$
\begin{aligned}
\inf_{\lambda\ge0}\sum_k\lambda_k(\tau\theta_k-\mathbf1^{\top}\bar\beta_k)>-\infty
&\iff\mathbf1^{\top}\bar\beta_k\le\tau\theta_k,\\
\inf_{w_0,w_1\ge0}\sum_k w_k^{\top}[\bar\beta_k+\theta_k\bar\mu-A_k(\bar\Pi)]>-\infty
&\iff A_k(\bar\Pi)\le\bar\beta_k+\theta_k\bar\mu,
\quad k=0,1.
\end{aligned}
\tag{B.7}
$$

All remaining linear infima are zero. Eliminating $\bar\beta$ requires

$$
\begin{aligned}
\bar\beta_{kj}\ge0,\quad
A_{kj}(\bar\Pi)\le\bar\beta_{kj}+\theta_k\bar\mu_j
&\iff\bar\beta_{kj}\ge(A_{kj}(\bar\Pi)-\theta_k\bar\mu_j)_+,\\
\mathbf1^{\top}\bar\beta_k\le\tau\theta_k
&\implies\sum_j(A_{kj}(\bar\Pi)-\theta_k\bar\mu_j)_+\le\tau\theta_k.
\end{aligned}
\tag{B.8}
$$

$$
\bar\beta_k=(A_k(\bar\Pi)-\theta_k\bar\mu)_+.
$$

This choice is feasible exactly when the last inequality in (B.8) holds. Thus

$$
\begin{aligned}
\max_{\tau,\bar\Pi,\bar\mu\ge0}\quad&H(\bar\Pi)\\
\text{subject to}\quad&\bar\Pi\mathbf1=\tau q,\qquad\tau+\bar\mu^{\top}z=1,\\
&\sum_j(A_{kj}(\bar\Pi)-\theta_k\bar\mu_j)_+\le\tau\theta_k,
\quad k=0,1.
\end{aligned}
\tag{B.9}
$$

### B.2 Show that the dual reaches the primal value

Primal attainment follows from Appendix A.3; the bound $B$ there is not an added primal constraint. Remove the coordinates forced to zero:

$$
z_j=0\implies0\le\theta_0w_{0j}+\theta_1w_{1j}\le\eta z_j=0
\implies w_{0j}=w_{1j}=0.
$$

After removing fixed prices and redundant inequalities, choose $0<\alpha<1$ and

$$
\begin{aligned}
u&=0,\qquad a_i=\Phi(0)+1,\qquad\lambda_k=1,\qquad w_k=\alpha z,\\
J&=2[\Phi(0)+1]+\theta_0+\theta_1,\qquad
\eta=J+1+\alpha(\theta_0+\theta_1).
\end{aligned}
\tag{B.10}
$$

The residuals are

$$
\begin{aligned}
\Phi(y_i u_\ell)-a_i-\mathbf d_{i\ell}^{\top}w_{c_i}
&=-1-\alpha\mathbf d_{i\ell}^{\top}z<0,\\
w_{kj}-\lambda_k&=\alpha z_j-1<0,\\
J-\eta&=-1-\alpha(\theta_0+\theta_1)<0,\\
\theta_0w_{0j}+\theta_1w_{1j}-\eta z_j
&=-(J+1)z_j<0\quad(z_j>0).
\end{aligned}
\tag{B.11}
$$

The retained prices and $\lambda_k$ are strictly positive. Convex strong duality therefore yields equality and dual attainment for the reduced problem. Restore deleted coordinates by

$$
z_j=0:\quad\bar\beta_{kj}=0,\qquad
\bar\mu_j=\max_k A_{kj}(\bar\Pi)/\theta_k.
$$

$$
\bar\beta_{kj}+\theta_k\bar\mu_j\ge A_{kj}(\bar\Pi),\qquad\bar\mu_jz_j=0\quad(z_j=0).
$$

The budgets and normalization are unchanged, so the full dual (B.9) also attains $v(z)$.

### B.3 Normalize the masses to obtain the fraction

Positive scaling of the masses gives

$$
\begin{aligned}
\Psi(\tau p_0,\tau p_1)
&=\inf_s\tau[p_0\Phi(-s)+p_1\Phi(s)]
=\tau\Psi(p_0,p_1),\\
H(\tau\Pi)&=\sum_\ell\Psi(\tau p_{0\ell},\tau p_{1\ell})=\tau H(\Pi).
\end{aligned}
\tag{B.12}
$$

At $\tau=0$,

$$
\bar\Pi\mathbf1=0\implies\bar\Pi=0\implies H(\bar\Pi)=\sum_\ell\inf_s0=0.
$$

$$
\ell_*\in\{1,\ldots,N\},\qquad\Pi:=q e_{\ell_*}^{\top}.
$$

Here $e_{\ell_*}$ is the coordinate unit vector. Both classes send unit mass to target $\ell_*$; convexity gives

$$
\Phi(-s)+\Phi(s)\ge2\Phi(0),\qquad
H(\Pi)=\Psi(1,1)=2\Phi(0)=B>0.
$$

A positive normalized-dual objective is obtained with

$$
\mu_j=\max_k\frac{A_{kj}(\Pi)}{\theta_k},\qquad
\tau=\frac1{1+\mu^{\top}z},\qquad
\bar\Pi=\tau\Pi,\qquad\bar\mu=\tau\mu.
\tag{B.13}
$$

$$
\begin{aligned}
\bar\Pi\mathbf1&=\tau q,\qquad
\tau+\bar\mu^{\top}z=\tau(1+\mu^{\top}z)=1,\\
A_k(\Pi)-\theta_k\mu&\le0
\implies(A_k(\bar\Pi)-\theta_k\bar\mu)_+=0,\\
H(\bar\Pi)&=\tau H(\Pi)=\tau B>0.
\end{aligned}
$$

Thus every attained optimum has $\tau>0$. Set

$$
\Pi:=\bar\Pi/\tau,\qquad\mu:=\bar\mu/\tau.
$$

Then

$$
\begin{aligned}
\bar\Pi\mathbf1=\tau q&\iff\Pi\mathbf1=q,\\
A_k(\bar\Pi)&=\tau A_k(\Pi),\\
(A_k(\bar\Pi)-\theta_k\bar\mu)_+
&=\tau(A_k(\Pi)-\theta_k\mu)_+,\\
\sum_j(A_{kj}(\bar\Pi)-\theta_k\bar\mu_j)_+\le\tau\theta_k
&\iff\sum_j(A_{kj}(\Pi)-\theta_k\mu_j)_+\le\theta_k.
\end{aligned}
\tag{B.14}
$$

$$
(\tau t)_+=\tau t_+\quad(\tau>0).
$$

The resulting constraints are (6), independent of $z$; normalization determines

$$
\begin{aligned}
\tau+\bar\mu^{\top}z=1
&\iff\tau(1+\mu^{\top}z)=1
\iff\tau=(1+\mu^{\top}z)^{-1},\\
H(\bar\Pi)&=\tau H(\Pi)=\frac{H(\Pi)}{1+\mu^{\top}z}.
\end{aligned}
\tag{B.15}
$$

$$
1+\mu^\top z>0,\qquad(\bar\Pi,\bar\mu)=\tau(\Pi,\mu),\qquad\tau=(1+\mu^\top z)^{-1}.
$$

Conversely, this scaling maps every feasible pair in (6) to (B.9) by (B.14). The objectives agree, establishing (7) and attainment.

### B.4 Connect the representation to selection and global cuts

For binary $z$ and a feasible point of (2), set

$$
\eta:=J,\qquad w_k:=\lambda_kz.
$$

$$
\begin{aligned}
0\le w_k&\le\lambda_k\mathbf1,\\
\theta_0w_{0j}+\theta_1w_{1j}
&=z_j\theta^{\top}\lambda\le z_jJ=z_j\eta,\\
a_i+\mathbf d_{i\ell}^{\top}w_{c_i}
&=a_i+\lambda_{c_i}\mathbf d_{i\ell}^{\top}z.
\end{aligned}
\tag{B.16}
$$

$$
v(z)\le V(z).
$$

Conversely, every feasible point of (4) satisfies

$$
\begin{aligned}
z_j=0&\implies\theta_0w_{0j}+\theta_1w_{1j}\le0
\implies w_{0j}=w_{1j}=0,\\
z_j=1&\implies w_{kj}\le\lambda_k,\\
&\hspace{1em}\Longrightarrow\quad w_k\le\lambda_kz.
\end{aligned}
\tag{B.17}
$$

$$
\Phi(y_i u_\ell)\le a_i+\mathbf d_{i\ell}^{\top}w_{c_i}
\le a_i+\lambda_{c_i}\mathbf d_{i\ell}^{\top}z.
$$

$$
V(z)\le J\le\eta\quad\Longrightarrow\quad V(z)\le v(z)\quad\Longrightarrow\quad v(z)=V(z).
$$

For a fixed feasible pair $(\Pi,\mu)$, define

$$
h:=H(\Pi)\ge0,\qquad r(z):=1+\mu^\top z>0,\qquad f(z):=h/r(z).
$$

$$
\begin{aligned}
\frac{\partial r}{\partial z_j}&=\mu_j,\qquad
\frac{\partial f}{\partial z_j}
=h[-r(z)^{-2}]\mu_j=-\frac{h\mu_j}{r(z)^2},\\
\frac{\partial^2f}{\partial z_m\partial z_j}
&=-h\mu_j[-2r(z)^{-3}]\mu_m
=\frac{2h\mu_j\mu_m}{r(z)^3},\\
\nabla^2f(z)&=\frac{2h\mu\mu^{\top}}{r(z)^3}.
\end{aligned}
\tag{B.18}
$$

For every $b\in\mathbb R^D$,

$$
b^{\top}\nabla^2f(z)b
=\frac{2h}{r(z)^3}(\mu^{\top}b)^2\ge0.
$$

Hence $f$ is convex and

$$
f(z^s)+\nabla f(z^s)^{\top}(z-z^s)\le f(z)\le v(z).
$$

The feasible family in (7) is independent of $z$; its maximum of convex functions is convex.

## Appendix C. Proof of Theorem 3: valid bounds and finite termination

Assume the feasibility and accuracy requirements of Section 5.

### C.1 A full solve supplies reliable lower functions

For an evaluated subset $z^s$, let $e_s\ge0$ denote the numerical downward shift.

$$
\begin{aligned}
f_s(z)&=\frac{h_s}{1+(\mu^s)^\top z},
&0\le h_s&\le H(\Pi^s),\\
\ell_s&=f_s(z^s)-e_s,
&\ell_s\le V(z^s)&\le R_s,\qquad R_s-\ell_s\le\delta.
\end{aligned}
$$

Convexity gives a global lower function without changing the original gradient:

$$
\begin{aligned}
g_s&=\nabla f_s(z^s)
=-\frac{h_s\mu^s}{[1+(\mu^s)^\top z^s]^2},\\
L_s(z)&=\ell_s+g_s^\top(z-z^s)\\
&=f_s(z^s)+\nabla f_s(z^s)^\top(z-z^s)-e_s\\
&\le f_s(z)-e_s\le f_s(z)\le v(z).
\end{aligned}
$$

For the same certificate, define

$$
\begin{aligned}
d_s&:=1+(\mu^s)^\top z^s>0,\\
n_s&:=[\ell_s]_+d_s
=\left[\frac{h_s}{d_s}-e_s\right]_+d_s
=[h_s-e_sd_s]_+,\\
0&\le n_s\le h_s,\\
F_s(z)&=\frac{n_s}{1+(\mu^s)^\top z}
\le\frac{h_s}{1+(\mu^s)^\top z}
=f_s(z)\le v(z).
\end{aligned}
$$

The gradient is $\nabla f_s(z^s)$, not the gradient obtained from $n_s$. At $e_s=0$, $F_s=f_s\ge L_s$; otherwise both remain valid without a universal ordering. Since $v=V$ on binary subsets,

$$
a_t(z)=\max\{0,L_s(z),F_s(z):s=1,\ldots,t\}\le V(z),\qquad
\mathrm{OPT}\le U_t=\min\{B,R_1,\ldots,R_t\}.
$$

### C.2 The master certifies stopping or supplies a candidate

Let $\widehat z_t$ be the retained test's subset with risk bound $U_t$; at $t=0$, use the constant test on any full-budget subset. Then

$$
\min_{z\in\mathcal Z_d}a_t(z)
\le a_t(\widehat z_t)\le V(\widehat z_t)\le U_t.
$$

$$
\begin{aligned}
\eta\ge0,\quad\eta\ge L_s(z),\quad\eta\ge F_s(z)\quad(1\le s\le t)
&\iff\eta\ge a_t(z),\\
\min_{\substack{z\in\mathcal Z_d,\eta\\0\le\eta\le U_t,\ \eta\ge a_t(z)}}\eta
&=\min_{z\in\mathcal Z_d}a_t(z).
\end{aligned}
$$

The finite selection set ensures attainment. A master witness is $(\widehat z_t,U_t)$. For any $z\in\mathcal Z_d$, complete it to $z'\ge z$ with $\mathbf1^\top z'=d$. Since $g_s\le0$, $n_s\ge0$ and $\mu^s\ge0$,

$$
\begin{aligned}
L_s(z')-L_s(z)&=g_s^\top(z'-z)\le0,\\
F_s(z')&=\frac{n_s}{1+(\mu^s)^\top z'}
\le\frac{n_s}{1+(\mu^s)^\top z}=F_s(z),\\
a_t(z')&\le a_t(z).
\end{aligned}
$$

Thus

$$
\min_{z\in\mathcal Z_d}a_t(z)
=\min_{\substack{z\in\mathcal Z_d\\\mathbf1^\top z=d}}a_t(z).
$$

The original risk is also nonincreasing under coordinate addition, since the existing test remains admissible. Positivity of the fraction denominator gives

$$
\begin{aligned}
\eta\ge F_s(z)
&\iff \eta\bigl[1+(\mu^s)^\top z\bigr]\ge n_s\\
&\iff \eta+\sum_j\mu_j^s(\eta z_j)\ge n_s\\
&\iff \eta+(\mu^s)^\top b\ge n_s.
\end{aligned}
$$

For $0\le\eta\le U_t$, the product bounds give

$$
\begin{aligned}
z_j=0:&\quad 0\le b_j\le0\quad\Longrightarrow\quad b_j=0=\eta z_j,\\
z_j=1:&\quad b_j\le\eta,\quad b_j\ge\eta\quad
\Longrightarrow\quad b_j=\eta=\eta z_j.
\end{aligned}
$$

$$
\begin{aligned}
b_j=\eta z_j,\ z_j=0&\implies b_j=0,\quad0\ge\eta-U_t,\\
b_j=\eta z_j,\ z_j=1&\implies b_j=\eta\le U_t.
\end{aligned}
$$

Conversely these products satisfy all bounds, proving exact equivalence at binary selections. The implied rows are

$$
\sum_jb_j=\eta\sum_jz_j=d\eta,\qquad
(\mu^s)^\top z\ge n_s/U_t-1\quad(U_t>0).
$$

$$
n_s\le\eta[1+(\mu^s)^\top z]\le U_t[1+(\mu^s)^\top z].
$$

The first implied row follows by summing products, and the second from this inequality. Hence

$$
\mathrm{LB}_t\le\min_{z\in\mathcal Z_d}a_t(z)
\le\min_{z\in\mathcal Z_d}V(z)=\mathrm{OPT}\le U_t.
$$

$$
a_{t+1}\ge a_t,\qquad
U_t-\mathrm{LB}_t\le\varepsilon
\ \Longrightarrow\ U_t\le\mathrm{OPT}+\varepsilon.
$$

Earlier certified bounds remain valid as lower functions are retained. At every nonterminal master solve, the next subset must satisfy

$$
a_t(z)\le U_t-\varepsilon/2.
\tag{C.1}
$$

$$
a_t(z_{\mathrm M})-\mathrm{LB}_t\le\xi\le\varepsilon/4,\qquad U_t-\mathrm{LB}_t>\varepsilon.
$$

Consequently

$$
\begin{aligned}
a_t(z_{\mathrm M})
&\le\mathrm{LB}_t+\xi
<U_t-\varepsilon+\xi\\
&\le U_t-3\varepsilon/4
<U_t-\varepsilon/2.
\end{aligned}
$$

Thus every nonterminal master solve supplies a subset satisfying (C.1), which is then fully evaluated.

### C.3 Successive evaluated subsets are separated

A completed subset cannot be selected by a later nonterminal master:

$$
a_t(z^s)\ge L_s(z^s)=\ell_s
\ge R_s-\delta\ge U_t-\delta
>U_t-\varepsilon/2.
$$

Source row masses bound the movement coordinates:

$$
\begin{aligned}
0\le A_{kj}(\Pi)
&=\sum_{i\in I_k,\ell}\Pi_{i\ell}d_{i\ell,j}\\
&\le\sum_{i\in I_k}\Bigl(\sum_\ell\Pi_{i\ell}\Bigr)
                       \max_\ell d_{i\ell,j}\\
&=\sum_{i\in I_k}q_i\max_\ell d_{i\ell,j}\le\theta_kM_j.
\end{aligned}
$$

$$
\mu_j>M_j\quad\Longrightarrow\quad
(A_{kj}(\Pi)-\theta_k\mu_j)_+
=(A_{kj}(\Pi)-\theta_kM_j)_+=0\quad(k=0,1).
$$

Clipping preserves (6) and decreases the denominator. With $0\le\mu^s\le M$, score zero in (5) gives

$$
\begin{aligned}
0\le h_s\le H(\Pi^s)
&\le\Phi(0)\sum_\ell(p_{0\ell}+p_{1\ell})=2\Phi(0)=B,\\
0\le -g_{s,j}
&=\frac{h_s\mu_j^s}{[1+(\mu^s)^\top z^s]^2}\le BM_j.
\end{aligned}
$$

$$
s<t,\qquad U_{t-1}\le R_s,\qquad L_s\le a_{t-1}.
$$

Inner accuracy and (C.1) imply

$$
\ell_s\ge R_s-\delta\ge U_{t-1}-\delta,\qquad
L_s(z^t)\le a_{t-1}(z^t)\le U_{t-1}-\varepsilon/2.
$$

Subtracting and discarding negative coordinate contributions gives

$$
\begin{aligned}
\varepsilon/2-\delta
&\le\ell_s-L_s(z^t)
=-g_s^\top(z^t-z^s)\\
&\le\sum_j(-g_{s,j})(z_j^t-z_j^s)_+\\
&\le B\sum_jM_j(z_j^t-z_j^s)_+
\le B\rho(z^t,z^s).
\end{aligned}
$$

Hence

$$
\rho(z^t,z^s)\ge r_\varepsilon
=\frac{\varepsilon/2-\delta}{B}>0.
$$

### C.4 Count the distinct patterns and conclude

$$
M_{(1)}\ge\cdots\ge M_{(D)}.
$$

Choose the smallest retained prefix with strictly smaller omitted weight:

$$
s_\varepsilon=\min\left\{s\in\{0,\ldots,D\}:
\sum_{j>s}M_{(j)}<r_\varepsilon\right\}.
$$

If two evaluated subsets agree on this prefix, with $z_{(j)}$ denoting the entry at weight $M_{(j)}$, then

$$
\rho(z,z')=\sum_{j>s_\varepsilon}M_{(j)}|z_{(j)}-z'_{(j)}|
\le\sum_{j>s_\varepsilon}M_{(j)}<r_\varepsilon,
$$

This contradicts separation, so each evaluation has a distinct retained pattern. The budget permits at most $d$ retained ones, giving

$$
T_\varepsilon\le
\min\left\{|\mathcal Z_d|,
\sum_{i=0}^{\min(d,s_\varepsilon)}\binom{s_\varepsilon}{i}\right\}
\le2^{s_\varepsilon}.
$$

If the gap remained above $\varepsilon$, the master would supply another eligible, distinct pattern, contradicting this finite bound. Thus the required gap is attained. $T_\varepsilon$ includes the first evaluation; same-point refinements count within it. The result requires the specified solve accuracies, including certificate and comparison errors. A premature budget exit returns the available feasible test, valid interval and exit status.

## Appendix D. Loss formulas for Section 6

$$
F(s):=p_0\Phi(-s)+p_1\Phi(s),\qquad\Psi(p_0,p_1)=\inf_{s\in\mathbb R}F(s),\qquad p_0,p_1>0.
$$

$s^*$ denotes a minimizing score when attained. For half hinge,

$$
\Phi(s)=\tfrac12(1+s)_+,\qquad
s>1\implies F(s)\ge F(1),\qquad
s<-1\implies F(s)\ge F(-1).
$$

Hence the score can be restricted to $[-1,1]$.

$$
\begin{aligned}
F(s)&=\frac{p_0(1-s)+p_1(1+s)}2
=\frac{p_0+p_1}2+\frac{p_1-p_0}2s,\\
\Psi(p_0,p_1)&=
\begin{cases}F(-1)=p_0,&p_1\ge p_0,\\F(1)=p_1,&p_1<p_0,\end{cases}
=\min(p_0,p_1).
\end{aligned}
\tag{D.1}
$$

The sign of the linear coefficient selects the endpoint; for $p_0=p_1$, all interval scores minimize. For logistic,

$$
\Phi(s)=\log(1+e^s).
$$

$$
\begin{aligned}
\frac d{ds}\bigl[p_0\log(1+e^{-s})\bigr]
&=-\frac{p_0e^{-s}}{1+e^{-s}}=-\frac{p_0}{1+e^s},\\
\frac d{ds}\bigl[p_1\log(1+e^s)\bigr]
&=\frac{p_1e^s}{1+e^s},\\
F'(s)=\frac{-p_0+p_1e^s}{1+e^s}=0
&\quad\Longleftrightarrow\quad -p_0+p_1e^{s^*}=0\\
&\quad\Longleftrightarrow\quad e^{s^*}=p_0/p_1,\quad s^*=\log(p_0/p_1).
\end{aligned}
$$

$$
\operatorname{sgn}F'(s)=\operatorname{sgn}(s-s^*).
$$

Substituting the minimizing score gives

$$
\begin{aligned}
1+e^{-s^*}&=(p_0+p_1)/p_0,
&1+e^{s^*}&=(p_0+p_1)/p_1,\\
\Psi(p_0,p_1)&=p_0\log\frac{p_0+p_1}{p_0}
+p_1\log\frac{p_0+p_1}{p_1}\\
&=p_0[\log(p_0+p_1)-\log p_0]
+p_1[\log(p_0+p_1)-\log p_1]\\
&=(p_0+p_1)\log(p_0+p_1)-p_0\log p_0-p_1\log p_1.
\end{aligned}
\tag{D.2}
$$

$$
\Phi(s)=e^s.
$$

$$
\begin{aligned}
F(s)-2\sqrt{p_0p_1}
&=\left(\sqrt{p_0}e^{-s/2}-\sqrt{p_1}e^{s/2}\right)^2\ge0,\\
\sqrt{p_0}e^{-s^*/2}=\sqrt{p_1}e^{s^*/2}
&\quad\Longleftrightarrow\quad
e^{s^*}=\sqrt{p_0/p_1},\quad s^*=\tfrac12\log(p_0/p_1),\\
\Psi(p_0,p_1)&=F(s^*)=2\sqrt{p_0p_1}.
\end{aligned}
\tag{D.3}
$$

$$
\Phi(s)=(1+s)_+^2.
$$

The same clipping inequalities restrict scores to $[-1,1]$; completing the square gives

$$
\begin{aligned}
F(s)&=p_0(1-s)^2+p_1(1+s)^2\\
&=(p_0+p_1)s^2+2(p_1-p_0)s+(p_0+p_1)\\
&=(p_0+p_1)\left(s-\frac{p_0-p_1}{p_0+p_1}\right)^2
+(p_0+p_1)-\frac{(p_0-p_1)^2}{p_0+p_1},\\
(p_0+p_1)-\frac{(p_0-p_1)^2}{p_0+p_1}
&=\frac{(p_0+p_1)^2-(p_0-p_1)^2}{p_0+p_1}
=\frac{4p_0p_1}{p_0+p_1},\\
s^*&=\frac{p_0-p_1}{p_0+p_1}\in[-1,1],
\qquad\Psi(p_0,p_1)=F(s^*)=\frac{4p_0p_1}{p_0+p_1}.
\end{aligned}
\tag{D.4}
$$

The nonnegative square vanishes at $s^*$. All four losses satisfy $\Phi(s)\to0$ as $s\to-\infty$, so the boundary values are

$$
\begin{aligned}
0\le\Psi(p_0,0)&\le p_0\Phi(-s)\longrightarrow0\quad(s\to+\infty),\\
0\le\Psi(0,p_1)&\le p_1\Phi(s)\longrightarrow0\quad(s\to-\infty),\\
\Psi(p_0,0)&=\Psi(0,p_1)=\Psi(0,0)=0.
\end{aligned}
$$

For half hinge and squared hinge, the first zero is attained at $s\ge1$ and the second at $s\le-1$. With a positive remaining mass, logistic and exponential attain zero only in the limit. Use $0\log0=0$ and squared-hinge value zero at $(0,0)$.
