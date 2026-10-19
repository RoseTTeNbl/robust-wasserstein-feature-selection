# Variable Selection for Wasserstein Robust Hypothesis Testing

## Abstract

Variable selection for Wasserstein robust testing admits an exact finite formulation and a maximum-affine dual representation. Dual couplings generate global cuts.

## 1. Problem formulation

Throughout this paper, we consider the variable selection problem for robust hypothesis testing. Let $[D]=\{1,\ldots,D\}$ and $d$ be a positive integer denoting the sparsity budget.
Given a detector $f:~\mathbb{R}^D\to\mathbb{R}$, we follow the convention in [2] to define the worst-case risk as

$$
R(f)=\sup_{\mathbb{P}_0\in\mathcal{P}_0, \mathbb{P}_1\in\mathcal{P}_1}~
\left\{\mathbb E_{\mathbb P_0}\ell(-f(X))+
       \mathbb E_{\mathbb P_1}\ell(f(X))\right\},
\tag{1}
$$

where $\mathcal{P}_k, k=0,1$ denote the Wasserstein distance-based ambiguity sets, and $\ell(\cdot)$ is the generating function that is nonnegative-valued, non-decreasing, convex, and satisfies $\ell(0)=1, \lim_{t\to-\infty}\ell(t)=0$.
For $k=0,1$, assume we have $n$ available observations in the group $k$, which are denoted as $\{x_i^{(k)}\}_{i=1}^n$.
We denote the corresponding empirical distribution as $\widehat{\mathbb{P}}_k$, and construct the Wasserstein ambiguity set as

$$
\mathcal{P}_k=\Big\{
\mathbb{P}_k:~\mathcal{W}_1(
\widehat{\mathbb{P}}_k, \mathbb{P}_k
)\le \rho_k
\Big\}.
$$

Here, $\mathcal{W}_1(\cdot,\cdot)$ refers to the $1$-Wasserstein distance equipped with the $\ell_1$-norm.

For $S\subseteq[D]$, define the collection of detectors whose input only depends on coordinates belonging to $S$:

$$
\mathcal{F}_S = \Big\{
f:~\mathbb{R}^D\to\mathbb{R}, f(x) = g(x[S])\text{ for some measurable function $g:~\mathbb{R}^{|S|}\to\mathbb{R}$}
\Big\}.
$$

Note that for a given vector $x\in\mathbb{R}^D$, we use $x[i]$ to denote its $i$-th coordinate, and $x[S]$ to denote the concatenation of its coordinates whose indices belong to $S$.
Our proposed variable selection problem can be formulated as the following combinatorial optimization problem

$$
\min_{S\subseteq[D], |S|\le d}\min_{f\in\mathcal{F}_S}~R(f).
\tag{2}
$$

## 2. Finite reformulation and duality

### 2.1. Fixed-selection reformulation

Fix $S\subseteq[D]$ and write $V(S)=\min_{f\in\mathcal F_S}R(f)$.
Assume $\rho_0,\rho_1>0$ and that $\ell$ is finite on $\mathbb R$.
Following the LFD formulation in the related manuscript listed in
[2, Lemma 2, Eq. (7)],
relabel the pooled observations as $x_1,\ldots,x_{2n}$, with
$x_{kn+i}=x_i^{(k)}$. Set

$$
c_{ir}^{k,S}=\|x_i^{(k)}[S]-x_r[S]\|_1,
\qquad i\in[n],\ r\in[2n],\ k=0,1.
$$

The referenced lemma gives the following transport problem for randomized
testing on the selected coordinates:

$$
\begin{aligned}
\max_{p_0,p_1,\pi^0,\pi^1\ge0}\quad&
\sum_{r=1}^{2n}\min\{p_{0r},p_{1r}\}\\
\text{subject to}\quad&
\sum_{r=1}^{2n}\pi^k_{ir}=\frac1n,\quad i\in[n],\ k=0,1,\\
&p_{kr}=\sum_{i=1}^n\pi^k_{ir},\quad r\in[2n],\ k=0,1,\\
&\sum_{i=1}^n\sum_{r=1}^{2n}c_{ir}^{k,S}\pi^k_{ir}\le\rho_k,
\quad k=0,1.
\end{aligned}
\tag{3}
$$

Here $\pi^k_{ir}$ transports class-$k$ mass from $x_i^{(k)}[S]$ to
$x_r[S]$, and $p_{kr}$ is the resulting mass at that target. Let
$\mathcal T_k(S)$ denote the nonnegative matrices satisfying the row-sum
and transport-budget constraints for class $k$ in (3).
The column-sum equalities then determine $p_k$.

For the generating loss in Section 1, write
$\ell_0(s)=\ell(-s)$ and $\ell_1(s)=\ell(s)$. At target $r$, the optimal
weighted loss is $\inf_s\{p_{0r}\ell(-s)+p_{1r}\ell(s)\}$.
The corresponding extension of the LFD formulation is

$$
\begin{aligned}
V(S)
&=\max_{\pi^0\in\mathcal T_0(S),\,\pi^1\in\mathcal T_1(S)}
\sum_{r=1}^{2n}\inf_{s\in\mathbb R}
\{p_{0r}\ell(-s)+p_{1r}\ell(s)\}\\
&=\inf_{u\in\mathbb R^{2n}}\sum_{k=0}^1
\max_{\pi^k\in\mathcal T_k(S)}
\sum_{i=1}^n\sum_{r=1}^{2n}\pi^k_{ir}\ell_k(u_r).
\end{aligned}
$$

The first equality requires a general-loss extension of the lemma;
Appendix A proves it by constructing detectors and transport
plans in both directions. The second equality follows from finite-dimensional
minimax duality. Its expected-loss term follows directly from the column sums:
$\sum_r p_{kr}\ell_k(u_r)=\sum_{i,r}\pi^k_{ir}\ell_k(u_r)$.

For fixed $u$, dualize the class-$k$ transport LP. Assign $a_i^k\in\mathbb R$
to its row-sum constraint and $\lambda_k\ge0$ to its budget. Its Lagrangian is

$$
\frac1n\sum_i a_i^k+\rho_k\lambda_k
+\sum_{i,r}\pi^k_{ir}
\bigl[\ell_k(u_r)-a_i^k-\lambda_kc_{ir}^{k,S}\bigr].
$$

The supremum over $\pi^k\ge0$ is finite precisely when every bracket is
nonpositive. LP strong duality, followed by minimization over $u$, therefore gives

$$
\begin{aligned}
V(S)=\min_{a,u,\lambda}\quad&
\underbrace{\frac1n\sum_{k=0}^1\sum_{i=1}^n a_i^k
+\sum_{k=0}^1\rho_k\lambda_k}_{J(a,\lambda)}\\
\text{subject to}\quad&
\ell_k(u_r)\le a_i^k+\lambda_kc_{ir}^{k,S},
\quad i\in[n],\ r\in[2n],\ k=0,1,\\
&a\in\mathbb R^{2\times n},\quad u\in\mathbb R^{2n},\quad
\lambda\in\mathbb R_+^2.
\end{aligned}
\tag{4}
$$

Thus (4) has exactly the optimal risk of detectors using
$S$; minimizing it over $|S|\le d$ solves the problem in
Section 1. The proof in Appendix A also defines
an optimal detector outside the pooled observations.

### 2.2. Mixed-integer reformulation

Let $z=\mathbf1_S$ and define the coordinate costs

$$
\Delta^k_{ir,j}=|x_i^{(k)}[j]-x_r[j]|,
\qquad c_{ir}^{k,S}=\sum_{j\in S}\Delta^k_{ir,j}.
$$

Following the logical-constraint formulation of
Bertsimas et al. [1], introduce coordinate prices
$w_{kj}$ that vanish when $z_j=0$. For fixed $z\in[0,1]^D$, define

$$
\begin{aligned}
F(z)=\min_{a,u,\lambda,w}\quad&J(a,\lambda)\\
\text{subject to}\quad&
\ell_k(u_r)\le a_i^k+\sum_{j=1}^D\Delta^k_{ir,j}w_{kj},
\quad i\in[n],\ r\in[2n],\ k=0,1,\\
&0\le w_{kj}\le\lambda_k,\quad k=0,1,\ j\in[D],\\
&\rho_0w_{0j}+\rho_1w_{1j}\le z_jJ(a,\lambda),\quad j\in[D],\\
&a\in\mathbb R^{2\times n},\quad u\in\mathbb R^{2n},\quad
\lambda\in\mathbb R_+^2.
\end{aligned}
\tag{5}
$$

Choosing the source itself as target gives $a_i^k\ge0$.
Consequently, when $z_j=1$, its selection inequality is redundant:

$$
\rho_0w_{0j}+\rho_1w_{1j}
\le\rho_0\lambda_0+\rho_1\lambda_1\le J(a,\lambda).
$$

When $z_j=0$, it forces $w_{0j}=w_{1j}=0$.
A feasible solution of (4) lifts to (5) through
$w_{kj}=\lambda_kz_j$. Conversely, at $z=\mathbf1_S$,

$$
\ell_k(u_r)\le a_i^k+\sum_{j\in S}\Delta^k_{ir,j}w_{kj}
\le a_i^k+\lambda_kc_{ir}^{k,S}.
$$

Both directions preserve $J$, so $F(\mathbf1_S)=V(S)$.
This proves the exactness of the reformulation for binary selections.

### 2.3. Dual representation

Let $\Pi^k_{ir}\ge0$ and $\mu_j\ge0$ be multipliers of the loss and
selection constraints in (5). The dual objective is

$$
H(\Pi)=\sum_{r=1}^{2n}\inf_{s\in\mathbb R}
\sum_{k=0}^1\sum_{i=1}^n\Pi^k_{ir}\ell_k(s).
$$

It is the sum of the optimal weighted losses at the targets, with weights
given by the column masses of $\Pi^0$ and $\Pi^1$.
Minimizing the Lagrangian over $a,u,\lambda,w$ gives

$$
\begin{aligned}
F(z)=\max_{\Pi,\alpha,\mu}\quad&H(\Pi)\\
\text{subject to}\quad&(\Pi,\alpha,\mu)\in\mathcal D,\qquad
\alpha+\mu^\top z=1,
\end{aligned}
\tag{6}
$$

where $(v)_+=\max\{v,0\}$ and

$$
\mathcal D=\left\{(\Pi,\alpha,\mu)\ge0:
\begin{array}{ll}
\displaystyle\sum_{r=1}^{2n}\Pi^k_{ir}=\alpha/n,&i\in[n],\ k=0,1,\\[3pt]
\displaystyle\sum_{j=1}^D
\left(\sum_{i,r}\Pi^k_{ir}\Delta^k_{ir,j}-\rho_k\mu_j\right)_+
\le\alpha\rho_k,&k=0,1
\end{array}\right\}.
\tag{7}
$$

The row sums show that $\Pi^k$ is a transport plan with its source masses
scaled by $\alpha$. Appendix B derives the positive-part
constraints by eliminating the multipliers of $w_{kj}\le\lambda_k$.

The set $\mathcal D$ is a cone and $H(\tau\Pi)=\tau H(\Pi)$ for
$\tau\ge0$. For any normalized tuple,

$$
\max_{\tau\ge0}\{2\sqrt{\tau H(\Pi)}-\tau\}=H(\Pi).
$$

Optimizing this scale removes the normalization and yields

$$
\boxed{F(z)=\max_{(\Pi,\alpha,\mu)\in\mathcal D}
\{2\sqrt{H(\Pi)}-\alpha-\mu^\top z\}.}
\tag{8}
$$

An optimal normalized tuple $(\bar\Pi,\bar\alpha,\bar\mu)$ is multiplied
by $F(z)$ to obtain an optimizer of (8). Thus its
affine slope is $-F(z)\bar\mu$.
Proposition B.1 proves this representation and the resulting
subgradient formula.

## 3. Cutting-plane algorithm

The affine representation allows the outer-approximation strategy of
Bertsimas et al. [1, Section 3.1]: solve the
continuous problem for one selection, retain its affine lower bound, and
use a binary linear program to choose the next selection.
Write $\mathrm{OPT}=\min_{|S|\le d}V(S)$ and let $\mathcal Z_d=\{z\in\{0,1\}^D:\mathbf1^\top z\le d\}$.
At $z^t$, take an optimizer $(\Pi^t,\alpha^t,\mu^t)$ of
(8). It defines

$$
\begin{aligned}
c_t(z)&=2\sqrt{H(\Pi^t)}-\alpha^t-(\mu^t)^\top z,\\
F(z)&\ge c_t(z)=F(z^t)-(\mu^t)^\top(z-z^t),\qquad
c_t(z^t)=F(z^t).
\end{aligned}
\tag{9}
$$

Dual feasibility gives the lower bound, and dual optimality gives equality
at $z^t$. Thus this is a supporting affine function, not merely an
arbitrary lower bound; no differentiability of $F$ is required.
The next selection and the master lower bound solve

$$
\begin{aligned}
(z^{t+1},\eta^{t+1})\in
\operatorname*{arg\,min}_{z\in\mathcal Z_d,\ \eta\ge0}\quad&\eta\\
\text{subject to}\quad&
\eta\ge F(z^s)-(\mu^s)^\top(z-z^s),\qquad s=1,\ldots,t.
\end{aligned}
\tag{10}
$$

With $U_t=\min_{s\le t}F(z^s)$, all retained cuts lie below $F$, so

$$
\eta^{t+1}\le\mathrm{OPT}\le U_t.
$$

Stop when $U_t-\eta^{t+1}\le\varepsilon$ and return the best evaluated
selection. Appendix A recovers its detector from a solution
of the finite convex problem.

**Algorithm 1. Exact cutting-plane method for variable selection.**

**Input:** Problem data, budget $d$, tolerance $\varepsilon>0$, and $z^1\in\mathcal Z_d$.

**Output:** A subset $S^*$, its optimal detector $f^*$, and a risk interval of width at most $\varepsilon$.

1. $t\gets1$, $U\gets+\infty$.

2. **while** true **do**

    3. Solve (8) at $z^t$ and take an optimizer $(\Pi^t,\alpha^t,\mu^t)$.

    4. **if** $F(z^t)<U$ **then**

        5. $(U,z^*)\gets(F(z^t),z^t)$.

    6. **end if**

    7. Retain the cut (9) with index $t$.

    8. Solve (10) to obtain $(z^{t+1},\eta^{t+1})$.

    9. **if** $U-\eta^{t+1}\le\varepsilon$ **then**

        10. $S^*\gets\{j:z_j^*=1\}$.

        11. Solve (4) for $S^*$ and construct $g^*$ by (14).

        12. Set $f^*(x)=g^*(x[S^*])$ and

        13. **return** $(S^*,f^*,[\eta^{t+1},U])$.

    14. **end if**

    15. $t\gets t+1$.

16. **end while**

## Appendix A. Finite reformulation

**Proposition A.1 (Finite reformulation).**

For every $S\subseteq[D]$, the minimum in (4) is attained and

$$
\begin{aligned}
V(S)
&=\min_{u\in\mathbb R^{2n}}\sum_{k=0}^1
\max_{\pi^k\in\mathcal T_k(S)}
\sum_{i=1}^n\sum_{r=1}^{2n}\pi^k_{ir}\ell_k(u_r)\\
&=\max_{\substack{\pi^0\in\mathcal T_0(S)\\
                  \pi^1\in\mathcal T_1(S)}}
\sum_{r=1}^{2n}\inf_{s\in\mathbb R}
\{p_{0r}\ell(-s)+p_{1r}\ell(s)\}.
\end{aligned}
\tag{11}
$$

Here $p_{kr}=\sum_{i=1}^n\pi^k_{ir}$.
An optimal solution of (4) also yields an optimal detector
$f(x)=g(x[S])$ on $\mathbb R^D$.

**Proof.**

For fixed $u$ and class $k$, dualize the source-mass constraints of the
transport LP with $a_i^k\in\mathbb R$ and its budget constraint with
$\lambda_k\ge0$. Its Lagrangian is

$$
\begin{aligned}
L_k={}&\sum_{i,r}\pi^k_{ir}\ell_k(u_r)
+\sum_i a_i^k\left(\frac1n-\sum_r\pi^k_{ir}\right)
+\lambda_k\left(\rho_k-\sum_{i,r}c_{ir}^{k,S}\pi^k_{ir}\right)\\
={}&\frac1n\sum_i a_i^k+\rho_k\lambda_k
+\sum_{i,r}\pi^k_{ir}
 [\ell_k(u_r)-a_i^k-\lambda_kc_{ir}^{k,S}].
\end{aligned}
\tag{12}
$$

The supremum over $\pi^k\ge0$ is finite exactly when every bracket is
nonpositive. Linear programming duality therefore gives

$$
\begin{aligned}
\max_{\pi^k\in\mathcal T_k(S)}
\sum_{i,r}\pi^k_{ir}\ell_k(u_r)
=\min_{a^k\in\mathbb R^n,\,\lambda_k\ge0}\quad&
\frac1n\sum_i a_i^k+\rho_k\lambda_k\\
\text{subject to}\quad&
\ell_k(u_r)\le a_i^k+\lambda_kc_{ir}^{k,S},
\quad i\in[n],\ r\in[2n].
\end{aligned}
\tag{13}
$$

Consequently, the infimum $W$ in (4) equals the infimum of
the first line of (11).

For any $f(x)=g(x[S])$, take $u_r=g(x_r[S])$. A plan
$\pi^k\in\mathcal T_k(S)$ defines a distribution in $\mathcal P_k$ by
moving mass $\pi^k_{ir}$ from $x_i^{(k)}$ to the point whose selected
coordinates are $x_r[S]$ and whose other coordinates remain
$x_i^{(k)}[S^c]$. Its transport cost is $\sum_{i,r}c_{ir}^{k,S}\pi^k_{ir}$,
and its expected loss is $\sum_{i,r}\pi^k_{ir}\ell_k(u_r)$. Hence

$$
W\le\sum_{k=0}^1\max_{\pi^k\in\mathcal T_k(S)}
\sum_{i,r}\pi^k_{ir}\ell_k(g(x_r[S]))\le R(f).
$$

Taking the infimum over $f\in\mathcal F_S$ proves $W\le V(S)$.

For the reverse inequality, let $(a,u,\lambda)$ be feasible in
(4). For $y\in\mathbb R^{|S|}$, define

$$
\begin{gathered}
E_k(y)=\min_{i\in[n]}
\{a_i^k+\lambda_k\|x_i^{(k)}[S]-y\|_1\},\\
i_k=\min\operatorname*{arg\,min}_{i\in[n]}
\{a_i^k+\lambda_k\|x_i^{(k)}[S]-y\|_1\}.
\end{gathered}
$$

Put $r_k=\|x_{i_k}^{(k)}[S]-y\|_1$ and
$b=\|x_{i_0}^{(0)}[S]-x_{i_1}^{(1)}[S]\|_1$. Set

$$
g(y)=\begin{cases}
(1-\gamma)u_{i_0}+\gamma u_{n+i_1},\quad
\gamma=\min\{1,r_0/b\},&b>0,\\
u_{i_0},&b=0.
\end{cases}
\tag{14}
$$

For $b>0$, the triangle inequality gives
$\gamma b\le r_0$ and $(1-\gamma)b\le r_1$. Convexity and the
source--target constraints in (4) yield

$$
\begin{aligned}
\ell_0(g(y))
&\le(1-\gamma)\ell_0(u_{i_0})+\gamma\ell_0(u_{n+i_1})
\le a_{i_0}^0+\lambda_0\gamma b\le E_0(y),\\
\ell_1(g(y))
&\le(1-\gamma)\ell_1(u_{i_0})+\gamma\ell_1(u_{n+i_1})
\le a_{i_1}^1+\lambda_1(1-\gamma)b\le E_1(y).
\end{aligned}
$$

When $b=0$, the same bounds follow from the two constraints at target
$i_0$. The finite tie rule makes $g$ Borel measurable, and
$|g(y)|\le\max_r|u_r|$. For any $\mathbb P_k\in\mathcal P_k$,
integrating these bounds against a coupling from
$\widehat{\mathbb P}_k$ to $\mathbb P_k$ of cost at most $\rho_k$ gives

$$
\mathbb E_{\mathbb P_k}\ell_k(g(X[S]))
\le\frac1n\sum_i a_i^k+\rho_k\lambda_k,
\qquad R(f)\le J(a,\lambda),\quad f(x)=g(x[S]).
$$

Thus $V(S)\le W$, proving equality with the finite convex problem.

The product $\mathcal T_0(S)\times\mathcal T_1(S)$ is compact and convex.
The function $\sum_{k,i,r}\pi^k_{ir}\ell_k(u_r)$ is continuous, affine
in the plans, and convex in $u$. Finite-dimensional minimax duality gives

$$
\begin{aligned}
W
&=\inf_u\max_{\pi^0,\pi^1}
  \sum_{k,i,r}\pi^k_{ir}\ell_k(u_r)
 =\max_{\pi^0,\pi^1}\inf_u
  \sum_{k,i,r}\pi^k_{ir}\ell_k(u_r)\\
&=\max_{\pi^0,\pi^1}\sum_r\inf_s
  \{p_{0r}\ell(-s)+p_{1r}\ell(s)\},
\end{aligned}
$$

where each pair of plans ranges over the same product set. This proves
the general-loss formulation starting from the transport constraints of
(3).

Finally, $(a,u,\lambda)=(\mathbf1,0,0)$ is feasible with $J=2$.
On the sublevel $J\le2$, the diagonal constraints give
$0\le a_i^k\le2n$, while $0\le\lambda_k\le2/\rho_k$.
The remaining constraints bound both $\ell(u_r)$ and $\ell(-u_r)$.
Since $\ell(t)\to\infty$ as $t\to\infty$, this sublevel is compact,
so the minimum is attained. The recovered detector then attains $V(S)$,
and its finite scores attain the first minimum in
(11). The last maximum is attained because its
objective is upper semicontinuous on the compact transport set.

$\square$

## Appendix B. Duality

**Proposition B.1 (Exactness and dual representation).**

(i) For every $S\subseteq[D]$, $F(\mathbf1_S)=V(S)$.
For every $z\in[0,1]^D$, the primal minimum and the normalized dual
maximum are attained, and

$$
0<F(z)=\max_{\substack{(\Pi,\alpha,\mu)\in\mathcal D\\
\alpha+\mu^\top z=1}}H(\Pi)\le2.
\tag{15}
$$

(ii) The same value is attained in

$$
F(z)=\max_{(\Pi,\alpha,\mu)\in\mathcal D}
\{2\sqrt{H(\Pi)}-\alpha-\mu^\top z\}.
\tag{16}
$$

If $(\bar\Pi,\bar\alpha,\bar\mu)$ is a normalized optimizer, then
an optimizer of (16) is

$$
(\Pi,\alpha,\mu)=F(z)(\bar\Pi,\bar\alpha,\bar\mu),\qquad
H(\Pi)=F(z)^2.
\tag{17}
$$

(iii) The function $F$ is convex on $[0,1]^D$. An optimizer
$(\Pi^t,\alpha^t,\mu^t)$ of (16) at $z^t$ gives

$$
\begin{aligned}
c_t(z)&=2\sqrt{H(\Pi^t)}-\alpha^t-(\mu^t)^\top z\le F(z),\\
c_t(z^t)&=F(z^t),\qquad
F(z)\ge F(z^t)-(\mu^t)^\top(z-z^t).
\end{aligned}
\tag{18}
$$

Hence $-\mu^t\in\partial F(z^t)$, with the subgradient inequality
understood on $[0,1]^D$.

**Proof.**

(i) Taking the source itself as the target in the loss
constraints gives $a_i^k\ge0$ in both primal formulations.
For a feasible $(a,u,\lambda)$ of (4), set
$z=\mathbf1_S$ and $w_{kj}=\lambda_kz_j$. Then

$$
\ell_k(u_r)\le a_i^k+\sum_{j=1}^D\Delta_{ir,j}^kw_{kj},\qquad
\sum_{k=0}^1\rho_kw_{kj}
=z_j\sum_{k=0}^1\rho_k\lambda_k\le z_jJ(a,\lambda).
$$

Conversely, feasibility in (5) with $z=\mathbf1_S$
and $\rho_k>0$ gives

$$
w_{kj}=0\quad(j\notin S),\qquad
\ell_k(u_r)\le a_i^k+\sum_{j\in S}\Delta_{ir,j}^kw_{kj}
\le a_i^k+\lambda_kc_{ir}^{k,S}.
$$

Thus both formulations have the same feasible $(a,u,\lambda)$ after
eliminating $w$, and the same objective. This proves
$F(\mathbf1_S)=V(S)$.

Assign nonnegative multipliers $\Pi_{ir}^k$, $\beta_{kj}$, and $\mu_j$
to the loss constraints, the inequalities $w_{kj}\le\lambda_k$, and
the selection constraints in (5), respectively.
Keeping $a,u$ free and $\lambda,w\ge0$, write
$\alpha=1-\mu^\top z$. The Lagrangian is

$$
\begin{aligned}
\mathcal L={}&J(a,\lambda)
+\sum_{k,i,r}\Pi_{ir}^k
  \left[\ell_k(u_r)-a_i^k-\sum_j\Delta_{ir,j}^kw_{kj}\right]\\
&+\sum_{k,j}\beta_{kj}(w_{kj}-\lambda_k)
+\sum_j\mu_j\left[\sum_k\rho_kw_{kj}-z_jJ(a,\lambda)\right]\\
={}&\sum_{k,i}a_i^k\left(\frac\alpha n-\sum_r\Pi_{ir}^k\right)
+\sum_{r=1}^{2n}\sum_{k=0}^1\sum_{i=1}^n
  \Pi_{ir}^k\ell_k(u_r)\\
&+\sum_k\lambda_k\left(\alpha\rho_k-\sum_j\beta_{kj}\right)
+\sum_{k,j}w_{kj}
  \left[\beta_{kj}+\rho_k\mu_j-\sum_{i,r}\Pi_{ir}^k\Delta_{ir,j}^k\right].
\end{aligned}
\tag{19}
$$

Minimization over $a,\lambda,w$ is finite if and only if

$$
\begin{gathered}
\sum_{r=1}^{2n}\Pi_{ir}^k=\frac\alpha n
\quad(k=0,1,\ i\in[n]),\qquad
\sum_{j=1}^D\beta_{kj}\le\alpha\rho_k\quad(k=0,1),\\
\beta_{kj}\ge
\left(\sum_{i=1}^n\sum_{r=1}^{2n}\Pi_{ir}^k\Delta_{ir,j}^k
-\rho_k\mu_j\right)_+
\quad(k=0,1,\ j\in[D]).
\end{gathered}
\tag{20}
$$

The row equalities force $\alpha\ge0$. Minimization over $u$ separates
across the $2n$ targets and yields

$$
\inf_u\sum_{r=1}^{2n}\sum_{k=0}^1\sum_{i=1}^n
\Pi_{ir}^k\ell_k(u_r)
=\sum_{r=1}^{2n}\inf_{s\in\mathbb R}
\sum_{k=0}^1\sum_{i=1}^n\Pi_{ir}^k\ell_k(s)=H(\Pi).
$$

Eliminating $\beta$ from (20) gives

$$
\sum_{j=1}^D
\left(\sum_{i=1}^n\sum_{r=1}^{2n}\Pi_{ir}^k\Delta_{ir,j}^k
-\rho_k\mu_j\right)_+\le\alpha\rho_k,\qquad k=0,1.
$$

These are precisely the constraints defining $\mathcal D$;
$\alpha=1-\mu^\top z$ supplies the normalization in
(15).

To establish strong duality also when some $z_j=0$, eliminate the
corresponding variables $w_{kj}$ and their redundant cap and selection
constraints. The reduced problem has the strictly feasible point

$$
u=0,\qquad a_i^k=2,\qquad\lambda_k=1,\qquad
w_{kj}=z_j/2\quad(z_j>0).
$$

Indeed, $\ell_k(0)=1<2$, $0<w_{kj}<1$, and

$$
\sum_{k=0}^1\rho_kw_{kj}
=\frac{z_j}{2}(\rho_0+\rho_1)
<z_j(4+\rho_0+\rho_1)=z_jJ(a,\lambda)
\quad(z_j>0).
$$

Slater's theorem gives equality of the reduced primal and dual values
and attainment of the dual maximum. For each eliminated coordinate,
restore the full dual by setting

$$
\mu_j=\max_{k=0,1}\frac1{\rho_k}
\sum_{i=1}^n\sum_{r=1}^{2n}\Pi_{ir}^k\Delta_{ir,j}^k,
\qquad\beta_{kj}=0\quad(z_j=0).
$$

This preserves both the objective and the normalization and satisfies
(20). Hence the normalized dual attains $F(z)$
for every $z\in[0,1]^D$.

The feasible point $a_i^k=1$, $u=0$, $\lambda=w=0$ gives $F(z)\le2$.
On the sublevel $J\le2$, nonnegativity gives
$a_i^k\le2n$, $\lambda_k\le2/\rho_k$, and $w_{kj}\le2/\rho_k$.
The loss constraints then bound both $\ell(u_r)$ and $\ell(-u_r)$.
Convexity, monotonicity, and $\ell(0)=1$ together with
$\lim_{t\to-\infty}\ell(t)=0$ imply
$\ell(t)\to\infty$ as $t\to\infty$. Thus the sublevel is compact,
and the primal minimum is attained. If its value were zero, then
$a=\lambda=w=0$, contradicting

$$
2=2\ell(0)\le\ell(-u_r)+\ell(u_r)\le0.
$$

This proves (15).

(ii) The set $\mathcal D$ is a cone and
$H(\tau\Pi)=\tau H(\Pi)\ge0$ for $\tau\ge0$.
For $(\Pi,\alpha,\mu)\in\mathcal D$, let
$b=\alpha+\mu^\top z$. If $b>0$, dividing the tuple by $b$ and
applying part (i) gives $H(\Pi)/b\le F(z)$. Consequently,

$$
2\sqrt{H(\Pi)}-b
=\frac{H(\Pi)}b-
\left(\sqrt{\frac{H(\Pi)}b}-\sqrt b\right)^2\le F(z).
\tag{21}
$$

If $b=0$, then $\alpha=0$ and the row equalities imply $\Pi=0$,
so the affine objective is zero. Conversely, for a normalized
optimizer $(\bar\Pi,\bar\alpha,\bar\mu)$,

$$
H(F(z)\bar\Pi)=F(z)^2,\qquad
2\sqrt{H(F(z)\bar\Pi)}
-F(z)(\bar\alpha+\bar\mu^\top z)=F(z).
$$

The affine upper bound is therefore attained, proving
(16) and (17).

(iii) The feasible set $\mathcal D$ is independent of $z$,
so (16) expresses $F$ as a maximum of affine
functions and proves convexity. Feasibility of
$(\Pi^t,\alpha^t,\mu^t)$ gives $c_t(z)\le F(z)$ for every
$z\in[0,1]^D$, while optimality at $z^t$ gives
$c_t(z^t)=F(z^t)$. Therefore,

$$
F(z)\ge c_t(z)=c_t(z^t)-(\mu^t)^\top(z-z^t)
=F(z^t)-(\mu^t)^\top(z-z^t).
$$

$\square$

## References

[1] Dimitris Bertsimas, Ryan Cory-Wright, and Jean Pauphilet. A unified approach to mixed-integer optimization problems with logical constraints. *SIAM Journal on Optimization*, 31(3):2340–2367, 2021.

[2] Rui Gao, Liyan Xie, Yao Xie, and Huan Xu. Robust hypothesis testing using Wasserstein uncertainty sets. In *Advances in Neural Information Processing Systems*, volume 31. Curran Associates, Inc., 2018. For Lemma 2 and Eq. (7), see the related manuscript by Liyan Xie, Rui Gao, and Yao Xie, *Robust Hypothesis Testing with Wasserstein Uncertainty Sets*, arXiv:2105.14348, 2021.
