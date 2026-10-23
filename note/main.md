# Variable Selection for Wasserstein Robust Hypothesis Testing

## Abstract

We derive an exact finite convex formulation for Wasserstein robust testing with coordinate selection. A uniform risk bound introduces the selection variables without changing binary optimal values. The resulting convex value function admits affine dual cuts for optimization over coordinate subsets.

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
For $k=0,1$, assume we have $n_k$ available observations in the group $k$, which are denoted as $\{x_i^{(k)}\}_{i=1}^{n_k}$.
We denote the corresponding empirical distribution as $\widehat{\mathbb{P}}_k=\frac1{n_k}\sum_{i=1}^{n_k}\delta_{x_i^{(k)}}$, and construct the Wasserstein ambiguity set as

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

## 2. Finite reformulation

Assume $n_0,n_1\ge1$, $\rho_0,\rho_1>0$, and that $\ell$ is finite
on $\mathbb R$. All distributions have finite first moments.
Throughout, $i\in[n_0]$ indexes class $0$, $j\in[n_1]$ indexes class $1$,
and $m\in[D]$ indexes a coordinate. For a fixed selection, define

$$
V(S)=\min_{f\in\mathcal F_S}R(f),\qquad
c_{ij}(S)=\|x_i^{(0)}[S]-x_j^{(1)}[S]\|_1.
$$

The least-favorable-distribution viewpoint of [2]
suggests transporting empirical mass between observations. We derive the
finite problem directly for the generating loss in Section 1.
For observation $r\in[n_k]$ of class $k$, introduce an auxiliary score
$u_r^{(k)}$ and a source loss bound $a_r^{(k)}$; let $\lambda_k\ge0$ price the transport
budget of class $k$. Write

$$
J(a,\lambda)=\frac1{n_0}\sum_i a_i^{(0)}+
\frac1{n_1}\sum_j a_j^{(1)}+\rho_0\lambda_0+\rho_1\lambda_1.
$$

**Proposition 2.1 (Finite reformulation).**

For every $S\subseteq[D]$, the following convex problem attains $V(S)$:

$$
\begin{aligned}
\min_{a,u,\lambda}\quad&J(a,\lambda)\\
\text{subject to}\quad&\ell(-u_i^{(0)})\le a_i^{(0)},&&i\in[n_0],\\
&\ell(u_j^{(1)})\le a_j^{(1)},&&j\in[n_1],\\
&\ell(-u_j^{(1)})\le a_i^{(0)}+\lambda_0c_{ij}(S),
&&i\in[n_0],\ j\in[n_1],\\
&\ell(u_i^{(0)})\le a_j^{(1)}+\lambda_1c_{ij}(S),
&&i\in[n_0],\ j\in[n_1],\\
&a^{(k)},u^{(k)}\in\mathbb R^{n_k},\quad\lambda_k\ge0,&&k=0,1.
\end{aligned}
\tag{3}
$$

An optimal solution determines a measurable detector on the whole
observation space with risk $V(S)$.

The first two constraints bound the losses of mass that stays at its
source. The last two bound transfers to the opposite class. For example,
class-$0$ mass moved from $x_i^{(0)}[S]$ to $x_j^{(1)}[S]$ has loss
$\ell(-u_j^{(1)})$, bounded by $a_i^{(0)}$ plus the priced cost
$\lambda_0c_{ij}(S)$. Appendix A proves exactness in both
directions: these restricted transports give a lower bound on the original
risk, and every feasible finite solution gives a detector whose risk is
at most its objective. This also covers coincident projections.

The zero detector has risk $2$. Since the diagonal constraints imply
$a^{(k)}\ge0$, an optimum satisfies

$$
\rho_0\lambda_0+\rho_1\lambda_1\le J(a,\lambda)=V(S)\le2.
\tag{4}
$$

This bound will let a binary variable switch a coordinate on or off.

## 3. Selection variables and duality

Let $z=\mathbf1_S$, and define

$$
\mathcal Z_d=\{z\in\{0,1\}^D:\mathbf1^\top z\le d\},\qquad
\Delta_{ij,m}=|x_i^{(0)}[m]-x_j^{(1)}[m]|.
$$

Then $c_{ij}(S)=\sum_m\Delta_{ij,m}z_m$.
Introduce $w_{km}$ to replace the products $\lambda_kz_m$.
The constraints $0\le w_{km}\le\lambda_k$ and
$\rho_0w_{0m}+\rho_1w_{1m}\le2z_m$ force $w_{0m}=w_{1m}=0$ when
$z_m=0$. When $z_m=1$, the choice $w_{km}=\lambda_k$ is permitted
at an optimum by (4).
For $z\in[0,1]^D$, define

$$
\begin{aligned}
F(z)=\min_{a,u,\lambda,w}\quad&J(a,\lambda)\\
\text{subject to}\quad&
\ell(-u_i^{(0)})\le a_i^{(0)},\quad
\ell(u_j^{(1)})\le a_j^{(1)},\\
&\ell(-u_j^{(1)})\le a_i^{(0)}+\sum_m\Delta_{ij,m}w_{0m},\\
&\ell(u_i^{(0)})\le a_j^{(1)}+\sum_m\Delta_{ij,m}w_{1m},\\
&0\le w_{km}\le\lambda_k,\qquad k=0,1,\ m\in[D],\\
&\rho_0w_{0m}+\rho_1w_{1m}\le2z_m,\qquad m\in[D],\\
&a^{(k)},u^{(k)}\in\mathbb R^{n_k},\quad\lambda_k\ge0,
\qquad k=0,1.
\end{aligned}
\tag{5}
$$

The loss constraints hold for every $i\in[n_0],j\in[n_1]$.
This has the two-stage logical structure of problems (2.1)--(2.2) in
[1]: each coordinate activates the pair
$(w_{0m},w_{1m})$ of a convex continuous problem, and the outer problem
chooses at most $d$ such pairs. The bound $2$ follows from the risk;
it is not a tuning parameter.

**Proposition 3.1 (Selection formulation).**

The minimum in (5) is attained. The function $F$ is convex
on $[0,1]^D$, and

$$
F(\mathbf1_S)=V(S),\qquad
\min_{|S|\le d}V(S)=\min_{z\in\mathcal Z_d}F(z).
\tag{6}
$$

**Proof.**

An optimum of (3) extends to (5) by
$w_{km}=\lambda_kz_m$ at $z=\mathbf1_S$, using
(4). Conversely, feasibility at that binary point
implies $w_{km}=0$ outside $S$ and

$$
\sum_m\Delta_{ij,m}w_{km}\le
\lambda_k\sum_{m\in S}\Delta_{ij,m}=\lambda_kc_{ij}(S).
$$

It therefore implies feasibility in (3) with the same
objective. This proves equality of the binary optimal values.
For any $z$, the zero detector gives a feasible objective of $2$.
On $J\le2$, the source bounds and prices are bounded; both signs of
each score are bounded as in Appendix A, and
$0\le w_{km}\le\lambda_k$. Thus an optimum exists.
All constraints in (5) are jointly convex, including in
$z$. A convex combination of optimal solutions at $z,z'$ is feasible
at $tz+(1-t)z'$, so

$$
F(tz+(1-t)z')\le tF(z)+(1-t)F(z'),\qquad 0\le t\le1.
$$

$\square$

### 3.1. Dual representation

Let $\Gamma_{ij}^{(0)}\ge0$ be the multiplier of the class-$0$ cross
constraint and $\Gamma_{ij}^{(1)}\ge0$ that of the class-$1$ cross
constraint. They represent mass transported from source $i$ to target
$j$, and from source $j$ to target $i$, respectively. Write
$\Gamma=(\Gamma^{(0)},\Gamma^{(1)})$.
At a location with class masses $p,q\ge0$, the best weighted loss is

$$
h(p,q)=\inf_{v\in\mathbb R}\{p\ell(-v)+q\ell(v)\}.
$$

At a class-$0$ observation, the remaining class-$0$ mass is
$1/n_0-\sum_j\Gamma_{ij}^{(0)}$, and the incoming class-$1$ mass is
$\sum_j\Gamma_{ij}^{(1)}$. Applying the same reasoning to class $1$
gives the total optimized loss

$$
\begin{aligned}
H(\Gamma)={}&\sum_i
h\left(\frac1{n_0}-\sum_j\Gamma_{ij}^{(0)},\sum_j\Gamma_{ij}^{(1)}\right)\\
&+\sum_j
h\left(\sum_i\Gamma_{ij}^{(0)},\frac1{n_1}-\sum_i\Gamma_{ij}^{(1)}\right).
\end{aligned}
\tag{7}
$$

Let $\mu_m\ge0$ multiply the coordinate constraint
$\rho_0w_{0m}+\rho_1w_{1m}\le2z_m$, and set

$$
\mathcal D=\left\{(\Gamma,\mu)\ge0:
\begin{aligned}
&\sum_j\Gamma_{ij}^{(0)}\le1/n_0&&(i\in[n_0]),\\
&\sum_i\Gamma_{ij}^{(1)}\le1/n_1&&(j\in[n_1]),\\
&\sum_m\left(\sum_{i,j}\Gamma_{ij}^{(k)}\Delta_{ij,m}
-\rho_k\mu_m\right)_+\le\rho_k&&(k=0,1)
\end{aligned}\right\},
\tag{8}
$$

where $(v)_+=\max\{v,0\}$ and every sum uses the index ranges defined
above. The first two constraints limit outgoing mass. The last two
arise by minimizing the Lagrangian over $\lambda,w$, as shown in
Appendix B. The set $\mathcal D$ is independent of $z$.

**Proposition 3.2 (Dual representation).**

For every $z\in[0,1]^D$, the following maximum is attained:

$$
F(z)=\max_{(\Gamma,\mu)\in\mathcal D}
\{H(\Gamma)-2\mu^\top z\}.
\tag{9}
$$

An optimizer $(\Gamma^t,\mu^t)$ at $z^t$ gives the supporting function

$$
\begin{aligned}
c_t(z)&=H(\Gamma^t)-2(\mu^t)^\top z\\
&=F(z^t)-2(\mu^t)^\top(z-z^t)\le F(z),
\end{aligned}
\tag{10}
$$

with $c_t(z^t)=F(z^t)$.

The function $h$ is concave in its weights, so $H$ is concave, and
$\mathcal D$ is convex. Thus the dual at fixed $z$ is a convex
optimization problem. Its affine dependence on $z$ gives the
cutting-plane construction underlying Theorem 2.5 and Algorithm 3.1 of
[1]. The capacity constraints of this model are
handled directly in Appendix B.

## 4. Cutting-plane algorithm

Write $\mathrm{OPT}=\min_{z\in\mathcal Z_d}F(z)$.
After evaluating $z^1,\ldots,z^t$ and obtaining their dual optimizers,
let $U_t=\min_{s\le t}F(z^s)$ and solve

$$
\begin{aligned}
(z^{t+1},L_t)\in\operatorname*{arg\,min}_{z\in\mathcal Z_d,\,\eta\ge0}
\quad&\eta\\
\text{subject to}\quad&
\eta\ge F(z^s)-2(\mu^s)^\top(z-z^s),\qquad s=1,\ldots,t.
\end{aligned}
\tag{11}
$$

Every retained cut is a global lower bound and is exact at its evaluated
selection. Consequently,

$$
L_t\le\mathrm{OPT}\le U_t.
\tag{12}
$$

Starting from any $z^1\in\mathcal Z_d$, alternate the convex subproblem
(5)--(9) and the binary linear master
(11). Stop when $U_t-L_t\le\varepsilon$ and return the
best evaluated selection. Its detector is recovered by
(15), with risk $U_t$ and optimality gap at most
$\varepsilon$. For an inexact inner solve, use the actual feasible dual
expression $H(\Gamma^t)-2(\mu^t)^\top z$ as the cut and a primal
feasible objective as the upper bound; the master must provide a global
lower bound. The binary formulation is exact, although its continuous
relaxation can be weak.

## Appendix A. Proof of the finite reformulation

**Proof of Proposition 2.1.**

Let $W(S)$ be the infimum in (3). For $f(x)=g(x[S])$, set
$u_r^{(k)}=g(x_r^{(k)}[S])$. Restrict transport to retaining mass
$s_i^{(0)},s_j^{(1)}\ge0$ at its source or moving mass
$\Gamma_{ij}^{(0)},\Gamma_{ij}^{(1)}\ge0$ to an opposite-class
selected observation, in the directions of Section 3.1:

$$
\begin{gathered}
s_i^{(0)}+\sum_j\Gamma_{ij}^{(0)}=1/n_0,\qquad
s_j^{(1)}+\sum_i\Gamma_{ij}^{(1)}=1/n_1,\\
\sum_{i,j}c_{ij}(S)\Gamma_{ij}^{(k)}\le\rho_k,
\qquad k=0,1.
\end{gathered}
\tag{13}
$$

The class-$0$ objective is

$$
\sum_i s_i^{(0)}\ell(-u_i^{(0)})
+\sum_{i,j}\Gamma_{ij}^{(0)}\ell(-u_j^{(1)}).
$$

With free mass multipliers $a_i^{(0)}$ and budget multiplier
$\lambda_0\ge0$, its maximization Lagrangian is

$$
\begin{aligned}
&\frac1{n_0}\sum_i a_i^{(0)}+\rho_0\lambda_0
+\sum_i s_i^{(0)}[\ell(-u_i^{(0)})-a_i^{(0)}]\\
&\hspace{12pt}
+\sum_{i,j}\Gamma_{ij}^{(0)}
[\ell(-u_j^{(1)})-a_i^{(0)}-\lambda_0c_{ij}(S)].
\end{aligned}
\tag{14}
$$

Its supremum is finite exactly when both bracketed coefficients are
nonpositive. The class-$1$ calculation gives the other two constraints
in (3). Both transport LPs are feasible and bounded;
LP duality therefore identifies their maximum risk with the minimum
of $J(a,\lambda)$ at these fixed scores. Replacing only a source's
selected coordinates by the target coordinates lifts each plan to the
original ambiguity set, with cost $c_{ij}(S)$ and the same loss.
Minimizing also over $u$ gives $W(S)\le R(f)$, hence $W(S)\le V(S)$.

Conversely, for feasible $(a,u,\lambda)$ define

$$
E_k(y)=\min_{r\in[n_k]}
\{a_r^{(k)}+\lambda_k\|x_r^{(k)}[S]-y\|_1\},\qquad k=0,1,
$$

for $y\in\mathbb R^{|S|}$. Choose the smallest minimizing indices
$i=i(y)$, $j=j(y)$ for $E_0(y)$, $E_1(y)$, and set

$$
r_0=\|x_i^{(0)}[S]-y\|_1,\qquad
r_1=\|x_j^{(1)}[S]-y\|_1,\qquad b=c_{ij}(S).
$$

Define

$$
g(y)=\begin{cases}
(1-\gamma)u_i^{(0)}+\gamma u_j^{(1)},\quad
\gamma=\min\{1,r_0/b\},&b>0,\\
u_i^{(0)},&b=0.
\end{cases}
\tag{15}
$$

For $b>0$, the triangle inequality gives $\gamma b\le r_0$ and
$(1-\gamma)b\le r_1$. Convexity and the four constraints yield

$$
\begin{aligned}
\ell(-g(y))
&\le(1-\gamma)\ell(-u_i^{(0)})+\gamma\ell(-u_j^{(1)})\\
&\le a_i^{(0)}+\lambda_0\gamma b\le E_0(y),\\
\ell(g(y))
&\le(1-\gamma)\ell(u_i^{(0)})+\gamma\ell(u_j^{(1)})\\
&\le a_j^{(1)}+\lambda_1(1-\gamma)b\le E_1(y).
\end{aligned}
\tag{16}
$$

For $b=0$, the class-$0$ diagonal and class-$1$ cross constraints give
the same bounds. The finite tie rule makes $g$ Borel measurable, and
$|g|\le\max_{k,r}|u_r^{(k)}|$. For every full-space input $x$ and source,

$$
\begin{aligned}
\ell(-g(x[S]))&\le a_i^{(0)}+
\lambda_0\|x_i^{(0)}[S]-x[S]\|_1
\le a_i^{(0)}+\lambda_0\|x_i^{(0)}-x\|_1,\\
\ell(g(x[S]))&\le a_j^{(1)}+
\lambda_1\|x_j^{(1)}[S]-x[S]\|_1
\le a_j^{(1)}+\lambda_1\|x_j^{(1)}-x\|_1.
\end{aligned}
$$

For $\delta>0$ and any $\mathbb P_k\in\mathcal P_k$, integrate against
empirical-to-$\mathbb P_k$ couplings of cost at most $\rho_k+\delta$:

$$
\begin{aligned}
\mathbb E_{\mathbb P_0}\ell(-g(X[S]))
&\le\frac1{n_0}\sum_i a_i^{(0)}+\lambda_0(\rho_0+\delta),\\
\mathbb E_{\mathbb P_1}\ell(g(X[S]))
&\le\frac1{n_1}\sum_j a_j^{(1)}+\lambda_1(\rho_1+\delta).
\end{aligned}
$$

Letting $\delta\downarrow0$ and setting $f(x)=g(x[S])$ gives
$R(f)\le J(a,\lambda)$.
Thus $V(S)\le W(S)$, proving equality.

The feasible point $a^{(k)}=\mathbf1$, $u^{(k)}=0$, $\lambda=0$ has
value $2$. On $J\le2$, $0\le a_r^{(k)}\le2n_k$ and
$0\le\lambda_k\le2/\rho_k$. Each score's two losses are bounded by
its diagonal and an opposite-class constraint. Since
$\ell(t)\to\infty$ as $t\to\infty$, the scores are bounded.
This sublevel is compact, so a minimizer exists; its recovered
detector attains $V(S)$.

$\square$

## Appendix B. Proof of the dual representation

**Proof of Proposition 3.2.**

Besides $\Gamma,\mu$ from Section 3.1, assign nonnegative
multipliers $s_i^{(0)},s_j^{(1)}$ to the diagonal loss constraints and
$\beta_{km}$ to $w_{km}\le\lambda_k$. Keeping $a,u$ free and
$\lambda,w\ge0$, the minimization Lagrangian is

$$
\begin{aligned}
\mathcal L={}&J(a,\lambda)
+\sum_i s_i^{(0)}[\ell(-u_i^{(0)})-a_i^{(0)}]
+\sum_j s_j^{(1)}[\ell(u_j^{(1)})-a_j^{(1)}]\\
&+\sum_{i,j}\Gamma_{ij}^{(0)}
\left[\ell(-u_j^{(1)})-a_i^{(0)}-\sum_m\Delta_{ij,m}w_{0m}\right]\\
&+\sum_{i,j}\Gamma_{ij}^{(1)}
\left[\ell(u_i^{(0)})-a_j^{(1)}-\sum_m\Delta_{ij,m}w_{1m}\right]\\
&+\sum_{k,m}\beta_{km}(w_{km}-\lambda_k)
+\sum_m\mu_m(\rho_0w_{0m}+\rho_1w_{1m}-2z_m).
\end{aligned}
\tag{17}
$$

The infimum over $a$ is finite exactly when

$$
s_i^{(0)}+\sum_j\Gamma_{ij}^{(0)}=1/n_0,\qquad
s_j^{(1)}+\sum_i\Gamma_{ij}^{(1)}=1/n_1.
$$

Eliminating $s\ge0$ gives the first two constraints of
(8). The score-dependent terms are

$$
\begin{aligned}
&\sum_i\left[s_i^{(0)}\ell(-u_i^{(0)})+
\left(\sum_j\Gamma_{ij}^{(1)}\right)\ell(u_i^{(0)})\right]
\\&\quad+\sum_j\left[\left(\sum_i\Gamma_{ij}^{(0)}\right)\ell(-u_j^{(1)})+
s_j^{(1)}\ell(u_j^{(1)})\right],
\end{aligned}
$$

whose infimum is $H(\Gamma)$ in (7). The coefficients
of $\lambda,w\ge0$ are

$$
\lambda_k:\quad\rho_k-\sum_m\beta_{km},\qquad
w_{km}:\quad\beta_{km}+\rho_k\mu_m-
\sum_{i,j}\Gamma_{ij}^{(k)}\Delta_{ij,m}.
$$

They are nonnegative, with $\beta\ge0$, exactly when

$$
\sum_m\beta_{km}\le\rho_k,\qquad
\beta_{km}\ge
\left(\sum_{i,j}\Gamma_{ij}^{(k)}\Delta_{ij,m}-\rho_k\mu_m\right)_+.
$$

Eliminating $\beta$ gives the last two constraints of
(8), leaving the dual objective
$H(\Gamma)-2\mu^\top z$.

For fixed $z\in[0,1]^D$, eliminate $w_{km}=0$ wherever $z_m=0$
and their cap and capacity constraints. The remaining system has the
Slater point

$$
u^{(k)}=0,\qquad a^{(k)}=2\mathbf1,\qquad
w_{km}=\frac{z_m}{2\rho_k},\qquad
\lambda_k=1+\frac1{2\rho_k}:
$$

all loss constraints are strict, $0<w_{km}<\lambda_k$, and
$\rho_0w_{0m}+\rho_1w_{1m}=z_m<2z_m$.
Strong duality and dual attainment follow. For eliminated coordinates,
restore

$$
\mu_m=\max_{k=0,1}\frac1{\rho_k}
\sum_{i,j}\Gamma_{ij}^{(k)}\Delta_{ij,m},\qquad
\beta_{km}=0\quad(z_m=0).
$$

All coefficient conditions hold and the objective is unchanged;
the full dual therefore attains $F(z)$.

Any $(\Gamma^t,\mu^t)\in\mathcal D$ gives the global lower bound
$H(\Gamma^t)-2(\mu^t)^\top z\le F(z)$. Optimality at $z^t$ gives
equality there, hence

$$
F(z)\ge F(z^t)-2(\mu^t)^\top(z-z^t).
$$

$\square$

## References

[1] Dimitris Bertsimas, Ryan Cory-Wright, and Jean Pauphilet. A unified approach to mixed-integer optimization problems with logical constraints. *SIAM Journal on Optimization*, 31(3):2340–2367, 2021.

[2] Rui Gao, Liyan Xie, Yao Xie, and Huan Xu. Robust hypothesis testing using Wasserstein uncertainty sets. In *Advances in Neural Information Processing Systems*, volume 31. Curran Associates, Inc., 2018.
