# Variable Selection for Wasserstein Robust Hypothesis Testing

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

## 2. Convex formulation

Assume $n_0,n_1\ge1$, $\rho_0,\rho_1>0$, and that $\ell$ is finite
on $\mathbb R$. All distributions have finite first moments. Fix
$S\subseteq[D]$ and write

$$
V(S)=\min_{f\in\mathcal F_S}R(f),\qquad
\mathcal X=\{x_i^{(0)}:i\in[n_0]\}\cup
\{x_j^{(1)}:j\in[n_1]\}.
$$

The following lemma extends the finite-support formulation of
[3, Lemma 2, equation (7)] to the generating loss
in Section 1. Its proof, including projection and
detector extension, is given in Appendix A.

**Lemma 2.1 (Finite LFD reformulation).**

The fixed-selection risk is

$$
\begin{aligned}
V(S)=\max_{p,\gamma\ge0}\quad&
\sum_{y\in\mathcal X}\inf_{v\in\mathbb R}
\{p_0(y)\ell(-v)+p_1(y)\ell(v)\}\\
\text{subject to}\quad&
\sum_{y\in\mathcal X}\gamma_{i,y}^{(k)}=\frac1{n_k},
&&i\in[n_k],\ k=0,1,\\
&p_k(y)=\sum_{i=1}^{n_k}\gamma_{i,y}^{(k)},
&&y\in\mathcal X,\ k=0,1,\\
&\sum_{i=1}^{n_k}\sum_{y\in\mathcal X}
\gamma_{i,y}^{(k)}\|x_i^{(k)}[S]-y[S]\|_1\le\rho_k,
&&k=0,1.
\end{aligned}
\tag{3}
$$

Here $\gamma_{i,y}^{(k)}$ is the mass transported from
$x_i^{(k)}[S]$ to $y[S]$, and $p_k(y)$ is the target mass.

Write $g_y$ for the score assigned to $y[S]$. Substituting the target
masses gives

$$
\sum_{y\in\mathcal X}\inf_{v\in\mathbb R}
\{p_0(y)\ell(-v)+p_1(y)\ell(v)\}
=\inf_g\left\{
\sum_{i,y}\gamma_{i,y}^{(0)}\ell(-g_y)
+\sum_{j,y}\gamma_{j,y}^{(1)}\ell(g_y)\right\}.
$$

The objective is convex in $g$ and linear in $\gamma$, and the transport
set is compact and convex. The finite-dimensional minimax theorem gives

$$
V(S)=\inf_g\left\{
\max_{\gamma^{(0)}}\sum_{i,y}\gamma_{i,y}^{(0)}\ell(-g_y)
+\max_{\gamma^{(1)}}\sum_{j,y}\gamma_{j,y}^{(1)}\ell(g_y)
\right\},
$$

where each maximum satisfies its source-mass and transport-budget
constraints in (3).

For a fixed score vector $g$, associate multipliers
$a_i^{(0)}\in\mathbb R$ and $\lambda_0\ge0$ with the mass and
transport constraints in (3), respectively. The Lagrangian
of the maximization over $\gamma^{(0)}$ is

$$
\begin{aligned}
\mathcal L_0={}&\sum_{i,y}\gamma_{i,y}^{(0)}\ell(-g_y)
+\sum_i a_i^{(0)}\left(\frac1{n_0}-\sum_y\gamma_{i,y}^{(0)}\right)\\
&+\lambda_0\left(\rho_0-\sum_{i,y}\gamma_{i,y}^{(0)}
\|x_i^{(0)}[S]-y[S]\|_1\right)\\
={}&\frac1{n_0}\sum_i a_i^{(0)}+\rho_0\lambda_0\\
&+\sum_{i,y}\gamma_{i,y}^{(0)}
\left[\ell(-g_y)-a_i^{(0)}
-\lambda_0\|x_i^{(0)}[S]-y[S]\|_1\right].
\end{aligned}
\tag{4}
$$

Thus

$$
\sup_{\gamma^{(0)}\ge0}\mathcal L_0<\infty
\iff
\ell(-g_y)\le a_i^{(0)}+\lambda_0\|x_i^{(0)}[S]-y[S]\|_1,
\quad i\in[n_0],\ y\in\mathcal X.
$$

Linear programming duality applied to both classes gives

$$
\begin{aligned}
V(S)=\min_{g,a,\lambda}\quad&
\frac1{n_0}\sum_i a_i^{(0)}+\frac1{n_1}\sum_j a_j^{(1)}
+\rho_0\lambda_0+\rho_1\lambda_1\\
\text{subject to}\quad&
\ell(-g_y)\le a_i^{(0)}+\lambda_0\|x_i^{(0)}[S]-y[S]\|_1,
&&i\in[n_0],\ y\in\mathcal X,\\
&\ell(g_y)\le a_j^{(1)}+\lambda_1\|x_j^{(1)}[S]-y[S]\|_1,
&&j\in[n_1],\ y\in\mathcal X,\\
&g\in\mathbb R^{\mathcal X},\quad a^{(k)}\in\mathbb R^{n_k},
\quad\lambda_k\ge0,&&k=0,1.
\end{aligned}
\tag{5}
$$

Denote this objective by $J(a,\lambda)$.
The equivalence with the risk over measurable detectors is established
in Appendix A.

Let $z=\mathbf1_S$ and
$\mathcal Z_d=\{z\in\{0,1\}^D:\mathbf1^\top z\le d\}$.
The transport cost decomposes directly into coordinates:

$$
\|x_i^{(k)}[S]-y[S]\|_1
=\sum_{m=1}^D z_m|x_i^{(k)}[m]-y[m]|.
$$

Introduce $w_{km}$ to encode coordinate activation and extend the
selection indicators to $z\in[0,1]^D$. Define

$$
\begin{aligned}
F(z)=\min_{g,a,\lambda,w}\quad&J(a,\lambda)\\
\text{subject to}\quad&
\ell(-g_y)\le a_i^{(0)}+\sum_m w_{0m}|x_i^{(0)}[m]-y[m]|,
&&i\in[n_0],\ y\in\mathcal X,\\
&\ell(g_y)\le a_j^{(1)}+\sum_m w_{1m}|x_j^{(1)}[m]-y[m]|,
&&j\in[n_1],\ y\in\mathcal X,\\
&0\le w_{km}\le\lambda_k,&&k=0,1,\ m\in[D],\\
&\rho_0w_{0m}+\rho_1w_{1m}\le2z_m,&&m\in[D],\\
&g\in\mathbb R^{\mathcal X},\quad a^{(k)}\in\mathbb R^{n_k},
\quad\lambda_k\ge0,&&k=0,1.
\end{aligned}
\tag{6}
$$

The constraints are jointly convex in $(z,g,a,\lambda,w)$, so partial
minimization defines a convex function $F$ on $[0,1]^D$.
At $z=\mathbf1_S$, the constraints impose $w_{km}=0$ for $m\notin S$
and admit an optimum with $w_{km}=\lambda_k$ for $m\in S$.
Consequently,

$$
F(\mathbf1_S)=V(S),\qquad
\min_{|S|\le d}V(S)=\min_{z\in\mathcal Z_d}F(z).
\tag{7}
$$

The continuous extension preserves the objective at every binary
selection; minimizing it over the continuous budget set gives a
relaxation. The discrete problem has the form of the logical-constraint
models in [1, problems (2.1)--(2.2)].
The exactness and attainment arguments are included in
Appendix B.

Associate multipliers $\gamma_{i,y}^{(k)}\ge0$ with the loss
constraints in (6) and $\mu_m\ge0$ with the coordinate
constraints. The resulting dual is as follows.

**Proposition 2.2 (Dual representation).**

For every $z\in[0,1]^D$,

$$
F(z)=\max_{(\gamma,\mu)\in\mathcal D}
\{H(\gamma)-2\mu^\top z\},
\tag{8}
$$

where

$$
H(\gamma)=\sum_{y\in\mathcal X}\inf_{v\in\mathbb R}
\left\{\sum_{i=1}^{n_0}\gamma_{i,y}^{(0)}\ell(-v)
+\sum_{j=1}^{n_1}\gamma_{j,y}^{(1)}\ell(v)\right\},
\tag{9}
$$

and

$$
\mathcal D=\left\{(\gamma,\mu)\ge0:
\begin{aligned}
&\sum_{y\in\mathcal X}\gamma_{i,y}^{(k)}=1/n_k,
&&i\in[n_k],\ k=0,1,\\
&\sum_{m=1}^D\left(
\sum_{i=1}^{n_k}\sum_{y\in\mathcal X}
\gamma_{i,y}^{(k)}|x_i^{(k)}[m]-y[m]|-\rho_k\mu_m
\right)_+\le\rho_k,
&&k=0,1
\end{aligned}\right\},
\tag{10}
$$

with $(v)_+=\max\{v,0\}$. The maximum is attained.
For an optimizer $(\gamma^t,\mu^t)$ at $z^t$,

$$
\begin{aligned}
F(z)&\ge c_t(z):=H(\gamma^t)-2(\mu^t)^\top z\\
&=F(z^t)-2(\mu^t)^\top(z-z^t),\qquad z\in[0,1]^D,
\end{aligned}
\tag{11}
$$

and $c_t(z^t)=F(z^t)$.

The proof is given in Appendix B.

## 3. Outer-approximation approach

Proposition 2.2 yields an outer-approximation scheme
of the form in [1, Theorem 2.5 and Algorithm 3.1].
Let $\mathrm{OPT}=\min_{z\in\mathcal Z_d}F(z)$.
After evaluating $z^1,\ldots,z^t$ and obtaining their dual optimizers,
let $U_t=\min_{s\le t}F(z^s)$ and solve

$$
\begin{aligned}
(z^{t+1},L_t)\in\operatorname*{arg\,min}_{z\in\mathcal Z_d,\,\eta\ge0}
\quad&\eta\\
\text{subject to}\quad&
\eta\ge F(z^s)-2(\mu^s)^\top(z-z^s),\qquad s=1,\ldots,t.
\end{aligned}
\tag{12}
$$

By (11),

$$
L_t\le\mathrm{OPT}\le U_t.
\tag{13}
$$

Starting from any $z^1\in\mathcal Z_d$, alternate the convex subproblem
(6)--(8) and the binary linear master
(12). Stop when $U_t-L_t\le\varepsilon$ and return the
best evaluated selection. Appendix A recovers its
detector, with risk $U_t$ and optimality gap at most $\varepsilon$.
For an inexact inner solve, use the actual feasible dual expression
$H(\gamma^t)-2(\mu^t)^\top z$ as the cut and a primal feasible objective
as the upper bound; the master must provide a global lower bound.

## Appendix A. Proof of Lemma 2.1

**Proof.**

Maximization over $\gamma$ and minimization over $(g,a,\lambda)$
below are subject to (3) and (5), respectively.
Let $S^c=[D]\setminus S$. For each source--target
pair, define the lifted target by

$$
\widetilde y_i^{(k)}[S]=y[S],\qquad
\widetilde y_i^{(k)}[S^c]=x_i^{(k)}[S^c],\qquad
\mathbb P_k^\gamma=\sum_{i,y}\gamma_{i,y}^{(k)}
\delta_{\widetilde y_i^{(k)}}.
$$

Then

$$
\mathcal W_1(\widehat{\mathbb P}_k,\mathbb P_k^\gamma)
\le\sum_{i,y}\gamma_{i,y}^{(k)}
\|x_i^{(k)}-\widetilde y_i^{(k)}\|_1
=\sum_{i,y}\gamma_{i,y}^{(k)}
\|x_i^{(k)}[S]-y[S]\|_1\le\rho_k.
$$

Consequently, for every $f(x)=g(x[S])$,

$$
\begin{aligned}
R(f)&\ge\sum_y\{p_0(y)\ell(-g(y[S]))+
p_1(y)\ell(g(y[S]))\}\\
&\ge H(\gamma),
\qquad
\max_\gamma H(\gamma)\le V(S),
\end{aligned}
\tag{14}
$$

where $H$ is the objective in (9).
Finite-dimensional minimax and the linear programming duality in
(4)--(5) give

$$
\begin{aligned}
\max_\gamma H(\gamma)
&=\inf_{g\in\mathbb R^{\mathcal X}}\max_\gamma
\sum_y\{p_0(y)\ell(-g_y)+p_1(y)\ell(g_y)\}\\
&=\inf_{(g,a,\lambda)}J(a,\lambda).
\end{aligned}
\tag{15}
$$

Now fix any feasible $(g,a,\lambda)$. Coincident projected targets
have identical constraint columns, so their scores may be chosen equal.
Set $g(y[S])=g_y$ and define

$$
E_k(\xi)=\min_{r\in[n_k]}
\{a_r^{(k)}+\lambda_k\|x_r^{(k)}[S]-\xi\|_1\},
\qquad k=0,1.
$$

Outside the projected target set, choose the smallest minimizing
indices $i,j$ for $E_0(\xi),E_1(\xi)$, respectively, and put

$$
r_0=\|x_i^{(0)}[S]-\xi\|_1,\quad
r_1=\|x_j^{(1)}[S]-\xi\|_1,\quad
b=\|x_i^{(0)}[S]-x_j^{(1)}[S]\|_1,
$$

$$
\tau=
\begin{cases}
\min\{1,r_0/b\},&b>0,\\
0,&b=0,
\end{cases}
\qquad
g(\xi)=(1-\tau)g_{x_i^{(0)}}+\tau g_{x_j^{(1)}}.
\tag{16}
$$

The triangle inequality gives

$$
b\le r_0+r_1,\qquad
\tau b\le r_0,\qquad (1-\tau)b\le r_1.
$$

Hence convexity and the source and cross-target constraints imply

$$
\begin{aligned}
\ell(-g(\xi))
&\le(1-\tau)\ell(-g_{x_i^{(0)}})+\tau\ell(-g_{x_j^{(1)}})\\
&\le a_i^{(0)}+\lambda_0\tau b
\le a_i^{(0)}+\lambda_0r_0=E_0(\xi),\\
\ell(g(\xi))
&\le(1-\tau)\ell(g_{x_i^{(0)}})+\tau\ell(g_{x_j^{(1)}})\\
&\le a_j^{(1)}+\lambda_1(1-\tau)b
\le a_j^{(1)}+\lambda_1r_1=E_1(\xi).
\end{aligned}
\tag{17}
$$

At projected targets the same bounds follow directly from
(5). The minimizing-index rule is Borel measurable; assigning the finitely many
observed values preserves measurability and $|g|\le\max_y|g_y|$.
Integrating (17) against optimal empirical-to-$\mathbb P_k$
couplings yields, for every $\mathbb P_k\in\mathcal P_k$,

$$
\begin{aligned}
\mathbb E_{\mathbb P_0}\ell(-g(X[S]))
&\le\frac1{n_0}\sum_i a_i^{(0)}+\rho_0\lambda_0,\\
\mathbb E_{\mathbb P_1}\ell(g(X[S]))
&\le\frac1{n_1}\sum_j a_j^{(1)}+\rho_1\lambda_1.
\end{aligned}
$$

Therefore

$$
\max_\gamma H(\gamma)
\le V(S)
\le\inf_{(g,a,\lambda)}J(a,\lambda)
=\max_\gamma H(\gamma).
$$

Since $J(\mathbf1,0)=2$ at $g=0$, the sublevel $J\le2$ is nonempty;
its feasible points satisfy

$$
0\le a_i^{(k)}\le2n_k,\qquad
0\le\lambda_k\le2/\rho_k.
$$

The loss constraints bound both $\ell(g_y)$ and $\ell(-g_y)$;
$\ell(t)\to\infty$ as $t\to\infty$ therefore bounds every $g_y$.
The sublevel is compact. Thus the finite minimum and its recovered
detector attain $V(S)$, and every optimal solution satisfies

$$
\rho_0\lambda_0+\rho_1\lambda_1\le J(a,\lambda)=V(S)\le2.
\tag{18}
$$

The finite LFD maximum is attained because $H$ is upper semicontinuous
on the compact transport set.

$\square$

## Appendix B. Proof of Proposition 2.2

**Proof.**

The bounds in Appendix A and
$0\le w_{km}\le\lambda_k$ make the nonempty sublevel $J\le2$
compact, so $F(z)$ attains its minimum. At $z=\mathbf1_S$,
(18) permits the lift $w_{km}=\lambda_kz_m$
of a fixed-selection optimum. Conversely,

$$
0\le w_{km}\le\lambda_kz_m
\quad\Longrightarrow\quad
\sum_m w_{km}|x_i^{(k)}[m]-y[m]|
\le\lambda_k\|x_i^{(k)}[S]-y[S]\|_1,
$$

which proves (7).

Associate nonnegative multipliers $\gamma,\beta,\mu$ with the loss
constraints, $w_{km}\le\lambda_k$, and the coordinate-capacity
constraints, respectively. With $g,a$ free and $\lambda,w\ge0$,
collecting terms in the Lagrangian gives

$$
\begin{aligned}
\mathcal L={}&-2\mu^\top z
+\sum_{y\in\mathcal X}\left[
\ell(-g_y)\sum_{i=1}^{n_0}\gamma_{i,y}^{(0)}
+\ell(g_y)\sum_{j=1}^{n_1}\gamma_{j,y}^{(1)}\right]\\
&+\sum_{k=0}^1\sum_{i=1}^{n_k}a_i^{(k)}
\left(\frac1{n_k}-\sum_{y\in\mathcal X}\gamma_{i,y}^{(k)}\right)
+\sum_{k=0}^1\lambda_k\left(\rho_k-\sum_m\beta_{km}\right)\\
&+\sum_{k,m}w_{km}\left[
\beta_{km}+\rho_k\mu_m
-\sum_{i=1}^{n_k}\sum_{y\in\mathcal X}
\gamma_{i,y}^{(k)}|x_i^{(k)}[m]-y[m]|\right].
\end{aligned}
\tag{19}
$$

Minimization over $a$ is finite precisely when
$\sum_y\gamma_{i,y}^{(k)}=1/n_k$; minimization over $g$ gives
$H(\gamma)$. Minimization over $\lambda,w\ge0$ is finite precisely when

$$
\sum_m\beta_{km}\le\rho_k,\qquad
\beta_{km}\ge
\sum_{i=1}^{n_k}\sum_{y\in\mathcal X}
\gamma_{i,y}^{(k)}|x_i^{(k)}[m]-y[m]|-\rho_k\mu_m.
$$

A nonnegative $\beta$ satisfying these inequalities exists if and only if

$$
\sum_{m=1}^D\left(
\sum_{i=1}^{n_k}\sum_{y\in\mathcal X}
\gamma_{i,y}^{(k)}|x_i^{(k)}[m]-y[m]|-\rho_k\mu_m
\right)_+\le\rho_k,\qquad k=0,1.
$$

The dual feasible set and objective are therefore $\mathcal D$ and
$H(\gamma)-2\mu^\top z$.

For every $z\in[0,1]^D$, the choices

$$
g=0,\qquad \lambda=w=0,\qquad a_i^{(k)}>\ell(0)
$$

satisfy all affine constraints and strictly satisfy all loss constraints. Refined Slater duality, with nonnegativity included among
the affine constraints, gives equality and dual attainment in
(8), including when $z_m=0$. Consequently, an optimizer
$(\gamma^t,\mu^t)$ at $z^t$ satisfies

$$
\begin{aligned}
c_t(z)&=H(\gamma^t)-2(\mu^t)^\top z\le F(z),\\
c_t(z^t)&=H(\gamma^t)-2(\mu^t)^\top z^t=F(z^t),\\
c_t(z)&=F(z^t)-2(\mu^t)^\top(z-z^t),
\end{aligned}
$$

which proves (11).

$\square$

## References

[1] Dimitris Bertsimas, Ryan Cory-Wright, and Jean Pauphilet. A unified approach to mixed-integer optimization problems with logical constraints. *SIAM Journal on Optimization*, 31(3):2340–2367, 2021.

[2] Rui Gao, Liyan Xie, Yao Xie, and Huan Xu. Robust hypothesis testing using Wasserstein uncertainty sets. In *Advances in Neural Information Processing Systems*, volume 31. Curran Associates, Inc., 2018.

[3] Liyan Xie, Rui Gao, and Yao Xie. Robust hypothesis testing with Wasserstein uncertainty sets, 2021. arXiv:2105.14348.
