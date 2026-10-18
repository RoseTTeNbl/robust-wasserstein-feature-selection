# Variable Selection for Wasserstein Robust Hypothesis Testing

18 October 2026

## Abstract

Variable selection for Wasserstein robust testing admits an exact finite formulation and a maximum-affine dual representation. Dual couplings generate global cuts. The analysis includes certificates for inexact solves and data-dependent bounds on the evaluated selections.

## 1. Problem formulation

Let $[D]=\{1,\ldots,D\}$ and let $0\le d\le D$ be an integer.
For a measurable score $f:\mathbb R^D\to\mathbb R$, define

$$
R(f)=\sup_{\substack{P_0\in\mathcal U_0\\P_1\in\mathcal U_1}}
\left\{\mathbb E_{P_0}\ell(-f(X))+
       \mathbb E_{P_1}\ell(f(X))\right\}.
\tag{1}
$$

The loss $\ell$ is finite, nonnegative, convex and nondecreasing, with
$\ell(0)>0$. Write $\ell_0(s)=\ell(-s)$ and $\ell_1(s)=\ell(s)$.
For $S\subseteq[D]$, let $\operatorname{Proj}_S:\mathbb R^D\to\mathbb R^{|S|}$
be the coordinate projection and define

$$

\mathcal G_S=\{g:\mathbb R^{|S|}\to\mathbb R:\ g\text{ is Borel measurable}\}.

$$

The variable-selection problem is

$$
\boxed{V(S)=\inf_{g\in\mathcal G_S}R(g\circ\operatorname{Proj}_S),
\qquad \mathrm{OPT}=\min_{|S|\le d}V(S).}
\tag{2}
$$

The $N$ observations are $x_1,\ldots,x_N\in\mathbb R^D$. Nonempty index sets
$I_0,I_1$ partition $[N]$ by class, with within-class masses $q_i>0$:

$$

\sum_{i\in I_k}q_i=1,\qquad
\widehat P_k=\sum_{i\in I_k}q_i\delta_{x_i},\qquad
\mathcal U_k=\{P:W_1(P,\widehat P_k)\le\theta_k\},\quad k=0,1.

$$

Here $\delta_x$ is unit mass at $x$, $\theta_k>0$, and $W_1$ uses
$\ell_1$ cost on distributions with finite first moments.
Unweighted samples have $q_i=1/|I_k|$ for $i\in I_k$.

## 2. Dual reformulation

Fix $S$ and write

$$

x_i^S=\operatorname{Proj}_Sx_i,\qquad \Delta_{ir,j}=|x_{ij}-x_{rj}|,\qquad
c_{ir}^S=\|x_i^S-x_r^S\|_1=\sum_{j\in S}\Delta_{ir,j}.

$$

Here $i,r\in[N]$ and $j\in[D]$.
A coupling $\pi_{ir}\ge0$ transports mass from source $i$ to target $r$. Define

$$

\mathcal T_k(S)=\left\{\pi\ge0:
\sum_r\pi_{ir}=q_i,\quad
\sum_{i\in I_k,r}\pi_{ir}c_{ir}^S\le\theta_k\right\}.

$$

With target-score variables $u=(u_1,\ldots,u_N)$, the finite transport
representation is

$$
V(S)=\inf_{u\in\mathbb R^N}
\sum_{k=0}^1\max_{\pi\in\mathcal T_k(S)}
\sum_{i\in I_k,r}\pi_{ir}\ell_k(u_r).
\tag{3}
$$

Dualizing each transport LP introduces source-loss bounds $a_i\in\mathbb R$
and transport prices $\lambda_k\ge0$. With
$J(a,\lambda)=\sum_iq_i a_i+\sum_k\theta_k\lambda_k$, this gives

$$
\begin{aligned}
V(S)=\min_{a,u\in\mathbb R^N,\ \lambda\ge0}\quad&J(a,\lambda)\\
\text{subject to}\quad&
\ell_k(u_r)\le a_i+\lambda_kc_{ir}^S,
\quad i\in I_k,\ r\in[N],\ k=0,1.
\end{aligned}
\tag{4}
$$

Appendix A proves exactness and recovers an optimal test.

For the binary indicator $z=\mathbf1_S$, replace the switched prices $\lambda_kz_j$
by coordinate prices $w_{kj}$. Since $a_i\ge0$,

$$

\sum_k\theta_kw_k=z\sum_k\theta_k\lambda_k\le zJ(a,\lambda)
\quad\text{when }w_k=\lambda_kz.

$$

For fixed $z\in[0,1]^D$, define

$$
\begin{aligned}
F(z)=\min_{a,u,\lambda,w}\quad&J(a,\lambda)\\
\text{subject to}\quad&
\ell_k(u_r)\le a_i+\sum_j\Delta_{ir,j}w_{kj},
\quad i\in I_k,\ r\in[N],\ k=0,1,\\
&0\le w_k\le\lambda_k\mathbf1,\qquad k=0,1,\\
&\theta_0w_0+\theta_1w_1\le zJ(a,\lambda),\\
&\lambda\ge0,\qquad a,u\in\mathbb R^N.
\end{aligned}
\tag{5}
$$

Here $w_k=(w_{k1},\ldots,w_{kD})^\top$ and $\mathbf1=(1,\ldots,1)^\top$;
vector inequalities are coordinatewise. For binary $z$, under the other constraints,

$$

\theta_0w_0+\theta_1w_1\le zJ
\quad\Longleftrightarrow\quad
w_{kj}=0\quad\text{for }z_j=0,\ k=0,1.

$$

When $z_j=1$, that particular inequality is redundant and may be omitted:

$$

\theta_0w_{0j}+\theta_1w_{1j}
\le\sum_k\theta_k\lambda_k\le J.

$$

Deleting the entire family also removes the zero-coordinate restrictions,
giving $V([D])$ for every $z$. Retaining it gives $F(\mathbf1_S)=V(S)$; Proposition B.1(i) proves equality of the projected feasible sets and preservation of the objective.

A dual of (5), with a scaled transport coupling $\Pi\ge0$,
a scalar $\alpha\ge0$, and coordinate multipliers $\mu\ge0$, is

$$
\begin{aligned}
F(z)=\max_{\Pi,\alpha,\mu}\quad&H(\Pi)\\
\text{subject to}\quad&(\Pi,\alpha,\mu)\in\mathcal D,\qquad
\alpha+\mu^\top z=1.
\end{aligned}
\tag{6}
$$

Here

$$

H(\Pi)=\sum_{r=1}^N\inf_{s\in\mathbb R}
\sum_{k=0}^1\sum_{i\in I_k}\Pi_{ir}\ell_k(s),

$$

and, with $(v)_+=\max\{v,0\}$,

$$
\mathcal D=\left\{(\Pi,\alpha,\mu)\ge0:
\begin{array}{l}
\displaystyle\sum_r\Pi_{ir}=\alpha q_i,\quad i\in[N],\\[3pt]
\displaystyle\sum_j
\left(\sum_{i\in I_k,r}\Pi_{ir}\Delta_{ir,j}-\theta_k\mu_j\right)_+
\le\alpha\theta_k,\quad k=0,1
\end{array}\right\}.
\tag{7}
$$

The coupling has source masses $\alpha q_i$; $H$ minimizes its weighted
loss over one score per target. The Lagrangian derivation is in
Proposition B.1(ii).

For a normalized tuple $\alpha+\mu^\top z=1$, scaling all three dual
variables by $\tau\ge0$ preserves $\mathcal D$ and gives

$$

H(\tau\Pi)=\tau H(\Pi),\qquad
\max_{\tau\ge0}\{2\sqrt{\tau H(\Pi)}-\tau\}=H(\Pi),
\quad \tau^*=H(\Pi).

$$

Optimizing this scale on every ray removes the normalization:

$$
\boxed{F(z)=\max_{(\Pi,\alpha,\mu)\in\mathcal D}
\{2\sqrt{H(\Pi)}-\alpha-\mu^\top z\}.}
\tag{8}
$$

Proposition B.1(iii) proves this equivalence and attainment. A normalized optimizer $(\bar\Pi,\bar\alpha,\bar\mu)$ gives the affine optimizer $F(z)(\bar\Pi,\bar\alpha,\bar\mu)$; the scaling must also be applied to the cut slope.

## 3. Cutting-plane algorithm

Let $\mathcal Z_d=\{z\in\{0,1\}^D:\mathbf1^\top z\le d\}$.
At $z^t$, evaluate $F(z^t)$ by (8) and take an optimizer
$(\Pi^t,\alpha^t,\mu^t)$ with the bounds $\mu_j^t\le M_j$ from
Proposition C.1(ii). Its subgradient $-\mu^t$ gives

$$
F(z)\ge2\sqrt{H(\Pi^t)}-\alpha^t-(\mu^t)^\top z
=F(z^t)-(\mu^t)^\top(z-z^t).
\tag{9}
$$

The next selection and lower bound solve

$$
\begin{aligned}
(z^{t+1},\eta^{t+1})\in
\operatorname*{arg\,min}_{z\in\mathcal Z_d,\ \eta\ge0}\quad&\eta\\
\text{subject to}\quad&
\eta\ge F(z^s)-(\mu^s)^\top(z-z^s),
\qquad s=1,\ldots,t.
\end{aligned}
\tag{10}
$$

For exact solves, writing $U_t=\min_{s\le t}F(z^s)$ and $T$ for the last
iteration, Appendix C proves

$$
\begin{aligned}
&\eta^{t+1}\le\mathrm{OPT}\le U_t,\\
&0\le R(f^*)-\mathrm{OPT}\le U_T-\eta^{T+1}\le\varepsilon,
\qquad T\le\sum_{r=0}^d\binom Dr.
\end{aligned}
\tag{11}
$$

For inexact solves, a feasible recovered score supplies an upper bound $U_t$,
dual feasible cuts remain global lower bounds, and a global master bound
$L_t$ gives

$$

0\le R(f_{\rm inc})-\mathrm{OPT}\le U_t-L_t.

$$

Proposition C.1(i) proves the certificate and the sufficient
accuracy condition $\delta+\beta\le\varepsilon$ for finite termination.
Parts (ii) and (iii) of the same proposition give
bounded slopes and a data-dependent separation bound for exact search.

**Algorithm 1. Exact cutting-plane method for variable selection.**

**Input:** Problem data, budget $d$, tolerance $\varepsilon>0$, and $z^1\in\mathcal Z_d$.

**Output:** A subset $S^*$, its optimal test $f^*$, and a risk interval of width at most $\varepsilon$.

1. $t\gets1$, $U\gets+\infty$.

2. **While** true:

    3. Solve (8) exactly at $z^t$; choose $\mu^t$ with $0\le\mu_j^t\le M_j$ as in Proposition C.1(ii).

    4. **If** $F(z^t)<U$:

        5. $(U,z^*)\gets(F(z^t),z^t)$.

    6. Retain the cut (9) with index $t$.

    7. Solve (10) exactly to obtain $(z^{t+1},\eta^{t+1})$.

    8. **If** $U-\eta^{t+1}\le\varepsilon$:

        9. $S^*\gets\{j:z_j^*=1\}$.

        10. Solve (4) for $S^*$; construct $g^*$ by (17) and set $f^*=g^*\circ\operatorname{Proj}_{S^*}$.

        11. **Return** $(S^*,f^*,[\eta^{t+1},U])$.

    12. $t\gets t+1$.

## Appendix A. Proof of the finite representation

Let $x_i\in\mathbb R^D$, $i\in[N]$, and let nonempty sets $I_0,I_1$
partition $[N]$. Let $q_i>0$, $\sum_{i\in I_k}q_i=1$, and $\theta_k>0$.
The loss $\ell:\mathbb R\to[0,\infty)$ is finite, convex and nondecreasing,
with $\ell(0)>0$; put $\ell_0(s)=\ell(-s)$ and $\ell_1(s)=\ell(s)$.
For $S\subseteq[D]$, let $\operatorname{Proj}_S$ be coordinate
projection; $\delta_x$ denotes unit mass at $x$. Define

$$

\begin{gathered}
x_i^S=\operatorname{Proj}_Sx_i,\qquad
c_{ir}^S=\|x_i^S-x_r^S\|_1,\qquad
\widehat P_k=\sum_{i\in I_k}q_i\delta_{x_i},\\
\mathcal U_k=\{P:W_1(P,\widehat P_k)\le\theta_k\},\qquad
\mathcal G_S=\{g:\mathbb R^{|S|}\to\mathbb R:\ g\text{ is Borel measurable}\},\\
R(f)=\sum_{k=0}^1\sup_{P\in\mathcal U_k}\mathbb E_P\ell_k(f(X)),
\qquad
V(S)=\inf_{g\in\mathcal G_S}R(g\circ\operatorname{Proj}_S).
\end{gathered}

$$

Here $W_1$ uses $\ell_1$ transport cost and finite-first-moment
distributions. Set

$$
\begin{aligned}
T(S)=\inf_{a,u\in\mathbb R^N,\ \lambda\in\mathbb R_+^2}\quad&
J(a,\lambda):=\sum_iq_i a_i+\sum_{k=0}^1\theta_k\lambda_k\\
\text{subject to}\quad&
\ell_k(u_r)\le a_i+\lambda_kc_{ir}^S,
\quad i\in I_k,\ r\in[N],\ k=0,1.
\end{aligned}
\tag{12}
$$

$T(S)$ is the finite value; equality with $V(S)$ is proved below.

**Proposition A.1.**

For every $S\subseteq[D]$, $V(S)=T(S)$, the infimum defining $T(S)$
is attained, and each optimal finite solution produces a Borel score
$g^*:\mathbb R^{|S|}\to\mathbb R$ with
$R(g^*\circ\operatorname{Proj}_S)=V(S)$.

**Proof.**

*Finite transport duality.*
Fix $u\in\mathbb R^N$. Define

$$

\mathcal T_k(S)=\left\{\pi\in\mathbb R_+^{|I_k|\times N}:
\sum_r\pi_{ir}=q_i\ (i\in I_k),\quad
\sum_{i\in I_k,r}\pi_{ir}c_{ir}^S\le\theta_k\right\}.

$$

The diagonal plan is feasible and $0\le\pi_{ir}\le q_i$;
thus $\mathcal T_k(S)$ is nonempty and compact.
With $a_i\in\mathbb R$, $\lambda_k\ge0$, and indices $i\in I_k$, $r\in[N]$,

$$
\begin{aligned}
L_k(\pi,a,\lambda_k;u)
={}&\sum_{i,r}\pi_{ir}\ell_k(u_r)
+\sum_i a_i\left(q_i-\sum_r\pi_{ir}\right)
+\lambda_k\left(\theta_k-\sum_{i,r}\pi_{ir}c_{ir}^S\right)\\
={}&\sum_iq_i a_i+\theta_k\lambda_k
+\sum_{i,r}\pi_{ir}[\ell_k(u_r)-a_i-\lambda_kc_{ir}^S].
\end{aligned}
\tag{13}
$$

$$

\sup_{\pi\ge0}L_k=
\begin{cases}
\sum_iq_i a_i+\theta_k\lambda_k,
&\ell_k(u_r)\le a_i+\lambda_kc_{ir}^S\quad\forall i,r,\\
+\infty,&\text{otherwise}.
\end{cases}

$$

LP strong duality gives

$$
\begin{aligned}
\max_{\pi\in\mathcal T_k(S)}\sum_{i\in I_k,r}\pi_{ir}\ell_k(u_r)
=\min_{(a_i)_{i\in I_k},\ \lambda_k\ge0}\quad&
\sum_{i\in I_k}q_i a_i+\theta_k\lambda_k\\
\text{subject to}\quad&
\ell_k(u_r)\le a_i+\lambda_kc_{ir}^S,\quad i\in I_k,\ r\in[N].
\end{aligned}
\tag{14}
$$

$$

T(S)=\inf_{u\in\mathbb R^N}\sum_{k=0}^1
\max_{\pi\in\mathcal T_k(S)}\sum_{i\in I_k,r}\pi_{ir}\ell_k(u_r).

$$

Only the finite LP has been dualized; no minimax exchange is used.

*Lower bound: $T(S)\le V(S)$.*
For $g\in\mathcal G_S$ and $\pi\in\mathcal T_k(S)$, choose
$\widetilde x_{ir}\in\mathbb R^D$ with

$$

\operatorname{Proj}_S\widetilde x_{ir}=x_r^S,\qquad
\operatorname{Proj}_{S^c}\widetilde x_{ir}=\operatorname{Proj}_{S^c}x_i,
\qquad \|\widetilde x_{ir}-x_i\|_1=c_{ir}^S.

$$

$$
\begin{aligned}
P_k^\pi&=\sum_{i\in I_k,r}\pi_{ir}\delta_{\widetilde x_{ir}},
&W_1(P_k^\pi,\widehat P_k)&\le\sum_{i\in I_k,r}\pi_{ir}c_{ir}^S\le\theta_k,\\
\mathbb E_{P_k^\pi}\ell_k(g(\operatorname{Proj}_S X))
&=\sum_{i\in I_k,r}\pi_{ir}\ell_k(g(x_r^S)).
\end{aligned}
\tag{15}
$$

$$
\begin{aligned}
V(S)&\ge\inf_{g\in\mathcal G_S}\sum_k
\max_{\pi\in\mathcal T_k(S)}\sum_{i\in I_k,r}\pi_{ir}\ell_k(g(x_r^S))\\
&\ge\inf_{u\in\mathbb R^N}\sum_k
\max_{\pi\in\mathcal T_k(S)}\sum_{i\in I_k,r}\pi_{ir}\ell_k(u_r)=T(S).
\end{aligned}
\tag{16}
$$

*Upper bound: $V(S)\le T(S)$.*
For any finite feasible $(a,u,\lambda)$ and $y\in\mathbb R^{|S|}$, set

$$

\begin{gathered}
E_k(y)=\min_{i\in I_k}\{a_i+\lambda_k\|x_i^S-y\|_1\},\qquad
i_k(y)=\min\operatorname*{arg\,min}_{i\in I_k}
\{a_i+\lambda_k\|x_i^S-y\|_1\},\\
r_k=\|x_{i_k}^S-y\|_1,\qquad
\rho=\|x_{i_0}^S-x_{i_1}^S\|_1\le r_0+r_1.
\end{gathered}

$$

$$
g(y)=\begin{cases}
(1-\gamma)u_{i_0}+\gamma u_{i_1},\quad
\gamma=\min\{1,r_0/\rho\},&\rho>0,\\
u_{i_0},&\rho=0.
\end{cases}
\tag{17}
$$

For $\rho>0$, $\gamma\rho\le r_0$ and $(1-\gamma)\rho\le r_1$.
Convexity and finite feasibility yield

$$

\begin{aligned}
\ell_0(g(y))&\le(1-\gamma)\ell_0(u_{i_0})+\gamma\ell_0(u_{i_1})
\le a_{i_0}+\lambda_0\gamma\rho\le E_0(y),\\
\ell_1(g(y))&\le(1-\gamma)\ell_1(u_{i_0})+\gamma\ell_1(u_{i_1})
\le a_{i_1}+\lambda_1(1-\gamma)\rho\le E_1(y).
\end{aligned}

$$

For $\rho=0$, the source--target pairs $(i_0,i_0)$ and $(i_1,i_0)$
give the same two bounds. The finite tie rule makes $g$ Borel, with
$|g(y)|\le\max_r|u_r|$; interpolation of the $u_r$ is unnecessary.
For $P\in\mathcal U_k$, disintegrate an optimal coupling as
source-indexed measures $\nu_i$ (splitting repeated sources by $q_i$):

$$

\nu_i(\mathbb R^D)=q_i,\qquad
\sum_{i\in I_k}\nu_i=P,\qquad
\sum_{i\in I_k}\int\|x-x_i\|_1\,\nu_i(dx)\le\theta_k.

$$

$$
\begin{aligned}
\mathbb E_P\ell_k(g(\operatorname{Proj}_S X))
&\le\sum_{i\in I_k}\int
[a_i+\lambda_k\|\operatorname{Proj}_Sx-x_i^S\|_1]\,\nu_i(dx)\\
&\le\sum_{i\in I_k}q_i a_i+\theta_k\lambda_k.
\end{aligned}
\tag{18}
$$

$$

R(g\circ\operatorname{Proj}_S)\le J(a,\lambda),\qquad V(S)\le T(S).

$$

The feasible point $(a,u,\lambda)=(\ell(0)\mathbf1,0,0)$ has value
$C=2\ell(0)$. On the closed sublevel $J\le C$,

$$

0\le a_i\le C/q_i,\qquad 0\le\lambda_k\le C/\theta_k.

$$

The loss constraints bound both $\ell(u_r)$ and $\ell(-u_r)$.
If $\ell$ is nonconstant, $\ell(t)\to\infty$ as $t\to\infty$, so $u$ is bounded;
if $\ell$ is constant, restrict to $u=0$.
Weierstrass's theorem gives a minimizer, whose extension attains $V(S)$.

$\square$

## Appendix B. Proof of the dual reformulation

Let $x_i\in\mathbb R^D$, let nonempty $I_0,I_1$ partition $[N]$, and let
$q_i,\theta_k>0$ with $\sum_{i\in I_k}q_i=1$.
Let $\ell:\mathbb R\to[0,\infty)$ be finite, convex and nondecreasing,
with $\ell(0)>0$; write $\ell_0(s)=\ell(-s)$ and $\ell_1(s)=\ell(s)$.
Set

$$

\Delta_{ir,j}=|x_{ij}-x_{rj}|,\qquad
c_{ir}^S=\sum_{j\in S}\Delta_{ir,j},\qquad
J(a,\lambda)=\sum_iq_i a_i+\sum_k\theta_k\lambda_k.

$$

Let $V(S)$ and $F(z)$ be the values of (4) and
(5), respectively. For $\Pi\in\mathbb R_+^{N\times N}$, define

$$

H(\Pi)=\sum_{r=1}^N\inf_{s\in\mathbb R}
\sum_{k=0}^1\sum_{i\in I_k}\Pi_{ir}\ell_k(s),\qquad
d_{kj}(\Pi)=\sum_{i\in I_k,r}\Pi_{ir}\Delta_{ir,j}.

$$

$$
\mathcal D=\left\{(\Pi,\alpha,\mu)\ge0:
\begin{array}{l}
\sum_r\Pi_{ir}=\alpha q_i\quad(i\in[N]),\\[2pt]
\sum_j(d_{kj}(\Pi)-\theta_k\mu_j)_+\le\alpha\theta_k
\quad(k=0,1)
\end{array}\right\},\qquad (v)_+=\max\{v,0\}.
\tag{19}
$$

**Proposition B.1 (Exact extension and dual representation).**

(i) For $z=\mathbf1_S$, (4) and (5)
have the same feasible $(a,u,\lambda)$ after eliminating $w$, with the same
objective. Hence $F(\mathbf1_S)=V(S)$.

(ii) For every $z\in[0,1]^D$, the primal and dual optima are attained and

$$
0<F(z)=\max_{\substack{(\Pi,\alpha,\mu)\in\mathcal D\\
\alpha+\mu^\top z=1}}H(\Pi)\le2\ell(0).
\tag{20}
$$

(iii) The equivalent affine dual is

$$
F(z)=\max_{(\Pi,\alpha,\mu)\in\mathcal D}
\{2\sqrt{H(\Pi)}-\alpha-\mu^\top z\}.
\tag{21}
$$

A normalized optimizer $(\bar\Pi,\bar\alpha,\bar\mu)$ yields an affine optimizer

$$
(\Pi,\alpha,\mu)=F(z)(\bar\Pi,\bar\alpha,\bar\mu),\qquad
H(\Pi)=F(z)^2.
\tag{22}
$$

The function $F$ is convex. Every affine optimizer
$(\Pi^t,\alpha^t,\mu^t)$ at $z^t$ satisfies

$$

F(z)\ge F(z^t)-(\mu^t)^\top(z-z^t)\quad(z\in[0,1]^D),
\qquad -\mu^t\in\partial F(z^t).

$$

**Proof.**

(i) Fix $z=\mathbf1_S$. For a feasible point of (4),
$a_i\ge\ell_k(u_i)\ge0$. With $w_k=\lambda_kz$,

$$

\ell_k(u_r)\le a_i+\sum_j\Delta_{ir,j}w_{kj},\qquad
\sum_k\theta_kw_{kj}=z_j\sum_k\theta_k\lambda_k\le z_jJ(a,\lambda).

$$

This gives a feasible point of (5) with unchanged objective.
Conversely, feasibility of (5) implies

$$

\begin{gathered}
w_{kj}=0\quad(j\notin S),\qquad 0\le w_{kj}\le\lambda_k\quad(j\in S),\\
\ell_k(u_r)\le a_i+\sum_{j\in S}\Delta_{ir,j}w_{kj}
\le a_i+\lambda_kc_{ir}^S.
\end{gathered}

$$

Thus $(a,u,\lambda)$ is feasible for (4) with unchanged $J$.
The two maps prove (i).

(ii) Assign nonnegative multipliers $\Pi_{ir},\beta_{kj},\mu_j$
to the loss, price-cap and selector inequalities in (5).
Keeping $\lambda,w\ge0$ and writing $\alpha=1-\mu^\top z$, the Lagrangian is

$$
\begin{aligned}
\mathcal L={}&J+\sum_{k,i\in I_k,r}\Pi_{ir}
\left[\ell_k(u_r)-a_i-\sum_j\Delta_{ir,j}w_{kj}\right]
+\sum_{k,j}\beta_{kj}(w_{kj}-\lambda_k)\\
&+\sum_j\mu_j\left[\sum_k\theta_kw_{kj}-z_jJ\right]\\
={}&\sum_i a_i\left(\alpha q_i-\sum_r\Pi_{ir}\right)
+\sum_{k,i\in I_k,r}\Pi_{ir}\ell_k(u_r)\\
&+\sum_k\lambda_k\left(\alpha\theta_k-\sum_j\beta_{kj}\right)
+\sum_{k,j}w_{kj}[\beta_{kj}+\theta_k\mu_j-d_{kj}(\Pi)].
\end{aligned}
\tag{23}
$$

The infimum over $a,u\in\mathbb R^N$ and $\lambda,w\ge0$ equals $H(\Pi)$
exactly under the conditions

$$
\sum_r\Pi_{ir}=\alpha q_i,\qquad
\sum_j\beta_{kj}\le\alpha\theta_k,\qquad
\beta_{kj}\ge(d_{kj}(\Pi)-\theta_k\mu_j)_+.
\tag{24}
$$

The row sums give $\alpha\ge0$; eliminating $\beta$ yields $\mathcal D$
and $\alpha+\mu^\top z=1$.
For $J_0=\{j:z_j=0\}$, the primal forces $w_{kj}=0$.
After deleting these variables and their selector rows, the point

$$

u=0,\qquad a_i=\ell(0)+1,\qquad\lambda_k=1,\qquad
w_{kj}=z_j/2\quad(j\notin J_0).

$$

All remaining inequalities are strict, including

$$

\sum_k\theta_kw_{kj}=\frac{z_j}{2}\sum_k\theta_k
<z_j\left[2(\ell(0)+1)+\sum_k\theta_k\right]=z_jJ.

$$

Slater's theorem gives strong duality and dual attainment for the reduced
problem, including $J_0=[D]$. Restoring

$$

\beta_{kj}=0,\qquad \mu_j=\max_{k=0,1}d_{kj}(\Pi)/\theta_k
\quad(j\in J_0)

$$

preserves the objective and normalization and satisfies (24).
The primal sublevel $J\le C:=2\ell(0)$ is nonempty.
Since $w_k\le\lambda_k\mathbf1$, its $(a,u,\lambda)$ satisfy the bounds
in the proof of Proposition A.1 with $S=[D]$; $w$ is bounded as well.
The same compactness argument gives primal attainment and $F(z)\le C$.
If $F(z)=0$, nonnegativity gives $a=\lambda=w=0$, contradicting

$$

0<2\ell(0)\le\ell(-u_r)+\ell(u_r)\le0.

$$

(iii) The set $\mathcal D$ is a cone and
$H(\tau\Pi)=\tau H(\Pi)\ge0$ for $\tau\ge0$.
For $(\Pi,\alpha,\mu)\in\mathcal D$, put $b=\alpha+\mu^\top z$.
If $b>0$, (ii) gives $H(\Pi)/b\le F(z)$ and

$$
2\sqrt{H(\Pi)}-b
=\frac{H(\Pi)}b-
\left(\sqrt{\frac{H(\Pi)}b}-\sqrt b\right)^2\le F(z).
\tag{25}
$$

If $b=0$, then $\alpha=0$, $\Pi=0$, and the affine objective is zero.
Conversely, for a normalized optimizer,

$$

H(F(z)\bar\Pi)=F(z)^2,\qquad
F(z)(\bar\alpha+\bar\mu^\top z)=F(z),\qquad
2\sqrt{H(F(z)\bar\Pi)}-F(z)=F(z).

$$

This proves (21) and (22).
As a supremum of affine functions, $F$ is convex, and an optimizer at $z^t$ gives

$$

F(z)\ge2\sqrt{H(\Pi^t)}-\alpha^t-(\mu^t)^\top z
=F(z^t)-(\mu^t)^\top(z-z^t).

$$

The rescaling gives $-\mu^t=-F(z^t)\bar\mu^t$.
Since $H$ is a nonnegative infimum of linear functions, both $H$ and
$\sqrt H$ are concave; the fixed-$z$ dual is a convex program.

$\square$

## Appendix C. Proof of the search guarantees

For the value function $F$ and dual objects $H,\mathcal D$ of
Proposition B.1, set

$$

\mathcal Z_d=\{\mathbf1_S:S\subseteq[D],\ |S|\le d\},\qquad
\mathrm{OPT}=\min_{z\in\mathcal Z_d}F(z).

$$

At $z^s=\mathbf1_{S_s}\in\mathcal Z_d$, let $p_s$ be a feasible objective
of (5), with recovered score $f_s=g_s\circ\operatorname{Proj}_{S_s}$.
For $(\Pi^s,\alpha_s,\mu^s)\in\mathcal D$ and a certified bound
$0\le h_s\le H(\Pi^s)$, define

$$

\begin{gathered}
c_s(z)=2\sqrt{h_s}-\alpha_s-(\mu^s)^\top z,\qquad
m_t(z)=\max\{0,c_1(z),\ldots,c_t(z)\},\\
U_t=\min_{s\le t}p_s,\qquad s_t\in\operatorname*{arg\,min}_{s\le t}p_s,
\qquad f_{\rm inc}=f_{s_t}.
\end{gathered}

$$

Let $L_t$ be a global lower bound on $\min_{z\in\mathcal Z_d}m_t(z)$ and
let $(z^{t+1},\eta^{t+1})$ be a feasible master point:
$z^{t+1}\in\mathcal Z_d$, $\eta^{t+1}\ge m_t(z^{t+1})$.

**Proposition C.1 (Risk certificate and finite search).**

(i) For every $t$,

$$
L_t\le\mathrm{OPT}\le R(f_{\rm inc})\le U_t,\qquad
0\le R(f_{\rm inc})-\mathrm{OPT}\le U_t-L_t.
\tag{26}
$$

If, at every iteration, for $\delta,\beta\ge0$ and $\varepsilon>0$,

$$

p_s-c_s(z^s)\le\delta\quad(s\le t),\qquad
\eta^{t+1}-L_t\le\beta,\qquad\delta+\beta\le\varepsilon,

$$

then stopping at $U_t-L_t\le\varepsilon$ requires at most $|\mathcal Z_d|$
evaluations.

(ii) Put

$$
C=2\ell(0),\qquad
A_{kj}=\sum_{i\in I_k}q_i\max_{r\in[N]}\Delta_{ir,j},\qquad
M_j=C\max_{k=0,1}\frac{A_{kj}}{\theta_k}.
\tag{27}
$$

At every $z\in[0,1]^D$ there exists an affine optimizer with
$0\le\mu_j\le M_j$. Consequently,

$$

|F(z)-F(z')|\le\sum_jM_j|z_j-z_j'|\qquad(z,z'\in[0,1]^D).

$$

(iii) Under exact solves and the bounded optimal multipliers in
(ii), Algorithm 1 returns $f^*$ at iteration $T$ with

$$
0\le R(f^*)-\mathrm{OPT}\le U_T-\eta^{T+1}\le\varepsilon.
\tag{28}
$$

Any two actually evaluated selections satisfy

$$
\sum_{j\in S_t\setminus S_s}M_j>\varepsilon\qquad(s<t).
\tag{29}
$$

For the pseudodistance $d_M(S,S')=\sum_{j\in S\triangle S'}M_j$, define

$$

\mathcal P_\varepsilon=\max\left\{|\mathcal A|:
\begin{array}{l}
\mathcal A\subseteq\{S\subseteq[D]:|S|\le d\},\\
d_M(S,S')>\varepsilon\quad(S\ne S',\ S,S'\in\mathcal A)
\end{array}\right\}.

$$

Then, allowing $M_j=0$,

$$
T\le\mathcal P_\varepsilon\le\sum_{r=0}^d\binom Dr,\qquad
N_{\rm dual}=N_{\rm master}=T,\qquad N_{\rm recovery}=1.
\tag{30}
$$

Here $N_{\rm recovery}$ counts the final fixed-selection primal solve and
score construction together.

**Proof.**

(i) Proposition B.1(i) and the extension in
Proposition A.1 give $R(f_s)\le p_s$.
Dual feasibility and $h_s\le H(\Pi^s)$ give $0\le m_t\le F$, hence

$$

L_t\le\min_{z\in\mathcal Z_d}m_t(z)\le\mathrm{OPT}
\le R(f_{s_t})\le p_{s_t}=U_t.

$$

If $z^{t+1}=z^s$ for some $s\le t$, then

$$

U_t-L_t\le p_s-c_s(z^s)+\eta^{t+1}-L_t
\le\delta+\beta\le\varepsilon.

$$

Thus every nonterminal iteration selects a previously unevaluated point,
proving the evaluation bound.

(ii) Scale a normalized optimizer as in (22).
Then $0\le\alpha=F(z)\bar\alpha\le C$ and

$$

d_{kj}(\Pi)=\sum_{i\in I_k,r}\Pi_{ir}\Delta_{ir,j}
\le\sum_{i\in I_k}\alpha q_i\max_r\Delta_{ir,j}=\alpha A_{kj}.

$$

Set $b_j=\max_k d_{kj}(\Pi)/\theta_k$ and
$\mu_j'=\min\{\mu_j,b_j\}$. If $\mu_j>b_j$, then

$$

(d_{kj}(\Pi)-\theta_k\mu_j')_+
=(d_{kj}(\Pi)-\theta_k\mu_j)_+=0\quad(k=0,1).

$$

All other coordinates are unchanged, so $(\Pi,\alpha,\mu')\in\mathcal D$ and

$$

\begin{gathered}
F(z)\ge2\sqrt{H(\Pi)}-\alpha-(\mu')^\top z
\ge2\sqrt{H(\Pi)}-\alpha-\mu^\top z=F(z),\\
0\le\mu_j'\le b_j\le\alpha\max_k A_{kj}/\theta_k\le M_j.
\end{gathered}

$$

Applying the resulting supporting cuts at $z$ and $z'$ gives

$$

-\sum_jM_j|z_j-z_j'|\le F(z)-F(z')\le\sum_jM_j|z_j-z_j'|.

$$

(iii) In (i), take $p_s=c_s(z^s)=F(z^s)$,
$\delta=\beta=0$, and $L_t=\eta^{t+1}$.
The recovered optimum has risk $U_T$, proving (28).
For $s<t$, the decision to evaluate $z^t$ requires

$$

\begin{aligned}
c_s(z^t)&\le\eta^t<U_{t-1}-\varepsilon\le F(z^s)-\varepsilon,\\
\varepsilon&<(\mu^s)^\top(z^t-z^s)
=\sum_{j\in S_t\setminus S_s}\mu_j^s
-\sum_{j\in S_s\setminus S_t}\mu_j^s
\le\sum_{j\in S_t\setminus S_s}M_j\le d_M(S_s,S_t).
\end{aligned}

$$

The evaluated selections therefore form an $\varepsilon$-packing, yielding
$T\le\mathcal P_\varepsilon\le|\mathcal Z_d|$.
Each iteration solves one dual and one master; the last step solves one
fixed-selection primal and constructs its score, proving (30).

$\square$
