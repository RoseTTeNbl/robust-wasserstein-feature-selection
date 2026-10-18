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
giving $V([D])$ for every $z$. Retaining it gives $F(\mathbf1_S)=V(S)$; Proposition B.1 proves equality of the projected feasible sets and preservation of the objective.

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
Proposition B.2.

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

Proposition B.3 proves this equivalence and attainment. A normalized optimizer $(\bar\Pi,\bar\alpha,\bar\mu)$ gives the affine optimizer $F(z)(\bar\Pi,\bar\alpha,\bar\mu)$; the scaling must also be applied to the cut slope.

## 3. Cutting-plane algorithm

Let $\mathcal Z_d=\{z\in\{0,1\}^D:\mathbf1^\top z\le d\}$.
At $z^t$, evaluate $F(z^t)$ by (8) and take an optimizer
$(\Pi^t,\alpha^t,\mu^t)$ with the bounds $\mu_j^t\le M_j$ from
Proposition C.2. Its subgradient $-\mu^t$ gives

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

Proposition C.1 proves the certificate and the sufficient
accuracy condition $\delta+\beta\le\varepsilon$ for finite termination.
Propositions C.2 and C.3 additionally give
bounded slopes and a data-dependent separation bound for exact search.

**Algorithm 1. Exact cutting-plane method for variable selection.**

**Input:** Problem data, budget $d$, tolerance $\varepsilon>0$, and $z^1\in\mathcal Z_d$.

**Output:** A subset $S^*$, its optimal test $f^*$, and a risk interval of width at most $\varepsilon$.

1. $t\gets1$, $U\gets+\infty$.

2. **While** true:

    3. Solve (8) exactly at $z^t$; choose $\mu^t$ with $0\le\mu_j^t\le M_j$ as in Proposition C.2.

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

*Attainment.*
The feasible point $(a,u,\lambda)=(\ell(0)\mathbf1,0,0)$ has value
$C=2\ell(0)$. Diagonal feasibility and $J\le C$ give

$$

0\le a_i\le C/q_i,\qquad 0\le\lambda_k\le C/\theta_k,\qquad
\ell_k(u_r)\le\min_{i\in I_k}\{C/q_i+(C/\theta_k)c_{ir}^S\}.

$$

A nonconstant convex nondecreasing $\ell$ satisfies
$\ell(t)\to\infty$ as $t\to\infty$; the two losses bound $u$ on both sides.
For constant loss take $u=0$. A nonempty compact restriction of this
closed sublevel therefore contains a minimizer; its extension attains $V(S)$.

$\square$

## Appendix B. Proofs of the three dual reformulations

Let $I_0,I_1$ partition $[N]$, with $q_i>0$,
$\sum_{i\in I_k}q_i=1$, and $\theta_k>0$.
Let $\ell:\mathbb R\to[0,\infty)$ be finite, convex and nondecreasing,
with $\ell(0)>0$, $\ell_0(s)=\ell(-s)$ and $\ell_1(s)=\ell(s)$.
For data $x_i\in\mathbb R^D$, put

$$

\Delta_{ir,j}=|x_{ij}-x_{rj}|,\qquad
c_{ir}^S=\sum_{j\in S}\Delta_{ir,j},\qquad
J(a,\lambda)=\sum_iq_i a_i+\sum_k\theta_k\lambda_k.

$$

Define the finite feasible set

$$

\mathcal C_S=\{(a,u,\lambda):
a,u\in\mathbb R^N,\ \lambda\ge0,\
\ell_k(u_r)\le a_i+\lambda_kc_{ir}^S
\ (i\in I_k,\ r\in[N],\ k=0,1)\}.

$$

Set $V(S):=\min_{\mathcal C_S}J$.
Proposition A.1 identifies this value with the original
fixed-subset distributional risk.
For $z\in[0,1]^D$, define

$$
\begin{aligned}
\mathcal E_z=\{(a,u,\lambda,w):\quad&
a,u\in\mathbb R^N,\ \lambda\ge0,\quad
0\le w_{kj}\le\lambda_k,\\
&\ell_k(u_r)\le a_i+\sum_j\Delta_{ir,j}w_{kj}
\ (i\in I_k,\ r\in[N],\ k=0,1),\\
&\sum_k\theta_kw_{kj}\le z_jJ(a,\lambda)\ (j\in[D])\},
\qquad F(z)=\min_{\mathcal E_z}J.
\end{aligned}
\tag{19}
$$

**Proposition B.1 (Binary extension).**

For every $S\subseteq[D]$, projection onto the original variables gives

$$

\operatorname{Proj}_{(a,u,\lambda)}\mathcal E_{\mathbf1_S}
=\mathcal C_S,\qquad F(\mathbf1_S)=V(S).

$$

Both directions preserve the same objective $J(a,\lambda)$.

**Proof.**

Let $z=\mathbf1_S$. For $(a,u,\lambda)\in\mathcal C_S$,
diagonal feasibility gives $a_i\ge0$. Set $w_k=\lambda_kz$; then

$$

\ell_k(u_r)\le a_i+\sum_j\Delta_{ir,j}w_{kj},\qquad
\sum_k\theta_kw_{kj}=z_j\sum_k\theta_k\lambda_k\le z_jJ(a,\lambda).

$$

Thus $(a,u,\lambda,w)\in\mathcal E_z$ with the same $J$.
Conversely, $(a,u,\lambda,w)\in\mathcal E_z$ gives

$$

w_{kj}=0\quad(j\notin S),\qquad
0\le w_{kj}\le\lambda_k\quad(j\in S),\qquad
\ell_k(u_r)\le a_i+\sum_{j\in S}\Delta_{ir,j}w_{kj}
\le a_i+\lambda_kc_{ir}^S.

$$

Hence $(a,u,\lambda)\in\mathcal C_S$, again with the same $J$.
The two inclusions prove the claim. The selector row is redundant at
$z_j=1$ and enforces $w_{kj}=0$ at $z_j=0$; deleting all selector rows
would instead give $V([D])$ at every $z$.

$\square$

For $\Pi\in\mathbb R_+^{N\times N}$, define

$$

H(\Pi)=\sum_{r=1}^N\inf_{s\in\mathbb R}
\sum_{k=0}^1\sum_{i\in I_k}\Pi_{ir}\ell_k(s),\qquad
d_{kj}(\Pi)=\sum_{i\in I_k,r}\Pi_{ir}\Delta_{ir,j}.

$$

The symbol $\Pi$ is a scaled transport coupling, not necessarily a
probability coupling. With $(v)_+=\max\{v,0\}$, set

$$
\mathcal D=\left\{(\Pi,\alpha,\mu)\ge0:
\sum_r\Pi_{ir}=\alpha q_i\ (i\in[N]),\quad
\sum_j(d_{kj}(\Pi)-\theta_k\mu_j)_+
\le\alpha\theta_k\ (k=0,1)\right\}.
\tag{20}
$$

**Proposition B.2 (Normalized dual).**

For every $z\in[0,1]^D$, the extended minimum is attained and

$$
0<F(z)=
\max_{\substack{(\Pi,\alpha,\mu)\in\mathcal D\\
\alpha+\mu^\top z=1}}H(\Pi)\le2\ell(0).
\tag{21}
$$

The normalized dual maximum is attained.

**Proof.**

*Lagrangian.*
Assign nonnegative multipliers $\Pi_{ir},\beta_{kj},\mu_j$ to the loss,
price-cap and selector inequalities, respectively; keep $\lambda,w\ge0$.

$$
\begin{aligned}
\mathcal L={}&J(a,\lambda)
+\sum_{k,i\in I_k,r}\Pi_{ir}
\left[\ell_k(u_r)-a_i-\sum_j\Delta_{ir,j}w_{kj}\right]\\
&+\sum_{k,j}\beta_{kj}(w_{kj}-\lambda_k)
+\sum_j\mu_j\left[\sum_k\theta_kw_{kj}-z_jJ(a,\lambda)\right].
\end{aligned}
\tag{22}
$$

Writing $\alpha=1-\mu^\top z$,

$$
\begin{aligned}
\mathcal L={}&\sum_i a_i\left(\alpha q_i-\sum_r\Pi_{ir}\right)
+\sum_{k,i\in I_k,r}\Pi_{ir}\ell_k(u_r)\\
&+\sum_k\lambda_k\left(\alpha\theta_k-\sum_j\beta_{kj}\right)
+\sum_{k,j}w_{kj}[\beta_{kj}+\theta_k\mu_j-d_{kj}(\Pi)].
\end{aligned}
\tag{23}
$$

The infimum over free $a$ and nonnegative $\lambda,w$ is finite exactly when

$$
\sum_r\Pi_{ir}=\alpha q_i,\qquad
\sum_j\beta_{kj}\le\alpha\theta_k,\qquad
\beta_{kj}\ge d_{kj}(\Pi)-\theta_k\mu_j.
\tag{24}
$$

The row sums force $\alpha\ge0$. Elimination of $\beta\ge0$ is exact:

$$

\min_{\beta_{kj}\ge(d_{kj}(\Pi)-\theta_k\mu_j)_+}\sum_j\beta_{kj}
=\sum_j(d_{kj}(\Pi)-\theta_k\mu_j)_+.

$$

Minimizing over $u$ gives $H(\Pi)$, yielding the normalized dual.

*Strong duality.*
For $J_0=\{j:z_j=0\}$, feasibility forces $w_{kj}=0$.
Delete those variables and selector rows. A strict feasible point of the reduced problem is

$$

u=0,\qquad a_i=\ell(0)+1,\qquad\lambda_k=1,\qquad
w_{kj}=z_j/2\quad(j\notin J_0),

$$

$$

\ell_k(0)<a_i+\sum_j\Delta_{ir,j}w_{kj},\qquad
0<w_{kj}<\lambda_k,\qquad
\sum_k\theta_kw_{kj}<z_jJ(a,\lambda).

$$

This also covers $J_0=[D]$ after omitting $w$.
Slater's theorem gives strong duality and dual attainment for the reduced problem.
Restore the deleted-coordinate inequalities with

$$

\beta_{kj}=0,\qquad \mu_j=\max_{k=0,1}d_{kj}(\Pi)/\theta_k
\quad(j\in J_0).

$$

Neither normalization nor objective changes, proving full dual attainment.

*Primal attainment and positivity.*
The feasible point $(a,u,\lambda,w)=(\ell(0)\mathbf1,0,0,0)$ has
value $C=2\ell(0)$. On $J\le C$,

$$

0\le a_i\le C/q_i,\qquad
0\le w_{kj}\le\lambda_k\le C/\theta_k,\qquad
\ell_k(u_r)\le C/q_i+(C/\theta_k)\sum_j\Delta_{ir,j}.

$$

As in the finite problem, these bounds give a compact restriction
containing a minimizer (set $u=0$ for constant loss).
If its value were zero, $a=\lambda=w=0$ would give

$$

0<2\ell(0)\le\ell(-u_r)+\ell(u_r)\le0,

$$

a contradiction.

$\square$

**Proposition B.3 (Affine dual and supporting cuts).**

For every $z\in[0,1]^D$,

$$
F(z)=\max_{(\Pi,\alpha,\mu)\in\mathcal D}
\{2\sqrt{H(\Pi)}-\alpha-\mu^\top z\}.
\tag{25}
$$

If $(\bar\Pi,\bar\alpha,\bar\mu)$ is a normalized dual optimizer at $z$,
an affine dual optimizer is

$$
(\Pi,\alpha,\mu)=F(z)(\bar\Pi,\bar\alpha,\bar\mu),
\qquad H(\Pi)=F(z)^2.
\tag{26}
$$

Consequently $F$ is convex. For an affine optimizer
$(\Pi^t,\alpha^t,\mu^t)$ at $z^t$,

$$

-\mu^t\in\partial F(z^t),\qquad
F(z)\ge F(z^t)-(\mu^t)^\top(z-z^t)\quad(z\in[0,1]^D).

$$

Here the subgradient inequality is understood on the cube.

**Proof.**

The set $\mathcal D$ is a cone and $H(\tau\Pi)=\tau H(\Pi)\ge0$
for $\tau\ge0$. For $(\Pi,\alpha,\mu)\in\mathcal D$, put
$b=\alpha+\mu^\top z$. If $b>0$, normalization by $b$ gives
$H(\Pi)/b\le F(z)$, hence

$$
2\sqrt{H(\Pi)}-b
=\frac{H(\Pi)}b
-\left(\sqrt{\frac{H(\Pi)}b}-\sqrt b\right)^2
\le F(z).
\tag{27}
$$

If $b=0$, nonnegativity and the row sums give $\alpha=0$, $\Pi=0$,
so the affine value is zero. Conversely, a normalized optimizer has
$H(\bar\Pi)=F(z)>0$, and

$$

H(F(z)\bar\Pi)=F(z)^2,\qquad
F(z)(\bar\alpha+\bar\mu^\top z)=F(z),\qquad
2\sqrt{H(F(z)\bar\Pi)}-F(z)=F(z).

$$

Thus the affine maximum is attained at the stated rescaling.
For its optimizer at $z^t$,

$$

F(z)\ge2\sqrt{H(\Pi^t)}-\alpha^t-(\mu^t)^\top z
=F(z^t)-(\mu^t)^\top(z-z^t).

$$

A supremum of affine functions is convex; this inequality gives
the subgradient $-\mu^t=-F(z^t)\bar\mu^t$.
Since $H$ is a nonnegative infimum of linear functions,
$H$ and $\sqrt H$ are concave, so the fixed-$z$ dual is a convex program.

$\square$

## Appendix C. Risk certificates and search cost

Let

$$

\mathcal Z_d=\{\mathbf1_S:S\subseteq[D],\ |S|\le d\},\qquad
\mathrm{OPT}=\min_{z\in\mathcal Z_d}F(z),\qquad F(\mathbf1_S)=V(S).

$$

At $z^s$, let $p_s$ be a primal feasible value, $f_s$ its recovered score,
and $c_s$ a dual feasible affine cut:

$$

R(f_s)\le p_s,\qquad c_s(z)\le F(z)\quad(z\in[0,1]^D).

$$

The cut uses exact $H(\Pi)$ or a certified lower bound $0\le h\le H(\Pi)$.
A trial score supplies an upper bound on this infimum and cannot certify a cut.

**Proposition C.1 (Certificate with inexact solves).**

After $t$ evaluations define

$$

U_t=\min_{s\le t}p_s,\qquad
m_t(z)=\max\{0,c_1(z),\ldots,c_t(z)\}.

$$

Let $(z^{t+1},\eta^{t+1})$ be any feasible master solution with
$\eta^{t+1}\ge m_t(z^{t+1})$, and let $L_t$ be a valid global lower
bound on $\min_{z\in\mathcal Z_d}m_t(z)$.
If $f_{\rm inc}$ is the score attached to the smallest $p_s$, then

$$
L_t\le\mathrm{OPT}\le R(f_{\rm inc})\le U_t,\qquad
0\le R(f_{\rm inc})-\mathrm{OPT}\le U_t-L_t.
\tag{28}
$$

For $\delta,\beta\ge0$ and $\varepsilon>0$, if every evaluated point satisfies

$$

p_s-c_s(z^s)\le\delta,\qquad
\eta^{t+1}-L_t\le\beta,\qquad
\delta+\beta\le\varepsilon,

$$

then a repeated selection $z^{t+1}=z^s$ implies
$U_t-L_t\le\varepsilon$. Thus an implementation that stops when
$U_t-L_t\le\varepsilon$ makes at most $|\mathcal Z_d|$ evaluations,
provided these accuracies hold at each iteration.

**Proof.**

Validity of the cuts and primal feasibility give

$$

0\le m_t\le F,\qquad
L_t\le\min_{\mathcal Z_d}m_t\le\mathrm{OPT}
\le R(f_{\rm inc})\le U_t.

$$

If $z^{t+1}=z^s$, then $U_t\le p_s$ and
$\eta^{t+1}\ge c_s(z^s)$, so

$$

U_t-L_t\le p_s-c_s(z^s)+\eta^{t+1}-L_t
\le\delta+\beta\le\varepsilon.

$$

Thus every nonterminal iteration selects a new element of
$\mathcal Z_d$. The bound uses the global master bound $L_t$;
the feasible master objective $\eta^{t+1}$ alone is insufficient.

$\square$

**Proposition C.2 (Data-dependent bounded slopes).**

Let $\Delta_{ir,j}=|x_{ij}-x_{rj}|$, $q_i>0$ with
$\sum_{i\in I_k}q_i=1$, and $\theta_k>0$, as in the extended model.
Put

$$
C=2\ell(0),\qquad
A_{kj}=\sum_{i\in I_k}q_i\max_{r\in[N]}\Delta_{ir,j},\qquad
M_j=C\max_{k=0,1}\frac{A_{kj}}{\theta_k}.
\tag{29}
$$

At every $z\in[0,1]^D$, there exists an affine dual optimizer with
$0\le\mu_j\le M_j$ for every $j$. These choices imply

$$

|F(z)-F(z')|\le\sum_jM_j|z_j-z_j'|
\qquad(z,z'\in[0,1]^D).

$$

**Proof.**

For the scaled optimizer of Proposition B.3,
$0\le\bar\alpha\le1$ gives $0\le\alpha=F(z)\bar\alpha\le F(z)\le C$.
The row sums imply
$d_{kj}(\Pi)=\sum_{i\in I_k,r}\Pi_{ir}\Delta_{ir,j}\le\alpha A_{kj}$.
Set $g_j=\max_k d_{kj}(\Pi)/\theta_k$ and
$\mu_j'=\min\{\mu_j,g_j\}$. Clipped coordinates have
$(d_{kj}(\Pi)-\theta_k\mu_j')_+=0$; all other coordinates are unchanged.
Thus $(\Pi,\alpha,\mu')\in\mathcal D$, and

$$

F(z)\ge2\sqrt{H(\Pi)}-\alpha-(\mu')^\top z
\ge2\sqrt{H(\Pi)}-\alpha-\mu^\top z=F(z),\qquad
0\le\mu_j'\le g_j\le\alpha\max_k A_{kj}/\theta_k\le M_j.

$$

Supporting cuts with these multipliers at $z$ and $z'$ give

$$

-\sum_jM_j|z_j-z_j'|\le F(z)-F(z')\le\sum_jM_j|z_j-z_j'|.

$$

$\square$

**Proposition C.3 (Exact search and separation).**

Suppose every fixed-selection problem and master problem is solved
exactly. At each evaluated point choose the bounded affine optimizer
of Proposition C.2, and stop when
$U_t-\eta^{t+1}\le\varepsilon$, where
$U_t=\min_{s\le t}F(z^s)$.
For the returned score $f^*$ and final iteration $T$,

$$
0\le R(f^*)-\mathrm{OPT}\le U_T-\eta^{T+1}\le\varepsilon.
\tag{30}
$$

For every later evaluated selection $S_t$ and earlier selection
$S_s$,

$$
\sum_{j\in S_t\setminus S_s}M_j>\varepsilon,\qquad s<t.
\tag{31}
$$

Define the weighted pseudodistance
$d_M(S,T)=\sum_{j\in S\triangle T}M_j$ and the finite packing number

$$

\mathcal P_\varepsilon=
\max\{|\mathcal A|:\mathcal A\subseteq\{S:|S|\le d\},\
d_M(S,T)>\varepsilon\text{ for distinct }S,T\in\mathcal A\}.

$$

Allowing $M_j=0$,

$$
T\le\mathcal P_\varepsilon\le\sum_{r=0}^d\binom Dr,\qquad
\#\mathrm{oracle}=T,\quad
\#\mathrm{master}=T,\quad
\#\mathrm{recovery}=1.
\tag{32}
$$

This data-dependent packing bound need not be polynomial.

**Proof.**

In Proposition C.1, take
$p_s=c_s(z^s)=F(z^s)$, $\delta=\beta=0$, and $L_t=\eta^{t+1}$.
The recovered optimum has risk $U_T$, proving the certificate.
For $s<t$, nontermination before evaluating $S_t$ gives

$$

\begin{aligned}
c_s(\mathbf1_{S_t})&\le\eta^t<U_{t-1}-\varepsilon
\le F(\mathbf1_{S_s})-\varepsilon,\\
\varepsilon&<(\mu^s)^\top(\mathbf1_{S_t}-\mathbf1_{S_s})
=\sum_{j\in S_t\setminus S_s}\mu_j^s
-\sum_{j\in S_s\setminus S_t}\mu_j^s
\le\sum_{j\in S_t\setminus S_s}M_j\le d_M(S_s,S_t).
\end{aligned}

$$

The evaluated selections form an $\varepsilon$-packing, so $T\le\mathcal P_\varepsilon$.
Algorithm 1 gives the solve counts.
Master $t$ has $D$ binary variables, one continuous variable and $t$ cuts;
these counts do not bound individual solve times.

$\square$
