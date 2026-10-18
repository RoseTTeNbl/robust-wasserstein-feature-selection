# Variable Selection for Wasserstein Robust Hypothesis Testing

18 October 2026

## Abstract

Variable selection for Wasserstein robust testing admits an exact finite formulation and a maximum-affine dual representation. Dual couplings generate global cuts. The analysis includes error bounds and iteration bounds in terms of solve accuracy and coordinate influence.

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
Proposition B.1(i).

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

Proposition B.1(ii) proves this equivalence and attainment. A normalized optimizer $(\bar\Pi,\bar\alpha,\bar\mu)$ gives the affine optimizer $F(z)(\bar\Pi,\bar\alpha,\bar\mu)$; the scaling must also be applied to the cut slope.

## 3. Cutting-plane algorithm

Let $\mathcal Z_d=\{z\in\{0,1\}^D:\mathbf1^\top z\le d\}$.
At $z^t$, evaluate $F(z^t)$ by (8) and take an optimizer
$(\Pi^t,\alpha^t,\mu^t)$. It gives a global affine lower bound attained at $z^t$:

$$
\begin{aligned}
c_t(z)&:=2\sqrt{H(\Pi^t)}-\alpha^t-(\mu^t)^\top z,\\
F(z)&\ge c_t(z)=F(z^t)-(\mu^t)^\top(z-z^t),\qquad
c_t(z^t)=F(z^t).
\end{aligned}
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

For the iteration bound, define

$$
C=2\ell(0),\qquad
A_{kj}=\sum_{i\in I_k}q_i\max_{r\in[N]}\Delta_{ir,j},\qquad
M_j=C\max_{k=0,1}\frac{A_{kj}}{\theta_k}.
\tag{11}
$$

Order these bounds as $M_{(1)}\ge\cdots\ge M_{(D)}$ and put, for $m=0,\ldots,D$,

$$
\rho_m=\sum_{r=m+1}^{\min(D,m+d)}M_{(r)},\qquad
K_m=\sum_{r=0}^{\min(d,m)}\binom mr.
\tag{12}
$$

Here $\rho_m$ bounds the total influence of at most $d$ coordinates outside
the first $m$, and $K_m$ counts the possible selections on those first $m$ coordinates.

For inexact solves, retain all global affine lower bounds $c_s(z)\le F(z)$ and let $p_s$ be
their paired primal feasible values. Define

$$
\begin{gathered}
U_t=\min_{s\le t}p_s,\qquad
m_t(z)=\max\{0,c_1(z),\ldots,c_t(z)\},\\
\underline L_t\le\min_{z\in\mathcal Z_d}m_t(z),\qquad
L_t=\max\{0,\underline L_1,\ldots,\underline L_t\},\qquad
G_t=U_t-L_t.
\end{gathered}
$$

Thus $\underline L_t$ is a global master lower bound and $G_t$ is the
computed optimality gap. Let $f_t$ be the recovered score paired with $U_t$.

**Proposition 3.1 (Convergence).**

(i) At every $z\in[0,1]^D$, adding $\alpha\le C$ and
$0\le\mu_j\le M_j$ to the affine dual preserves its attained optimal value $F(z)$.

(ii) Fix $\delta,\beta\ge0$. Suppose the returned affine lower bounds have slopes $-\mu^s$,
and each inner solve and master solve satisfies

$$
\begin{gathered}
0\le\mu^s\le M,\qquad 0\le p_s-c_s(z^s)\le\delta,\\
z^{t+1}\in\mathcal Z_d,\qquad
\eta^{t+1}\ge m_t(z^{t+1}),\qquad
0\le\eta^{t+1}-L_t\le\beta.
\end{gathered}
$$

For every completed iteration $t$ and every $m$ with $t\ge K_m$,

$$
0\le R(f_t)-\mathrm{OPT}\le G_t\le\delta+\beta+\rho_m.
\tag{13}
$$

For $\varepsilon\ge\delta+\beta$ with $\varepsilon>0$, stopping at
$G_t\le\varepsilon$ requires at most

$$
T_\varepsilon\le K_{m_\varepsilon},\qquad
m_\varepsilon=\min\{m\in\{0,\ldots,D\}:\rho_m\le\varepsilon-\delta-\beta\}.
\tag{14}
$$

Exact solves correspond to $\delta=\beta=0$ and
$L_t=\eta^{t+1}$. The proof is in Appendix C.

**Algorithm 1. Exact cutting-plane method for variable selection.**

**Input:** Problem data, budget $d$, tolerance $\varepsilon>0$, and $z^1\in\mathcal Z_d$.

**Output:** A subset $S^*$, its optimal test $f^*$, and a risk interval of width at most $\varepsilon$.

1. $t\gets1$, $U\gets+\infty$.

2. **while** true **do**

    3. Solve (8) exactly at $z^t$; choose $\mu^t$ with $0\le\mu_j^t\le M_j$ as in Proposition 3.1(i).

    4. **if** $F(z^t)<U$ **then**

        5. $(U,z^*)\gets(F(z^t),z^t)$.

    6. **end if**

    7. Retain the cut (9) with index $t$.

    8. Solve (10) exactly to obtain $(z^{t+1},\eta^{t+1})$.

    9. **if** $U-\eta^{t+1}\le\varepsilon$ **then**

        10. $S^*\gets\{j:z_j^*=1\}$.

        11. Solve (4) for $S^*$; construct $g^*$ by (20) and set $f^*=g^*\circ\operatorname{Proj}_{S^*}$.

        12. **return** $(S^*,f^*,[\eta^{t+1},U])$.

    13. **end if**

    14. $t\gets t+1$.

15. **end while**

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
\tag{15}
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
\tag{16}
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
\tag{17}
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
\tag{18}
$$

$$
\begin{aligned}
V(S)&\ge\inf_{g\in\mathcal G_S}\sum_k
\max_{\pi\in\mathcal T_k(S)}\sum_{i\in I_k,r}\pi_{ir}\ell_k(g(x_r^S))\\
&\ge\inf_{u\in\mathbb R^N}\sum_k
\max_{\pi\in\mathcal T_k(S)}\sum_{i\in I_k,r}\pi_{ir}\ell_k(u_r)=T(S).
\end{aligned}
\tag{19}
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
\tag{20}
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
\tag{21}
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
Put

$$
\Delta_{ir,j}=|x_{ij}-x_{rj}|,\qquad
c_{ir}^S=\sum_{j\in S}\Delta_{ir,j},\qquad
J(a,\lambda)=\sum_iq_i a_i+\sum_{k=0}^1\theta_k\lambda_k.
$$

For $S\subseteq[D]$ and $z\in[0,1]^D$, consider

$$
\begin{aligned}
V(S)=\min_{a,u,\lambda}\quad&J(a,\lambda)\\
\text{subject to}\quad&
\ell_k(u_r)\le a_i+\lambda_kc_{ir}^S,
\quad i\in I_k,\ r\in[N],\ k=0,1,\\
&a,u\in\mathbb R^N,\qquad\lambda\in\mathbb R_+^2,
\end{aligned}
\tag{22}
$$

$$
\begin{aligned}
F(z)=\min_{a,u,\lambda,w}\quad&J(a,\lambda)\\
\text{subject to}\quad&
\ell_k(u_r)\le a_i+\sum_{j=1}^D\Delta_{ir,j}w_{kj},
\quad i\in I_k,\ r\in[N],\ k=0,1,\\
&0\le w_{kj}\le\lambda_k,
\quad k=0,1,\ j\in[D],\\
&\sum_{k=0}^1\theta_kw_{kj}\le z_jJ(a,\lambda),
\quad j\in[D],\\
&a,u\in\mathbb R^N,\qquad\lambda\in\mathbb R_+^2.
\end{aligned}
\tag{23}
$$

For $\Pi\in\mathbb R_+^{N\times N}$, define

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
\end{array}\right\},\qquad(v)_+=\max\{v,0\}.
\tag{24}
$$

**Proposition B.1 (Dual representations and subgradients).**

(i) For every $S\subseteq[D]$, $F(\mathbf1_S)=V(S)$.
For every $z\in[0,1]^D$, the primal minimum and the normalized dual maximum
are attained, and

$$
0<F(z)=\max_{\substack{(\Pi,\alpha,\mu)\in\mathcal D\\
\alpha+\mu^\top z=1}}H(\Pi)\le2\ell(0).
\tag{25}
$$

(ii) The same value is attained by the affine representation

$$
F(z)=\max_{(\Pi,\alpha,\mu)\in\mathcal D}
\{2\sqrt{H(\Pi)}-\alpha-\mu^\top z\}.
\tag{26}
$$

For any normalized optimizer $(\bar\Pi,\bar\alpha,\bar\mu)$ at $z$,
an affine optimizer is

$$
(\Pi,\alpha,\mu)=F(z)(\bar\Pi,\bar\alpha,\bar\mu),\qquad
H(\Pi)=F(z)^2.
\tag{27}
$$

(iii) The function $F$ is convex. For any affine optimizer
$(\Pi^t,\alpha_t,\mu^t)$ at $z^t$, its affine function satisfies

$$
\begin{aligned}
c_t(z)&:=2\sqrt{H(\Pi^t)}-\alpha_t-(\mu^t)^\top z
\le F(z)\quad(z\in[0,1]^D),\\
c_t(z^t)&=F(z^t),\qquad
c_t(z)=F(z^t)-(\mu^t)^\top(z-z^t).
\end{aligned}
\tag{28}
$$

Thus $-\mu^t\in\partial F(z^t)$, with subgradients understood on
$[0,1]^D$; differentiability is not assumed.

**Proof.**

(i) Diagonal feasibility gives $a_i\ge\ell_k(u_i)\ge0$.
For a feasible triple in (22), let
$z=\mathbf1_S$ and $w_k=\lambda_kz$. Then

$$
\ell_k(u_r)\le a_i+\sum_j\Delta_{ir,j}w_{kj},\qquad
\sum_k\theta_kw_{kj}=z_j\sum_k\theta_k\lambda_k\le z_jJ(a,\lambda).
$$

Conversely, (23) at $z=\mathbf1_S$ gives

$$
\begin{gathered}
w_{kj}=0\quad(j\notin S),\qquad0\le w_{kj}\le\lambda_k\quad(j\in S),\\
\ell_k(u_r)\le a_i+\sum_{j\in S}\Delta_{ir,j}w_{kj}
\le a_i+\lambda_kc_{ir}^S.
\end{gathered}
$$

Both maps preserve $J$, proving binary equivalence.

Assign multipliers $\Pi_{ir},\beta_{kj},\mu_j\ge0$ to the loss,
price-cap and selector inequalities of (23).
Keep $a,u$ free and $\lambda,w\ge0$; put $\alpha=1-\mu^\top z$.
The Lagrangian is

$$
\begin{aligned}
\mathcal L={}&J+\sum_{k,i\in I_k,r}\Pi_{ir}
\left[\ell_k(u_r)-a_i-\sum_j\Delta_{ir,j}w_{kj}\right]
+\sum_{k,j}\beta_{kj}(w_{kj}-\lambda_k)\\
&+\sum_j\mu_j\left[\sum_k\theta_kw_{kj}-z_jJ\right]\\
={}&\sum_i a_i\left(\alpha q_i-\sum_r\Pi_{ir}\right)
+\sum_{r,k,i\in I_k}\Pi_{ir}\ell_k(u_r)\\
&+\sum_k\lambda_k\left(\alpha\theta_k-\sum_j\beta_{kj}\right)
+\sum_{k,j}w_{kj}[\beta_{kj}+\theta_k\mu_j-d_{kj}(\Pi)].
\end{aligned}
\tag{29}
$$

Its infimum over $a,\lambda,w$ is finite precisely when

$$
\sum_r\Pi_{ir}=\alpha q_i,\qquad
\sum_j\beta_{kj}\le\alpha\theta_k,\qquad
\beta_{kj}\ge(d_{kj}(\Pi)-\theta_k\mu_j)_+.
\tag{30}
$$

The row sums force $\alpha\ge0$. By separability over target columns,

$$
\begin{aligned}
\inf_{u\in\mathbb R^N}
\sum_{r=1}^N\sum_{k=0}^1\sum_{i\in I_k}\Pi_{ir}\ell_k(u_r)
&=\sum_{r=1}^N\inf_{s\in\mathbb R}
\sum_{k=0}^1\sum_{i\in I_k}\Pi_{ir}\ell_k(s)\\
&=H(\Pi).
\end{aligned}
\tag{31}
$$

Eliminating $\beta$ is exact because its smallest feasible entries are
$\beta_{kj}=(d_{kj}(\Pi)-\theta_k\mu_j)_+$.
This gives $\mathcal D$ and $\alpha+\mu^\top z=1$.

For $J_0=\{j:z_j=0\}$, feasibility forces $w_{kj}=0$.
After deleting these variables and selector rows, a strict feasible point is

$$
u=0,\qquad a_i=\ell(0)+1,\qquad\lambda_k=1,\qquad
w_{kj}=z_j/2\quad(j\notin J_0).
$$

The loss and price-cap inequalities are strict, and

$$
\sum_k\theta_kw_{kj}=\frac{z_j}{2}\sum_k\theta_k
<z_j\left[2(\ell(0)+1)+\sum_k\theta_k\right]=z_jJ
\quad(j\notin J_0).
$$

Slater's theorem gives strong duality and dual attainment for the reduced
problem, also when $J_0=[D]$. Restore the deleted coordinates with

$$
\beta_{kj}=0,\qquad
\mu_j=\max_{k=0,1}d_{kj}(\Pi)/\theta_k\quad(j\in J_0).
$$

These choices satisfy (30) and change neither the
objective nor normalization.

The constant feasible point $(a,u,\lambda,w)=(\ell(0)\mathbf1,0,0,0)$
has value $C=2\ell(0)$. Its closed sublevel $J\le C$ satisfies

$$
\begin{gathered}
0\le a_i\le C/q_i,\qquad
0\le w_{kj}\le\lambda_k\le C/\theta_k,\\
\ell_k(u_r)\le C/q_i+(C/\theta_k)\sum_j\Delta_{ir,j}
\quad(i\in I_k,\ r\in[N],\ k=0,1).
\end{gathered}
$$

For nonconstant $\ell$, convexity and monotonicity give
$\ell(t)\to\infty$ as $t\to\infty$, so the two losses bound $u$ on both sides.
For constant $\ell$, take $u=0$.
A compact restriction therefore contains a primal minimizer.
If its value were zero, $a=\lambda=w=0$ would imply

$$
0<2\ell(0)\le\ell(-u_r)+\ell(u_r)\le0,
$$

a contradiction. This proves (25) and primal
attainment; binary equivalence gives attainment of (22).

(ii) The set $\mathcal D$ is a cone and
$H(\tau\Pi)=\tau H(\Pi)\ge0$ for $\tau\ge0$.
For a feasible tuple put $b=\alpha+\mu^\top z$.
If $b>0$, normalization and (i) give $H(\Pi)/b\le F(z)$, hence

$$
2\sqrt{H(\Pi)}-b
=\frac{H(\Pi)}b-
\left(\sqrt{\frac{H(\Pi)}b}-\sqrt b\right)^2\le F(z).
\tag{32}
$$

If $b=0$, then $\alpha=0$, $\Pi=0$, and the affine value is zero.
Conversely, any normalized optimizer satisfies

$$
H(F(z)\bar\Pi)=F(z)^2,\qquad
F(z)(\bar\alpha+\bar\mu^\top z)=F(z),\qquad
2\sqrt{H(F(z)\bar\Pi)}-F(z)=F(z).
$$

Thus (26) is attained by (27).

(iii) The feasible set $\mathcal D$ does not depend on $z$;
(26) is a maximum of affine functions and hence convex.
Dual feasibility gives $c_t(z)\le F(z)$, and optimality at $z^t$ gives
$c_t(z^t)=F(z^t)$. Subtracting these values yields
(28) and the stated subgradient inequality.
The normalized multiplier must be scaled as
$\mu^t=F(z^t)\bar\mu^t$.

For an inexact solve, let $(\Pi,\alpha,\mu)\in\mathcal D$,
$0\le h\le H(\Pi)$ and $c(z)=2\sqrt h-\alpha-\mu^\top z$.
If $p_t$ is primal feasible at $z^t$ and $p_t-c(z^t)\le\delta$, then

$$
\begin{gathered}
c(z)\le F(z),\qquad
0\le F(z^t)-c(z^t)\le p_t-c(z^t)\le\delta,\\
F(z)\ge F(z^t)-\mu^\top(z-z^t)-\delta
\quad(z\in[0,1]^D).
\end{gathered}
$$

$\square$

## Appendix C. Proof of convergence

Let $x_i\in\mathbb R^D$, let nonempty $I_0,I_1$ partition $[N]$, and let
$q_i,\theta_k>0$ with $\sum_{i\in I_k}q_i=1$.
The loss $\ell:\mathbb R\to[0,\infty)$ is finite, convex and nondecreasing,
with $\ell(0)>0$. Let $d\in\{0,\ldots,D\}$. Write

$$
F(z)=\max_{(\Pi,\alpha,\mu)\in\mathcal D}
\{2\sqrt{H(\Pi)}-\alpha-\mu^\top z\},\qquad
\mathcal Z_d=\{z\in\{0,1\}^D:\mathbf1^\top z\le d\},\qquad
\mathrm{OPT}=\min_{z\in\mathcal Z_d}F(z),
$$

where $H$ and $\mathcal D$ are defined in Appendix B.
With $\Delta_{ir,j}=|x_{ij}-x_{rj}|$, set

$$
C=2\ell(0),\qquad
A_{kj}=\sum_{i\in I_k}q_i\max_{r\in[N]}\Delta_{ir,j},\qquad
M_j=C\max_{k=0,1}\frac{A_{kj}}{\theta_k}.
$$

Let $J_m$ be the indices of the $m$ largest $M_j$, breaking ties by index.
For $M_{(1)}\ge\cdots\ge M_{(D)}$, define

$$
\rho_m=\sum_{r=m+1}^{\min(D,m+d)}M_{(r)},\qquad
K_m=\sum_{r=0}^{\min(d,m)}\binom mr,\qquad 0\le m\le D.
$$

Fix $\delta,\beta\ge0$ and $\varepsilon>0$.
At each evaluated $z^s=\mathbf1_{S_s}$, take a feasible primal value
$p_s$ and recover $f_s=g_s\circ\operatorname{Proj}_{S_s}$, so that $R(f_s)\le p_s$.
The paired affine lower bound is

$$
\begin{gathered}
c_s(z)=2\sqrt{h_s}-\alpha_s-(\mu^s)^\top z,\qquad
(\Pi^s,\alpha_s,\mu^s)\in\mathcal D,\qquad 0\le h_s\le H(\Pi^s),\\
0\le\mu^s\le M,\qquad 0\le p_s-c_s(z^s)\le\delta.
\end{gathered}
$$

Retain all cuts and define

$$
\begin{gathered}
m_t(z)=\max\{0,c_1(z),\ldots,c_t(z)\},\qquad
U_t=\min_{s\le t}p_s,\qquad f_t=f_{s_t},\quad
s_t\in\operatorname*{arg\,min}_{s\le t}p_s,\\
\underline L_t\le\min_{z\in\mathcal Z_d}m_t(z),\qquad
L_t=\max\{0,\underline L_1,\ldots,\underline L_t\},\qquad G_t=U_t-L_t.
\end{gathered}
$$

The master returns $z^{t+1}\in\mathcal Z_d$ and $\eta^{t+1}$ satisfying

$$
\eta^{t+1}\ge m_t(z^{t+1}),\qquad
0\le\eta^{t+1}-L_t\le\beta.
$$

**Proof of Proposition 3.1.**

(i) Scale a normalized optimizer by $F(z)$ as in Appendix B.
Since $\bar\alpha+\bar\mu^\top z=1$ and $F(z)\le C$,

$$
0\le\alpha=F(z)\bar\alpha\le C,\qquad
d_{kj}(\Pi)=\sum_{i\in I_k,r}\Pi_{ir}\Delta_{ir,j}
\le\alpha\sum_{i\in I_k}q_i\max_r\Delta_{ir,j}=\alpha A_{kj}.
$$

Put $b_j=\max_k d_{kj}(\Pi)/\theta_k$ and
$\mu_j'=\min\{\mu_j,b_j\}$. Whenever $\mu_j>b_j$,

$$
(d_{kj}(\Pi)-\theta_k\mu_j')_+
=(d_{kj}(\Pi)-\theta_k\mu_j)_+=0\quad(k=0,1).
$$

Thus $(\Pi,\alpha,\mu')\in\mathcal D$ and

$$
\begin{gathered}
F(z)\ge2\sqrt{H(\Pi)}-\alpha-(\mu')^\top z
\ge2\sqrt{H(\Pi)}-\alpha-\mu^\top z=F(z),\\
0\le\mu_j'\le b_j\le\alpha\max_k A_{kj}/\theta_k\le M_j.
\end{gathered}
$$

The bounded dual therefore retains an optimizer and the same value.

(ii) Primal feasibility, dual feasibility and $h_s\le H(\Pi^s)$ give

$$
0\le m_t\le F,\qquad
L_t\le\min_{z\in\mathcal Z_d}m_t(z)\le\mathrm{OPT}
\le R(f_t)\le U_t.
\tag{33}
$$

Because all cuts are retained, $m_t$ and $L_t$ are nondecreasing, whereas
$U_t$ and $G_t$ are nonincreasing.
For $s<r\le t+1$, the master point $z^r=\mathbf1_{S_r}$ satisfies

$$
\begin{aligned}
G_{r-1}
&\le p_s-c_s(z^r)+\eta^r-L_{r-1}\\
&=p_s-c_s(z^s)+(\mu^s)^\top(z^r-z^s)+\eta^r-L_{r-1}\\
&\le\delta+\beta+\sum_{j\in S_r\setminus S_s}M_j.
\end{aligned}
\tag{34}
$$

Fix $m$ and suppose $G_t>\delta+\beta+\rho_m$.
Monotonicity and (34) imply

$$
\sum_{j\in S_r\setminus S_s}M_j>\rho_m\qquad(s<r\le t+1).
$$

If $S_r\cap J_m=S_s\cap J_m$, then
$S_r\setminus S_s\subseteq[D]\setminus J_m$ and $|S_r\setminus S_s|\le d$, so

$$
\sum_{j\in S_r\setminus S_s}M_j
\le\sum_{q=m+1}^{\min(D,m+d)}M_{(q)}=\rho_m,
$$

a contradiction. Thus $S_1\cap J_m,\ldots,S_{t+1}\cap J_m$ are pairwise
distinct, but there are only $K_m$ such subsets. Hence $t+1\le K_m$.
For $t\ge K_m$ this is impossible, proving

$$
0\le R(f_t)-\mathrm{OPT}\le G_t\le\delta+\beta+\rho_m
\qquad(t\ge K_m).
$$

For $\varepsilon\ge\delta+\beta$, set
$m_\varepsilon=\min\{m\in\{0,\ldots,D\}:\rho_m\le\varepsilon-\delta-\beta\}$.
This gives $G_{K_{m_\varepsilon}}\le\varepsilon$ and $T_\varepsilon\le K_{m_\varepsilon}$,
unless the method stopped earlier. The last master candidate is included in
the counting argument without an additional inner solve. In particular,

$$
\rho_D=0,\quad K_D=\sum_{r=0}^d\binom Dr,\qquad
d=0\ \text{or}\ M=0\quad\Longrightarrow\quad G_1\le\delta+\beta.
$$

$\square$
