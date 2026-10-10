# 同一点的切线选择：risk core_point 的数学含义

审核范围仅为现有 `risk` oracle 的二阶段选择，不改变候选子集排序，也不增加分数点求解。仍然完整求解当前 binary z，最后只保存当前 z 的一条切线。

## 归一化为什么给出当前二阶段目标

固定已选 z，归一化对偶满足源行质量 \(\bar\Gamma\mathbf1=\tau q\) 及

\[
\tau+\bar\mu^Tz=1.
\]

记 \(h=H(\bar\Gamma)\)。该证书在其它选择点 x 的原始分式及其在 z 的切线是

\[
f(x)=\frac{h}{\tau+\bar\mu^Tx},\qquad
f(z)=h,\qquad \nabla f(z)=-h\bar\mu,
\]
\[
L(x)=h-h\bar\mu^T(x-z).
\]

这里的 \(\bar\mu\) 是归一化后的变量。若改用未归一化 \(\mu\)，必须同时保留分母及导数的缩放，不能直接套同一个目标。

第一阶段取得精确最大值 \(h_*>0\)。限制在真正最优面 \(h=h_*\) 上，给定内部参考点 \(c=(d/D)\mathbf1\)，有

\[
\arg\max L(c)=\arg\min \bar\mu^T(c-z),
\qquad L(c)=h_*[1-\bar\mu^T(c-z)].
\]

这正是当前 risk 二阶段：overlap LP 的 `second[mu0:tau0] = core - z`（oracle.py:370），logistic cone 的 `second_objective = -mu @ (core-z)`（oracle.py:543）。若最优值为零，切线也为零，此选择没有非零改进可言。

直观上，很多对偶解在当前已求解点同样准确，但它们朝其它子集延伸的斜率不同。二阶段挑选在参考点 c 上给出最高下界的切线。它选择同一点的证书，既不产生新的候选点评价，也不使用旧点评价替代当前完整求解。

## 当前数值实现能保证什么

实现使用的是容差面：LP 允许线性目标距第一阶段目标约1e-9（oracle.py:375–377），cone 使用 \(h\ge h_1-10^{-8}\)（oracle.py:541）。第一阶段本身也由浮点求解器提供。因此 h 并非严格固定；只优化 \(\bar\mu^T(c-z)\) 不再严格等于优化全部 \(h[1-\bar\mu^T(c-z)]\)。

原始 mu 未直接返回。`_repair_dual`（oracle.py:576）从 plans、beta、tau 重建最小覆盖价格，应用数据界，重新归一化并重算 h，最后把切线下移 margin（oracle.py:623）。这些步骤保证修复后的证书有效，但没有显式约束其仍处于数值最优面，或保证它在 c 比第一阶段修复后的切线更高。评价精度仍应检查实际的 \(R_s-\ell_s\le\delta\)，包括下移量。

有一个精确算术下的限定结论：若原证书严格可行、恰在正目标真最优面，重建得到 \(\mu_{min}\le\bar\mu\)。此时归一化因子 \(n=\tau+\mu_{min}^Tz\le1\)，而新的可行目标是 \(h_*/n\)。最优性迫使 n=1。因此这类理想修复保持真最优值；消除的仅为不影响当前点的冗余价格。这个结论依赖真最优性和精确可行性，不能直接用于当前容差面与浮点修复。

## 在参考点更高不意味着所有子集都更高

两条同值切线的差为

\[
L_1(x)-L_2(x)=-h_*(\bar\mu_1-\bar\mu_2)^T(x-z).
\]

只在 c 优化一个内积，不会逐坐标约束这个差在所有可行 x 上的符号。因此 core-point 不保证支配默认切线，也不保证更少完整评价。mu clipping 的目的也是界与可行性，不是证明不同证书之间的全域支配。上述理想最小价格修复可以不降低其自己的 raw cut；这仍不推出 core-point cut 支配另一条 default cut。

## 真实成本与归类

- overlap：通常从 primal LP＋dual LP 两次变成另加一次二阶段 LP；仅在第一阶段 optimal 时启动。
- logistic：通常另加一次二阶段 cone solve，且 `core_point=True` 停用现有 risk persistent primal/dual 路径（oracle.py:465、530），每次重新构建/canonicalize；这也是额外成本。
- 二阶段共享该 oracle 的剩余 deadline，不增加允许时间；可能因多一次求解削弱最终精度或耗尽时间。
- 无论二阶段成功与否，修复与实际 gap 检查保留，外层仍只获得一条 own-point cut；不能把收益记为候选排序改进。

应报告完整评价数、实际 primal/dual/secondary 连续求解数、构建成本与总时间。该解释只支持‘同点切线选择’，不支持未测量的查询或运行时间提升。
