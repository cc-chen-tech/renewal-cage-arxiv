# Delayed Renewal Cage Model 中文推导

这份笔记给出当前最有发表潜力的理论进展：一个闭式可解的 delayed renewal cage
model。它比前面的 static mixture 和 switching harmonic cage 更进一步，因为它同时给出：

```text
1. local cage plateau
2. cage renewal 后的 long-time diffusion
3. NGP 从 0 起步
4. NGP 出现有限时间 peak
5. NGP 长时间按 1/t 衰减
```

## 1. 物理图像

粒子在短时间内被邻居形成的 cage 限制，做近似 harmonic Ornstein-Uhlenbeck motion。
经过一段延迟后，cage rearrangement 发生，cage center 跳到新位置。多次 cage
renewal 后，粒子恢复长时间扩散。

把位移写成：

```text
Delta x(t) = local cage displacement + sum of cage-center jumps
```

其中 local cage displacement 是 Gaussian，cage-center jumps 也是 Gaussian，但 jump
的个数是随机的。因此整体分布是 Gaussian variance mixture，会产生 NGP。

## 2. Local Cage Variance

local cage 部分取一维 Gaussian，方差为：

```text
L(t) = A [1 - exp(-t/tau_c)]
```

其中：

```text
A     = cage plateau variance
tau_c = local cage relaxation time
```

极限：

```text
t << tau_c:  L(t) ~ (A/tau_c)t
t >> tau_c:  L(t) -> A
```

这给出 MSD plateau。

## 3. Delayed Cage Renewal

如果 cage renewal 是普通 Poisson process，短时间就有非零跳跃概率，会导致 NGP 在
`t -> 0` 时出现不物理的奇异行为。为避免这个问题，引入 delayed renewal intensity：

```text
r(t) = lambda [1 - exp(-t/tau_d)]^2
```

其中：

```text
lambda = long-time renewal rate
tau_d  = renewal onset delay time
```

短时：

```text
r(t) ~ lambda t^2/tau_d^2
```

所以 cage renewal 在短时间被抑制。

renewal count `N(t)` 服从 inhomogeneous Poisson process，平均 renewal count 为：

```text
R(t) = integral_0^t r(u) du
```

积分可得闭式：

```text
R(t)
  = lambda [t - 2 tau_d(1 - exp(-t/tau_d))
            + (tau_d/2)(1 - exp(-2t/tau_d))]
```

短时展开：

```text
R(t) ~ lambda t^3/(3 tau_d^2)
```

长时：

```text
R(t) ~ lambda t
```

## 4. Conditional Gaussian Variance

设每次 cage-center jump 的一维方差为：

```text
q
```

给定 renewal count `N(t)=n`，总位移仍然是 Gaussian，其方差为：

```text
V(t | n) = L(t) + n q
```

因为 `N(t)` 是 Poisson variable，满足：

```text
E[N] = R(t)
Var[N] = R(t)
```

所以：

```text
E[V] = L(t) + q R(t)
Var[V] = q^2 R(t)
```

## 5. MSD 和 NGP 的闭式表达

二阶矩：

```text
M2(t) = <Delta x^2> = L(t) + q R(t)
```

四阶矩：

给定 `V` 时：

```text
<Delta x^4 | V> = 3V^2
```

所以：

```text
M4(t)
  = 3 E[V^2]
  = 3(E[V]^2 + Var[V])
  = 3[(L(t)+qR(t))^2 + q^2 R(t)]
```

一维 NGP：

```text
alpha_2(t)
  = M4(t)/(3M2(t)^2) - 1
```

代入得：

```text
alpha_2(t)
  = q^2 R(t) / [L(t) + qR(t)]^2
```

这是模型的核心结果。

## 5.1 三维 NGP

如果位移是三维 isotropic Gaussian variance mixture，那么给定 scalar variance `V`
时：

```text
<r^2 | V> = 3V
<r^4 | V> = 15V^2
```

论文中常用的三维 NGP 是：

```text
alpha_2^3D(t)
  = 3<r^4> / [5<r^2>^2] - 1
```

代入可得：

```text
alpha_2^3D(t)
  = E[V^2]/E[V]^2 - 1
  = Var[V]/E[V]^2
  = q^2 R(t) / [L(t) + qR(t)]^2
```

所以一维 NGP 和三维标准 NGP 在这个模型中有同一个核心表达式。

## 5.2 Van Hove Distribution

三维 radial van Hove distribution 可以写成 Poisson 加权的 Maxwell 分布混合：

```text
G_s(r,t)
  = sum_n P[N(t)=n] sqrt(2/pi) r^2 / V_n(t)^(3/2)
      exp[-r^2/(2V_n(t))]
```

其中：

```text
V_n(t) = L(t) + nq
```

这给出一个可直接画图的预测：在 NGP peak 附近，`G_s(r,t)` 会有比单一 Gaussian
更宽的尾部；长时间 renewal count 变大以后，相对方差下降，分布重新接近 Gaussian。

## 6. 短时和长时极限

短时：

```text
L(t) ~ (A/tau_c)t
R(t) ~ lambda t^3/(3 tau_d^2)
```

所以：

```text
alpha_2(t)
  ~ [q^2 lambda tau_c^2 / (3 A^2 tau_d^2)] t
```

因此：

```text
alpha_2(0) = 0
```

并且 NGP 从 0 线性起步。

长时：

```text
L(t) -> A
R(t) ~ lambda t
```

当 `q lambda t >> A` 时：

```text
alpha_2(t) ~ 1/(lambda t)
```

因此模型自动恢复 Gaussian long-time limit。

## 7. Peak 条件

核心公式：

```text
alpha_2(t) = q^2 R(t) / [L(t) + qR(t)]^2
```

求导可得 peak 条件：

```text
R'(t)[L(t) - qR(t)] - 2R(t)L'(t) = 0
```

如果 local cage 已经接近 plateau，即：

```text
L'(t) ~ 0
L(t) ~ A
```

则 peak 条件简化为：

```text
qR(t*) = A
```

此时：

```text
alpha_2(t*) = q/(4A)
```

这给出两个可检验预测：

```text
1. renewal delay tau_d 主要移动 peak time
2. jump variance q 主要控制 peak height
```

还有一个更强的无数据 consistency check。令：

```text
beta = q/A ~= 4 alpha_2(t*)
y(t) = beta R(t)
```

在 late branch 上：

```text
alpha_2(t_l) = beta y_l/(1 + y_l)^2
```

这个反演有两个分支。只有当 `0 < alpha_l <= beta/4` 时才可行；`y<1`
对应 peak 前，`y>1` 对应 peak 后。因此 late-time consistency check 必须选
`y>1` 分支。

因此一个晚时间 NGP 值可以反解出 `y_l`，再给出：

```text
lambda_l ~= y_l/[beta tau_d F(t_l/tau_d)]
```

如果 `lambda_l` 和 peak 反推出的 `lambda_*` 不一致，模型就被 falsify。

脚本 `generate_renewal_cage_results.py` 的参数扫描正是验证这两个预测。

## 7.5. Self-intermediate scattering function

更贴近 glass literature 的量是：

```text
F_s(k,t) = <exp[i k · Delta r(t)]>
```

在本模型中，条件在 renewal count `N(t)=n` 下位移仍然是 Gaussian，因此：

```text
F_s(k,t) = exp[-k^2 L(t)/2 + R(t)(exp(-k^2 q/2)-1)]
```

这可以拆成 cage Debye-Waller plateau 和 alpha relaxation：

```text
F_s(k,t) = exp[-k^2 L(t)/2] Phi_alpha(k,t)
Phi_alpha(k,t) = exp[-Gamma_k R(t)]
Gamma_k = 1 - exp(-k^2 q/2)
```

长时间 `R(t) ~ lambda t`，所以：

```text
tau_alpha(k)^-1 ~= lambda [1 - exp(-k^2 q/2)]
```

这比 random diffusivity 的说法更具体：alpha relaxation 受离散 cage-center
renewal count 控制。

## 7.6. 温度依赖和 Stokes-Einstein violation

最小温度扩展不是完整 microscopic glass theory，而是一个可检验的 phenomenological
law。令：

```text
Delta_T = 1/T - 1/T0
```

取：

```text
lambda(T) = lambda0 exp[-E_lambda Delta_T]
tau_d(T)  = tau_d0 exp[ E_d Delta_T]
A(T)      = A0 exp[-E_A Delta_T]
q(T)/A(T) = beta0 exp[E_beta Delta_T]
```

冷却时 `lambda` 下降，`tau_d` 增长，cage 变硬，`q/A` 可以增强。

长时间 diffusion coefficient 是：

```text
D(T) = lambda(T) q(T) / 2
```

而 cage-normalized alpha relaxation 满足：

```text
Phi_alpha(k,t) = exp[-Gamma_k(T) R(t;T)]
Gamma_k(T) = 1 - exp[-k^2 q(T)/2]
Gamma_k(T) R(tau_alpha;T) = 1
```

因为：

```text
R(t;T) = lambda(T) tau_d(T) F[t/tau_d(T)]
```

所以：

```text
tau_alpha(k,T)
  = tau_d(T) F^{-1}[1/(Gamma_k lambda(T) tau_d(T))]
```

于是 Stokes-Einstein product 的闭式判据是：

```text
D(T) tau_alpha(k,T)
  = lambda(T) q(T) tau_d(T)
    F^{-1}[1/(Gamma_k lambda(T) tau_d(T))] / 2
```

关键结论：只让 `lambda(T)` 下降，通常只是同时放慢 diffusion 和 alpha relaxation，
不一定产生强 SE violation。真正的 decoupling 来自 `tau_d(T)` 相对 `1/lambda(T)`
增长，也就是 delayed renewal onset 变成 structural relaxation 的额外瓶颈。

玻璃文献里常用 fractional Stokes-Einstein exponent 表示这种 decoupling：

```text
D ~ tau_alpha^(-xi_SE)
xi_SE(T) = - d log D(T) / d log tau_alpha(k,T)
```

普通 Stokes-Einstein scaling 对应：

```text
xi_SE = 1
```

fractional SE violation 对应：

```text
0 < xi_SE < 1
```

当前默认温度扫描给出的 `xi_SE` 从热端约 `0.725` 降到冷端约 `0.568`，说明模型不只
让 `D tau_alpha` 变大，也能产生文献常见的 fractional SE slope。

## 7.7. Activated barrier 和 renewal susceptibility

上面的温度律可以从一个最小 activated barrier 图像得到。设长时间 renewal rate 的
barrier 是 `E_lambda`，delayed onset 的 persistence barrier 是 `E_d`，则：

```text
lambda(T) tau_d(T)
  = lambda0 tau_d0 exp[(E_d - E_lambda)(1/T - 1/T0)]
```

所以只要：

```text
E_d > E_lambda
```

冷却时 `lambda tau_d` 就会增长，structural relaxation 会比 long-time diffusion
更强地受到 delayed cage renewal 限制。这是 SE violation 的最小 barrier 判据。

模型还给出一个和四点 susceptibility 有关的 renewal-count 部分。定义：

```text
W_k(t|N) = exp[-k^2 L(t)/2] a_k^N
a_k = exp[-k^2 q/2]
```

则：

```text
F_s(k,t) = E_N[W_k(t|N)]
```

renewal count fluctuation 贡献的 susceptibility 是：

```text
chi_R(k,t)
  = Var_N[W_k(t|N)]
  = exp[-k^2 L(t)] {exp[R(t)(a_k^2-1)] - exp[2R(t)(a_k-1)]}
```

也可以写成更简单的相对形式：

```text
chi_R(k,t) / F_s(k,t)^2
  = exp[R(t)(a_k-1)^2] - 1
```

这不是完整空间四点函数 `chi_4(t)`，因为它没有包含不同粒子之间的空间相关长度。
但它是 delayed renewal count 对 dynamic heterogeneity 的闭式贡献，可以和 NGP peak
以及 `F_s` relaxation time 对比。

## 7.8. 从可观测量反演参数和证伪条件

更强的可发表价值在于：模型不只是正向画曲线，也能从常见 glass observable 反推
参数并给出无解判据。

如果实验或模拟给出：

```text
f_k          F_s(k,t) 的 cage Debye-Waller plateau
D            long-time diffusion coefficient
tau_alpha    cage-normalized alpha relaxation time
tau_d        delayed cage-breaking onset time
h            alpha threshold, 常用 h=e^-1
```

则 plateau 直接给出：

```text
A = -2 log(f_k) / k^2
```

long-time diffusion 给出：

```text
D = lambda q / 2
```

而 alpha relaxation 条件是：

```text
lambda tau_d F(tau_alpha/tau_d) [1 - exp(-k^2 q/2)] = -log(h)
```

消去 `lambda` 后得到一个只关于 `q` 的方程：

```text
[1 - exp(-k^2 q/2)] / q
  = -log(h) / [2 D tau_d F(tau_alpha/tau_d)]
```

由于左边最大只能趋近于 `k^2/2`，所以模型存在正 jump variance 解的必要条件是：

```text
M_inv = D tau_d F(tau_alpha/tau_d) k^2 / [-log(h)] > 1
```

如果 `M_inv <= 1`，那么 delayed renewal cage model 无法同时解释给定的
`F_s` plateau、`D`、`tau_alpha` 和 `tau_d`。如果 `M_inv > 1`，则 `q` 被唯一确定，
然后：

```text
lambda = 2D/q
```

此时 NGP peak 不再是自由拟合参数，而是 out-of-sample prediction：

```text
R(t*) = A/q
alpha_2(t*) = q/(4A)
```

这就是当前版本最强的证伪点：先用 scattering 和 transport 反演参数，再用 NGP peak
检查模型是否自洽。

## 8. 与 Glass Transition 的联系

在 glass transition 语境下：

```text
A       cage size / plateau amplitude
tau_c   local rattling relaxation time
tau_d   cage-breaking onset time
lambda  long-time cage renewal rate
q       cage jump length variance
```

冷却时，通常可以预期：

```text
tau_d 增大
lambda 减小
q/A 的有效对比增强
```

因此 NGP peak 会移动到更长时间，并且动态异质性变得更明显。这和 thesis 中
NGP 在 Tg 以上温度就出现强变化的观察相容：NGP peak 反映的是 cage-renewal
heterogeneity 的增强，而不是简单的长时间 diffusion coefficient。

## 9. 当前结果文件

实现：

```text
src/renewal_cage.py
```

测试：

```text
tests/test_renewal_cage.py
```

主结果脚本：

```text
scripts/generate_renewal_cage_results.py
```

输出：

```text
data/renewal_cage_main.csv
data/renewal_cage_sweeps.csv
data/renewal_cage_dimensionless.csv
data/renewal_cage_van_hove.csv
data/renewal_cage_inversion.csv
figures/renewal_cage_results.svg
figures/renewal_cage_dimensionless.svg
figures/renewal_cage_inversion.svg
```

运行：

```bash
python3 -m unittest tests/test_renewal_cage.py -v
python3 scripts/generate_renewal_cage_results.py
```
