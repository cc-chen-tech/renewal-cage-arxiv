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

更强的一点是，NGP peak time 和 alpha relaxation time 被同一个 renewal
clock 连接。如果 `Phi_alpha(k,tau_alpha)=h`，则：

```text
R_alpha(k,h) = -log(h) / Gamma_k
R_peak       = A/q

R_alpha / R_peak
  = -q log(h) / [A(1 - exp(-k^2 q/2))]

tau_alpha / t_peak
  = R^{-1}(R_alpha) / R^{-1}(R_peak)
```

因此只要知道 `A,q,lambda,tau_d`，`tau_alpha/t_peak` 不是自由拟合量。默认参数
和 `k=1.1,h=e^-1` 下，`R_alpha=2.606`，`R_peak=1.250`，所以
`tau_alpha/t_peak=1.678`。这给出一个很直接的 glass-literature 判据：NGP peak
早于 structural relaxation，但二者是否由同一个 renewal process 控制可以被检验。

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

另一个直接对应 glass transition 的温度诊断是 alpha relaxation 的 apparent activation
energy：

```text
E_app(T) = d log tau_alpha(k,T) / d(1/T)
```

如果 relaxation 是 Arrhenius 的，`E_app` 是常数；如果冷却时 `E_app` 增长，就表示
有效 relaxation barrier 正在升高。可以定义一个局部 Angell-style fragility proxy：

```text
m_loc(T) = E_app(T) / [T log(10)]
```

当前默认温度扫描中：

```text
E_app: 2.69 -> 3.43
m_loc: 1.17 -> 2.41
```

所以这个模型已经能同时给出三类温度可检验量：`D tau_alpha` 增长、`xi_SE<1`、
以及 `E_app/m_loc` 随冷却增强。

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

如果一个 correlated renewal domain 中有 `N_corr` 个粒子共享同一个 renewal count
history，那么 renewal 部分对 per-particle 四点 susceptibility 的贡献是：

```text
chi_4^R(k,t) = N_corr chi_R(k,t)
```

因此如果实验或模拟给出 renewal-dominated 的四点峰值，可以反推 cooperative size：

```text
N_corr = chi_4,peak^obs / max_t chi_R(k,t)
```

这不是完整空间相关长度理论，但它给出一个很直接的桥：用闭式 `chi_R` 把观测到的
`chi_4` 峰值转换成 renewal-domain 粒子数。默认合成例子用 `N_corr=12` 生成四点峰，
反演会回到 `N_corr=12.0`。

## 7.8. Finite-exchange heterogeneity extension

单一 Poisson renewal clock 是最小模型，但真实 glassy dynamics 还有 trajectory-to-trajectory
mobility heterogeneity。一个静态 renewal rate 分布确实能拉宽 alpha decay，但会留下
非零长时间 NGP，破坏 Gaussian recovery。更合理的最小闭式扩展是让 renewal count 做
finite-exchange gamma mixing。

条件在 mobility `M` 下：

```text
N(t) | M ~ Poisson[M R(t)]
```

令有效 gamma shape 随着采样过的 renewal count 增大：

```text
kappa_eff(t) = kappa0 [1 + R(t)/R_x]
```

则 renewal-count PGF 是负二项式形式：

```text
G_N(z;t) = [1 + R(t)(1-z)/kappa_eff(t)]^[-kappa_eff(t)]
```

它保持平均 renewal count 不变，但增加有限时间 overdispersion：

```text
E N(t)   = R(t)
Var N(t) = R(t) + R(t)^2/kappa_eff(t)
```

因此：

```text
alpha_2^hx(t) = q^2 Var N(t) / [L(t)+qR(t)]^2

Phi_alpha^hx(k,t)
  = G_N(exp[-k^2 q/2];t)
  = [1 + Gamma_k R(t)/kappa_eff(t)]^[-kappa_eff(t)]
```

局部拉伸指数可以定义为：

```text
beta_loc(t) = d log[-log Phi_alpha^hx(k,t)] / d log t
```

默认例子 `kappa0=0.4, R_x=10` 下，NGP peak 增强到 `1.195`，出现在
`t=32.07`；alpha window 中 `beta_loc` 的中位数为 `0.805`，说明出现
stretched-like alpha decay；同时到 `t=3.0e4` 时 NGP 已恢复到 `0.00480`。
这比静态 heterogeneity 更适合 glass transition：它允许强 dynamic heterogeneity，
但仍保留长时间 Gaussian recovery。

这个扩展还给出一个很强的 late-time consistency check。令：

```text
c = R_x/kappa0
```

则长时间：

```text
R(t) alpha_2^hx(t) -> 1 + c

-log Phi_alpha^hx(k,t) / R(t)
  -> log(1 + Gamma_k c) / c
```

因此 late NGP amplitude 给出：

```text
c = R alpha_2 - 1
```

而 long-time alpha slope 也能独立反推出同一个 `c`。默认参数中
`c=25`，所以 late NGP amplitude 预测为 `26.0`；同时 alpha decay per renewal
从 Poisson 的 `Gamma_1.1=0.384` 降到 `0.0944`，用 alpha-rate 反演也回到
`c=25.0`。这把 stretched-like alpha decay、enhanced NGP 和 Gaussian recovery
连成一个可证伪三联判据。

这一步还能做成 diagnostic map：固定 `k` 后，实验或模拟测到的
`(R alpha_2, -log Phi_alpha/R)` 必须落在同一条由 `c` 参数化的曲线上。
如果 late NGP 反推出的 `c` 和 alpha slope 反推出的 `c` 不一致，就说明
finite-exchange renewal 不是充分解释。默认阈值
`R alpha_2 >= 3` 且 alpha-rate renormalization `< 0.75` 时，联合可观测窗口
从大约 `c=2` 开始。

实际使用时，可以把它压缩成一个 residual：

```text
c_NGP = R_l alpha_2(t_l) - 1

log(1 + Gamma_k c_alpha) / c_alpha
  = -log Phi_alpha(k,t_l) / R_l

Delta_c = log(c_alpha / c_NGP)
```

如果 `|Delta_c|` 小，late NGP recovery 和 alpha slowing 由同一个 exchange
scale 控制；如果它很大，就 falsify 这个一参数 finite-exchange renewal 图像。
默认例子在 `t_l=3.0e4` 给出 `c_NGP=24.94`、`c_alpha=25.00`、
`Delta_c=0.0023`，通过；如果只把 alpha slope 换成对应 `c_alpha=2` 的值，
则 `Delta_c=-2.52`，失败。

如果有测量误差，还可以直接给出统计显著性：

```text
sigma_cNGP^2 = alpha_2^2 sigma_R^2 + R_l^2 sigma_alpha2^2

f'(c) = [Gamma_k c/(1+Gamma_k c) - log(1+Gamma_k c)] / c^2

sigma_calpha = sigma_salpha / |f'(c_alpha)|

z_c = |Delta_c| / sigma_Delta
```

在 `R_l` 和 `alpha_2(t_l)` 各有 `1%` 误差、`sigma_salpha=0.002` 的示例中，
一致案例 `z_c=0.062`，错配案例 `z_c=76.4`。这使判据从“看起来一致”
变成了统计意义上的 falsification protocol。

如果有多个 wave number 的 `F_s(k,t)`，还可以做更强的 collapse test：
每个 `k` 都通过

```text
log(1 + Gamma_k c_alpha(k)) / c_alpha(k) = s_alpha(k)
```

独立反推出一个 `c_alpha(k)`。finite-exchange renewal 预言所有
`c_alpha(k)` 必须 collapse 到同一个 `c_NGP`。默认 `k=0.6,1.1,1.8`
的相容例子给出 weighted `c_alpha=25.00`、`z=0.096`；如果只把 `k=1.1`
的 alpha slope 换成对应 `c=2` 的值，则 weighted `c_alpha=2.23`、
`z=61.1`，跨 wave number reduced chi-square 为 `756`，强烈失败。

## 7.9. 从可观测量反演参数和证伪条件

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

还可以进一步消去外部输入的 `tau_d`。如果同时使用 NGP peak height 和 peak time：

```text
q = 4 A alpha_2(t*)
lambda = 2D/q
```

peak time 满足：

```text
lambda tau_d F(t*/tau_d) = A/q
```

令：

```text
s = t*/tau_d
```

则：

```text
F(s)/s = (A/q)/(lambda t*)
```

只要：

```text
(A/q)/(lambda t*) < 1
```

就能唯一解出 `s`，进而得到：

```text
tau_d = t*/s
```

这样 `A, q, lambda, tau_d` 都可以从 `F_s` plateau、long-time `D`、NGP peak
height 和 NGP peak time 反演出来；`tau_alpha` 不再是输入，而是 held-out
consistency check。默认合成数据中完整反演恢复：

```text
A = 1.00
q = 0.80
lambda = 0.18
tau_d = 3.00
log tau_alpha residual < 1e-12
```

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

## 8.5 Static Gamma Null Model

为了说明 novelty 不是“random diffusivity 也能产生非 Gaussian”，当前版本加入了
一个明确的 competing null model：static gamma mobility disorder。

如果每条 trajectory 有一个固定 mobility `M`，且 `M` 服从均值为 1、shape 为
`kappa0` 的 gamma 分布，那么 renewal count 的 generating function 是：

```text
G_N^st(z;t) = [1 + R(t)(1-z)/kappa0]^(-kappa0)
```

所以：

```text
E N(t) = R(t)
Var N(t) = R(t) + R(t)^2/kappa0
```

这给出：

```text
alpha_2^st(t)
  = q^2 [R(t)+R(t)^2/kappa0] / [L(t)+qR(t)]^2
  -> 1/kappa0
```

同时：

```text
Phi_alpha^st(k,t)
  = [1 + Gamma_k R(t)/kappa0]^(-kappa0)

-log Phi_alpha^st(k,t) / R(t) -> 0
```

结论是：static gamma disorder 可以让 alpha relaxation 变宽，但它必然留下长时间
NGP plateau，不能解释 Gaussian recovery。finite-exchange 模型的关键改动是让
`kappa_eff(t) ~ R(t)`，所以既能保留 stretched-like alpha，又能让 NGP 回到 0。

默认例子中 `kappa0=0.4`。在 `t=30000`：

```text
static gamma NGP       = 2.499
finite-exchange NGP   = 0.00480
static alpha slope/R   = 6.34e-4
finite-exchange slope  = 0.0945
```

这就是最直接的判据：如果数据同时有 broadened alpha decay 和 long-time Gaussian
recovery，static mobility disorder 不够，必须有 mobility exchange 或等价的
self-averaging 机制。

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
data/renewal_cage_static_null.csv
data/renewal_cage_inversion.csv
figures/renewal_cage_results.svg
figures/renewal_cage_dimensionless.svg
figures/renewal_cage_static_null.svg
figures/renewal_cage_inversion.svg
```

运行：

```bash
python3 -m unittest tests/test_renewal_cage.py -v
python3 scripts/generate_renewal_cage_results.py
```
