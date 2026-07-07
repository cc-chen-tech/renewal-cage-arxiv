# Langevin Coarse-Graining Bridge

This note records the conditional Langevin coarse-graining bridge used by the
repository. The goal is not to claim that the whole effective theory follows
from a no-input many-body Langevin equation. The goal is narrower: derive the
intra-cage Gaussian layer and Kramers escape clocks from a local Langevin
landscape once a metastable basin partition and its barriers are specified.

## 1. Microscopic Starting Point

Start from Langevin dynamics for a many-particle configuration
`R=(r_1,...,r_N)`:

```text
m_i d^2 r_i / dt^2 = -grad_i U(R) - gamma d r_i / dt + xi_i(t)
```

with thermal noise

```text
<xi_i(t) xi_j(t')> = 2 gamma T delta_ij delta(t-t')
```

in reduced units with `k_B=1`. In the overdamped limit this becomes

```text
gamma dR/dt = -grad U(R) + xi(t)
```

and the probability density obeys the Smoluchowski equation

```text
partial_t P(R,t)
  = sum_i D_0 div_i [grad_i P + beta P grad_i U],
D_0 = T/gamma.
```

This gives the first microscopic scale used by the effective theory:

```text
D_0 = T/gamma.
```

## 2. Local Cage From a Harmonic Basin

Near a metastable cage basin minimum, approximate the free-energy landscape by a
local quadratic form. For a one-dimensional local coordinate `x`,

```text
F(x) = F_a + (kappa_c/2) x^2.
```

The overdamped dynamics is an Ornstein-Uhlenbeck process:

```text
gamma dx/dt = -kappa_c x + xi(t).
```

Therefore

```text
cage_variance = T/kappa_c,
cage_tau      = gamma/kappa_c.
```

These are the effective renewal-cage `cage_variance` and `cage_tau`. The MSD
plateau is therefore inherited from equipartition inside the harmonic basin,
not inserted as an unrelated fitting constant.

## 3. Escape Rate From Kramers Theory

Let a basin have curvature `kappa_c`, the transition saddle have unstable
curvature magnitude `kappa_s`, and the effective free-energy barrier be
`Delta F`. In the overdamped local-quadratic Kramers limit,

```text
k_escape =
  sqrt(kappa_c kappa_s) / (2 pi gamma) * exp(-Delta F / T).
```

This turns continuous Langevin dynamics into a coarse-grained jump process
between metastable cages.

This part is close to established exit-rate theory: in a metastable basin, the
overdamped Langevin first-passage problem can be approximated by an
Eyring-Kramers escape rate and then by a Markov jump model. The glass-specific
cage-jump and CTRW literature also uses intermittent cage motion and jumps to
describe transport. The non-standard step in this repository is narrower: the
delayed hazard `r(t)=lambda[1-exp(-t/tau_d)]^2` is a coarse-grained model of
post-cage-entry softening or precursor build-up, not an automatic consequence
of ordinary quasi-stationary Kramers theory.

The periodic-softness bridge makes that non-standard step explicit. Start from
a periodic cage potential

```text
U(x) = DeltaU [1 - cos(2 pi x / L)] / 2.
```

Both the basin curvature and saddle-curvature magnitude are

```text
kappa = 2 pi^2 DeltaU / L^2.
```

This gives the long-time Kramers hopping rate. Now add two collective
precursor gates representing slow many-body cage softening:

```text
p_i(t) = 1 - exp(-t/tau_d).
```

If escape requires both gates, the two precursor readiness probabilities
multiply:

```text
r(t) = lambda p_1(t) p_2(t)
     = lambda [1 - exp(-t/tau_d)]^2.
```

Thus the square delayed hazard is derived from an effective periodic
Langevin/Kramers landscape plus two slow collective gates. The many-body origin
of those gates remains an explicit coarse-grained assumption.

## Extended Landscape Scope

A single static one-dimensional potential is not enough to generate every
effective module in the renewal-cage theory. The more useful microscopic
picture is an extended coarse-grained free-energy landscape,

```text
U(x,C,s1,s2,zeta)
  = (kappa/2)|x-C|^2
    + [DeltaU0 + chi zeta - eps s1 s2] B(x-C)
    + W1(s1) + W2(s2) + Wzeta(zeta).
```

Here `x` is the particle coordinate, `C` is the cage center, `s1` and `s2` are
slow escape-precursor or softness coordinates, and `zeta` is a mobility or
barrier environment. The harmonic projection gives the OU cage; the periodic
projection gives the Vorselaars-type cage-to-cage baseline; the softness-gate
projection gives the delayed hazard; and the mobility-environment projection
gives finite-exchange heterogeneity.

This is closer to a Langevin/free-energy derivation than a hand-set renewal
hazard, but it is still not a complete many-body first-principles derivation.
The thermodynamic configurational-entropy layer would need an additional
inherent-state density of basins and its own partition function.
The same scope boundary is exported as
`data/renewal_cage_potential_taxonomy.csv`, where each row records a potential
projection, the effective parameters it can supply, the observables it supports,
and the many-body input that still remains external.
The numerical companion `data/renewal_cage_landscape_parameterization.csv`
shows how two of those rows become calculations: a basin adjacency graph gives
`q`, while a discrete inherent-state density `Omega(e)` gives
`Z_conf`, `F_conf`, `s_c`, and `Delta c_p`.

In the current effective model, cage rearrangement is not inserted as an ordinary Langevin drift. The local displacement and cage-center motion are kept as two different stochastic layers:

```text
dy_t = -(1/tau_c) y_t dt + sqrt(2D_c) dW_t
dC_t = eta_t dN_t
x_t = y_t + C_t
```

Here `y_t` is the intra-cage OU-like vibration, `C_t` is the cage center, and
`N_t` is the delayed renewal count. This distinction matters because the OU
layer is always Gaussian; the NGP peak comes from the trajectory-to-trajectory
spread in how many cage-center jumps have occurred by time `t`.

For the persistence/exchange theory, the first escape and subsequent exchanges
may cross different effective barriers:

```text
Delta F_p = Delta F + Delta F_p_extra,
Delta F_x = Delta F + Delta F_x_extra.
```

Then

```text
tau_p = 1/k_p,
tau_x = 1/k_x.
```

This is the microscopic bridge to persistence/exchange decoupling. If
`Delta F_p > Delta F_x`, then `tau_p > tau_x`: the first cage escape is harder
than later exchanges.

## 4. Effective Renewal-Cage Parameters

Given a typical cage-jump length `ell` in `d` dimensions, the one-dimensional
jump variance is

```text
jump_variance = ell^2 / d.
```

The Langevin/Kramers bridge therefore maps

```text
(T, gamma, kappa_c, kappa_s, Delta F_p, Delta F_x, ell)
```

to

```text
PersistenceExchangeParams(
  cage_variance    = T/kappa_c,
  cage_tau         = gamma/kappa_c,
  jump_variance    = ell^2/d,
  persistence_mean = 1/k_p,
  exchange_mean    = 1/k_x
)
```

The repository implements this map in `langevin_to_persistence_exchange`.

## 5. Observable Consequences

Once the effective clocks are derived, the existing renewal-cage formulas give
the observable predictions:

```text
D ~ jump_variance / (2 tau_x)
tau_alpha ~ tau_p
D tau_alpha ~ jump_variance * tau_p / (2 tau_x)
```

Thus growth of `tau_p/tau_x` gives a Stokes-Einstein violation.

The self-intermediate scattering decay follows the renewal count generating
function:

```text
F_s(k,t) = G_N(phi(k),t),
phi(k) = exp[-k^2 jump_variance / 2].
```

At intermediate times, a mixture of unescaped and escaped particles produces a
non-Gaussian displacement distribution and an NGP peak. At long times, finite
exchange gives repeated independent jumps, so the displacement distribution
recovers toward Gaussian behavior.

## 6. What Is Still Assumed

Only the intra-cage OU layer and the local escape-rate layer are derived from
the Langevin/Kramers approximation. The rest of the renewal-cage effective
theory still needs the following coarse-grained inputs:

```text
1. a metastable-basin partition,
2. local basin and saddle curvatures,
3. persistence and exchange barrier estimates,
4. a jump-length scale,
5. a Markov/Kramers approximation for basin-to-basin escapes.
```

Therefore the repository sets

```text
entire_effective_theory_from_langevin_claim_allowed = 0
```

in `renewal_cage_langevin_bridge.csv`. A stronger future result would measure
these basin, barrier, and jump statistics directly from molecular or Langevin
trajectories, then use the same formulas to predict `MSD`, `NGP`, `F_s(k,t)`,
and `D tau_alpha` without refitting the effective clocks.
