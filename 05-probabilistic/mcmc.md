# MCMC

`mcmc.py` · Probabilistic models · Tier 2 · 7 variants

## Intuition

A Markov chain that forgets where it started and settles into the distribution
you cannot integrate. You never evaluate the posterior — you only compare it at
two points, and a ratio of densities is all a Metropolis rule needs.

$$P(x \to \mathrm{d}x') = q(x \to x')\,a(x, x')\,\mathrm{d}x' \qquad (1)$$

If `a` is chosen so that the chain is reversible with respect to $\pi$,

$$\pi(x)\,q(x \to x')\,a(x, x') \;=\; \pi(x')\,q(x' \to x)\,a(x', x) \qquad (2)$$

then $\pi$ is stationary, and under mild conditions the chain converges to it
from any starting point. That single equation generates every sampler in the
file: the variants differ only in which proposal `q` they can afford and how
much structure they exploit to make `a` large.

The economics are the whole subject. A random-walk proposal is free and accepts
often but moves $O(\sqrt{d})$ per step, so it needs $O(d)$ steps to decorrelate.
A gradient tells you which way is uphill, and following a Hamiltonian
trajectory moves $O(1)$ per step regardless of `d` — the same posterior for
orders of magnitude fewer density evaluations, at the price of a gradient.
Everything else is engineering that trade.

## Variants

| Variant | Proposal `q` | Cost per draw | Use when |
|:--|:--|:--|:--|
| `metropolis_hastings` | $N(x, s^2 I)$ | 1 density | nothing is differentiable, `d` is small, you need a baseline |
| `gibbs` | the exact full conditional | 1 conditional per coordinate | conditionals are conjugate or one-dimensional |
| `hmc` | leapfrog trajectory of length $L\varepsilon$ | $L$ gradients + $L$ densities | differentiable, `d` from 10 to 10⁶ |
| `nuts` | HMC, trajectory length chosen by u-turns | `2^depth` gradients | you do not know the trajectory length — the default choice |
| `slice_sampling` | uniform under the density along a line | $O(1)$ densities | no gradients, awkward shapes, heavy tails |
| `adaptive_metropolis` | $N(x, s_t^2 I)$ with $s_t$ tuned | 1 density | you want Metropolis without a tuning sweep |
| `diagnostics` | — (uses the random walk) | — | always, before you believe any of the above |

`diagnostics` is not a kernel. It is the three numbers — R-hat, effective sample
size, acceptance — that tell you whether the other six produced anything. It
lives here because a sampler shipped without them is a random number generator
with a story attached.

`gibbs` is Metropolis with acceptance probability exactly one applied to a
proposal you can only make when the conditional is available. `nuts` is `hmc`
with the hardest hyper-parameter — trajectory length — removed. `slice_sampling`
is the only kernel here that needs no tuning at all and the only one that is
happy with a target whose mean does not exist.

## Derivation

### Metropolis–Hastings

Propose $x' \sim q(\cdot \mid x)$, accept with

$$a(x, x') \;=\; \min\left(1, \frac{\pi(x')\,q(x \mid x')}{\pi(x)\,q(x' \mid x)}\right)
\;=\; \min\left(1, e^{\ell(x') - \ell(x)}\right) \qquad (3)$$

for a symmetric proposal. The second equality is why no normalising constant is
ever needed: only the difference in log density appears, and it is compared with
$\log U$. Substituting (3) into (2) gives detailed balance in one line — the
numerator of one direction is the denominator of the other.

### Gibbs

If $q$ is the full conditional $p(x_j \mid x_{-j})$, then
$\ell(x') - \ell(x) = \log p(x_j' \mid x_{-j}) - \log p(x_j \mid x_{-j})$, and
the Hastings ratio is exactly the inverse of the proposal ratio. The acceptance
probability is 1. Gibbs is not an approximation of Metropolis; it is the case
where the proposal is already correct.

The corollary is the failure: if $x_1$ and $x_2$ are correlated at $\rho = 0.99$,
the conditional for $x_1$ is a very narrow distribution around
$\rho x_2$, so each sweep moves a distance proportional to $1 - \rho^2$. Gibbs
is a random walk that has quietly chosen a terrible step size.

### Hamiltonian Monte Carlo

Add a momentum $p \sim N(0, M)$ and integrate Hamilton's equations for the
augmented density $\pi(x)N(p; 0, M)$:

$$H(x, p) \;=\; -\ell(x) + \tfrac{1}{2} p^\top M^{-1} p, \qquad
\dot{x} = M^{-1}p, \quad \dot{p} = \nabla \ell(x) \qquad (4)$$

The leapfrog integrator alternates half-kicks and drifts, is symplectic and
reversible, and therefore preserves the stationary distribution *exactly* up to
the Metropolis correction

$$a \;=\; \min\left(1, e^{H_0 - H_1}\right) \qquad (5)$$

In the demo, a target with standard deviations $(1, 10)$ and the matching metric
$M = \mathrm{diag}(1, 10^{-2})$ keeps the mean energy error at $0.05$ and
delivers 21× the random walk's effective samples per evaluation on the slow
coordinate. With the identity metric the same code would still be correct, and
would take a step size small enough for the narrow coordinate that the wide one
barely moves — the metric, not the integrator, decides whether HMC works.

### NUTS

Trajectory length is the remaining knob, and it is target-dependent. NUTS grows
the trajectory by doubling, in a random direction, and stops when either end
turns back towards the other:

$$(x^{+} - x^{-}) \cdot p^{-} < 0 \quad \text{or} \quad (x^{+} - x^{-}) \cdot p^{+} < 0
\qquad (6)$$

The proposal is drawn from the new subtree with probability proportional to its
number of states, which keeps the doubling unbiased. The step size is tuned by
dual averaging during warm-up,

$$\log \varepsilon_{t+1} = \mu - \frac{\sqrt{t}}{\gamma}\,\bar h_t, \qquad
\bar h_t = \left(1 - \frac{1}{t+t_0}\right)\bar h_{t-1} + \frac{1}{t+t_0}\left(\delta - \bar\alpha_t\right)
\qquad (7)$$

and initialised by doubling from $\varepsilon = 1$ until a single leapfrog
accepts about half the time. The target $\delta$ is not 0.234: random-walk
optimal scaling says nothing about Hamiltonian dynamics, which want
$\delta \approx 0.8$ because a rejected trajectory has already paid for every
leapfrog step.

### Slice sampling

Sample uniformly from the region under the density. Draw a level
$y = \ell(x_0) - e$ with $e \sim \mathrm{Exp}(1)$, then a point uniformly from
$\{x : \ell(x) > y\}$ along a line. The returned point is uniform on the slice,
so the kernel is reversible with acceptance 1, and the only tuning is a window
width that adapts to the local scale as it steps out. It is the only sampler
here that handles a Cauchy target — whose mean does not exist — without any
special casing.

### Adaptation and diagnostics

Robbins–Monro tunes the proposal during burn-in:

$$\log s_{t+1} = \log s_t + \gamma_t\,(\alpha_t - \alpha^{*}), \qquad \gamma_t \propto t^{-0.6}
\qquad (8)$$

A decaying gain is not a detail. Adapting forever makes the kernel
non-homogeneous and destroys the stationary distribution; decaying to zero makes
the adaptation vanish asymptotically, so the chain converges. Freeze the scale
after warm-up and the guarantee holds.

Two chains that disagree have not mixed. Compare between- and within-chain
variance:

$$\hat R = \sqrt{\frac{\frac{n-1}{n}W + \frac{B}{n}}{W}} \qquad (9)$$

and count how many independent draws the autocorrelation has cost you:

$$\mathrm{ESS} = \frac{mn}{1 + 2\sum_{t \geq 1}\rho_t} \qquad (10)$$

with Geyer's initial positive pair sum truncating the series where it stops
being informative. Identical chains give $\hat R = \sqrt{1 - 1/n} = 0.99975$,
not 1 — the $1/n$ factor is visible in the demo, which is a good sign the
formula was implemented rather than approximated.

## Code walkthrough

The whole atom is one class plus three functions. `VARIANTS` is the public API;
`sample` dispatches on it.

```python
VARIANTS = frozenset(
    ["metropolis_hastings", "gibbs", "hmc", "nuts", "slice_sampling",
     "adaptive_metropolis", "diagnostics"]
)
```

`sample` returns `(n, d)` and records the price of every draw in `n_eval_` and
`n_grad_`. Those counters are the reason the HMC comparison in the demo is
honest: kernels are compared per evaluation, not per iteration.

```python
        if self.variant == "gibbs":
            chain = self._gibbs(n_samples, x0, rng)
        elif self.variant in ("hmc", "nuts"):
            chain = self._hamiltonian(n_samples, x0, rng)
        elif self.variant == "slice_sampling":
            chain = self._slice(n_samples, x0, rng)
        else:
            chain = self._random_walk(n_samples, x0, rng)
```

The Metropolis core is four lines, and the one that would be wrong in a naive
implementation is the comparison: it is a log-space test against `log U`, not
`U < ratio`, because the ratio underflows for any meaningful move in high
dimensions.

```python
            prop = x + scale * rng.standard_normal(d)
            lpp = self._lp(prop)
            ok = np.log(rng.random()) < lpp - lp
            if ok:
                x, lp = prop, lpp
                accepted += 1
```

Gibbs is the same loop with the proposal replaced by a draw from the
conditional, which is why it has no accept step at all.

```python
        for i in range(self.n_warmup + n):
            for j, draw in enumerate(samplers):
                x[j] = draw(x, rng)
```

When a closed-form conditional is not available, the same variant slices it.
This is the useful trick: Gibbs-with-slice is a valid Gibbs sampler whose
conditionals are each one-dimensional, so nothing is rejected and no algebra is
required.

```python
        y = float(logp(x0)) - float(rng.exponential())
        left = x0 - self.width * float(rng.random())
        right = left + self.width
```

The slice's shrinkage loop is the invariant: every rejected draw moves an
endpoint inward, so the interval always contracts around points above the level
and cannot wander into the tail.

```python
        for _ in range(1000):
            x1 = left + (right - left) * float(rng.random())
            if float(logp(x1)) > y:
                return x1
```

HMC's leapfrog is where the mass matrix must be handled in momentum form.
Updating a *velocity* as if it were a momentum is off by a factor of `M`, which
shows up as energy that is not conserved on an anisotropic target — exactly the
bug the demo's `|ΔH| = 0.05` assertion catches.

```python
        p = p + 0.5 * eps * g
        x = x + eps * p / m
        lp = self._lp(x)
        g = self._grad(x)
        p = p + 0.5 * eps * g
```

The NUTS leaf is the slice test and the Metropolis ratio in Hamiltonian form.
Getting the sign of `logu` wrong makes every tree empty and the chain frozen at
its starting point — a bug that produces a plausible-looking trace and zero
mixing.

```python
            H1 = -lp1 + self._kinetic(p1)
            n = int(logu <= -H1)
            s = int(logu < -H1 + 1000.0)
            alpha = min(1.0, float(np.exp(H0 - H1)))
```

The recursion reuses the same `_tree` for both directions and joins two
subtrees, stopping when the endpoints turn back.

```python
        delta = xp - xm
        no_uturn = int(float(delta @ pm) >= 0.0 and float(delta @ pp) >= 0.0)
```

Step-size initialisation is a doubling search, not a constant. Starting dual
averaging from an arbitrary step is the difference between converging in 500
warm-up draws and wandering for 5,000.

```python
            if trial == 0:
                direction = 1 if better else -1
            elif better != (direction == 1):
                break
            eps *= 2.0**direction
```

The acceptance target is variant-aware, and that is not cosmetic: with the
random-walk value 0.234, NUTS settles on a step where most trajectories are
rejected and reports a mean tree depth of 1.

```python
        if self.variant in ("hmc", "nuts"):
            return 0.8
        return 0.44 if d == 1 else 0.234
```

The ESS estimator sums autocorrelation in pairs and stops at the first
non-positive pair. Truncating at a fixed lag either throws away real
correlation or adds noise; the pair rule is the standard fix.

```python
    total = 0.0
    lag = 0
    while lag + 1 < rho.size:
        pair = rho[lag] + rho[lag + 1]
        if pair <= 0.0:
            break
        total += pair
        lag += 2
```

R-hat is between-chain variance added to within-chain variance:

```python
    W = float(np.mean(chains.var(axis=1, ddof=1)))
    B = float(n * np.var(chains.mean(axis=1), ddof=1))
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))
```

## Complexity

Per draw, with `d` the dimension and `L` the trajectory length:

| Variant | Time | Memory | Density evaluations per draw |
|:--|:--|:--|--:|
| `metropolis_hastings` | `O(d)` | `O(n d)` | 1 |
| `gibbs` | `O(d)` per sweep with closed forms, `O(d²)` to evaluate them | `O(n d)` | 0 (conditionals) |
| `hmc` | `O(L d)` | `O(n d)` | `2L` (plus `L` gradients) |
| `nuts` | `O(2^depth d)` | `O(n d)` | `≤ 2^{depth+1}` |
| `slice_sampling` | `O(d)` expected, `O(1)` per coordinate | `O(n d)` | ~3–5 per coordinate |
| `adaptive_metropolis` | `O(d)` | `O(n d + warmup)` | 1 |

The stored chain dominates memory: 100,000 draws of 1,000 parameters is 800 MB
in float64. Nothing here streams; subsample or checkpoint if that matters.

What actually dominates wall-clock is not the per-draw cost but the number of
draws needed, and that is set by the *effective* sample size. The demo's
anisotropic target shows the whole point: the random walk spends 10,000
evaluations to get 39 effective draws of the slow coordinate; HMC spends 160,000
to get 10,000, which is 21× cheaper per effective draw.

## Failure modes

**The trace looks stationary and the diagnostics lie.** R-hat compares chains to
each other, so four chains started in the same mode agree perfectly about a
multimodal target. The demo constructs the failure deliberately: two chains
frozen at ±6 give R-hat = 849, but if both had started at −6 the number would
have been 1.000 and the posterior mean wrong by 6. *Fix:* overdispersed
starting points, and plot the trace.

**Acceptance is fine and mixing is not.** In the anisotropic demo the random
walk accepts 40% of its proposals while the slow coordinate gets 39 effective
samples out of 10,000. Acceptance measures the proposal relative to the local
scale, not progress through the posterior. *Fix:* look at ESS per coordinate,
not at the acceptance rate.

**Energy explodes instead of decorrelating.** A leapfrog step above the
stability limit $2/\omega$ turns a trajectory into a diverging orbit; the
symptom is a sudden collapse of acceptance when the step size grows, and a
`|ΔH|` in the tens. The demo's first implementation had exactly this bug — a
velocity updated as if it were a momentum, off by the mass matrix — and the
assertion `energy_error_ < 0.2` is what caught it. *Fix:* `ΔH` percentiles per
draw; divergences concentrate in the funnel of a hierarchical model.

**Adapting forever.** Any tuning that does not decay keeps the kernel
non-homogeneous and the chain does not converge to π. *Fix:* freeze after
warm-up, as `adaptive_metropolis` and `nuts` do here, and check that the
adaptation has settled before the first stored draw.

**Correlated conditionals.** Gibbs on a Gaussian with ρ = 0.99 needs
`O((1-ρ²)^{-1})` sweeps, which is 5,000 per effective draw. Nothing about the
code is wrong. *Fix:* reparameterise, or use HMC on the joint.

**No burn-in is not "none needed".** Every kernel here is started from `x0`, and
the first draws are not from π. The demo discards `n_warmup`; the diagnostics for
a chain with `n_warmup = 0` and a bad start will show it as a shift, not as an
R-hat.

## Applications

- **Hierarchical and multilevel models.** The reason MCMC is in every Bayesian's
  toolkit: a model with partial pooling has no closed-form posterior, and the
  funnel geometry is exactly what NUTS was designed to survive.
- **Latent variable and mixture models.** Label switching, unknown $K$, and
  posteriors over assignments are all sampling problems (see
  [`dirichlet_process.py`](dirichlet_process.py)).
- **Gaussian process hyperparameters.** The marginal likelihood is available,
  but its gradient with respect to a handful of kernel parameters is often not;
  slice sampling or NUTS on the log-hyperparameters is the standard fallback
  (see [`gaussian_process.py`](gaussian_process.py)).
- **Posterior predictive checks.** Drawing parameters and then data is the
  cheapest model criticism that exists, and it requires only a sampler.
- **Calibration of expensive simulators.** When the likelihood is a simulation,
  approximate Bayesian computation wraps a sampler around it.

Where it is a trap: a conjugate model with three parameters. The posterior is
normal, beta, or Dirichlet in closed form, and 10,000 draws buy you Monte Carlo
error on a number you could have written down. [`bayesian_inference.py`](bayesian_inference.py)
exists for those cases. MCMC is for the posterior you cannot write down, and its
cost is measured in gradients, not in elegance.

## References

- Metropolis, Rosenbluth, Rosenbluth, Teller & Teller (1953). *Equation of state
  calculations by fast computing machines.* J. Chem. Phys. 21, 1087. The original.
- Hastings (1970). *Monte Carlo sampling methods using Markov chains and their
  applications.* Biometrika 57, 97. The asymmetric-proposal correction.
- Geman & Geman (1984). *Stochastic relaxation, Gibbs distributions, and the
  Bayesian restoration of images.* IEEE TPAMI 6, 721. Gibbs sampling, from
  image restoration.
- Neal (2003). *Slice sampling.* Annals of Statistics 31, 705. Stepping out and
  shrinkage; still the best no-tuning sampler.
- Duane, Kennedy, Pendleton & Roweth (1987). *Hybrid Monte Carlo.* Physics
  Letters B 195, 216. HMC's origin, in lattice QCD.
- Hoffman & Gelman (2014). *The No-U-Turn Sampler.* JMLR 15, 1593. NUTS,
  dual averaging and the u-turn criterion implemented here.
- Betancourt (2017). *A conceptual introduction to Hamiltonian Monte Carlo.*
  arXiv:1701.02434. The best explanation of why the geometry matters.
- Gelman & Rubin (1992). *Inference from iterative simulation using multiple
  sequences.* Statistical Science 7, 457. R-hat.
- Vehtari, Gelman, Simpson, Carpenter & Bürkner (2021). *Rank-normalization,
  folding, and localization: an improved R-hat.* Bayesian Analysis 16, 667.
  What to use instead of the version in this file when the posterior is heavy
  tailed.
- Brooks, Gelman, Jones & Meng (2011). *Handbook of Markov Chain Monte Carlo.*
  CRC Press. The survey to keep on the shelf.
