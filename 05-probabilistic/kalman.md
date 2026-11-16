# Kalman filter

`kalman.py` · Probabilistic models · Tier 2 · 6 variants

## Intuition

A hidden state drifts; a noisy sensor reports on it. Both halves are Gaussian,
so the posterior stays Gaussian, and a Gaussian is two numbers. That is the
whole reason this filter is forty lines instead of an MCMC run: the belief never
needs a representation more complicated than a mean and a covariance.

Two sentences drive the recursion. **Predict** is the law of total expectation:
the state moves through the dynamics, so the mean moves through $A$ and the
covariance is pushed through $A$ and inflated by process noise $Q$. **Update** is
conditioning on one observation: among all state values consistent with the
prediction, weight them by how well they would have predicted what you saw.

$$m_t^{-} = A m_{t-1}, \qquad P_t^{-} = A P_{t-1} A^{\top} + Q \qquad (1)$$

$$S_t = C P_t^{-} C^{\top} + R, \qquad K_t = P_t^{-} C^{\top} S_t^{-1} \qquad (2)$$

$$m_t = m_t^{-} + K_t\,(z_t - C m_t^{-}), \qquad P_t = P_t^{-} - K_t C P_t^{-} \qquad (3)$$

$S_t$ is the covariance of the *innovation* $z_t - C m_t^{-}$: the observation
noise $R$ plus the state uncertainty mapped into observation space. $K_t$ is the
fraction of the innovation you accept. When $R$ is tiny the filter believes the
sensor and $P$ collapses; when $P^{-}$ is tiny it believes the model and
$K \to 0$.

The filter is worth more than its estimates, because the same recursion gives the
likelihood. Each innovation is Gaussian under the model,

$$p(z_{1..T}) = \prod_{t=1}^{T} N\!\left(z_t;\; C m_t^{-},\, S_t\right) \qquad (4)$$

so the log-likelihood is a running sum with no extra work. That is how the
parameters $Q$ and $R$ are usually fitted, and it is why the demo checks (4)
against the exact joint Gaussian of the whole sequence — a check that catches a
wrong $S$ immediately, where the means alone might not.

## Variants

| Variant | What changes | Cost | Use when |
|:--|:--|:--|:--|
| `kf` | the recursion (1)–(3) | $O(n^3 + m^3)$ per step | the model is linear and Gaussian — this is exact |
| `ekf` | $f$ and $h$ linearised at the mean | one Jacobian per step | mildly nonlinear, Jacobians available |
| `ukf` | $2n+1$ sigma points through $f$ and $h$ | $2n+1$ evaluations per step | nonlinear, no Jacobian, or the EKF's linearisation is degenerate |
| `information_filter` | the same posterior as $(y, J) = (P^{-1}m, P^{-1})$ | $O(n^3)$ per step, two inversions | many sensors fused, or a distributed estimate |
| `rts_smoother` | backward pass after the forward pass | one extra $O(n^3)$ per step | the data is offline; you want $p(z_t \mid z_{1..T})$ |
| `missing_obs` | update restricted to the observed rows | same as `kf` | sensors drop out, irregular sampling, ragged panels |

The split is real, not cosmetic. `kf`, `information_filter`, `rts_smoother` and
`missing_obs` compute the *exact* posterior and differ only in algebra or in what
they do with the answer; `ekf` and `ukf` are approximations whose error is second
order in the curvature of $f$ and $h$.

## Derivation

### 1. Why the update is a matrix identity

Everything follows from the Gaussian conditioning identity. If
$(x, z) \sim N(\mu, \Sigma)$ with blocks, then

$$x \mid z \;\sim\; N\!\left(\mu_x + \Sigma_{xz}\Sigma_{zz}^{-1}(z - \mu_z),\;\; \Sigma_{xx} - \Sigma_{xz}\Sigma_{zz}^{-1}\Sigma_{zx}\right) \qquad (5)$$

For the state-space model, $z_t = C x_t + v$ with $v \sim N(0, R)$, so
$\Sigma_{xz} = P^{-}C^{\top}$, $\Sigma_{zz} = C P^{-} C^{\top} + R = S$, and
$\Sigma_{xx} = P^{-}$. Substituting into (5) gives (2)–(3) with nothing left over.
The reason $P$ shrinks is the Schur complement: conditioning removes the part of
the variance that the observation explains.

### 2. The likelihood is the product of innovations

Write $p(z_{1..T}) = \prod_t p(z_t \mid z_{1..t-1})$. Under the model,
$z_t \mid z_{1..t-1} \sim N(C m_t^{-}, S_t)$ — the predictive distribution the
filter already computed. Hence (4). This is also the cleanest way to see that a
filter and a smoother are different questions: the filter's product is
$p(z_{1..T})$, while its per-step posterior is $p(x_t \mid z_{1..t})$.

The demo verifies this the hard way, by assembling the $Tm \times Tm$ covariance
of the whole observation sequence from the state covariances and taking its
Cholesky log-density. The filter's sum matches to $10^{-8}$: same model, two
routes, one number.

### 3. Smoothing

The filter answers "where was the state given the past". The smoother answers
"given everything", and for offline data it is strictly better. The RTS recursion
runs backwards from the last filtered moment:

$$G_t = P_t A^{\top}\left(A P_t A^{\top} + Q\right)^{-1} \qquad (6)$$

$$m_t^{s} = m_t + G_t\left(m_{t+1}^{s} - A m_t\right), \qquad P_t^{s} = P_t + G_t\left(P_{t+1}^{s} - P_{t+1}^{-}\right)G_t^{\top} \qquad (7)$$

$G_t$ is the same matrix the forward pass built to predict $P_{t+1}^{-}$ — which
is why the smoother needs the filter's covariances, not just its means. The
variance can only shrink: $P_t^{s} \preceq P_t$, with equality at $t = T$ where
there is no future to condition on. The demo checks both facts, and the payoff is
visible: position RMSE $0.622$ filtered against $0.351$ smoothed on the same data.

### 4. The information form

Premultiply (3) by $P^{-1}$ and define $J = P^{-1}$, $y = Jm$. The update
becomes additive:

$$J_t = J_t^{-} + C^{\top}R^{-1}C, \qquad y_t = y_t^{-} + C^{\top}R^{-1}z_t \qquad (8)$$

Fusing $k$ sensors is $k$ additions — the reason decentralised and
multi-sensor systems are written this way. The prediction is the ugly half:

$$J_t^{-} = \left(A J_{t-1}^{-1} A^{\top} + Q\right)^{-1}, \qquad y_t^{-} = J_t^{-} A J_{t-1}^{-1} y_{t-1} \qquad (9)$$

Two inversions per step, so the form is a loss unless the sensor count is high or
the estimate is being split across machines. The demo fits both and requires
agreement to $10^{-9}$: same posterior, different arithmetic.

### 5. Linearisation: the EKF

Replace the model by its first-order Taylor expansion at the current mean:

$$f(x) \approx f(m) + F(m)(x - m), \qquad h(x) \approx h(m) + H(m)(x - m) \qquad (10)$$

and run (1)–(3) on the linearised model. The covariance transform is correct to
first order; the mean is biased by $\tfrac12\mathrm{tr}(F'' P)$ and
$\tfrac12\mathrm{tr}(H'' P)$ — terms that vanish on a linear model, which is why
the demo's linear-model assertion (EKF equals KF to $10^{-9}$) is the right unit
test and says nothing about accuracy on a curve.

The failure is not subtle when the linearisation point is wrong. The demo's
one-step test has $h(x) = x^2$ and a prior mean of zero, so $H = 2m = 0$: the
filter concludes the observation is uninformative and returns the prior. The true
evidence is $-1.323$ nats, the UKF reports $-1.290$, the EKF reports $-4.768$.

### 6. The unscented transform

Instead of differentiating, propagate a small deterministic set of points that
matches the first two moments exactly. With $n$ states, $2n+1$ points:

$$\mathcal{X}_0 = m, \qquad \mathcal{X}_{i}^{\pm} = m \pm \left[\sqrt{(n+\lambda)P}\right]_{i}, \qquad \lambda = \alpha^2(n+\kappa) - n \qquad (11)$$

$$w_0^{m} = \frac{\lambda}{n+\lambda}, \quad w_0^{c} = w_0^{m} + 1 - \alpha^2 + \beta, \quad w_i^{m} = w_i^{c} = \frac{1}{2(n+\lambda)} \qquad (12)$$

Push the points through $f$, take the weighted mean and covariance, and you have
the predicted moments to second order — no Jacobian, only $2n+1$ evaluations. The
update needs a cross-covariance, which the same points supply:

$$\Sigma_{xz} = \sum_i w_i^{c}\left(\mathcal{X}_i - m\right)\left(\mathcal{Z}_i - \hat z\right)^{\top}, \qquad K = \Sigma_{xz} S^{-1} \qquad (13)$$

Two details that are easy to get wrong, and both are in the code. The sigma points
used for $h$ must be **redrawn from the predicted moments**: the points that went
through $f$ describe the previous state, not this one. And $\beta = 2$ is the
correction for a Gaussian prior; with $\beta = 0$ the predicted covariance of a
quadratic is biased low.

$\alpha$ controls how far the points sit from the mean. Many implementations set
$\alpha = 10^{-3}$ so the points hug the mean, which is a numerical convenience in
high dimensions — it costs accuracy, and at $\alpha \to 0$ the transform
degenerates towards the EKF. This file defaults to $\alpha = 1$, where the
weights are $O(1)$ and (11) is exact for the moments it claims.

### 7. Missing observations

A missing entry is not a zero and not a NaN to be interpolated. It means the
update has no information from that coordinate, so the correct posterior is

$$m_t = m_t^{-} + K (z_t^{\text{obs}} - C^{\text{obs}} m_t^{-}), \qquad K = P^{-}(C^{\text{obs}})^{\top}\left(C^{\text{obs}} P^{-} (C^{\text{obs}})^{\top} + R^{\text{obs}}\right)^{-1} \qquad (14)$$

with $C^{\text{obs}}$ and $R^{\text{obs}}$ the observed rows and block. When no
row survives, the posterior is the prediction and $P$ grows by $Q$ — the filter
gets less certain because nothing arrived. The demo checks a sharp consequence:
observing one coordinate of a two-dimensional observation must give *exactly* the
same answer as a filter built for that coordinate alone.

## Code walkthrough

One frozenset, one dispatcher, and the two halves of the algorithm behind it.

```python
VARIANTS = frozenset(
    ["kf", "ekf", "ukf", "information_filter", "rts_smoother", "missing_obs"]
)
```

The dispatcher is where the API's one awkwardness lives: `transition` and
`observation` are matrices for four variants and callables for two. Rather than
two entry points, `kalman_filter` reads the variant and converts. `rts_smoother`
is the forward filter plus a backward pass, so it shares the matrix branch.

```python
    if variant in ("kf", "information_filter", "rts_smoother", "missing_obs"):
        A, C = _as_matrix(transition, n), _as_matrix(observation, n)
        if variant == "information_filter":
            return _information_filter(z, A, C, Q, R, m0, P0)
        out = _kf(z, A, C, Q, R, m0, P0, skip=variant == "missing_obs")
        if variant == "rts_smoother":
            out.smooth_mean, out.smooth_cov = _rts(out, A, Q)
        return out
```

The core is (1)–(3). The one line that is not the textbook formula is
`np.linalg.solve(St, Ct @ P).T`: the gain is $P C^{\top} S^{-1}$, and solving for
$S\,X = (CP)^{\top}$ then transposing avoids forming $S^{-1}$ explicitly — the
same number, half the work and a slightly better conditioned result.

```python
        Ct = C[obs]
        St = Ct @ P @ Ct.T + R[np.ix_(obs, obs)]
        innov = z[t, obs] - Ct @ mu
        loglik += _gauss_logpdf(innov, St)
        K = np.linalg.solve(St, Ct @ P).T  # P Cᵀ S⁻¹, without an explicit inverse
        mu = mu + K @ innov
        P = P - K @ Ct @ P
        P = 0.5 * (P + P.T)
```

That symmetrisation is not decoration. The update is a subtraction of two
positive-definite matrices, so round-off can leave $P$ with a small negative
eigenvalue; symmetrising does not cure it but keeps the error from compounding,
and a Cholesky inside `_gauss_logpdf` will raise rather than silently propagate a
non-positive matrix. In long runs, the Joseph form
$P = (I-KC)P(I-KC)^{\top} + KRK^{\top}$ is the answer, at twice the cost.

Missing data is a mask, not a special case, which is why `missing_obs` is one
boolean argument rather than a second filter. The `raise` is the interesting
branch: `kf` refuses a NaN and names the variant that handles it.

```python
        if skip:
            obs = ~np.isnan(z[t])
        else:
            if np.isnan(z[t]).any():
                raise ValueError(
                    "z has a missing observation; use variant='missing_obs', which "
                    "restricts the update to the observed rows"
                )
            obs = np.ones(z.shape[1], dtype=bool)
```

An empty observation set is a real state: the update is the identity, and the
filter has to record the prediction as the posterior. Conflating this with
"skip the step entirely" is the classic bug — the state would stop moving.

```python
        if not obs.any():
            # Nothing to condition on: the posterior *is* the prediction. The
            # update still happens, it is just the identity.
            n_skipped += 1
            mean[t], cov[t] = mu, P
            continue
```

The information form is (8)–(9) literally: two adds for the measurement, two
inversions for the prediction. This is also where the filter's arithmetic shows
its cost — note that the posterior is recovered by inverting $J$ at every step
purely so the two variants can be compared.

```python
            J = J_pred + Ct.T @ np.linalg.solve(Rt, Ct)
            y = y_pred + Ct.T @ np.linalg.solve(Rt, zt)
```

The smoother is (6)–(7) and nothing else. `P_pred` is recomputed rather than
stored: $A P_t A^{\top} + Q$ is exactly what the forward pass used, so the
smoother could read it off a saved array, and caching it would save one matrix
product per step.

```python
    for t in range(T - 2, -1, -1):
        P_pred = A @ out.cov[t] @ A.T + Q
        G = np.linalg.solve(P_pred, A @ out.cov[t]).T
        sm[t] = out.mean[t] + G @ (sm[t + 1] - A @ out.mean[t])
        sc[t] = out.cov[t] + G @ (sc[t + 1] - P_pred) @ G.T
        sc[t] = 0.5 * (sc[t] + sc[t].T)
```

The UKF is where the interesting mistakes live. The weights are (12), and the two
weight vectors differ only in the first entry — that single number is the
second-order correction for a Gaussian, and dropping it biases the predicted
variance low on any curved $f$.

```python
    lam = alpha**2 * (n + kappa) - n
    wm = np.full(2 * n + 1, 0.5 / (n + lam))
    wc = wm.copy()
    wm[0] = lam / (n + lam)
    wc[0] = wm[0] + (1.0 - alpha**2 + beta)
```

The update then redraws the sigma points from the predicted moments. Reusing the
points that went through $f$ — the tempting version, and the bug this file had
first — conditions on the *previous* state's distribution and is off by more than
a unit in the mean on the demo's linear model.

```python
            Xs = _sigma_points(mu, P, n + lam)
            Z = np.array([h(x) for x in Xs])
            zhat = wm @ Z
            Zt = Z[:, obs]
            S = _weighted_cov(Zt, Zt, zhat[obs], zhat[obs], wc) + R[np.ix_(obs, obs)]
            cross = _weighted_cov(Xs, Zt, mu, zhat[obs], wc)
```

`_weighted_cov` is public-minded about a detail that is easy to get wrong: the
cross-covariance is between *state* deviations and *observation* deviations, and
its transpose is what makes the gain $(n, m)$ rather than $(m, n)$.

```python
def _weighted_cov(a: np.ndarray, b: np.ndarray, ma: np.ndarray, mb: np.ndarray,
                  w: np.ndarray) -> np.ndarray:
    """`Σ_i w_i (a_i − ma)(b_i − mb)ᵀ` — the cross-covariance the gain needs.

    Computing the cross term directly, rather than reading it off a scatter
    matrix, is what lets the UKF update a state it never observed.
    """
    return (a - ma).T @ (w[:, None] * (b - mb))
```

The demo's exact likelihood check is the one piece of test code worth reading,
because it is a different algorithm: the joint covariance of $z_{1..T}$ built
from the state covariances, with $C A^{i-j}$ carrying a state $j$ forward to
observation $i$.

```python
    S = np.zeros((T * m, T * m))
    for i in range(T):
        for j in range(i + 1):
            Pij = states[j]
            for _ in range(i - j):
                Pij = A @ Pij
            block = C @ Pij @ C.T
```

## Complexity

With $n$ states, $m$ observations and $T$ steps:

| Variant | Time | Memory | Inversions per step |
|:--|:--|:--|--:|
| `kf` | $O(T(n^3 + m^3))$ | $O(T n^2)$ | 0 (two solves) |
| `information_filter` | $O(T n^3)$ | $O(T n^2)$ | 2 |
| `rts_smoother` | $O(T n^3)$ | $O(T n^2)$ | 0 (one solve) |
| `ekf` | $O(T(n^3 + c_J + m^3))$ | $O(T n^2)$ | 0 |
| `ukf` | $O(T(n^3 + (2n+1)(c_f + c_h)))$ | $O(T n^2)$ | 0 |
| `missing_obs` | $O(T(n^3 + m_{\text{obs}}^3))$ | $O(T n^2)$ | 0 |

The cubic is the covariance. It is why the filter is cheap for $n = 10$ and
hopeless for $n = 10^4$ — and why the ensemble and particle filters exist: they
replace $O(n^3)$ with $O(Nn)$ for $N$ samples, at the price of Monte Carlo error.
Storing every covariance is $O(T n^2)$; if you only need the final state, do not
allocate the arrays.

`ukf` is the most expensive variant per step in practice, because $h$ is
evaluated $2n+1$ times and each evaluation may be a numerical model. That is
still linear in $n$ and usually cheaper than deriving a Jacobian by hand.

## Failure modes

**The linearisation point is not where the probability is.** The EKF's $H = 0$ at
a symmetric prior makes a quadratic sensor look blind: $-4.768$ nats against a
true $-1.323$ on one observation. Nothing about the estimate looks broken — the
mean is right, the variance is untouched — but the likelihood is wrong by 3.4
nats, so any model comparison, any $Q$/$R$ fit by maximum likelihood, and any
gate that thresholds the innovation will be wrong. *Diagnostic:* rerun with
`ukf`; if the likelihood moves by more than a nat on a one-step problem, the
linearisation is the problem.

**The covariance stops being positive definite.** $P - KCP$ is a difference of
positive-definite matrices, and in float64 it loses definiteness after thousands
of steps or immediately when $R$ is tiny. The symptom is a `LinAlgError` from a
Cholesky deep inside a likelihood evaluation, or a covariance with a $-10^{-12}$
eigenvalue that the next update turns into a real failure. *Diagnostic:* track
the smallest eigenvalue of $P$; *fix:* the Joseph form, or a square-root filter
that propagates a Cholesky factor and cannot represent an indefinite matrix.

**Divergence on a strongly curved model.** A first version of this demo used
$f(x) = 0.5x + 25x/(1+x^2)$ with a quadratic sensor, where the derivative
vanishes near the mean: with no information arriving, $P$ grew geometrically
until it reached `inf` at step 109. The filter did not raise; it produced
`nan`s and a likelihood of `nan`. *Diagnostic:* log the innovation and $S$ —
a growing $S$ with a shrinking gain is the filter telling you the model and the
data disagree; use `ukf`, inflate $Q$, or fix the model.

**Treating a missing value as a number.** Interpolating a gap and feeding it to
`kf` as if it were data makes the filter *more* confident at exactly the steps
where it should be less. `missing_obs` exists so that this is a decision, not an
accident: the posterior at a gap is the prediction, and the trace of $P$ grows.

**A wrong $R$ that hides in the covariance.** $R$ too small makes $K \to 1$, the
filter chases noise, and $P$ collapses so hard that later real changes are
ignored ("filter lock"). $R$ too large does the opposite: the filter ignores the
sensor and follows the model, and the innovation becomes systematically biased
rather than white. *Diagnostic:* the normalised innovation squared
$e^{\top} S^{-1} e$ should average about $m$; the demo's exact likelihood check is
the same idea in one number.

## Applications

**Navigation and sensor fusion.** The origin of the method, and still its
largest use: an IMU integrated at high rate between GPS fixes, with the
information form when several sensors report on the same state. Apollo's
onboard computer ran a 15-state filter with a 24 Hz update.

**Tracking and radar.** Constant-velocity or constant-acceleration models with
position-only observations; `missing_obs` is the normal case, not an exception,
because a target drops out of a radar sweep for a few frames all the time.

**Econometrics and structural time series.** Local level, local linear trend and
seasonal components are all state-space models, fitted by maximising (4) over
$Q$ and $R$, and smoothed for historical estimates. `statsmodels`'
`UnobservedComponents` is this file with a parameter optimiser on top.

**Control.** The LQG controller is a Kalman filter feeding an LQR gain; the
separation principle says the two can be designed independently precisely
because the conditional mean is a sufficient statistic for a linear-Gaussian
system.

**Online learning of a linear model.** Recursive least squares is the Kalman
filter with $A = I$, $Q = 0$, $C$ the feature row and $R$ the noise variance —
the same recursion seen from a different literature. [`ridge.py`](../02-linear-models/ridge.py)
is the batch version.

Where it is a trap: any posterior that is not unimodal, and any model where the
nonlinearity is the point rather than a nuisance. A Gaussian belief cannot
represent "either here or there", and a filter that is confident and wrong is
worse than one that is uncertain. [`particle_filter.py`](particle_filter.py)
exists for that case, and [`mcmc.py`](mcmc.py) when the state is not sequential
at all.

## References

- Kalman (1960). *A new approach to linear filtering and prediction problems.*
  Journal of Basic Engineering 82, 35. Three pages, and the whole thing.
- Kalman & Bucy (1961). *New results in linear filtering and prediction theory.*
  Journal of Basic Engineering 83, 95. The continuous-time form and the
  optimality proof.
- Rauch, Tung & Striebel (1965). *Maximum likelihood estimates of linear dynamic
  systems.* AIAA Journal 3, 1445. The smoother.
- Anderson & Moore (1979). *Optimal Filtering.* Prentice-Hall. The reference for
  the information form, the Joseph form and stability analysis.
- Julier & Uhlmann (1997). *A new extension of the Kalman filter to nonlinear
  systems.* SPIE AeroSense. The unscented transform.
- Wan & van der Merwe (2000). *The unscented Kalman filter for nonlinear
  estimation.* IEEE AS-SPCC. The weights in (12), including $\beta$.
- Julier & Uhlmann (2004). *Unscented filtering and nonlinear estimation.*
  Proceedings of the IEEE 92, 401. Why $\alpha \to 0$ is a mistake, and what the
  transform is actually second-order in.
- Särkkä (2013). *Bayesian Filtering and Smoothing.* Cambridge. The one book to
  read next: it derives this file, the particle filter and the extended Kalman
  filter in the same notation.
- Durbin & Koopman (2012). *Time Series Analysis by State Space Methods.* Oxford.
  The applied view: structural models, diffuse initialisation, and the
  likelihood as an optimisation objective.
