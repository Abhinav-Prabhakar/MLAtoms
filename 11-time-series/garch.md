# GARCH — variance that clusters, reacts to bad news and remembers

`garch.py` · Time series · Tier 2 · 6 variants

## Intuition

Returns are hard to forecast; their *variance* is not. Large moves follow large
moves, small moves follow small moves, and the autocorrelation of squared
returns is one of the most reliable facts in finance. A GARCH model writes that
down as a recursion: today's conditional variance is a constant plus a weighted
average of last period's squared shock and last period's variance.

That single change — letting $\sigma_t^2$ depend on the past — turns a
homoskedastic regression into a model of volatility clustering. It also
explains why the model is fitted by maximum likelihood rather than least
squares: each observation has its own variance, and the likelihood is the
product of $\mathcal N(\mu,\sigma_t^2)$ densities with $\sigma_t^2$ generated
recursively from the parameters.

The variants each answer a different stylised fact:

- **`arch`** is the original: variance depends on recent squared shocks only.
  It needs many lags to capture persistence, which is why GARCH replaced it.
- **`garch`** adds lagged variance, giving a two-parameter filter that matches
  the slow decay of volatility with $\alpha+\beta$ near one.
- **`gjr_garch`** adds a term for negative shocks, the *leverage effect*: a
  fall raises future volatility more than an equally sized rise.
- **`egarch`** models $\log\sigma_t^2$, so variance stays positive without
  constraints and the asymmetry enters through a signed $z_{t-1}$.
- **`igarch`** sets $\alpha+\beta=1$ exactly: shocks never decay, and the
  variance forecast becomes a random walk with drift $\omega$.
- **`garch_t`** replaces the Gaussian innovation with a standardised Student-t,
  which is what makes the model usable on daily equity returns.

The demo fits each variant to data simulated from it, checks the conditional
variance recursion by hand, and checks the two limits that matter: GARCH
forecasts mean-revert to $\omega/(1-\alpha-\beta)$, while IGARCH forecasts grow
linearly.

## Variants

| variant | variance recursion | demo result |
|:--|:--|:--|
| `arch` | $\omega+\sum\alpha_i e^2_{t-i}$ | $\alpha=0.4403$ |
| `garch` | plus $\sum\beta_j\sigma^2_{t-j}$ | $\alpha+\beta=0.9780$ |
| `egarch` | $\log\sigma^2$ with signed $z$ | $\gamma=-0.0742$ |
| `gjr_garch` | plus $\gamma e^2\mathbf 1[e<0]$ | $\gamma=0.0734$ |
| `igarch` | $\beta=1-\alpha$ | increments $\omega=0.0899$ |
| `garch_t` | Student-t innovations | $\nu=4.1032$ |

**`arch`** estimates a variance that is a distributed lag of squared shocks.
On 1 000 points with true $\omega=0.5,\alpha=0.5$ it returns $0.4973$ and
$0.4403$; the one-step forecast is exactly $\omega+\alpha e_T^2$. It is the
right diagnostic for short memory and the wrong model for daily equities, where
the estimated $\alpha$ is small and the fit is poor: on the GARCH data below its
AIC is 3663.49 against 3526.52.

**`garch`** is the default. The demo's fit has $\alpha=0.1038$, $\beta=0.8742$
against a true $(0.08,0.9)$, and persistence $0.9780$. Its variance forecast
converges to $2.5438$, matching $\omega/(1-\alpha-\beta)$ to six digits, and its
standardised residuals have variance $0.993$ — the recursion is not leaving
structure behind.

**`egarch`** writes $\log\sigma_t^2$ as an AR(1) with an asymmetric term. On a
simulated path with $\gamma=-0.08$ it recovers $\gamma=-0.0742$ and
$\beta=0.9039$. The estimate of $\alpha$ is poor ($0.0123$ against $0.1$)
because the intercept and $\alpha$ trade off against each other; the leverage
sign, which is the point of the model, is recovered.

**`gjr_garch`** adds a non-negative $\gamma$ on negative squared shocks. The
demo's true $\gamma=0.1$ gives an estimate of $0.0734$, and the model beats the
symmetric one on AIC by only 1.63 points. That margin is a useful warning: with
1 000 observations, leverage is real but not precisely estimated.

**`igarch`** fixes $\beta=1-\alpha$, so a shock has a permanent effect on the
variance level. The demo recovers $\alpha=0.2240$ and $\beta=0.7760$, and every
forecast step adds exactly $\omega=0.0899$ — the linear growth that makes
IGARCH useful for risk over short horizons and dangerous over long ones.

**`garch_t`** estimates the degrees of freedom by likelihood. On data with
$\nu=5$ it returns $4.1032$ and improves the log-likelihood from $-1905.96$ to
$-1846.06$. The 95% interval at one step is $1.96$ times the Gaussian width;
that extra tail is the reason the Gaussian model understates risk on daily
returns.

## Derivation

### Mean and variance

Write the return as a constant mean plus a scaled innovation:

$$
r_t=\mu+e_t,\qquad e_t=\sigma_t z_t,\qquad
\mathbb E[z_t]=0,\quad \mathrm{Var}(z_t)=1 .
\qquad (1)
$$

The whole family differs only in how $\sigma_t^2$ is generated from
$\mathcal F_{t-1}$, the information available at $t-1$.

### GARCH(p, q)

The baseline recursion is

$$
\sigma_t^2=\omega+\sum_{i=1}^{p}\alpha_i e_{t-i}^2
+\sum_{j=1}^{q}\beta_j\sigma_{t-j}^2,
\qquad (2)
$$

with $\omega>0$, $\alpha_i\ge0$ and $\beta_j\ge0$. For $p=1,q=0$ this is
ARCH($q$), and for $p=q=1$ it is the familiar GARCH(1,1). Taking expectations
under stationarity gives

$$
\mathbb E[\sigma^2]=\frac{\omega}{1-\sum_i\alpha_i-\sum_j\beta_j},
\qquad (3)
$$

so the denominator must be positive. The quantity
$\sum\alpha_i+\sum\beta_j$ is the persistence: it is the decay rate of a shock
to the variance, and it is the number the demo prints first.

An equivalent way to see the memory is to substitute $e_t^2=\sigma_t^2z_t^2$:
for GARCH(1,1), $e_t^2=\omega+(\alpha+\beta)e_{t-1}^2+\eta_t$ with $\eta_t$
a martingale difference. Squared returns are therefore ARMA(1,1)-like, which is
why they have a slowly decaying autocorrelation even when returns themselves are
white noise.

### GJR-GARCH

The leverage variant adds an indicator for the sign of the last shock:

$$
\sigma_t^2=\omega+\alpha e_{t-1}^2+\beta\sigma_{t-1}^2
+\gamma e_{t-1}^2\mathbf 1[e_{t-1}<0].
\qquad (4)
$$

For symmetric $z$, $\mathbb E[e^2\mathbf 1[e<0]]=\sigma^2/2$, so the
persistence becomes $\alpha+\beta+\gamma/2$. The code uses exactly that
expression in the stationarity constraint and in `persistence_`. A positive
$\gamma$ means a negative shock raises variance more than a positive one of the
same size — the leverage effect.

### EGARCH

Nelson's exponential form models the log variance:

$$
\log\sigma_t^2=\log\omega
+\alpha\bigl(\lvert z_{t-1}\rvert-\mathbb E\lvert z\rvert\bigr)
+\gamma z_{t-1}+\beta\log\sigma_{t-1}^2,
\qquad (5)
$$

with $\mathbb E\lvert z\rvert=\sqrt{2/\pi}$ for Gaussian $z$. Positivity is
automatic because the right-hand side is exponentiated. A negative $\gamma$
means a negative $z_{t-1}$ raises log variance, so the leverage sign is the
opposite of GJR's $\gamma$. The $\alpha$ term captures the size of the shock
independent of its sign; the $\gamma$ term captures the sign.

### IGARCH

The integrated model imposes

$$
\alpha+\beta=1,
\qquad (6)
$$

so (2) has a unit root in the variance. The forecast does not revert to a
constant: at horizon $h$,

$$
\hat\sigma_{T+h}^2=\hat\sigma_{T+1}^2+(h-1)\omega,
\qquad (7)
$$

which is the linear growth the demo verifies by differencing the forecast path.
IGARCH is the volatility analogue of a random walk with drift, and it should be
read the same way: fine for a few steps, meaningless for a long horizon.

### Likelihood

For Gaussian innovations, the conditional log-likelihood is

$$
\ell(\theta)=-\frac12\sum_{t=s}^{n}\left[
\log2\pi+\log\sigma_t^2+\frac{e_t^2}{\sigma_t^2}\right],
\qquad (8)
$$

where the first $s$ observations are used to initialise the recursion but
excluded from the sum. The code sets $s=\max(p,q,10)$ capped at $n/4$.

For Student-t innovations, standardise $z=t_\nu/\sqrt{\nu/(\nu-2)}$ so that
$\mathrm{Var}(z)=1$. The density of $e_t$ is then

$$
f(e_t)=\frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}
{\Gamma\!\left(\frac{\nu}{2}\right)\sqrt{\pi(\nu-2)}\,\sigma_t}
\left(1+\frac{e_t^2}{(\nu-2)\sigma_t^2}\right)^{-\frac{\nu+1}{2}},
\qquad (9)
$$

and (8) is replaced by the sum of its logarithms. The degrees of freedom are
estimated jointly with the variance parameters; `nu_` is the fitted value.

### Forecasting

Because $e_{T+h}^2=\sigma_{T+h}^2z_{T+h}^2$ has conditional expectation
$\sigma_{T+h}^2$, the GARCH forecast is deterministic:

$$
\hat\sigma_{T+h}^2=\omega
+\sum_{i=1}^{p}\alpha_i\,\mathbb E[e_{T+h-i}^2]
+\sum_{j=1}^{q}\beta_j\,\hat\sigma_{T+h-j}^2,
\qquad (10)
$$

with observed values used where the index is at or before $T$ and forecasts
used afterwards. For GJR the future asymmetric term contributes
$\gamma\hat\sigma^2/2$, and for IGARCH the same recursion with
$\beta=1-\alpha$ produces (7).

For EGARCH, $\mathbb E[\alpha(\lvert z\rvert-\mathbb E\lvert z\rvert)+\gamma z]=0$,
so for $h\ge2$ the *log* variance follows
$\log\hat\sigma_{T+h}^2=\log\omega+\beta\log\hat\sigma_{T+h-1}^2$. The code
returns the exponential of this path. That is the conditional median of
$\sigma^2$, not its mean: exponentiating an expectation of a log is not the
expectation of the level. The distinction is small at short horizons and grows
with $\beta$, and it is the one place where `predict` returns an approximation
rather than an exact conditional moment.

### Information criteria

With $k$ estimated parameters — $\mu$, $\omega$, the $\alpha$'s, the $\beta$'s,
plus $\gamma$ or $\nu$ where present — the demo compares

$$
\mathrm{AIC}=-2\ell+2k,\qquad
\mathrm{BIC}=-2\ell+k\log n_{\text{used}} .
\qquad (11)
$$

The likelihoods are on the same return scale, so the comparison is valid within
a data set. It is not valid across different mean specifications, and it is not
a test: the GJR improvement of 1.63 AIC points is suggestive, not decisive.

## Code walkthrough

The parameter vector is a flat array, and `_unpack` turns it into named
quantities. The IGARCH constraint lives here rather than in the likelihood,
which keeps the search one-dimensional in $\alpha$:

```python
    def _unpack(self, theta: np.ndarray) -> dict:
        index = 2
        if self.variant == "arch":
            alpha = theta[index:index + self.q].copy()
            beta = np.zeros(0)
            index += self.q
        else:
            alpha = theta[index:index + self.p].copy()
            index += self.p
            if self.variant == "igarch":
                beta = np.asarray([1.0 - float(alpha[0])])
            else:
                beta = theta[index:index + self.q].copy()
                index += self.q
```

Index $0$ is $\mu$ and index $1$ is $\log\omega$; the logarithm is what keeps
$\omega$ positive without a constrained optimiser. For IGARCH the $\beta$ slot
is not even allocated, so the reported $\alpha+\beta$ is exactly one by
construction, not by a penalty.

The core recursion is the model written as code. The `t >= lag` guards are what
let the first few observations use the sample variance as a backcast instead of
wrapping around to the end of the array:

```python
            arch_lags = self.q if self.variant == "arch" else self.p
            for t in range(1, n):
                value = omega
                for lag in range(1, arch_lags + 1):
                    past = errors[t - lag] ** 2 if t >= lag else scale
                    value += alpha[lag - 1] * past
                if beta.size:
                    for lag in range(1, self.q + 1):
                        past = variance[t - lag] if t >= lag else scale
                        value += beta[lag - 1] * past
                if gamma:
                    value += gamma * errors[t - 1] ** 2 * (errors[t - 1] < 0.0)
```

The GJR term is the last two lines: `errors[t - 1] < 0.0` is `False` or `True`,
and multiplying by the boolean is the indicator. A naive implementation that
forgets this branch still fits a GARCH and looks plausible; only the asymmetry
diagnostics reveal the missing parameter.

EGARCH is a separate branch because the state is the log variance, not the
variance:

```python
            for t in range(1, n):
                z = errors[t - 1] / math.sqrt(variance[t - 1])
                log_variance = (math.log(omega) + alpha[0] * (abs(z) - SQRT_2_OVER_PI)
                                + gamma * z + beta[0] * log_variance)
                if not np.isfinite(log_variance) or log_variance > 50.0:
                    return None
                variance[t] = math.exp(log_variance)
```

The clip at $50$ is a guard against an explosive candidate during the search; a
log variance above $50$ means a variance above $5\times10^{21}$, which is
numerically useless and should be rejected rather than propagated.

The objective is the negative log-likelihood, with the stationarity constraints
checked before the recursion:

```python
        if (self.variant in ("garch", "garch_t")
                and params["alpha"].sum() + params["beta"].sum() >= 0.999):
            return math.inf
        if self.variant == "gjr_garch":
            total = (params["alpha"].sum() + params["beta"].sum()
                     + 0.5 * params["gamma"])
            if total >= 0.999:
                return math.inf
```

Returning `inf` rather than raising is deliberate: the pattern search treats an
infeasible point as a bad point and steps away, while an exception would abort
the whole fit. The `0.5 * gamma` is the expectation of the leverage term under
a symmetric innovation law.

The two likelihoods differ only in the per-observation constant and the power
on the ratio:

```python
        if self.variant == "garch_t":
            nu = params["nu"]
            constant = (math.lgamma(0.5 * (nu + 1.0)) - math.lgamma(0.5 * nu)
                        - 0.5 * math.log(math.pi * (nu - 2.0)))
            loglik = float(np.sum(constant - 0.5 * np.log(var)
                                  - 0.5 * (nu + 1.0) * np.log1p(ratio / (nu - 2.0))))
        else:
            loglik = float(-0.5 * np.sum(np.log(2.0 * math.pi) + np.log(var)
                                         + ratio))
```

`np.log1p(ratio / (nu - 2.0))` is the log of $1+e^2/((\nu-2)\sigma^2)$; using
`log1p` instead of `log(1 + ...)` matters when the ratio is tiny, which happens
for observations in a quiet period. The `lgamma` terms are the normalising
constant of the $t$ density.

Because the objective is cheap and low-dimensional, the fit uses coordinate
pattern search with step halving rather than a gradient method:

```python
        for _ in range(70):
            improved = False
            for index in range(best.size):
                for delta in (steps[index], -steps[index]):
                    trial = best.copy()
                    low, high = bounds[index]
                    trial[index] = min(max(trial[index] + delta, low), high)
                    if trial[index] == best[index]:
                        continue
                    value = self._objective(trial, r)
                    if value < score - 1e-10:
                        best, score, improved = trial, value, True
            if not improved:
                steps *= 0.5
                if steps.max() < 1e-5:
                    break
```

The bounds are the positivity and stationarity constraints in disguise, and
`steps.max() < 1e-5` is the stopping rule. Pattern search is slower than a
gradient by a constant factor but it does not need derivatives through the
recursion, which is where a hand-written GARCH gradient usually goes wrong.

Forecasting reuses the recursion with observed values for the early lags and
forecasts afterwards. The `future` list holds $\hat e^2$, which equals
$\hat\sigma^2$ under (10):

```python
            if self.beta_.size:
                for lag in range(1, self.q + 1):
                    past = out[h - lag] if lag <= h else variance[h - lag]
                    value += self.beta_[lag - 1] * past
            if self.gamma_:
                if h == 0:
                    value += (self.gamma_ * observed[-1]
                              * (errors[-1] < 0.0))
                else:
                    value += self.gamma_ * 0.5 * future[h - 1]
```

At $h=0$ the sign of the last observed shock is known, so the indicator is
used directly. For every later step the sign is unknown and the code uses the
symmetric expectation $1/2$, which is why the leverage forecast decays at the
rate $\alpha+\beta+\gamma/2$.

The interval uses the innovation law, not the variance recursion:

```python
        variance = self.predict(horizon)
        if self.variant == "garch_t":
            z = _t_quantile(0.5 + level / 2.0, self.nu_
                            ) * math.sqrt(self.nu_ / (self.nu_ - 2.0))
        else:
            z = NormalDist().inv_cdf(0.5 + level / 2.0)
        spread = z * np.sqrt(variance)
        return self.mu_ - spread, self.mu_ + spread
```

The extra factor $\sqrt{\nu/(\nu-2)}$ converts the raw $t$ quantile to the
quantile of the *standardised* $t$ used in (9). Omitting it makes the band too
wide by a factor that grows as $\nu$ falls, which is the opposite of the
intended correction.

## Complexity

With $n$ returns, $p+q$ variance lags, $H$ forecast steps and $P$ searched
parameters:

- **One variance path:** $O(n(p+q))$ time and $O(n)$ memory.
- **One likelihood:** the same, plus $O(n)$ for the log terms.
- **One fit:** about $70\times2P$ objective evaluations before halving; with
  $P\le6$ that is under a thousand paths, so $O(nP^2)$ in practice.
- **Forecast:** $O(H(p+q))$ after the fit.
- **Interval:** $O(H)$ plus one $t$ quantile, which is 120 bisection steps on
  the incomplete beta function.
- **Memory:** $O(n)$ for the variance, residuals and standardised residuals.

The demo's seven fits on 1 000 points take about four seconds in pure Python;
the recursion is a scalar loop, and that is the price of keeping it readable.
For $n$ in the millions the loop should be moved to a compiled kernel, but the
model is usually fitted to daily or monthly data, where $n$ is in the thousands.

## Failure modes

**Persistence near one.** A fitted $\alpha+\beta=0.99$ implies a half-life of
$\log0.5/\log0.99\approx69$ days and a long-run variance
$\omega/(1-\alpha-\beta)$ that is extremely sensitive to the denominator. The
demo's $0.9780$ is healthy; a value of $0.999$ on the same data usually means
the sample contains a structural break, and the forecast will look calm for a
while and then be wrong by a large factor. Check the persistence and the
long-run variance together, not just the likelihood.

**Leverage is hard to identify.** The GJR improvement in the demo is 1.63 AIC
points on 1 000 observations with a true $\gamma=0.1$. On shorter samples the
sign can flip. A positive $\gamma$ in one sample and a negative one in the next
is not evidence that the market changed; it is the standard error doing its job.

**The mean is still a model.** (1) assumes a constant mean. If returns have an
AR component, or a day-of-week effect, the residuals feed a biased $e_t^2$ into
the variance recursion and the persistence estimate absorbs the mean
misspecification. The standard fix is ARMA-GARCH: fit the conditional mean
first, then the variance on its residuals.

**EGARCH forecasts are medians.** As noted under (10), `predict` exponentiates
the expected log variance. $\mathbb E[\exp(X)]\ge\exp(\mathbb E[X])$, so the
reported variance is below the true conditional mean, and the gap grows with
the variance of $\log\sigma^2$. For risk applications, either simulate the
log-variance path or apply the lognormal correction; do not treat the printed
number as $\mathbb E[\sigma^2]$.

**Fat tails are not the same as volatility clustering.** A Student-t fit can
raise the likelihood substantially even when the data have no GARCH structure,
because the tail parameter is doing the work. Compare `garch_t` against
`garch` on the same data, as the demo does: the t model wins by 59.9
log-likelihood points here, but the variance parameters still have to be
significant on their own.

**Zero or constant returns.** The fit refuses a constant series because the
initial variance is zero. A series with long runs of exact zeros — a low-price
stock, a thinly traded asset — will fit a tiny $\omega$ and a persistence near
one for the wrong reason. Aggregate or model the zero process separately.

## Applications

GARCH is the standard language of financial risk. Value-at-Risk uses the
one-step variance forecast and a tail quantile; expected shortfall uses the
same recursion with a different tail integral. The 2008 crisis made the
distinction between Gaussian and Student-t innovations a practical matter:
Gaussian GARCH understated daily losses precisely when the tail mattered, and
the t variant is now the default in most risk systems.

Option pricing uses GARCH as a discrete-time stochastic volatility model: Duan's
(1995) transform gives a risk-neutral recursion, and the model reproduces the
volatility smile without continuous-time Brownian motion. Volatility targeting
in portfolio construction uses the forecast as a denominator, scaling exposure
inversely to predicted risk.

Outside finance, the same recursion models inflation uncertainty, the
volatility of electricity prices, and the variance of sensor noise. The model
is the wrong tool when the variance is driven by an observable covariate rather
than by its own past — then a regression with heteroskedastic errors, or a
stochastic volatility model with a latent AR(1) log variance, is the better
starting point. GARCH is also a poor choice for point forecasts of returns:
its mean forecast is $\mu$, and the interesting content is entirely in the
second moment.

## References

- Engle, R. F. (1982). *Autoregressive conditional heteroscedasticity with
  estimates of the variance of United Kingdom inflation.* Econometrica 50(4),
  987–1007. The ARCH model and the Lagrange-multiplier test.
- Bollerslev, T. (1986). *Generalized autoregressive conditional
  heteroskedasticity.* Journal of Econometrics 31(3), 307–327. GARCH and the
  stationarity condition behind (3).
- Glosten, L. R., Jagannathan, R., Runkle, D. E. (1993). *On the relation
  between the expected value and the volatility of the nominal excess return on
  stocks.* Journal of Finance 48(5), 1779–1801. The GJR asymmetry of (4).
- Nelson, D. B. (1991). *Conditional heteroskedasticity in asset returns: a new
  approach.* Econometrica 59(2), 347–370. EGARCH and the log-variance
  parameterisation of (5).
- Bollerslev, T. (1987). *A conditionally heteroskedastic time series model for
  speculative prices and rates of return.* Review of Economics and Statistics
  69(3), 542–547. The Student-t GARCH likelihood of (9).
- Engle, R. F., Bollerslev, T. (1986). *Modelling the persistence of
  conditional variances.* Econometric Reviews 5(1), 1–50. IGARCH and the
  unit-root interpretation behind (6).
