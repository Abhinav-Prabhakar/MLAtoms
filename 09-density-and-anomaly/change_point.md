# Change-point detection — where the process changed, not just that it is odd

`change_point.py` · Density & anomaly · Tier 2 · 5 variants

## Intuition

An anomaly detector asks whether a point is unusual. A change-point detector
asks whether the *process* that produced the points is still the same one. The
two questions have different answers. A single spike is anomalous and is not a
change point. A shift from mean $0$ to mean $0.01$ standard deviations is not
anomalous at any single point, and it is a change point if it lasts long enough.

```
  x_t
  2 ┤              ┌───────────┐
    │              │           │
  0 ┤──────────────┘           └──────────
    └──────────────┬───────────┬──────────
                   t₁          t₂
```

A spike is a point that does not belong to the segment it is in; a change point
is a place where the segment ends. Fitting the first does not find the second:
a robust anomaly detector will happily absorb a spike and see no change, and a
change-point detector will happily report a change for a run of mildly unusual
points that never rises to an anomaly.

Every variant in this atom answers three questions:

1. **What does one segment cost?** A negative log-likelihood of the points in
   it, given a mean and a variance of its own.
2. **What does a change cost?** A penalty $\beta$ per change, or a threshold on
   a test statistic, or a prior hazard rate.
3. **When do we decide?** After seeing the whole series, or at each step with
   only the past.

The answers to (1) and (2) are the offline methods, `pelt` and
`binary_segmentation`. The answer to (3) splits the online pair, `cusum` and
`bocpd`, from the rest. `window_based` is the cheap local test: compare two
adjacent windows and ask whether they look like the same process.

The demo's first series is four blocks of ninety points with means $0$, $2$,
$-1$ and $0.5$, unit noise, and true change points at $90$, $180$, $270$:

| method | change points | segments costed | objective |
|:--|:--|--:|--:|
| `pelt`, Gaussian cost | 89, 180, 270 | 20 800 | 545.63 |
| `pelt`, mean-only cost | 89, 180, 270 | 16 273 | 190.14 |
| `binary_segmentation` | 89, 180, 270 | 1 059 | 545.63 |
| `window_based`, $w=40$ | 89, 179, 270 | 360 | — |

All four recover the truth to within two samples. The exact optimiser and the
greedy one report the *same* objective, $545.63$, which is the only direct
evidence that the greedy recursion happened to be exact here. The window method
is sixty times cheaper in work than the exact optimiser and the least precise:
its resolution is its window, and at $w=20$ the same test reports only
`[89, 180]`, losing the third change.

The two objectives in the table are not comparable. The mean-only cost fixes one
variance for the whole series and therefore reports a smaller number for the
same segmentation; only differences within a column mean anything.

## Variants

| variant | decides | price of a change | online |
|:--|:--|:--|:--:|
| `pelt` | exact minimisation of (2) | penalty $\beta$ | no |
| `binary_segmentation` | greedy minimisation of (2) | penalty $\beta$ | no |
| `cusum` | sequential test on the running sum | threshold $h$ | yes |
| `bocpd` | posterior over the run length | hazard $H$ | yes |
| `window_based` | likelihood ratio between two windows | $\chi^2_2$ quantile | no |

**`pelt`** is the reference implementation. It computes the exact optimum of the
penalised objective by dynamic programming, and prunes candidate change points
that provably cannot be optimal later. The pruning is what brings the cost down
to linear in $n$ when the number of changes grows with $n$, instead of quadratic
always. Use it when you want the answer and can afford to look at the whole
series.

**`binary_segmentation`** optimises the same objective greedily: find the single
best split, recurse on both halves while the gain pays for the penalty. It costs
$O(n \log n)$ in the good case and it is not exact. The demo's five-level series
shows the failure exactly: it reports `[149, 200]` and pays $379.55$ where the
optimum `[56, 90, 149, 200]` pays $369.81$. Use it when $n$ is large enough that
PELT's bookkeeping matters, and check the answer against PELT on a subsample.

**`cusum`** is Page's cumulative sum. It accumulates the evidence for a shift of
a given size and alarms when the accumulator crosses a threshold. It is the
oldest detector here and still the one on the factory control chart: constant
memory, one pass, and a threshold with a known average run length between false
alarms. The cost is a detection delay and a blind spot for shifts much smaller
than the one it was tuned for.

**`bocpd`** keeps a probability distribution over the *run length* — how many
steps since the last change — and updates it with Bayes' rule at every step. It
needs no threshold, returns a distribution rather than a point, and is the right
choice when you want to say "the change probably happened between 88 and 92"
rather than "there is a change at 89". The price is $O(n)$ work per step, which
makes it $O(n^2)$ over a series.

**`window_based`** slides two adjacent windows along the series and tests
whether they come from the same distribution. It is the only variant here that
reacts to a change in variance as readily as one in mean, and it needs no model
of the segment cost. Its resolution is the window: two changes closer together
than $w$ are one change.

## Derivation

### The cost of one segment

For a segment covering indices $a+1$ to $b$, length $m = b-a$, with its own mean
and variance, the negative log-likelihood at the maximum-likelihood parameters is

$$
C(a,b) = \frac{m}{2}\left(\log 2\pi + 1 + \log\frac{\mathrm{SS}(a,b)}{m}\right)
\qquad (1)$$

where $\mathrm{SS}(a,b) = \sum_{t=a+1}^{b}(x_t - \bar x)^2$. The derivation is
three lines: the Gaussian negative log-likelihood is
$\frac{m}{2}\log(2\pi\sigma^2) + \mathrm{SS}/(2\sigma^2)$, the maximiser over
$\sigma^2$ is $\mathrm{SS}/m$, and substituting it gives (1). The $+1$ is the
$\log e$ from $\log(\mathrm{SS}/m)$ replacing $\log \sigma^2$ at the optimum.

Because the variance is profiled out, a segment that changes only its *spread*
is charged for it: a variance that doubles over a segment of $m$ points costs
about $0.35m$ nats, which is linear in the length and eventually pays any fixed
penalty. That is why `model="gaussian"` finds variance changes and
`model="mean"` cannot.

Both sums in $\mathrm{SS}$ come from prefix sums, so a segment costs $O(1)$
rather than $O(m)$:

$$
\mathrm{SS}(a,b) = S_2(b) - S_2(a) - \frac{(S_1(b)-S_1(a))^2}{b-a},
\qquad S_1(t) = \sum_{s\le t} x_s,\quad S_2(t) = \sum_{s\le t} x_s^2
$$

This is the single most important implementation decision in the file: it turns
every inner loop over a segment into two array lookups, and it is what lets PELT
cost tens of thousands of candidate segments in a few milliseconds.

The mean-only model is the same expression with the variance fixed at
$\sigma^2$, the robust estimate $\hat\sigma^2 = \frac{1}{2}\mathbb{E}[(x_t -
x_{t-1})^2]$. Successive differences are used because a change in the mean does
not change them: the estimator is blind to the very thing being detected, so it
is not inflated by the changes it is supposed to find.

### The objective and the price of a change

A segmentation is a set of change points $0 = t_0 < t_1 < \dots < t_m = n$, and
the objective is

$$
\min_{t_1,\dots,t_{m-1}} \; \sum_{k=1}^{m} C(t_{k-1}, t_k) + \beta m
\qquad (2)$$

with the convention that the last segment ends at $n$. The penalty is the whole
model-selection problem in one number. Too small and every noise fluctuation is
a change; too large and real changes are absorbed into their neighbours. The
default is BIC, $\beta = 2\log n$, because a change adds two parameters (a mean
and a variance) and the likelihood-ratio gain of a spurious split is
asymptotically $\frac{1}{2}\chi^2_2$, whose maximum over $n$ positions is
$O(\log n)$. A real change of size $\delta$ over $m$ points gains
$m\delta^2/(2\sigma^2)$, which grows linearly in $m$ and therefore beats any
$\log n$ penalty for long enough segments. That is the consistency argument, and
it is why the penalty must be in nats and must grow with $n$.

The demo's penalty sweep makes the trade-off concrete on a $300$-point series
with three changes:

| $\beta$ | changes | penalised objective | fit only |
|--:|--:|--:|--:|
| 0.0 | 137 | 89.43 | 89.43 |
| 4.0 | 27 | 429.12 | 321.12 |
| 12.0 | 3 | 487.20 | 451.20 |
| 24.0 | 2 | 521.30 | 473.30 |
| 48.0 | 1 | 554.83 | 506.83 |

At $\beta = 0$ the optimum is a segment every `min_size` points — 137 of
them — because splitting can only reduce the fit. The fit-only column is
monotone increasing in $\beta$ for the same reason, and the count falls
monotonically — the only thing a penalty is supposed to do. The objective
values themselves are not comparable across rows, because $\beta$ is part of
the objective.

### PELT: the same optimum for less work

Let $F(t)$ be the optimal objective for the prefix ending at $t$. Then

$$
F(t) = \min_{s \in \mathcal{R}_t} \big[ F(s) + C(s,t) + \beta \big],
\qquad F(0) = -\beta
\qquad (3)$$

and the optimal segmentation is recovered by following the argmin back from $n$.
The naive evaluation of (3) is $O(n^2)$. PELT's observation is that most
candidates $s$ can never be optimal again. Suppose at time $t$

$$
F(s) + C(s,t) > F(t).
$$

For any later $T > t$, splitting a segment can only lower its cost, so
$C(s,T) \ge C(s,t) + C(t,T)$. Therefore

$$
F(s) + C(s,T) + \beta \;\ge\; F(s) + C(s,t) + C(t,T) + \beta
\;>\; F(t) + C(t,T) \;\ge\; F(T) - \beta + \beta,
$$

so $s$ cannot be the minimiser at $T$ either, and it can be dropped for good.
The inequality $C(s,T) \ge C(s,t) + C(t,T)$ is exactly why the cost must be a
*profile* likelihood: two free segments fit at least as well as one, so the
pruning is a theorem about this cost and not a heuristic that happens to work.

The demo checks the theorem directly. On a $300$-point series PELT costs $18\,779$
candidate segments and the unpruned recursion costs $44\,850$, a factor of $2.4$,
and both return `[60, 139, 208]` with objectives equal to nine decimal places.
PELT's advantage is asymptotic in the number of change points, not in $n$: with
three changes in three hundred points, most of the quadratic work is still
being done.

### Binary segmentation and its failure

Greedy splitting takes the segment $[a,b)$, finds the split $t$ maximising

$$
C(a,b) - C(a,t) - C(t,b) - \beta,
$$

and recurses while that gain is positive. The demo's five-level series has
changes at $60$, $90$, $150$ and $200$ with levels $0, 1.2, -0.6, 2.0, 0.3$. The
first split is at $149$, which is right, and the recursion then looks at
$[0,149)$, which contains *two* changes. No single split of that half pays for
itself — the two-level model fits neither side well — so the recursion stops and
loses both. The exact optimiser pays $369.81$; the greedy one pays $379.55$ and
reports half the changes.

That is masking, and it is the reason to prefer PELT. The failure is not about
the penalty: the two missed changes are jointly worth more than $2\beta$ but
neither is worth $\beta$ on its own in the presence of the other.

### CUSUM: the sequential test

For a shift from $\mu_0$ to $\mu_0 + \delta$ with known unit variance, the
log-likelihood ratio of the last $m$ points is

$$
\sum_{s=t-m+1}^{t}\Big[\delta\,(x_s - \mu_0) - \frac{\delta^2}{2}\Big],
$$

so the running maximum over all starting points is the recursion

$$
g_t = \max\Big(0,\; g_{t-1} + x_t - \mu_0 - k\Big), \qquad k = \delta/2
\qquad (4)$$

with an alarm when $g_t > h$. The allowance $k$ is half the shift the test is
tuned for, and it is the only tuning in the recursion: too small and noise
accumulates, too large and the test is blind to the shift it was built for.

The threshold controls the average run length between false alarms, which grows
exponentially in $h$. With a known mean and $k = 0.5$, simulation gives a
one-sided ARL of $951$ at $h=5$, $2\,556$ at $h=6$, $16\,690$ at $h=8$ and
$34\,451$ at $h=10$; the default here is $h = 8$, so a two-sided test costs
about one false alarm in $8\,350$ samples. The detection delay is roughly
$h/(\delta-k)$, which is $16$ samples for a one-sigma shift and $5.3$ for a
two-sigma one; the demo measures $13$ and $4.5$.

Two details make the implementation usable. The reference mean is re-estimated
from a warm-up window after each alarm, because a detector that keeps comparing
against the original level will fire forever after one change. And the reported
change point is the last index at which the winning statistic was zero — the
start of the excursion that eventually crossed the threshold — not the alarm
time. The alarm time is the change point plus the delay, and reporting it would
bias every estimate late by exactly the quantity the test is worst at.

### BOCPD: a posterior over the run length

Let $r_t$ be the number of steps since the last change, so $r_t = 0$ means a
change happened at $t$. With a constant hazard $H$,

$$
p(r_t \mid x_{1:t}) \;\propto\; \sum_{r_{t-1}} p(r_t \mid r_{t-1})\,
p(x_t \mid r_{t-1}, x_{1:t-1})\, p(r_{t-1} \mid x_{1:t-1})
\qquad (5)$$

where $p(r_t = r_{t-1}+1 \mid r_{t-1}) = 1-H$ and $p(r_t = 0) = H$. The
predictive $p(x_t \mid r_{t-1}, \cdot)$ is where the model lives. With a
Normal-Inverse-Gamma prior on $(\mu, \sigma^2)$ the posterior after $r$
observations is conjugate, and the predictive is a Student-t:

$$
x_t \mid r \;\sim\; t_{2\alpha_r}\Big(\mu_r,\;
\frac{\beta_r(\kappa_r+1)}{\alpha_r\kappa_r}\Big)
$$

with the one-step updates

$$
\kappa_r = \kappa_{r-1}+1,\quad
\mu_r = \frac{\kappa_{r-1}\mu_{r-1} + x_t}{\kappa_r},\quad
\alpha_r = \alpha_{r-1}+\tfrac{1}{2},\quad
\beta_r = \beta_{r-1} + \frac{\kappa_{r-1}(x_t-\mu_{r-1})^2}{2\kappa_r}.
$$

The Student-t is not decoration. Marginalising the unknown variance gives heavy
tails, so one wild point does not by itself make a long run impossible — which
is what stops the detector from restarting on every outlier. It damps the
effect; it does not remove it (see failure modes).

The branch $r_t = 0$ gets the prior predictive, not a flat weight. Without it
the model would treat a restart as equally likely at every value, which is
wrong and, worse, asymmetric: a value near the prior mean would look like a
cheaper restart than one far from it. The prior is centred on the series mean
rather than on $x_0$ for the same reason — a prior centred on the first point
manufactures a change every time the series revisits its origin.

The detector reports a change when the *maximum a posteriori* run length
collapses, not when $p(r_t = 0)$ rises. A single surprising point lifts
$p(r_t=0)$ for one step and then forgets it; the mode moves only when the
evidence persists. In the demo the 3.8-sigma outlier at $t=303$ lifts the short
mass from $0.006$ to $0.1495$ for one step while the mode stays at $33$, and the
mode only collapses six steps later, at $309$, to a run length of $6$ — which
points back at $303$. The reported change is the outlier, not the true change at
$270$.

### The window test

Two adjacent windows of length $w$, with means $m_L, m_R$ and variances
$v_L, v_R$. Let $v$ be the variance of the $2w$ points pooled:

$$
v = \frac{v_L+v_R}{2} + \frac{(m_L-m_R)^2}{4},
\qquad
G = w\big[2\log v - \log v_L - \log v_R\big]
\qquad (6)$$

The second term in $v$ is the between-window mean difference converted to a
variance, which is what makes the same statistic sensitive to a mean shift and
to a spread change at once. Expanding (6) under no change, with
$a = \log v_L$ and $b = \log v_R$,

$$
G \;\approx\; \frac{w(m_L-m_R)^2}{2\sigma^2} + \frac{w}{4}(a-b)^2,
$$

two squares of standard normals, so $G \sim \chi^2_2$ asymptotically. That gives
a threshold with no tuning: since $P(\chi^2_2 > x) = e^{-x/2}$, the level-$\alpha$
quantile is $-2\log\alpha$, Bonferroni-corrected over the number of window
positions. For $400$ points and $w=40$ that is $-2\log(0.05/321) = 16.8$, and
the demo's twenty stationary series produce exactly one alarm over $6\,420$
positions against the one that $\alpha = 0.05$ predicts.

The statistic is smeared over about one window either side of a change, because
every window that straddles the change sees a mixture. Peaks are merged within
$w$, and the resolution is $w$: at $w=20$ the statistic at the third change is
$14.7$ against a threshold of $17.5$ and the change is lost.

## Code walkthrough

### Segment costs from prefix sums

Everything in the file is built on these six lines:

```python
    m = end - starts
    total = s1[end] - s1[starts]
    total2 = s2[end] - s2[starts]
    ss = np.maximum(total2 - total * total / m, 1e-300)
    if model == "mean":
        return ss / (2.0 * sigma2)
    return 0.5 * m * (_LOG2PI + 1.0 + np.log(ss / m))
```

`starts` is an array, so this costs every candidate start for a fixed `end` in
one vectorised pass. The `np.maximum(..., 1e-300)` is not cosmetic: a segment
whose values are all equal has $\mathrm{SS} = 0$ and a cost of $-\infty$, which
would make the optimiser split every constant stretch into single points. The
floor gives such a segment the cost of a very tight one instead.

### The pruning loop

```python
        reach = best[starts] + _costs(s1, s2, starts, t, model, sigma2)
        evaluations += starts.size
        k = int(np.argmin(reach))
        best[t] = reach[k] + penalty
        prev[t] = int(starts[k])
        # keep s only while it can still beat the best segmentation ending here
        keep = {int(s) for s, ok in zip(starts, reach <= best[t], strict=True) if ok}
        candidates = [s for s in candidates if s in keep or t - s < min_size]
```

`reach` is $F(s) + C(s,t)$ for every candidate at once, and `best[t]` is $F(t)$
once the penalty is added. The set comprehension is the pruning test of the
derivation: keep $s$ only if $F(s)+C(s,t) \le F(t)$. The `or t - s < min_size`
keeps candidates that are not yet eligible, which matters because a candidate
younger than `min_size` has not been tested and must not be discarded.

### The greedy split

```python
        base = _costs(s1, s2, np.array([a]), b, model, sigma2)[0]
        left = _costs(s1, s2, np.full(starts.size, a), starts, model, sigma2)
        right = _costs(s1, s2, starts, np.full(starts.size, b), model, sigma2)
        evaluations += starts.size
        gain = base - left - right - penalty
        k = int(np.argmax(gain))
        if gain[k] <= 0.0:
            continue
```

`np.full(starts.size, a)` is the vectorised way of saying "every candidate has
the same left edge". The recursion is an explicit stack rather than a Python
recursion, which keeps deep segmentations from hitting the recursion limit and
makes the order of the splits irrelevant to the result.

### CUSUM

```python
            up = max(0.0, up + (x[t] - mu) - k)
            down = max(0.0, down - (x[t] - mu) - k)
            if up == 0.0:
                last_up = t + 1
            if down == 0.0:
                last_down = t + 1
            if up > h or down > h:
                hit = (t, last_up if up > h else last_down)
                break
```

The two accumulators are the two-sided test: one for an upward shift, one for a
downward one, each reset at zero. `last_up` records where the current excursion
began, which is the change-point estimate, and it is updated whenever the
accumulator returns to zero. Note that the reset is `t + 1`, not `t`: after
processing $x_t$ the accumulator is zero, so an excursion starting at $t+1$ has
not yet seen a point.

### The run-length posterior

```python
        updated = np.empty(m + 1)
        updated[0] = hazard * prior_density
        updated[1:] = (1.0 - hazard) * grow
        total = float(updated.sum())
        if total <= 0.0 or not np.isfinite(total):
            mass[:m + 1] = 0.0
            mass[0] = 1.0
            continue
        mass[:m + 1] = updated / total
```

The whole recursion is those five lines. `grow` is the elementwise product of the
old masses with the Student-t predictive, computed for every run length at once;
the new mass at run length $r+1$ is $(1-H)$ times that, and the mass at $0$ is
$H$ times the prior predictive. Normalising by the total is Bayes' rule. The
`total <= 0` guard matters on long series: with a thousand run lengths the
predictive densities underflow to zero and the normalisation becomes $0/0$.

The sufficient statistics are updated in the same loop, shifted by one so that
row $r+1$ is the posterior after $r+1$ observations:

```python
        kappa_new = kappa[:m] + 1.0
        mu[1:m + 1] = (kappa[:m] * mu[:m] + x[t]) / kappa_new
        kappa[1:m + 1] = kappa_new
        alpha[1:m + 1] = alpha[:m] + 0.5
        beta[1:m + 1] = beta[:m] + 0.5 * (kappa[:m] * (x[t] - mu[:m]) ** 2
                                          / kappa_new)
```

This is the conjugate update written incrementally. The $\beta$ update is the
textbook $\beta + \frac{\kappa(x-\mu)^2}{2(\kappa+1)}$ with the new $\kappa$ in
the denominator, and the shift `[1:m + 1]` is what lets the run length grow by
one without recomputing anything from scratch.

### Deciding where the change is

```python
    found = []
    for t in range(1, n):
        previous = map_run[t - 1]
        if map_run[t] < previous and map_run[t] <= max(min_size, 0.2 * previous):
            found.append(max(0, t - int(map_run[t])))
```

A collapse is a drop to at most a fifth of the previous run length, with a floor
at `min_size` so that the first few steps after a change do not count as a
collapse from a run length of three. The reported index is `t - map_run[t]`,
because a run length of $r$ at time $t$ means the last change was at $t-r$. The
`max(0, ...)` is needed at the start of the series, where the estimated change
can point before the first sample.

### The penalty

```python
        parameters = 2.0 if self.model == "gaussian" else 1.0
        return parameters * math.log(max(self.n_, 2))
```

Two parameters per change for the Gaussian cost, one for the mean-only cost, and
`max(self.n_, 2)` so that a two-point series does not get a negative penalty.
The penalty is in nats, like the cost, and that is the only thing that makes
them addable.

## Complexity

Let $n$ be the series length, $m$ the number of changes found and $w$ the window.

| variant | time | memory |
|:--|:--|:--|
| `pelt` | $O(n)$ expected, $O(n^2)$ worst | $O(n)$ |
| `binary_segmentation` | $O(n\log n)$ typical, $O(n^2)$ worst | $O(n)$ |
| `cusum` | $O(n)$ per pass | $O(1)$ streaming |
| `bocpd` | $O(n^2)$ | $O(n)$ |
| `window_based` | $O(nw)$ | $O(n)$ or $O(w)$ streaming |

PELT's expected linearity is in the number of change points, not in $n$: the
pruning removes candidates at a rate that depends on how much better the
segmentation gets with more changes. On data with no changes at all, nothing
prunes and the recursion is fully quadratic — which is the honest worst case and
the reason the demo reports evaluation counts rather than claiming linearity.

`bocpd` is the expensive one: every step touches every run length, so a
thousand-point series costs a million predictive evaluations, vectorised into a
thousand numpy calls. The file's largest series is $360$ points and the whole
demo runs in a quarter of a second. For a long stream you would truncate the run
length distribution or use the online variant of PELT instead.

## Failure modes

**The penalty is in nats, and everything depends on it.** $\beta = 0$ produces
$137$ changes in $300$ points; $\beta = 48$ produces one. The default
$2\log n$ is right for the Gaussian cost and is *not* right for the mean-only
cost, which uses $\log n$ because it adds one parameter. Comparing objectives
across models — $190.14$ against $545.63$ in the first table — is meaningless:
they are different likelihoods, not different segmentations.

**Greedy segmentation masks.** The five-level series loses two of its four
changes and $9.74$ nats. The mechanism is that a single split of a segment
containing two changes fits neither, so the gain never clears the penalty and
the recursion stops. The diagnostic is cheap: run PELT on the same series. If
the objectives differ, the greedy answer is not the optimum.

**CUSUM's calibration assumes a known reference.** With a known mean and
$h=8$, the two-sided false-alarm rate is about one in $8\,350$ samples. The demo
measures eight alarms in eight thousand stationary points, an order of magnitude
more. The reference mean is estimated from a twenty-point warm-up, whose
standard error is $1/\sqrt{20} = 0.22$ standard deviations — a fifth of the
$0.5$-sigma allowance. A reference that is off by $0.22\sigma$ turns the
allowance into $0.28\sigma$ in one direction, and the ARL at $0.28$ is a
fraction of the ARL at $0.5$. Use a longer warm-up, or accept the rate and say
so.

**CUSUM's delay is not a property of the data.** At $h=8$ and $k=0.5$ a
half-sigma shift is caught $13$ times in twenty and takes $21$ samples when it
is; a one-sigma shift is caught every time in $13$; a three-sigma shift in $3$.
The threshold and the allowance set the delay, and the threshold sets the false
alarms. There is no setting that is fast and quiet at once; that trade-off *is*
the method.

**BOCPD cannot tell an outlier from a change.** The demo's series has one
3.8-sigma point at $t=303$ and BOCPD reports a change there, six steps after the
point, because the contaminated long run eventually loses to a fresh segment
that also contains the outlier. The Student-t predictive damps this and does not
prevent it. If your data has spikes, run an anomaly detector first and mask
them, or use `pelt`, whose segment cost absorbs a single point into a long
segment at a cost of a few nats.

**The window method's resolution is its window.** At $w=20$ the third change
scores $14.7$ against a threshold of $17.5$ and is lost; at $w=40$ it scores
$26.3$ and is found. Two changes closer together than $w$ merge into one, and a
change in the first or last $w$ points cannot be seen at all.

**A mean-only model on a variance change finds the wrong thing.** The demo's
series doubles its variance at $150$ and keeps its mean. `pelt/mean` reports
`[231, 234]` — spurious changes inside the noisy half — and misses $150$
entirely, while the Gaussian cost finds $150$ exactly. If the spread can change,
the cost must profile the variance.

**Stationarity within a segment is assumed.** Every variant here fits a constant
mean and variance per segment. A slowly drifting series will be chopped into
many segments by PELT, or produce a stream of alarms from CUSUM, because the
model has no way to say "drifting". Detrend or difference first.

## Applications

**Statistical process control.** CUSUM was invented for manufacturing
(Page, 1954) and is still the standard chart: one accumulator per quality
variable, a threshold set from the ARL the plant can tolerate, and an alarm that
means "stop the line". The demo's delay table is the production trade-off in
miniature.

**Telemetry and operations.** Offline, `pelt` segments a metric series for a
postmortem — "the latency changed at 14:32 and again at 15:10". Online, `bocpd`
puts a posterior over the run length on a dashboard, which is more useful than a
binary alarm because it degrades gracefully: an ambiguous point moves the
distribution rather than firing.

**Genomics.** Copy-number segmentation is change-point detection on a
log-ratio series along a chromosome, and circular binary segmentation (Olshen et
al., 2004) is the standard tool. The segments are the answer; the individual
probes are not.

**Climate and hydrology.** Regime shifts in temperature, rainfall and river
flow are change points in a noisy series, and the penalty is the scientific
claim: a segmentation with many changes says the climate is variable, one with
few says it shifted.

**Finance.** Volatility regimes are variance changes, which is why the Gaussian
cost and the window test matter more here than a mean-only detector. The
distinction between "a bad day" and "a new regime" is exactly the anomaly versus
change-point distinction.

**As a preprocessing step.** Segment first, then run an anomaly detector per
segment. `hbos` or `kde` fitted on a series that spans a regime change will
score the old regime as anomalous; fitted per segment, it will not. The two
atoms compose, and the order matters.

## References

- Page, E. S. (1954). Continuous inspection schemes. *Biometrika* 41(1),
  100–115. The CUSUM.
- Lorden, G. (1971). Procedures for reacting to a change in distribution.
  *Annals of Mathematical Statistics* 42(6), 1897–1908. The average run length
  and the optimality of the CUSUM for a given delay.
- Killick, R., Fearnhead, P. & Eckley, I. A. (2012). Optimal detection of
  changepoints with a linear computational cost. *JASA* 107(500), 1590–1598.
  PELT and the pruning theorem.
- Adams, R. P. & MacKay, D. J. C. (2007). Bayesian online changepoint detection.
  arXiv:0710.3742. The run-length recursion in (5).
- Barry, D. & Hartigan, J. A. (1993). A Bayesian analysis for change point
  problems. *JASA* 88(421), 309–319. The conjugate prior and the product
  partition.
- Olshen, A. B., Venkatraman, E. S., Lucito, R. & Wigler, M. (2004). Circular
  binary segmentation for the analysis of array-based DNA copy number data.
  *Biostatistics* 5(4), 557–572. Binary segmentation as it is actually used.
- Scott, A. J. & Knott, M. (1974). A cluster analysis method for grouping means
  in the analysis of variance. *Biometrics* 30(3), 507–512. The DP recursion
  that PELT prunes.
- Fearnhead, P. & Liu, Z. (2011). On-line inference for multiple changepoint
  problems. *JRSS-B* 69(4), 589–605. Particle filters for the online problem,
  and why the exact recursion is quadratic.
- Truong, C., Oudre, L. & Vayatis, N. (2020). Selective review of offline change
  point detection methods. *Signal Processing* 167, 107299. The survey that
  names the cost functions this file uses.
