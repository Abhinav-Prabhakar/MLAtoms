# Copulas

`copula.py` · Probabilistic models · Tier 3 · 6 variants

## Intuition

A joint distribution is two things at once: where each variable sits, and how
they move together. Most of the time only the second is interesting, and it is
the part that survives any change of units. Sklar's theorem says the split is
exact:

$$F(x_1,\dots,x_d) = C\big(F_1(x_1),\dots,F_d(x_d)\big), \qquad (1)$$

with each $F_j$ a marginal and $C$ a copula — a distribution on the unit cube
whose own margins are uniform. Apply $F_j$ to a column and the units, the
location, the skew and the scale all disappear; what is left is $C$.

$$C(u_1,\dots,u_d) = P(U_1 \le u_1, \dots, U_d \le u_d), \qquad U_j = F_j(X_j) \qquad (2)$$

Two consequences carry the whole subject.

- **Margins are free.** Fit them however you like — empirical, normal, GPD for
  the tail — and glue the dependence on afterwards. A copula fitted to ranks
  cannot be disturbed by a monotone transform of any column.
- **Dependence is one object.** Every variant below is a different $C$, and the
  differences show up where correlations are least informative: the corners.

```
   raw data                    ranks                   the copula
   ─────────                   ─────                   ──────────
   x₂ │      · ·             1 │     ·  ·              1 │    · ·
      │   ·  · ·                │   ·  ·                  │  · ·
      │ · ·                    │ · ·                     │ · ·
      └──────────► x₁       0 └──────────► u₁         0 └─────────► u₁
     units, location,          uniform on [0,1]²         the dependence,
     skew, outliers            by construction           nothing else
```

The picture in the middle is the point: after the rank transform every column
has the same distribution regardless of what it was, so the only thing a fit can
possibly see is the arrangement.

**Tail dependence** is the quantity that separates the families, and the reason
to prefer one over another:

$$\lambda_L = \lim_{u \to 0} \frac{C(u,u)}{u}, \qquad \lambda_U = \lim_{u \to 1} \frac{1 - 2u + C(u,u)}{1-u} \qquad (3)$$

$\lambda_L > 0$ means that a draw in the bottom $u$-quantile of one variable has
probability about $\lambda_L$ of being in the bottom $u$-quantile of the other,
*for every* $u$, however small. A Gaussian copula fails this for every
$|\rho| < 1$: its conditional probability of a joint crash does go to zero. If
the application is a joint default, a joint flood, or two assets selling off
together, the Gaussian copula asserts that the scenario has probability zero in
the limit, and no amount of fitting changes that.

The six variants are chosen to span that space:

| family | $\lambda_L$ | $\lambda_U$ | shape of the dependence |
|:--|:--|:--|:--|
| `gaussian` | 0 | 0 | symmetric, elliptical, tail-free |
| `t` | $>0$ | same | symmetric, elliptical, heavy in both tails |
| `clayton` | $2^{-1/\theta}$ | 0 | strong lower tail, no upper |
| `gumbel` | 0 | $2-2^{1/\theta}$ | strong upper tail, no lower |
| `frank` | 0 | 0 | symmetric, concentrated in the body |
| `vine` | pair-dependent | pair-dependent | one family per pair, per tree |

## Variants

| Variant | $C(u)$ | Parameter | Fitted by | Exact statement the demo checks |
|:--|:--|:--|:--|:--|
| `gaussian` | $\Phi_R(\Phi^{-1}(u_1),\dots)$ | $\rho$ or $R$ | $\tau \mapsto \sin(\pi\tau/2)$ | axioms, $\partial^2 C = c$, $\int c = 1$, KS |
| `t` | $t_{R,\nu}(t_\nu^{-1}(u_1),\dots)$ | $\rho$, $\nu$ | $\tau$ for $\rho$, profile for $\nu$ | $\tau$ analytic vs empirical, tail mixture |
| `clayton` | $(\sum_j u_j^{-\theta} - d + 1)^{-1/\theta}$ | $\theta>0$ | $\tau \mapsto 2\tau/(1-\tau)$ | $\lambda_L = 2^{-1/\theta}$ to $10^{-6}$ |
| `gumbel` | $\exp(-(\sum_j(-\log u_j)^\theta)^{1/\theta})$ | $\theta \ge 1$ | $\tau \mapsto 1/(1-\tau)$ | $\lambda_U = 2-2^{1/\theta}$ to $10^{-6}$ |
| `frank` | $-\frac{1}{\theta}\log\left(1+\frac{\prod_j(e^{-\theta u_j}-1)}{(e^{-\theta}-1)^{d-1}}\right)$ | $\theta \neq 0$ | bisection on $\tau(\theta)$ | $\tau$ from Debye quadrature |
| `vine` | pair-copula product | one per pair | sequential, tree by tree | $\int c_{123}\,du_3 = c_{12}$ |

**`gaussian`** — the default, and the one that taught everyone to distrust
copulas in 2008. Its density is the multivariate normal's with the marginal
factors divided out:

$$c(u) = |R|^{-1/2}\exp\left(-\tfrac{1}{2}z^\top (R^{-1} - I) z\right), \qquad z = \Phi^{-1}(u) \qquad (4)$$

The $+I$ inside is the marginal correction: the normal density has
$z^\top R^{-1} z$, and dividing by $\prod \phi(z_j)$ adds back $z^\top z$. Forget
it and every axiom test still passes — only the density check and $\int c = 1$
catch it.

**`t`** — the Gaussian's inverse-Wishart mixture. Draw $W \sim \chi^2_\nu$ once,
divide the whole normal vector by $\sqrt{W/\nu}$, and the result is multivariate
$t$. Elliptical like the Gaussian, but the shared divisor makes joint extremes
arrive together, so both tail dependences are positive and equal:

$$\lambda_L = \lambda_U = 2\,\bar t_{\nu+1}\!\left(-\sqrt{\frac{(\nu+1)(1-\rho)}{1+\rho}}\right) \qquad (5)$$

with $\bar t$ the Student-t survival function. As $\nu \to \infty$ this decays
to zero and the family becomes Gaussian.

**`clayton`** and **`gumbel`** are the asymmetric pair. Clayton is generated by
a shared Gamma frailty, Gumbel by a shared positive-stable frailty; both are
archimedean, $C(u) = \psi^{-1}(\sum \psi(u_j))$, with $\psi$ the generator.
Clayton's density diverges at the origin and vanishes at the far corner, Gumbel
the reverse. Pick Clayton for survival/joint-failure data, Gumbel for joint
booms — equity indices, hurricane counts, anything where the upside co-moves and
the downside does not.

**`frank`** has no tail dependence at all, but unlike the Gaussian its
dependence is not elliptical: it is strongest in the middle of the distribution.
It is the archimedean member that can model negative dependence ($\theta < 0$),
and it is symmetric under $u \mapsto 1-u$, which makes it the right null model
when you want dependence without a tail story.

**`vine`** replaces one parametric family with a product of pair copulas. For
three dimensions rooted at variable 1:

$$c_{123}(u_1,u_2,u_3) = c_{12}(u_1,u_2)\,c_{13}(u_1,u_3)\,c_{23\mid1}\big(h(u_2\mid u_1),\,h(u_3\mid u_1)\big) \qquad (6)$$

Each factor is an ordinary bivariate copula, so the families can be mixed — a
Gumbel for the first tree, Clayton for the second — and each pair gets its own
tail behaviour. That flexibility is the reason vines replaced the Gaussian
copula in energy and insurance; it is also why they are the hardest to fit
honestly, since a $d$-dimensional vine has $d(d-1)/2$ pairs and the later trees
are built from estimated h-functions, so errors compound.

## Derivation

### The density is the mixed derivative

Differentiating (1) once per coordinate,

$$c(u_1,\dots,u_d) = \frac{\partial^d C}{\partial u_1 \cdots \partial u_d}, \qquad
f(x) = c\big(F_1(x_1),\dots\big)\prod_j f_j(x_j). \qquad (7)$$

This is the definition the code tests: a central difference of `cdf` in two
coordinates must reproduce `logpdf`, and the double integral of $c$ over the
square must be one. The axioms $C(u,1)=u$, $C(u,0)=0$, $C(1,\dots,1)=1$ alone
admit any function agreeing with the margins, including ones whose mixed
derivative is negative somewhere; the density check is what pins a family down.

### Kendall's tau, and why it is the parameter

For a bivariate copula,

$$\tau = 4\iint_{[0,1]^2} C(u,v)\,dC(u,v) - 1 = 1 - 4\iint \frac{\partial C}{\partial u}\frac{\partial C}{\partial v}\,du\,dv. \qquad (8)$$

$\tau$ depends only on the copula, never on the margins, and it has a closed form
in every family here:

$$\tau_{\text{gauss}} = \frac{2}{\pi}\arcsin\rho, \qquad
\tau_{\text{clayton}} = \frac{\theta}{\theta+2}, \qquad
\tau_{\text{gumbel}} = 1 - \frac{1}{\theta}, \qquad
\tau_{\text{frank}} = 1 - \frac{4}{\theta} + \frac{4}{\theta^2}D_1(\theta) \qquad (9)$$

where $D_1(\theta) = \frac{1}{\theta}\int_0^\theta \frac{t}{e^t-1}\,dt$ is the
first Debye function. Inverting these turns an estimate of $\tau$ into an
estimate of the parameter, which is exactly what `fit` does. The inversion is
closed form for the first three and a bisection for Frank, which is monotone but
has no elementary inverse.

$\tau$ is also the reason the fit is margin-free. The sample version counts
concordant minus discordant pairs:

$$\hat\tau = \frac{n_c - n_d}{\binom{n}{2}}, \qquad (10)$$

and a pair is concordant exactly when the two rank orders agree, so the count
depends on ranks only. The demo refits on $x^2$ and $\log y$ and asserts the
estimate is unchanged to $10^{-12}$.

### Tail dependence, derived once

$\lambda_L$ is a limit of a conditional probability. For the t copula it follows
from the fact that both coordinates share the same $W$: conditionally on a small
$W$, both $z$'s are large together, and the probability that the *other*
coordinate is also extreme given one is extreme is the probability that a
$t_{\nu+1}$ variable — the conditional distribution of one coordinate given the
other — falls below a threshold that grows with $\nu$ and $\rho$. That yields
(5). For Clayton, $C(u,u) \approx (2u^{-\theta})^{-1/\theta} = 2^{-1/\theta}u$ as
$u \to 0$, so the ratio in (3) is the constant:

$$\lambda_L^{\text{clayton}} = 2^{-1/\theta}, \qquad
\lambda_U^{\text{gumbel}} = 2 - 2^{1/\theta}. \qquad (11)$$

Both are exact for every $u$ to leading order, which is why the demo can assert
them to $10^{-6}$ at $\varepsilon = 10^{-8}$ while the Gaussian's ratio is still
$4\times10^{-3}$ there.

### Sampling: the conditional, and the frailty

Two constructions cover all six variants.

**Conditional inversion.** If $h(u \mid v) = \partial C(u,v)/\partial v$ is the
conditional CDF of $U$ given $V=v$, then $U = h^{-1}(W \mid V)$ with
$W \sim \mathrm{Unif}(0,1)$ gives a draw from $C$. For the Gaussian,

$$h(u \mid v) = \Phi\!\left(\frac{\Phi^{-1}(u) - \rho\Phi^{-1}(v)}{\sqrt{1-\rho^2}}\right), \qquad (12)$$

which inverts in one line. Frank has a closed-form $h$ and $h^{-1}$ too, and its
sampler is this construction in $d=2$. Clayton and Gumbel have a closed-form $h$
but an awkward $h^{-1}$ (Gumbel's needs a bisection), so they use the next
construction instead.

**Marshall–Olkin frailty.** Write the copula as a mixture over a positive
frailty $V$: conditionally on $V$, the coordinates are independent with survival
$\exp(-V A_j(u_j))$. Then

$$C(u) = \mathbb{E}_V\left[\exp\left(-V\textstyle\sum_j A_j(u_j)\right)\right] = \mathcal{L}_V\!\left(\textstyle\sum_j A_j(u_j)\right). \qquad (13)$$

Clayton takes $A_j(u) = u^{-\theta}-1$ and $V \sim \mathrm{Gamma}(1/\theta,1)$,
whose Laplace transform is $(1+s)^{-1/\theta}$; that reproduces the Clayton
form. Sampling is then two lines: draw $V$, draw independent exponentials
$E_j$, and set $U_j = (1 + E_j/V)^{-1/\theta}$. Gumbel takes
$A_j(u) = (-\log u)^\theta$ and $V$ positive-stable with index $1/\theta$ and
Laplace transform $\exp(-s^{1/\theta})$, giving
$U_j = \exp(-(E_j/V)^{1/\theta})$. The stable draw is Kanter's:

$$V = \frac{\sin(\alpha\Phi)}{\sin(\Phi)^{1/\alpha}}\left(\frac{\sin((1-\alpha)\Phi)}{E}\right)^{(1-\alpha)/\alpha}, \qquad \Phi \sim \mathrm{Unif}(0,\pi),\; E \sim \mathrm{Exp}(1). \qquad (14)$$

The cosine form of the general stable law is the wrong branch here: it is the
$\beta \neq 1$ representation, and using it makes the margins non-uniform at the
third decimal. Kanter's sine form is the one that has Laplace transform exactly
$\exp(-s^\alpha)$, which is what (13) needs.

### The vine recursion

For three variables in a C-vine rooted at 1, the h-functions do the work:

```
  tree 1        (1,2)          (1,3)
                 │              │
  tree 2          └── (2,3 | 1) ┘
```

Sampling runs the tree in reverse. Draw $w_1,w_2,w_3$ uniform:

$$u_1 = w_1, \qquad u_2 = h_{12}^{-1}(w_2 \mid u_1), \qquad
u_3 = h_{13}^{-1}\Big(h_{23}^{-1}\big(w_3 \mid h_{12}(u_2 \mid u_1)\big) \;\Big|\; u_1\Big). \qquad (15)$$

Each $h$ is an ordinary bivariate conditional from one of the four families, so
the whole sampler is (12) and its Clayton/Gumbel/Frank cousins applied twice.
The density (6) is the product of the three pair densities evaluated at the
right arguments — the second tree's pair sees *conditional* uniforms, not the
raw ones.

## Code walkthrough

`Copula(variant, d=2, theta=..., rho=..., df=..., families=..., seed=0)` builds
the object; `sample(n)`, `cdf(U)`, `logpdf(U)`, `fit(X)`, `tau()`, `tail()` are
the surface. The demo runs the axioms, then the density identity, then
integration, then margins, then tau three ways, then the fit, then the tails,
then the vine, then the refusals.

### 1. The normal quantile is the load-bearing numeric

Every family except Clayton and Gumbel needs $\Phi^{-1}$, and a sampler needs it
accurate at $u = 10^{-12}$, not at $10^{-3}$. Acklam's rational approximation is
good to $10^{-9}$; two Newton steps on the exact CDF take it to machine
precision.

```python
    for _ in range(2):                     # 1e-9 is not enough for a sampler
        err = np.clip(_Phi(x) - u, -0.5, 0.5)
        x = x - err / np.maximum(np.exp(-0.5 * x * x - 0.5 * LOG2PI), 1e-300)
    return x
```

The clip matters: Acklam returns a value a few ulps outside the bracket at the
ends, and one unclipped Newton step there sends $x$ to infinity. `_Phi` itself
is a Python loop over `math.erfc` because that is the only correctly rounded
normal CDF in the standard library — a vectorised rational approximation would
cost accuracy exactly where the tail is measured.

### 2. The bivariate normal CDF, by Drezner's integral

```python
    a = math.asin(min(max(rho, -0.999999), 0.999999))
    x, w = np.polynomial.legendre.leggauss(nodes)
    th = 0.5 * a * (x + 1.0)
    c2 = np.cos(th) ** 2
    q = (np.outer(h * h, np.ones_like(th)) - 2.0 * np.outer(h * k, np.sin(th))
         + np.outer(k * k, np.ones_like(th)))
    return _Phi(h) * _Phi(k) + (np.exp(-q / (2.0 * c2)) @ (0.5 * a * w)) / (2.0 * math.pi)
```

$$\Phi_2(h,k;\rho) = \Phi(h)\Phi(k) + \frac{1}{2\pi}\int_0^{\arcsin\rho}
\exp\left(-\frac{h^2 - 2hk\sin\theta + k^2}{2\cos^2\theta}\right)d\theta. \qquad (16)$$

The one non-obvious line is the outer product: `np.outer(h * k, np.sin(th))`
rather than `np.outer(h, k) * np.sin(th)`. The second form is right for one
point and silently transposes the two axes for a batch, because `np.outer(h, k)`
loses the point index. With 200 nodes this agrees with direct trapezoid
quadrature of $\int \phi(x)\Phi((k-\rho x)/\sqrt{1-\rho^2})dx$ to
$10^{-11}$ at every corner tested, including $(2,-2.5)$ at $\rho = 0.95$.

### 3. Conditional CDFs, one arm per family

```python
    if family == "clayton":
        return v ** (-param - 1.0) * (u ** (-param) + v ** (-param) - 1.0) ** (-(1.0 + param) / param)
    if family == "gumbel":
        a, b = -np.log(u), -np.log(v)
        s = (a ** param + b ** param) ** (1.0 / param)
        return np.exp(-s) * s ** (1.0 - param) * b ** (param - 1.0) / v
    if family == "frank":
        a = np.exp(-param * u) - 1.0
        b = np.exp(-param * v) - 1.0
        return a * np.exp(-param * v) / ((math.exp(-param) - 1.0) + a * b)
```

These are the derivatives of each $C$ in its second argument, simplified by
hand. Gumbel's is the one to check: differentiating $\exp(-S)$ with
$S = (a^\theta+b^\theta)^{1/\theta}$ gives
$-e^{-S}S^{1-\theta}b^{\theta-1}\,\partial b/\partial v$, and
$\partial b/\partial v = -1/v$, so the two minus signs cancel and the expression
ends with `/ v`. Drop that and `h` is not a CDF — it does not even reach one at
$u=1$ — which the $h(1) \approx 1$ assertion catches.

### 4. Inverses: closed form where possible, bisection for Gumbel

```python
    if family == "clayton":
        a = (w * np.clip(v, EPS, 1.0) ** (param + 1.0)) ** (-param / (1.0 + param))
        return np.clip(a - np.clip(v, EPS, 1.0) ** (-param) + 1.0, EPS, None) ** (-1.0 / param)
    if family == "frank":
        b = np.exp(-param * np.clip(v, EPS, 1.0)) - 1.0
        s = w * (math.exp(-param) - 1.0) / (np.exp(-param * np.clip(v, EPS, 1.0)) - w * b)
        return np.clip(-np.log1p(s) / param, EPS, 1.0 - EPS)
```

Clayton inverts by raising to $-1/\theta$ twice; Frank by solving the linear
fractional form for $e^{-\theta u}$, which is why it needs `log1p` rather than
`log` — the argument is near zero in the lower tail and `log(1+x)` loses half
its digits there. Gumbel has no closed form, so `_h_inv` runs 60 bisection steps
on the monotone `_h`; 60 steps is $2^{-60}$ of the interval, far past double
precision, and it costs nothing because it is 60 vectorised passes.

### 5. Pair densities, and the sign trap

```python
    if family == "gumbel":
        a, b = -np.log(u), -np.log(v)
        s = (a ** param + b ** param) ** (1.0 / param)
        return (-s - np.log(u) - np.log(v) + (param - 1.0) * (np.log(a) + np.log(b))
                + (1.0 / param - 2.0) * np.log(a ** param + b ** param) + np.log(s + param - 1.0))
```

Gumbel's density is the one that took three attempts. The Jacobian of
$u \mapsto -\log u$ contributes $-\log u - \log v$, and the second derivative of
$S$ contributes the $S + \theta - 1$ factor; omitting either leaves a function
that is positive, integrates to one, and is wrong by a factor that depends on
$(u,v)$. The test is the mixed difference of the CDF at $(0.37,0.61)$, which
catches it at $10^{-8}$.

The same function is reused by `logpdf` for the bivariate Gumbel and Frank, so
there is one copy of each formula and the vine's pair densities come from the
same code path as the standalone families.

### 6. Ranks, and the tau that only sees them

```python
def pseudo_observations(X) -> np.ndarray:
    """`rank/(n+1)` per column — the margins removed, nothing else changed."""
    X = np.asarray(X, dtype=float)
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    n = X.shape[0]
    if n < 2:
        raise ValueError("need at least two rows")
    r = np.argsort(np.argsort(X, axis=0, kind="stable"), axis=0, kind="stable") + 1.0
    return r / (n + 1.0)
```

The double `argsort` is the rank; the $n+1$ denominator is the standard
correction (Genest–Favre) that keeps the pseudo-observations off the boundary,
where every family's density is infinite or zero. `kind="stable"` makes ties
resolve consistently instead of by memory order.

```python
def _kendall_pair(x, y) -> float:
    """Kendall's tau, ties-free, by counting inversions after sorting on `x`."""
    n = np.asarray(x).size
    order = np.argsort(np.asarray(x), kind="stable")
    ranks = np.argsort(np.argsort(np.asarray(y)[order], kind="stable"), kind="stable") + 1
    return 1.0 - 4.0 * _inversions(ranks) / (n * (n - 1.0))
```

Sort by $x$, then a pair is discordant exactly when $y$ decreases, i.e. when
the rank sequence has an inversion. Counting inversions with a Fenwick tree is
$O(n\log n)$; the naive double loop is $O(n^2)$ and at $n=20000$ that is the
difference between 6 ms and 20 seconds. This is also the reason there is no
`scipy.stats.kendalltau` import: the count is four lines and the tree is ten.

### 7. Inverting tau, including the Debye function

```python
    if family == "frank":
        if abs(param) < 1e-9:
            return 0.0                                   # independence, θ → 0
        x, w = np.polynomial.legendre.leggauss(200)      # Debye D₁(θ)
        t = 0.5 * param * (x + 1.0)
        # the ½·θ is the Jacobian of [−1,1] → [0,θ]; dropping it scales D₁ by θ
        d1 = 0.5 * param * np.sum(w * np.where(np.abs(t) < 1e-12, 1.0, t / np.expm1(t)))
        return 1.0 - 4.0 / param + 4.0 / param ** 2 * d1
```

Frank is the only family whose $\tau$ needs a quadrature, and the trap is the
Jacobian: Gauss–Legendre gives nodes on $[-1,1]$, the integral runs over
$[0,\theta]$, and forgetting the $\tfrac12\theta$ factor multiplies $D_1$ by
$\theta$. The symptom is subtle — $\tau(0.5)$ came out as $-0.12$ instead of
$0.31$ — and it only shows up when the fitted $\theta$ is compared to the truth.
The `abs(param) < 1e-9` guard exists because the bisection in `_param_from_tau`
starts at the midpoint, which is zero, and $4/\theta$ divides by it.

### 8. Sampling: frailty for the archimedeans

```python
        if v == "clayton":
            # Marshall–Olkin: V ~ Gamma(1/θ), E_j ~ Exp(1), U_j = (1+E_j/V)^(−1/θ)
            w = _gamma(rng, 1.0 / self.theta, n)
            e = rng.exponential(1.0, (n, d))
            return np.clip((1.0 + e / w[:, None]) ** (-1.0 / self.theta), EPS, 1.0 - EPS)
        if v == "gumbel":
            # Marshall–Olkin: S ~ Stable(1/θ), E_j ~ Exp(1), U_j = exp(−(E_j/S)^(1/θ))
            s = _stable(rng, 1.0 / self.theta, n)
            e = rng.exponential(1.0, (n, d))
            return np.clip(np.exp(-(e / s[:, None]) ** (1.0 / self.theta)), EPS, 1.0 - EPS)
```

Note `(n, d)`: one frailty per *row*, shared by every column. Drawing a fresh
$w$ per coordinate is the classic bug — it gives independent uniforms with the
right margins, so the KS test passes, and the joint CDF at $(0.3,0.6)$ is off by
several sigma. The demo's joint-CDF check is there precisely for that.

`_gamma` is Marsaglia–Tsang with the $a<1$ boost because `np.random` has no
gamma distribution; `_stable` is Kanter's form (14). Both are validated against
their moments in the demo's first run, not assumed.

### 9. The vine sampler is three h-inversions

```python
    def _sample_vine(self, n: int, rng) -> np.ndarray:
        """C-vine on root 1: u₁ = w₁, u₂ = h₁₂⁻¹(w₂|u₁), then u₃ through two h's."""
        (_, _, f12, p12), (_, _, f13, p13), (_, _, f23, p23) = self.pairs_
        w = rng.random((n, 3))
        u1 = w[:, 0]
        u2 = _h_inv(w[:, 1], u1, f12, p12)
        mid = _h_inv(_h_inv(w[:, 2], _h(u2, u1, f12, p12), f23, p23), u1, f13, p13)
        return np.clip(np.column_stack([u1, u2, mid]), EPS, 1.0 - EPS)
```

Read `mid` inside out: $h_{23}^{-1}$ takes the tree-2 conditional of $u_2$ as
its conditioning argument, then $h_{13}^{-1}$ maps back to the original scale
through $u_1$. The tree-2 `w` must be independent of tree 1, which is why it is a
fresh column of `rng.random` and not a function of the first two.

### 10. The CDF, and what is left to the user

```python
            if self.variant == "clayton":
                s = np.sum(U ** (-self.theta), axis=1) - self.d + 1.0
                return s ** (-1.0 / self.theta)
            if self.variant == "gumbel":
                a = -np.log(U)
                return np.exp(-np.sum(a ** self.theta, axis=1) ** (1.0 / self.theta))
            b = (math.exp(-self.theta) - 1.0) ** (self.d - 1)
            a = np.exp(-self.theta * U) - 1.0
            return -np.log1p(np.prod(a, axis=1) / b) / self.theta
```

Clayton and Gumbel are vectorised over $d$ with a `sum`, Frank with a `prod`;
all three are cheap and exact. The Gaussian branch is where the honesty lives:
for $d=2$ it calls `_bvn`, and for $d>2$ it raises, because the multivariate
normal CDF needs Genz's reduction and a half-page of code that would not teach
anything about copulas. The `t` and `vine` branches raise too. A returned number
would be indistinguishable from a correct one to every caller.

### 11. Fitting: tau first, then the profile

```python
        tri = self.tau_[np.triu_indices(self.d, 1)]
        tau = float(tri.mean())
        if self.variant in ("gaussian", "t"):
            self.rho = _param_from_tau(self.variant, tau)
            if self.variant == "t":
                self.df = self._fit_df(U)
```

With $d>2$ the model is exchangeable, so all pairwise taus are pooled into one
number — the estimate is one parameter and $d(d-1)/2$ observations of it. For
the t copula `rho` comes from tau first and `nu` is then profiled with `rho`
fixed:

```python
        grid = (1.0, 2.0, 3.0, 5.0, 6.0, 8.0, 12.0, 20.0, 40.0)
        step = max(1, U.shape[0] // 2000)
        return max(grid, key=lambda v: self._loglik_at(U[::step], v))
```

This is a deliberately weak estimator and the docstring says so. $\nu$ and
$\rho$ both control tail mass; pinning $\rho$ by tau and then choosing $\nu$ by
likelihood leaves the two partly unreconciled, and the profile is flat across a
factor of two in $\nu$. On 4000 draws with $\nu=6$ the grid picks 5 or 6 — good
enough to say "heavy-tailed", not good enough to report $\hat\nu$ to a decimal.

### 12. Tails, where each family's claim is exact or refused

```python
        if v == "t":
            r = float(np.asarray(self.rho).ravel()[0])
            lam = 2.0 * float(_t_cdf(np.array([-math.sqrt((self.df + 1.0) * (1.0 - r) / (1.0 + r))]),
                                     self.df + 1.0)[0])
            return {"lower": lam, "upper": lam}
        if v == "clayton":
            return {"lower": 2.0 ** (-1.0 / self.theta), "upper": 0.0}
        if v == "gumbel":
            return {"lower": 0.0, "upper": 2.0 - 2.0 ** (1.0 / self.theta)}
```

The t line is (5) with the survival function spelled as a CDF at a negative
argument — one `_t_cdf` call, no separate survival code to drift out of sync.
`vine` raises: its tail dependence is a property of the first tree's pair, but
no single number covers mixed families, and inventing one would be worse than
the error.

## Complexity

For $m$ evaluation points and $d$ dimensions:

| Operation | Time | Memory | Dominant term |
|:--|:--|:--|:--|
| `sample` Gaussian/t | $O(nd^2)$ | $O(nd)$ | Cholesky of $R$, once |
| `sample` Clayton/Gumbel | $O(nd)$ | $O(nd)$ | gamma/stable draws, $n$ of them |
| `sample` Frank | $O(n)$ | $O(n)$ | closed-form conditional, $d=2$ |
| `sample` vine | $O(nd)$ | $O(nd)$ | 2 h-inversions per row |
| `cdf` Gaussian | $O(mN)$ | $O(mN)$ | Drezner, $N = 200$ nodes per row |
| `cdf` Clayton/Gumbel/Frank | $O(md)$ | $O(m)$ | one pass over the cube |
| `logpdf` all | $O(md)$ | $O(md)$ | $\Phi^{-1}$ or $t^{-1}$ per entry |
| `logpdf` t | $O(I\,md)$ | $O(md)$ | $I \approx 10$ Newton steps |
| `fit` tau | $O(d^2 n\log n)$ | $O(n)$ | Fenwick tree per pair |
| `fit` t profile | $O(9\,nd)$ | $O(nd)$ | 9 grid points, ≤2000 rows |
| `tail`, `tau` | $O(d^2)$ | $O(d^2)$ | closed forms |

The Gaussian CDF is the only place where a quadrature appears in the hot path,
and it is why `cdf` on a grid is the slow part of the demo. Everything else is
linear in the data. The $n\log n$ in `fit` is the inversion count; at
$n = 20000$ it is 6 ms per pair, which is why the whole demo fits five families
in under two seconds.

## Failure modes

**1. Ties, and discrete margins.** Copulas are defined for continuous margins.
A column with repeated values gets arbitrary ranks, and $\hat\tau$ is biased
toward zero because the tie-breaking is noise. The diagnostic is a histogram of
`pseudo_observations`: if any value repeats, the ranks are not a permutation.
Fix by jittering, or by fitting a discrete copula — not by hoping.

**2. Reading the tail off a finite $u$.** The Gaussian copula has $\lambda = 0$,
yet at $u = 10^{-4}$ its conditional tail ratio is $4.9\times10^{-2}$, and at
$10^{-8}$ it is still $4\times10^{-3}$. If the risk number is "probability of a
joint 1-in-10000 event", a Gaussian copula gives a small positive number that
decays to zero only in a limit no dataset reaches. The diagnostic is to compute
the ratio at three decreasing $u$ and check whether it is stable (a true
$\lambda$) or shrinking (a fake one). The same test in reverse catches a t
copula whose $\nu$ is being read off too small a sample.

**3. Fitting $\nu$ and $\rho$ as if they were identified.** They are not: both
control how much mass sits in the corners, and fixing $\rho$ by $\tau$ before
profiling $\nu$ leaves the profile flat over a factor of two. The demo's fitted
$\hat\nu$ on data with $\nu=6$ is 5. Report "heavy-tailed, $\nu \lesssim 10$",
not a decimal. With $d > 2$ and exchangeable coordinates it is worse, since all
pairs share one $\nu$.

**4. The corner singularity in quadrature.** Clayton's density diverges at
$u_j \to 0$ and Gumbel's at $u_j \to 1$, so Gauss–Legendre converges
algebraically there, not spectrally. On a $200^2$ grid, $\int c = 1$ is good to
$4.6\times10^{-5}$ for the smooth families and only $1.3\times10^{-4}$ for
Clayton. Tighten the tolerance below that and the test fails on a correct
implementation.

**5. Vine depth.** The file implements $d=3$, one pair per tree, because a
general vine needs a tree-selection heuristic (which variable roots each tree)
and $d(d-1)/2$ pair fits. Scaling the code up without that heuristic produces
a vine whose later trees are fitted to h-transforms of already-noisy estimates,
and the compounding is not visible in any marginal or pairwise diagnostic. The
diagnostic that does work: hold out rows and compare the vine's log-density to
the Gaussian copula's on the same data. If the vine is not better out of sample,
the extra trees are fitting noise.

## Applications

- **Joint defaults and credit portfolios.** The Gaussian copula was the market
  standard for CDO tranche pricing, and the 2008 failure was a tail-dependence
  failure, not a calibration failure: the model fit the body of the dependence
  and asserted $\lambda = 0$. Current practice uses t or a vine with a
  Clayton/Gumbel first tree, and reports $\lambda$ separately.
- **Insurance and reinsurance.** Aggregate losses across lines (wind, flood,
  liability) are dependent in the tail. Clayton for the lower tail of claims
  frequency, Gumbel for co-occurring catastrophes, and a vine when there are
  more than two lines with different pairwise behaviour.
- **Hydrology.** Rainfall intensity and duration, or streamflow at neighbouring
  gauges, are strongly upper-tail dependent; Gumbel or a t with small $\nu$ is
  the usual choice, and the copula lets the marginal return-period models stay
  as they are.
- **Synthetic data with preserved margins.** Fit each column's marginal
  nonparametrically, fit a copula to the ranks, sample, and invert the margins.
  This is the cheapest way to generate data that has the right histograms *and*
  the right cross-correlation, which independent bootstrap destroys.
- **Dependence measures that are invariant.** $\tau$, tail dependence, and the
  copula density itself are unchanged by monotone transforms, so they are the
  right summary when the units are arbitrary (rankings, scores, indices).

Where it is a trap: using a Gaussian copula for a risk number that is defined
by a joint extreme. The model will fit, the marginals will look right, the
correlation will be estimated precisely, and the answer to "how often do these
two fail together" will be zero — because the family was chosen to make it zero.
The diagnostic is not a goodness-of-fit test on the body; it is (3) and it needs
data in the tail.

## References

- **Sklar (1959),** *Fonctions de répartition à n dimensions et leurs marges* —
  the theorem, in eight pages, with the uniqueness statement that makes the
  decomposition canonical.
- **Nelsen (2006),** *An Introduction to Copulas*, 2nd ed. — the reference for
  the axioms, the archimedean constructions, and the tail-dependence formulas.
- **Joe (2014),** *Dependence Modeling with Copulas* — the modern treatment;
  Chapter 4 is the vine construction used here.
- **Genest & Favre (2007),** *Everything you always wanted to know about copula
  modeling but were afraid to ask* — the rank-based estimation recipe
  ($\hat u = R/(n+1)$, tau inversion) that `fit` implements, with the
  finite-sample caveats.
- **Aas, Czado, Frigessi & Bakken (2009),** *Pair-copula constructions of
  multiple dependence* — the C-vine/D-vine sampling and estimation scheme.
- **Marshall & Olkin (1988),** *Families of multivariate distributions* — the
  frailty representations that make the Clayton and Gumbel samplers two lines.
- **Drezner (1978),** *Computation of the bivariate normal integral* — the
  quadrature in `_bvn`; **Genz (2004)** for the $d>2$ reduction that this file
  deliberately does not implement.
