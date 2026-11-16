# Interpretability

`interpretability.py` · Evaluation · Tier 1 · 9 variants

## Intuition

Two different questions hide behind "why did the model say that", and mixing them up is
the source of most bad explanations.

**Which features does the model rely on?** This is a property of the fitted function, and
it is answered by deleting or perturbing a feature and watching the loss. `permutation`
does exactly that. `pdp`, `ice` and `ale` answer a softer version — how does the
prediction move as one feature sweeps its range — which is the question a stakeholder
usually means by "what is the effect of age".

**Why *this* prediction?** This is a property of one row, and it needs an attribution:
a decomposition of the single number `f(x)` into per-feature contributions that add up.
`shap`, `kernel_shap`, `tree_shap`, `lime` and `integrated_gradients` all produce one, and
they differ only in what "contribution" is defined to mean.

Both families have a trap the others do not: the output is a chart, charts look
plausible, and nothing in the pipeline will fail when the chart is wrong. So every
variant here is verified against something exact in the demo:

| Variant | Verified against | Demo residual |
|:--|:--|:--|
| `permutation` | a pure-noise column must be worth nothing | `+0.005` vs max `7.32` |
| `shap` | efficiency: attributions sum to prediction − baseline | `4e-16` |
| `kernel_shap` | exact Shapley over the same coalitions | `4.4e-16` |
| `tree_shap` | brute-force enumeration of all `2^d` coalitions | `4.4e-16` |
| `pdp` | convexity of a known squared effect | `+0.057` per unit |
| `ice` | parallel curves for an additive model | `2.2e-15` |
| `ale` | centring, and an interaction PDP averages away | spans `3.33` vs `0.03` |
| `integrated_gradients` | exact for a linear model, completeness for a quadratic | `0.0` |
| `lime` | recovering a known linear coefficient vector | `3.2e-06` |

The demo's model is a 25-tree forest fitted to
`y = 2x₀ + x₁² − x₂ + 0.5x₀x₂ + noise`: one dominant linear feature, one convex feature,
one linear feature that also interacts, and one pure noise column.

## Variants

| Variant | Question | What a "contribution" is | Cost |
|:--|:--|:--|:--|
| `permutation` | global | loss increase when a column is shuffled | `d·R` predictions |
| `shap` | local | Shapley value over a background distribution | `2^d` coalition evaluations |
| `kernel_shap` | local | the same, by weighted least squares | `2^d` evaluations, `d²` solve |
| `tree_shap` | local | Shapley value under the tree's own conditional | `O(L·D²)` per tree |
| `pdp` | global | average prediction with one feature fixed | `G` prediction passes |
| `ice` | global | the same, one curve per row | `G` prediction passes |
| `ale` | global | average *change* within bins, accumulated | `2B` prediction passes |
| `integrated_gradients` | local | integral of the gradient along a path | `steps` gradient calls |
| `lime` | local | coefficients of a weighted local linear fit | `n_samples` predictions |

Three of the nine compute the same mathematical object. `shap`, `kernel_shap` and
`tree_shap` are the same Shapley value under three different computational strategies —
except that `tree_shap` uses a *different value function*, which is the subject of the
next section and the most important thing on this page.

`permutation` is not an attribution: it returns `d` numbers, one per feature, and they do
not sum to anything. It is the right tool for "which columns matter" and the wrong tool
for "why this row". The demo returns `[7.32, 1.16, 1.45, 0.005]` for the four features —
`x₀` dominates (coefficient `2.0` plus half the interaction), and `x₃` appears in no
term at all, so its importance is `0.005`, which is only sampling noise from re-predicting
the same rows.

`pdp` and `ice` differ only in whether the per-row curves are averaged away. `ice` is
strictly more informative and strictly harder to read, so the usual workflow is to plot
ICE and overlay the PDP. `ale` differs from both in that it computes *within-bin
differences* rather than *levels*, which is what makes it immune to the failure the demo
shows: for `y = x₀x₂` with `x₂ = 0.9x₀`, the PDP of `x₀` spans `0.03` because
`E[x₂] ≈ 0`, while the ALE spans `3.33` and is convex — the honest conditional effect.

`lime` is the only variant that fits a model. It is a local surrogate, not a
decomposition of `f`: its coefficients are approximately the gradient at `x`, and they
add up to nothing in particular. `integrated_gradients` also uses gradients but is exact
by construction for differentiable models, in the sense that its attributions satisfy
completeness.

## Derivation

### Shapley values

Let `N = {1, …, d}` be the features and `v(S)` the value of the model when only the
features in `S` are "known" — for a prediction task, the expected prediction given
that subset. The Shapley value is the unique attribution satisfying four axioms:

- **Efficiency** — `Σ_i φ_i = v(N) − v(∅)`: the parts add up to the whole.
- **Symmetry** — if `v(S ∪ {i}) = v(S ∪ {j})` for all `S` not containing `i, j`, then
  `φ_i = φ_j`.
- **Dummy** — if `v(S ∪ {i}) = v(S)` for all `S`, then `φ_i = 0`.
- **Additivity** — the attribution of `v + w` is the sum of the attributions.

and it is given by averaging each feature's marginal contribution over all orderings:

$$
\varphi_i(v) = \sum_{S \subseteq N \setminus \{i\}}
\frac{|S|!\,(d - |S| - 1)!}{d!}\,\bigl(v(S \cup \{i\}) - v(S)\bigr)
$$

Equivalently: pick a random permutation of the features, add them one at a time, and
record how much `v` moved when feature `i` arrived. The weight `|S|!(d−|S|−1)!/d!` is the
probability that the features before `i` are exactly `S`. Uniqueness of the axioms is
Shapley's 1953 theorem; the coefficient array is `shapley_coefficients(d)` in the file.

The axioms force the *formula* but not the *game*. `v` is where all disagreement lives,
and there is no canonical choice. Two conventions dominate:

**Interventional.** `v(S) = E_{X_bg}[f(x_S, X_{bg,\bar S})]` — break the dependence
between the features by substituting background values. This is what `shap` and
`kernel_shap` compute. It is well-defined for any model, it satisfies the axioms, and it
can evaluate `f` on rows that never occur in the data (a 30-year-old with a 40-year
tenure), which is why it is the default in the `shap` package.

**Path-dependent (conditional).** `v(S) = E[f(X) | X_S = x_S]` under the model's own
estimate of that conditional. For a tree, that estimate is exact and free: at a split
whose feature is not in `S`, average the two children weighted by their cover. This is
what `tree_shap` computes, and it is *not* the same number:

```text
exact Shapley, interventional background   [ 1.828, -0.350,  0.259, -0.002 ]  Σ = 1.734
tree_shap, path-dependent conditional       [ 1.675, -0.375,  0.281, -0.004 ]  Σ = 1.578
```

Both are internally consistent — each sums to its own `v(N) − v(∅)` — and both are
"SHAP". Reporting one and describing it as the other is the single most common error in
applied interpretability. The interventional version answers "what if this feature had
been different, holding the rest to the background"; the conditional version answers
"what if this feature had been different, given what the rest of this row tells us about
the correlations". With correlated features those are different questions.

### KernelSHAP

Evaluating `v` on all `2^d` coalitions is the bottleneck, and KernelSHAP's observation is
that the Shapley value is the solution of a *weighted least-squares* problem. Take the
regression

$$
\hat\varphi = \arg\min_{\varphi}\; \sum_{S \subseteq N}
\pi(S)\,\Bigl(v(S) - \sum_{i \in S}\varphi_i\Bigr)^2
\quad\text{subject to}\quad \sum_i \varphi_i = v(N) - v(\varnothing)
$$

with the SHAP kernel

$$
\pi(S) = \frac{d-1}{\binom{d}{|S|}\,|S|\,(d-|S|)}, \qquad 0 < |S| < d
$$

Lundberg & Lee's theorem is that this kernel is the unique weighting for which the
solution satisfies local accuracy, missingness and consistency — i.e. the solution *is*
the Shapley value. The intuition for the `1/(|S|(d−|S|)C(d,|S|))` shape: it is the
reciprocal of the number of orderings that produce a coalition of that size, so the
regression weights coalitions by how many permutations realise them, which is exactly
how the Shapley sum is organised.

The constraint matters numerically. Dropping it and hoping the kernel enforces efficiency
gives a badly conditioned matrix (the endpoints have weight `∞`); assigning them a huge
finite weight gives answers that are only accurate to that weight's reciprocal. The
implementation performs the elimination exactly:

```python
        # impose efficiency exactly by eliminating the last coordinate
        reduced = A[:, :d - 1] - A[:, d - 1:d]
        rhs = target - A[:, d - 1] * total
        root = np.sqrt(weight)
        coef, *_ = np.linalg.lstsq(reduced * root[:, None], rhs * root, rcond=None)
```

Substituting `φ_d = total − Σ_{i<d} φ_i` turns each row `Σ_i φ_i z_i` into
`Σ_{i<d} φ_i(z_i − z_d) + z_d·total`, which is a `(d−1)`-column regression with no
constraint. Efficiency then holds by construction rather than by tolerance, and the demo
matches the exact enumeration to `4.4e-16` over the `14` interior coalitions.

KernelSHAP is not asymptotically cheaper: it still needs all `2^d` values unless the
coalitions are *sampled*. Sampling is what makes it usable for `d = 1000`, and it is also
what makes the answer stochastic — the regression then has fewer rows than columns and
the solution depends on which coalitions were drawn.

### TreeSHAP

Trees make `v(S)` cheap to compute but the naive recursion is still exponential, because
a split whose feature is not in `S` must be averaged over both children. TreeSHAP's
contribution is to compute the whole attribution in `O(T L D²)` — linear in the number of
trees, leaves and depth.

The derivation that matches this file is a per-leaf decomposition. For a leaf `L`, let
`F(L)` be the set of distinct features on its root path and let

$$
Q_i(L) = \prod_{\text{positions of } L\text{ splitting on } i}
\frac{\text{cover}(\text{child on } L)}{\text{cover}(\text{parent})}
$$

so `Π_{i∈F} Q_i = cover(L)/cover(root)` telescopes. The path-dependent value function is

$$
v(S) = \sum_{L} v_L \cdot \mathbf{1}\bigl[S \cap D(L) = \varnothing\bigr]
\cdot \prod_{i \in F(L) \setminus S} Q_i(L)
$$

where `D(L)` is the set of features at which the leaf's path **diverges** from `x`'s
path. The indicator is the whole subtlety: if a feature in `S` is one where `L` went the
other way, then conditioning on that feature excludes `L` entirely. For a leaf on `x`'s
own path, `D(L) = ∅` and the indicator vanishes; for every other leaf it is what keeps
the sum honest.

Because the product factors over features, each leaf contributes independently to each
`φ_i`, and the only thing left is to count coalitions by size. Let `A = F∖D` be the
"agreeing" features. A coalition may put any subset of `A` into `S`, but none of `D`; the
`d − |F|` features not on the path are unconstrained. For a subset of size `k` from `A`,
the number of coalitions of total size `k + m` is `C(d−|F|, m)`, so the effective weight is

$$
W(k) = \sum_{m=0}^{d-|F|} \binom{d-|F|}{m}\, c(k+m),
\qquad c(j) = \frac{j!\,(d-j-1)!}{d!}
$$

and the sum over subsets of size `k` of `Π_{j∈A∖S} Q_j` is the coefficient of `t^k` in
`Π_{j∈A}(Q_j + t)`. That polynomial is built in `poly`, and the whole leaf collapses to
two dot products. Summing over all leaves gives the exact Shapley values for the tree;
summing over trees and dividing by the tree count gives the forest.

The demo verifies it the only way that matters: enumerate all `2^d` coalitions with
`_tree_value` (the definition, written naively and recursively), compute the Shapley sum
directly, and compare. The two agree to `4.4e-16` on every feature, and the efficiency
identity `Σφ = v(x's leaf) − mean` holds to `1e-10` at the forest level.

### Permutation importance

Shuffle column `j`, re-predict, and measure the loss increase:

$$
\mathrm{PI}_j = \mathbb{E}\bigl[\ell(y, f(X^{(j)}))\bigr] - \ell(y, f(X))
$$

with `X^{(j)}` the data with column `j` permuted. It is model-agnostic, needs only
predictions, and is unnormalised (units are loss units), which is why importance values
are usually compared as ratios rather than read absolutely.

The estimator is biased in two ways worth knowing. Permuting breaks the correlation
between `j` and the other features, so the model is evaluated on rows that never occur —
which *inflates* importance for correlated features (the model is being asked to predict
from a feature that no longer carries its usual companions). And the split is random, so
the estimate has the usual finite-sample noise, which is why the file averages
`n_repeats` shuffles per column.

### Partial dependence, ICE, ALE

For a feature `j` fixed at value `v`,

$$
\mathrm{PD}_j(v) = \mathbb{E}_{X_{-j}}\bigl[f(v, X_{-j})\bigr],
\qquad
\mathrm{ICE}_i(v) = f(v, x_{i,-j})
$$

ICE keeps the expectation undone, one curve per row; PDP is their average. Both are
estimated by substituting `v` into every row and averaging predictions — `G` prediction
passes for a grid of `G` values.

The failure mode is visible in the definition: `E_{X_{-j}}` is taken over the *marginal*
of the other features. If `x_j` is correlated with `x_{-j}`, the average includes
combinations that no real row has, and interactions get averaged into oblivion. The
demo's `y = x₀x₂` with `x₂ = 0.9x₀` is the extreme case: `PD₀(v) = v·E[x₂] ≈ 0` for all
`v`, so the PDP is flat and claims the feature does nothing.

ALE fixes this by never evaluating the model off the data manifold. Bin the feature into
`B` intervals, and within each bin compare the prediction at the upper edge with the
prediction at the lower edge, using only the rows in that bin:

$$
\Delta_k = \frac{1}{|I_k|}\sum_{i \in I_k}
\bigl[f(z_k^{+}, x_{i,-j}) - f(z_k^{-}, x_{i,-j})\bigr],
\qquad
\mathrm{ALE}(k) = \sum_{l \le k} \Delta_l - \text{centre}
$$

For `y = x₀x₂` the within-bin average of `x₂` is `0.9·E[x₀ | bin]`, so
`Δ_k = (z⁺−z⁻)·0.9·E[x₀|bin]` and the accumulation is convex — the demo gets a span of
`3.33` against the PDP's `0.03`, and a per-unit second difference that is positive.

The centring constant is the count-weighted mean of the bin midpoints. Without it the
curve depends on the arbitrary choice of first bin; with it, `Σ_b w_b · mid_b = 0` exactly,
which the demo checks at `2.9e-16`.

### Integrated gradients

For a differentiable `f`, pick a baseline `x'` (the "absence" of all features — zeros, or
the data mean) and integrate the gradient along the straight line:

$$
\mathrm{IG}_i(x) = (x_i - x'_i)\int_0^1
\frac{\partial f\bigl(x' + \alpha(x - x')\bigr)}{\partial x_i}\,d\alpha
$$

This is the unique attribution satisfying **completeness**,
`Σ_i IG_i = f(x) − f(x')`, which is just the fundamental theorem of calculus applied
component-wise. The demo verifies completeness for a quadratic at `0.0`, and verifies
exactness for a linear model: `IG_i = w_i(x_i − x'_i)` identically, because the gradient
is constant along the path.

The integral is approximated by a midpoint rule with `steps` points, which for a
piecewise-smooth `f` has error `O(1/steps²)`. Two limitations follow from the definition:
the gradient must exist (no trees, no ReLU kinks exactly on the path), and the answer
depends on `x'`. Completeness holds for *any* baseline, so it cannot tell you whether the
baseline was sensible — a fact that surprises people who expect an axiom to be a
guarantee.

Note what completeness does **not** imply: the attributions are not Shapley values, and
they can disagree sharply with SHAP for an interaction-heavy `f`. IG satisfies sensitivity
and implementation invariance (its own axioms, from Sundararajan et al.) and fails
dummy-ness in general — a feature that `f` ignores gets zero, but a feature that matters
only through an interaction gets a share that depends on the path.

### LIME

Fit a weighted linear model around `x`:

$$
\hat\beta = \arg\min_\beta \sum_{z \sim \mathcal{N}(x, \sigma)}
K(z, x)\,\bigl(f(z) - f(x) - \beta^\top(z - x)\bigr)^2 + \lambda\|\beta\|^2
$$

with `K` a distance kernel. The coefficients `β` are the explanation: "locally, moving
feature `i` by one unit moves the prediction by `β_i`".

For a linear `f`, every weight `K` gives `β = w` exactly, which is the demo's
`3.2e-06` (the residual is the ridge term). For a nonlinear `f`, `β` approximates the
gradient at `x`, and the kernel width is a genuine hyperparameter: too narrow and the
surrogate sees only noise, too wide and it approximates the global model instead of the
local one. The demo's kernel is `exp(−‖u‖²/(2·0.75²·d))` on standardised offsets.

LIME is *not* Shapley. If the kernel happened to be the SHAP kernel and the surrogate were
fitted on coalitions rather than Gaussian perturbations, the coefficients would converge
to Shapley values; as implemented they do not, and the demo's correlation with `tree_shap`
on the forest is `0.849` rather than `1.0`. That is the honest relationship between the
two methods: same direction, different numbers, different guarantees.

## Code walkthrough

The coalition value is one line of substitution, and it is the only place the background
distribution enters:

```python
        Z = self._background().copy()
        Z[:, mask] = x[mask]
        return float(np.mean(_predict(model, Z)))
```

`Z` is a copy of the background rows, and the boolean `mask` selects which columns are
taken from the row being explained. This is the *interventional* value: the substituted
features come from `x`, the rest from the background, and nothing models the correlation
between them. `_predict` accepts either a model with `.predict` or a bare callable, which
is what lets the demo explain an analytic function and a forest with the same code.

The exact Shapley sum is the definition, transcribed:

```python
        for i in range(d):
            bit = 1 << i
            for mask in range(1 << d):
                if mask & bit:
                    continue
                k = bin(mask).count("1")
                phi[i] += c[k] * (value[mask | bit] - value[mask])
```

`mask | bit` is `S ∪ {i}` and `mask` is `S`; the loop skips masks that already contain
`i`, so each marginal `v(S ∪ i) − v(S)` is counted once with the weight `c[|S|]`. Note
that `value` is precomputed for all `2^d` masks before the loop — the same array is reused
for every feature, which is what makes the naive implementation `O(2^d · d)` rather than
`O(2^d · d²)`.

KernelSHAP's kernel and its exact constraint handling:

```python
        k = A.sum(axis=1)
        weight = (d - 1) / (np.array([math.comb(d, int(ki)) for ki in k])
                            * k * (d - k))
        # impose efficiency exactly by eliminating the last coordinate
        reduced = A[:, :d - 1] - A[:, d - 1:d]
        rhs = target - A[:, d - 1] * total
```

`A` is the `(2^d−2) × d` matrix of coalition indicators, `k` is each coalition's size, and
`weight` is `π(S)`. `np.sqrt(weight)` multiplies both sides so that `lstsq` — which
minimises an unweighted residual — solves the weighted problem; this is the standard trick
for turning weights into a rescaling, and it avoids `diag(π)^{1/2}` as an explicit matrix.
The (d−1)-column reduction is the efficiency constraint, substituted in rather than
appended, so the answer is exact instead of depending on a penalty large enough.

TreeSHAP's per-leaf polynomial:

```python
    def poly(features: list[int], q: dict[int, float]) -> np.ndarray:
        """Coefficients of prod_{i in features} (Q_i + t), degree len(features)."""
        out = np.ones(1)
        for i in features:
            step = np.zeros(out.size + 1)
            step[:out.size] = out * q[i]        # feature i is not in S
            step[1:] += out                     # feature i is in S
            out = step
        return out
```

Multiplying by `(Q_i + t)` encodes exactly the two cases: feature `i` is not in `S`, which
contributes the factor `Q_i` and leaves the degree alone; or it is in `S`, which
contributes `1` and increments the degree. After processing every feature of the path,
coefficient `k` is the sum over `k`-subsets of the product of the *excluded* features'
`Q`s — which is precisely the quantity the Shapley sum needs. Duplicated features are
handled for free because `Q_i` is the product of that feature's per-position fractions.

The size marginalisation is the other half of the derivation:

```python
            size = np.array([
                sum(math.comb(n_off, m) * c[k + m]
                    for m in range(n_off + 1) if k + m <= n_features - 1)
                for k in range(len(inside) + 1)
            ])
```

`n_off = d − |F|` counts the features that never appear on this leaf's path — they can be
in or out of `S` without changing anything, so each coalition size `k` is multiplied by
`C(n_off, m)`. The guard `k + m <= d − 1` exists because a marginal is only defined up to
coalitions of size `d − 1`; the full coalition is the efficiency endpoint, not a term.

The leaf accumulation distinguishes the two signs:

```python
            for i in distinct:
                if i in diverge:
                    # forcing i into S kills this leaf entirely
                    total = float(np.dot(poly(inside, q), size))
                    phi[i] -= value * forced * total
                else:
                    others = [f for f in inside if f != i]
                    total = float(np.dot(poly(others, q), size[:len(others) + 1]))
                    phi[i] += value * forced * (1.0 - q[i]) * total
```

For a feature where the leaf *agrees* with `x`, the marginal is
`(1 − Q_i) · Π_{others}`: the leaf's value changes when `i` is added to `S` exactly by the
fraction of mass that `i` was diverting. For a feature where the leaf *diverges*, the
leaf's weight is `−Π` with no `(1−Q)` factor, because adding `i` to `S` removes the leaf
entirely rather than rescaling it. Both cases multiply `forced = Π_{j∈D} Q_j`, the
factors of the divergence features, which stay in the product no matter what `S` is.

ALE's bin differences, then the centring:

```python
        for k in range(edges.size - 1):
            rows = idx == k
            if not rows.any():
                continue
            Z = X[rows]
            lo = Z.copy()
            hi = Z.copy()
            lo[:, feature] = edges[k]
            hi[:, feature] = edges[k + 1]
            delta[k] = float(np.mean(_predict(model, hi) - _predict(model, lo)))
```

`Z = X[rows]` restricts to the rows in the bin, so both `lo` and `hi` keep each row's own
values of the other features — that is what "conditional on being in this bin" means, and
it is the difference from PDP. Subtracting before averaging is also deliberate: averaging
the two predictions and then subtracting would give the same number in exact arithmetic,
but the difference-of-means form cancels the row's baseline and keeps the floating-point
error proportional to the effect instead of to the prediction.

```python
        curve = np.concatenate([[0.0], np.cumsum(delta)])
        counts = np.bincount(idx, minlength=edges.size - 1).astype(float)
        centre = float(np.sum(counts / counts.sum()
                              * (curve[:-1] + curve[1:]) / 2.0))
```

`np.cumsum` accumulates the local effects into a curve anchored at zero at the lowest
edge, and `counts` weights the centring by how many rows each bin actually contains — the
bins are quantiles, so unweighted centring would be wrong whenever the edges are uneven.
The curve is defined at the *edges*, so the bin midpoint is `(curve[k] + curve[k+1])/2`.

Integrated gradients is the midpoint rule and one multiplication:

```python
        alphas = (np.arange(self.steps) + 0.5) / self.steps   # midpoint rule
        grads = np.array([grad_fn(base + a * delta) for a in alphas])
        return delta * grads.mean(axis=0)
```

`+ 0.5` shifts the nodes to interval centres, which is second-order accurate for a smooth
integrand; the endpoints would be first-order and would also evaluate the gradient exactly
at the baseline, where it is often degenerate. `delta` multiplies *after* the average,
because `(x_i − x'_i)` does not depend on `α` — the same factor can be pulled out of the
integral, and doing so avoids dividing by zero when `x = x'`.

LIME is a weighted ridge solve:

```python
        u = offset / scale
        weight = np.exp(-0.5 * np.sum(u * u, axis=1) / (0.75 ** 2 * x.size))
        A = offset * weight[:, None]
        gram = A.T @ offset
        ridge = self.l2 * float(np.trace(gram)) / x.size + EPS
        coef = np.linalg.solve(gram + ridge * np.eye(x.size), A.T @ target)
```

Distances are standardised by each feature's standard deviation, otherwise a feature
measured in thousands would always be "far" and would always get zero weight. Folding
`weight[:, None]` into `A` turns the weighted normal equations `AᵀWA` into an unweighted
product, so only one matrix is formed. The ridge term is scaled by `trace(AᵀWA)/d` rather
than used raw, which makes `l2` mean "shrink by this fraction" and keeps the fit
independent of `n_samples` and of the units.

## Complexity

| Variant | Cost | Memory |
|:--|:--|:--|
| `permutation` | `d·R` prediction passes over `n` rows | `O(n)` |
| `shap` | `2^d` coalition evaluations, each `O(B + n_bg)` | `O(2^d + n_bg·d)` |
| `kernel_shap` | `2^d` evaluations + one `(2^d × d)` lstsq | `O(2^d·d)` |
| `tree_shap` | `O(T·L·D²)`; `D ≤ d` | `O(L·D)` per tree |
| `pdp` | `G` prediction passes | `O(n·G)` for ICE, `O(G)` for PDP |
| `ice` | `G` prediction passes | `O(n·G)` |
| `ale` | `2B` prediction passes on subsets | `O(n)` |
| `integrated_gradients` | `steps` gradient evaluations | `O(steps·d)` |
| `lime` | `n_samples` predictions + `O(d³)` solve | `O(n_samples·d)` |

The `2^d` in the two exact Shapley variants is the whole story: at `d = 4` it is 16
coalitions and the demo is instant, at `d = 20` it is a million and at `d = 100` it is not
a number. KernelSHAP's practical form samples coalitions instead, which trades exactness
for tractability and gives an answer whose error you must estimate. TreeSHAP has no such
wall — its `L·D²` is why it is the default for gradient-boosted models — which is exactly
why the two implementations produce different numbers in the demo: not because one is
wrong, but because `tree_shap` is using a different value function.

Memory is the other asymmetry: `shap` and `kernel_shap` materialise a value for every
coalition, while `tree_shap` walks paths and returns `d` floats.

## Failure modes

1. **Correlated features, interventional attributions.** Shapley splits credit among
   correlated features by averaging over coalitions that substitute background values.
   With `x₁ ≈ x₂`, the pair gets roughly half the credit each no matter which one the
   model actually uses. This is not a bug in SHAP — it is what the axioms say — but it
   does mean "feature importance" and "causal effect" are different objects.
2. **Confusing path-dependent and interventional SHAP.** The demo's two attributions
   differ by `0.15` on the dominant feature and by `0.02` on the third, from the same
   forest and the same row. Pick one, name it, and never mix libraries in one report.
3. **PDP under dependence.** The demo's PDP of `x₀` spans `0.03` for a feature that is
   fully responsible for a `3.33`-wide ALE. Averaging over the marginal puts half the mass
   on rows that do not exist. If features are correlated, plot ALE, or plot ICE and look
   at the spread rather than the mean.
4. **PDP hides interactions.** A flat PDP is consistent with "no effect" and with
   "positive for half the rows, negative for the other half". Always check the ICE spread
   before concluding a feature is irrelevant — the demo's `x₂` curves have a slope spread
   of `85` around a mean that looks modest.
5. **Permutation importance on correlated features.** Shuffling `x₁` destroys its
   relationship with `x₂`, so the model is scored on impossible rows and `x₁` looks more
   important than it is. Group correlated features, or use a conditional permutation
   scheme.
6. **Attributions that do not sum.** `lime` coefficients, permutation importances and
   PDP spans are not decompositions, and adding them up is meaningless. Only `shap`,
   `kernel_shap`, `tree_shap` and `integrated_gradients` come with a stated efficiency
   property; if you need "these parts explain the prediction", check that property on one
   row before trusting the rest.
7. **Baseline games in integrated gradients.** Completeness holds for every baseline, so
   a convenient choice (zeros) silently defines "absence" as an all-zero input, which may
   be far outside the data. The attributions move when the baseline moves. Report it.
8. **LIME instability.** Perturbation sampling is random, so two runs give two different
   explanations; with `n_samples` too small and a kernel width too narrow, the surrogate
   fits noise. Increase the sample count, and average over seeds before believing a
   coefficient.
9. **Explaining a bad model.** All nine variants faithfully explain whatever function
   they are handed, including one that has learned a shortcut. An explanation is evidence
   about the model, not about the world, and a plausible waterfall plot of a leaky model
   is the most convincing wrong artifact in machine learning.
10. **Attributions as causal claims.** Shapley values distribute credit for a *prediction*
    under a stated value function. They say nothing about what happens if you intervene
    on a feature. That requires a causal graph, and no amount of averaging over coalitions
    substitutes for one.

## Applications

- **Debugging a surprising prediction.** A single row explained by `tree_shap` often
  reveals a leaked feature, a unit error or a bad default faster than any aggregate metric.
- **Model auditing and fairness review.** Per-group attributions show whether a decision
  leans on a protected attribute, or on a proxy that correlates with one.
- **Feature engineering triage.** Permutation importance ranks columns by how much the
  model actually relies on them, which is the cheapest way to find dead features.
- **Regulatory and clinical explanation.** Credit decisions and risk scores are legally
  required to give reasons; Shapley attributions are the most defensible default because
  the axioms are stated and the efficiency property is checkable.
- **Sanity-checking a pipeline.** If `pdp` shows a non-monotone effect for a feature you
  know is monotone, the bug is in the data pipeline, not the plot.
- **Scientific sensitivity analysis.** ALE and PDP are standard tools for reporting how a
  fitted response surface depends on each input, especially when interactions matter.
- **Distillation and rule extraction.** Attributions on a large model guide which features
  a small surrogate should keep, and where a local linear fit is good enough.

## References

- Shapley, *A Value for n-Person Games*, 1953, Contributions to the Theory of Games II —
  the axioms and the formula.
- Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions*, 2017, NeurIPS —
  SHAP, the SHAP kernel, and KernelSHAP's weighted-regression formulation.
- Lundberg, Erion & Lee, *Consistent Individualized Feature Attribution for Tree
  Ensembles*, 2019 — TreeSHAP, the path-dependent value function, and the `O(TLD²)`
  recursion.
- Štrumbelj & Kononenko, *Explaining Prediction Models and Individual Predictions with
  Feature Contributions*, 2014, Knowledge and Information Systems — the sampling
  estimator that KernelSHAP generalises.
- Ribeiro, Singh & Guestrin, *"Why Should I Trust You?": Explaining the Predictions of Any
  Classifier*, 2016, KDD — LIME and the local surrogate framing.
- Sundararajan, Taly & Yan, *Axiomatic Attribution for Deep Networks*, 2017, ICML —
  integrated gradients, completeness and sensitivity.
- Friedman, *Greedy Function Approximation: A Gradient Boosting Machine*, 2001, Annals of
  Statistics — partial dependence plots.
- Goldstein, Kapelner, Bleich & Pitkin, *Peeking Inside the Black Box: Visualizing
  Statistical Learning with Plots of Individual Conditional Expectation*, 2015, JCGS —
  ICE curves and their advantage over PDP.
- Apley & Zhu, *Visualizing the Effects of Predictor Variables in Black Box Supervised
  Learning Models*, 2020, JRSS-B — ALE, the within-bin differences, and the argument that
  PDP is biased under dependence.
- Hooker, Mentch & Zhou, *Unrestricted Permutation Forces Extrapolation: Variable
  Importance for Correlated Predictors*, 2021, AISTATS — why drop-column importance is
  biased when features are correlated.
