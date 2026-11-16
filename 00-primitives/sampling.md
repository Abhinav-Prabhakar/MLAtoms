# Sampling

`sampling.py` · Primitives · Tier 1 · 10 variants

## Intuition

Sampling is how a distribution becomes data. Four different jobs get called
"sampling", and mixing them up is the usual source of subtle bugs:

1. **Draw from a known distribution** — inverse CDF, alias, rejection.
2. **Draw from a stream you cannot store** — reservoir.
3. **Draw *without replacement* or from a truncated distribution** — Gumbel
   top-k, nucleus, negative sampling.
4. **Estimate an expectation with fewer draws** — stratified, systematic,
   importance sampling.

The first three must be exact; the fourth is allowed to be biased or
higher-variance as long as the estimator is right. Keeping that distinction
explicit is what separates a sampler from a heuristic.

## Variants

| Variant | Draws from | Cost per draw | The property that earns it a slot |
|:--|:--|:--|:--|
| `uniform` | `U(0,1)` | `O(1)` | Inverse-CDF template for every continuous law. |
| `weighted_alias` | categorical | `O(1)` after `O(k)` build | Constant time regardless of vocabulary size. |
| `reservoir` | uniform subset of a stream | `O(1)` amortised, `O(k)` memory | Unknown length, one pass. |
| `stratified` | one per stratum | `O(n)` | Kills the between-stratum component of variance. |
| `systematic` | one per stratum, shared offset | `O(n)` | Stratification with a single random number. |
| `gumbel_top_k` | categorical, without replacement | `O(k log k)` | Exact Plackett-Luce via a sort. |
| `top_p` | truncated categorical | `O(k log k)` | Truncation adapts to model confidence. |
| `negative_sampling` | categorical, excluding positives | `O(log k)` (alias) | Cheap, unbiased-ish negatives at scale. |
| `importance_sampling` | expectation under `p`, samples from `q` | `O(n)` | Lets you sample from the wrong distribution. |
| `rejection` | arbitrary density | `O(M/area)` proposals | Works when no inverse CDF exists. |

## Derivation

**Alias method.** Split the unit interval into `k` columns of width `1/k`, one per
item, and give item `i` a mass `k·w_i` (so the total is `k`). Columns heavier than
1 donate their excess to columns lighter than 1; each light column is then
"filled" by exactly one donor. A draw picks a column uniformly and accepts it with
probability `prob[i]`, otherwise takes `alias[i]`. Both the acceptance and the
fallback are `O(1)`, and the table is built in `O(k)` by the two-list sweep in
`alias_table`.

The invariant worth testing is *not* `Σ prob[i] = k` (it is not, in general).
What holds is that the implied distribution equals the weights:

$$
\hat w_i = \frac{1}{k}\left(\text{prob}[i] + \sum_{s:\,\text{alias}[s]=i}\bigl(1-\text{prob}[s]\bigr)\right) = w_i
$$

which is exactly what the demo checks.

**Reservoir (Algorithm R).** Claim: after processing `i` items, each is in the
reservoir with probability `k/i`. Base case `i = k` is clear. Inductive step:
item `i+1` enters with probability `k/(i+1)`; for an item already present, it
survives with probability

$$
\Bigl(1-\tfrac{k}{i+1}\Bigr) + \tfrac{k}{i+1}\cdot\tfrac{k-1}{k}
= \frac{i}{i+1}
$$

so its inclusion probability is `(k/i)·(i/(i+1)) = k/(i+1)`. No length is needed
anywhere — that is the whole point.

**Gumbel-max.** For logits `z`, let `G_i ~ Gumbel(0,1)` i.i.d. Then

$$
\Pr\!\left[\arg\max_i (z_i + G_i) = j\right] = \frac{e^{z_j}}{\sum_i e^{z_i}}
$$

Proof sketch: `argmax = j` iff `z_j + G_j > max_{i≠j}(z_i + G_i)`, and the max of
independent Gumbels is Gumbel with the log-sum-exp of the offsets. Taking the
top-`k` instead of the top-1 yields the **Plackett-Luce** distribution: sample the
first item proportionally to `w`, remove it, repeat. That identity is why
top-k/top-p decoding is a sort, not a sequence of categorical draws, and why the
same code is differentiable-friendly under the Gumbel-softmax relaxation.

**Nucleus (top-p).** Sort probabilities descending, keep the shortest prefix with
cumulative mass `≥ p`, renormalise, draw once. The truncation is a function of the
distribution's own shape: a peaked model keeps one token, a flat one keeps most of
the vocabulary. Top-k cannot do that with a fixed `k`.

**Rejection.** Propose `x ~ g`, accept with probability `f(x)/(M g(x))` for
`f(x) ≤ M g(x)`. The accepted points are distributed as `f`, and the acceptance
rate is `area(f)/(M·area(g))`. For the demo's `f(x)=2x`, `g=U(0,1)`, `M=2`, the
rate is exactly `1/2`, so half the proposals are wasted — the price of not having
an inverse CDF.

**Importance sampling.** For any `q` with `q(x) > 0` wherever `p(x)f(x) ≠ 0`:

$$
\mathbb{E}_p[f] = \int f(x)\frac{p(x)}{q(x)}q(x)\,dx
= \mathbb{E}_q\!\left[f(x)\frac{p(x)}{q(x)}\right]
$$

The estimator is unbiased but its variance is `E_q[(f·p/q)²] − E_p[f]²`, which
explodes when `q` is much narrower than `p` (the weights grow without bound). The
diagnostic is effective sample size, `ESS = (Σw)²/Σw²`: below ~10% of `n`, the
estimate is noise.

**Stratified vs systematic.** Partition the population into `n` strata and draw
`n_i` from each. The variance decomposition

$$
\operatorname{Var}(\bar X) = \underbrace{\sum_i \frac{\sigma_i^2}{n_i}}_{\text{within}} + \underbrace{\frac{1}{n^2}\sum_i (\mu_i - \mu)^2}_{\text{between}}
$$

shows the gain: stratification removes the between-stratum term entirely. With
`n_i = 1` in every stratum (the demo's case) the estimate is unbiased with variance
`O(1/n²)` for a smooth population, versus `O(1/n)` for i.i.d. draws — a factor of
`n` fewer samples for the same error. Systematic sampling is stratified with
`n_i = 1` and *one* shared `u`, which is `n−1` fewer random numbers.

## Code walkthrough

The file is split into "exact draws" and "designs and estimators", and the split is
functional: everything above `# Designs` returns samples from a specified law,
everything below returns indices or estimates whose *bias* is a design choice.

`alias_table` is the only non-trivial build, and it is a two-list sweep:

```python
    k = w.size
    p = w / w.sum() * k
    prob = np.zeros(k)
    alias = np.arange(k)
    small = [i for i in range(k) if p[i] < 1.0]
    large = [i for i in range(k) if p[i] >= 1.0]
    while small and large:
        s = small.pop()
        ell = large.pop()
        prob[s] = p[s]
        alias[s] = ell
        p[ell] -= 1.0 - p[s]
        (small if p[ell] < 1.0 else large).append(ell)
```

`p` is the mass in units of `1/k`, so the *total* is `k`, not 1. A column with
`p[i] < 1` keeps `p[i]` and is *filled* by one donor `ell`, which gives up
`1 − p[s]` of its mass; the donor's remaining mass shrinks and it re-enters whichever list it now belongs to. The line
`prob[s] = p[s]` records the *keep* probability, and `alias[s] = ell` records where
the remainder goes. Each pair removes one element from `small`, so the loop runs at
most `k` times.

The trailing line is easy to dismiss:

```python
    for i in large + small:  # floating-point leftovers
        prob[i] = 1.0
```

After the loop, any remaining entry is within `1e-16` of 1; setting `prob = 1` makes
it "always keep". Dropping the line leaves `prob[i] = 0` for those indices, so they
are *never* kept and always route to `alias[i]` — a bias of order `1e-16` per item
that is invisible in a histogram and reproducible in a seed. The demo checks the
implied distribution, not `Σprob`, because `Σprob = k` is false in general:

```python
    implied = prob / w.size
    np.add.at(implied, alias, (1.0 - prob) / w.size)
    assert np.allclose(implied, w / w.sum()), (implied, w / w.sum())
```

`np.add.at` again, for the same reason as in `metrics.py`: several columns can alias
to the same donor, and buffered `+=` would count only the last one. This three-line
reconstruction is the whole correctness proof in code form.

`weighted_alias` is two vectorised draws and a select:

```python
    i = rng.integers(0, k, size=n)
    take_alias = rng.random(n) >= prob[i]
    return np.where(take_alias, alias[i], i)
```

Column `i` is uniform; with probability `prob[i]` keep `i`, otherwise take
`alias[i]`. `>=` rather than `>` matches "accept with probability `prob`" only up to
a measure-zero event, but it matters for reproducibility across RNGs that can return
exactly 0.

`reservoir` is five lines and the off-by-one is the entire algorithm:

```python
    for i, x in enumerate(stream):
        if i < k:
            res.append(x)
        else:
            j = int(rng.integers(0, i + 1))
            if j < k:
                res[j] = x
```

At item index `i` (0-based), there are `i+1` items seen, and the new one must enter
with probability `k/(i+1)`, which requires `j` uniform on `0..i` inclusive —
`rng.integers(0, i+1)`. Writing `rng.integers(0, i)` gives `i` outcomes instead of
`i+1`, so later items are *over*-represented relative to early ones; the sample is
still "random-looking" and only an inclusion-frequency test (the demo's 20k trials)
reveals it.

`gumbel_top_k` turns a categorical draw into an argsort:

```python
    u = np.clip(rng.random(z.shape), 1e-300, 1.0 - 1e-16)
    g = -np.log(-np.log(u))
    return np.argsort(-(z + g), kind="stable")[:k]
```

`u` must be clipped at both ends: `u = 0` gives `log(0) = −inf` and then
`−log(inf) = −inf`; `u = 1` gives `−log(0) = +inf`. Either produces an infinite
perturbation and the corresponding item is always first (or produces `nan` via
`inf − inf` when two collide). The bounds are not symmetric because the float
nearest to 1 is `1 − 2⁻⁵³`, not `1 − 1e-16`; the clip is a safety net, not a
precision guarantee. `kind="stable"` matters when two perturbed scores tie exactly:
a non-stable sort can reorder equal scores and break Plackett-Luce's exchangeability.
The top-`k` of one Gumbel draw is *not* `k` independent draws — it is exactly the
without-replacement distribution, which is why the demo checks the pair probabilities
and not just the marginals.

`rejection` batches proposals because Python-level loops are the bottleneck:

```python
    while filled < n:
        batch = max(64, 2 * (n - filled))
        x = rng.uniform(lo, hi, size=batch)
        u = rng.uniform(0.0, envelope, size=batch)
        acc = x[u < pdf(x)]
```

`u ~ U(0, envelope)` and accept when `u < pdf(x)`: the accepted `x` has density
`pdf/envelope`, so after normalising over `[lo,hi]` it is `pdf/area`. If `envelope` is
not a true upper bound the accepted sample is *not* from `pdf` — it is from
`min(pdf, envelope)`, silently truncated. The batch factor `2×` is a guess at the
acceptance rate; if the rate is far below 1/2 the loop runs many times, which is
correct but slow.

Stratified and systematic are the same idea with one random number versus `n`:

```python
    bounds = np.linspace(0, population_size, n + 1)
    u = rng.random(n)
    return np.floor(bounds[:-1] + u * (bounds[1:] - bounds[:-1])).astype(int)
```

```python
    step = population_size / n
    start = rng.random() * step
    return np.floor(start + step * np.arange(n)).astype(int)
```

Both return *indices* into a population of size `population_size`, so the last
stratum is `[bounds[-2], bounds[-1])` and the `floor` is safe. `systematic` differs
by using a single `start` for every stratum; the spacing is exactly `step`, so
`np.unique(s // step)` has `n` elements by construction — the demo asserts this
rather than checking a mean. When `population_size` is not divisible by `n`, the
last index can be `population_size` (out of range) if `start` is close to `step`;
the demo uses a divisible case, and production code must clamp.

`top_p` is three lines of sorting and the `+1` is the off-by-one:

```python
    order = np.argsort(-q)
    cum = np.cumsum(q[order])
    keep = int(np.searchsorted(cum, p) + 1)
```

`searchsorted(cum, p)` with the default `side="left"` returns the first index where
`cum[i] >= p`, which is the *zero-based* count minus one, so `+1` converts it to a
count. If `p` is larger than the total mass (impossible after normalisation) it
returns `q.size`, and `+1` would overflow — hence the `min(keep, q.size)`. The draw
is over the *renormalised* support:

```python
    w = q[support]
    return int(support[rng.choice(keep, p=w / w.sum())])
```

Forgetting `w / w.sum()` and passing `w` directly is the failure mode the doc names:
`np.random.choice` requires probabilities that sum to 1 and raises, so that one is
loud; the quiet version is sampling from `w` after truncation *without* renormalising
by hand and reusing the original `p`, which biases toward the head.

`negative_sampling` is one exponentiation and a zeroing:

```python
    w = np.asarray(counts, dtype=float) ** exponent
    w[np.asarray(positives, dtype=int)] = 0.0
    w = w / w.sum()
```

`counts**0.75` flattens Zipf; `exponent=1` is exact unigram and over-samples the head.
Zeroing *after* the exponent is irrelevant numerically but keeps the normaliser
consistent. If every item is positive, `w.sum() == 0` and the division yields `nan`,
which `rng.choice` rejects — the demo always leaves at least one negative, and real
code should assert that.

`importance_sampling` is deliberately plain:

```python
    weights = np.asarray(p(x), dtype=float) / np.asarray(q(x), dtype=float)
    fx = x if f is None else f(x)
    return float(np.mean(fx * weights))
```

No self-normalisation, no clipping. The estimator is unbiased by construction, and
the failure is variance, not bias: when `q` is narrow, a single weight can dominate
the mean. The demo's `p(t)=2t, q=U(0,1)` has bounded weights `2t ≤ 2`, so 200k draws
land within `0.005`; a Gaussian `q` centred away from the mass of `p` would need an
ESS report to look trustworthy.

## Complexity

| Routine | Time | Space |
|:--|:--|:--|
| `uniform` | `O(n)` | `O(n)` |
| `alias_table` | `O(k)` build | `O(k)` |
| `weighted_alias` | `O(k + n)` | `O(k)` |
| `reservoir` | `O(N)` | `O(k)` |
| `stratified` / `systematic` | `O(n)` | `O(n)` |
| `gumbel_top_k` | `O(k log k)` | `O(k)` |
| `top_p` | `O(k log k)` | `O(k)` |
| `negative_sampling` | `O(log k)` per draw with an alias table | `O(k)` |
| `importance_sampling` | `O(n)` | `O(n)` |
| `rejection` | `O(n·M/area)` expected | `O(n)` |

The headline: alias turns categorical sampling from `O(k)` into `O(1)`, and that
constant is why it is the default in word2vec and in large-scale recommenders.

## Failure modes

1. **Alias with floating-point leftovers.** The two-list sweep can leave a column
   with `p ≈ 1 ± 1e-16` in both lists; the trailing `for i in large + small`
   matters. Drop it and the last item is under-sampled by a tiny, untraceable
   amount.
2. **Reservoir off-by-one.** `j = randint(0, i+1)` uses the *count including the
   current item*; `randint(0, i)` makes later items too likely. The symptom is a
   bias toward the head of the stream — invisible unless you test inclusion
   frequencies.
3. **Top-p without renormalising.** Sampling from `q_i` over the truncated support
   without dividing by the kept mass turns a decoding heuristic into a sampling
   bug; the model emits `<eos>` less often than intended.
4. **Negative sampling with `exponent=1`.** Exact unigram sampling spends almost
   all draws on the few most frequent items, so rare items never become negatives.
   The `3/4` exponent is a deliberate, well-tested bias.
5. **Importance sampling with a mismatched `q`.** Unbiased in theory, useless in
   practice when the weights are dominated by a handful of samples. Always report
   ESS, and prefer a defensive `q` with heavier tails than `p`.
6. **Rejection without a valid envelope.** If `M` is not a true upper bound, the
   accepted distribution is silently wrong (not just slow). Check `max(f/g)` on a
   dense grid before trusting the sampler.

## Applications

- **Decoding:** `top_p` (nucleus) and `gumbel_top_k` are the two standard LLM
  samplers; the Gumbel view is what makes them a single sort on device.
- **Embeddings:** `negative_sampling` with the 3/4 exponent, plus the alias table
  for `O(1)` draws over a 10⁵–10⁶ vocabulary.
- **Streaming:** reservoir sampling for unbiased evaluation sets, log sampling,
  and "keep a uniform sample of everything I have seen" counters.
- **Monte Carlo:** stratified/systematic for quasi-Monte-Carlo integrals and
  variance reduction; importance sampling for rare-event estimation and for
  off-policy evaluation.
- **Distributed training:** systematic sampling to shard without overlap — one
  offset per epoch, exactly one draw per shard.

## References

- Vose, *A linear algorithm for generating random numbers with a given distribution*, 1991. IEEE TSE.
- Vitter, *Random sampling with a reservoir*, 1985. ACM TOMS.
- Gumbel, *Statistical Theory of Extreme Values*, 1958 (the max-stability that gives the trick its name).
- Holtzman et al., *The Curious Case of Neural Text Degeneration* (nucleus sampling), 2019. arXiv:1904.09751
- Mikolov et al., *Distributed Representations of Words and Phrases*, 2013. arXiv:1310.4546 (negative sampling).
- Owen, *Monte Carlo theory, methods and examples*, 2013 (stratification, importance sampling, ESS).
