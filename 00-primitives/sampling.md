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
