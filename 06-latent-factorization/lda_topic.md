# Latent Dirichlet allocation

`lda_topic.py` · Latent structure · Tier 2 · 4 variants

## Intuition

A document is a bag of words. LSA takes that literally and decomposes the bag.
LDA asks a different question — not *how do I factor this matrix* but *what
process would produce it* — and the answer it proposes is the smallest one that
works:

```
   α ──▶ θ_d ──▶ z_dn ──▶ w_dn ◀── β_k
         ▲        ▲                 ▲
         │        └── one topic per token
         └── a document is a mixture over topics
                          K topics, one β_k each

   for each document d:
       θ_d ~ Dirichlet(α, …, α)              the mixture
       for each of the N_d words:
           z_dn ~ Categorical(θ_d)           pick a topic
           w_dn ~ Categorical(β_{z_dn})      draw a word from it
```

That is the whole model: two Dirichlet priors and two categorical draws. There
is no matrix in the generative story at all. The counts appear only because the
generator is run many times and then histogrammed, and the histogram is what a
corpus gives you:

$$x_{dw} \;=\; \#\{n : w_{dn} = w\}. \qquad (1)$$

The consequences of taking the story seriously are not cosmetic.

**It explains the zeros.** A term with count zero in a document is not "a small
value"; it is a term the document's mixture did not generate. LSA and NMF treat
the same zero as a small residual, and both are free to fill it in with whatever
the low-rank structure prefers. LDA cannot: a topic that puts mass on a term
contributes to that term's rate in every document that uses the topic.

**It makes the factors probabilities.** A row of `components_` is a
distribution over the vocabulary, so it sums to one and its entries are rates.
That is why topic printouts are readable — "the 15 most probable words" is a
meaningful summary — and why `doc_topic_` rows are mixtures, not loadings.

**It separates the two things a corpus has.** Document length is $N_d$, drawn
once and then fixed; the topic mixture is $\theta_d$. LSA's weighting has to
fight document length because it is baked into the counts. LDA conditions on it,
and a 2 000-word document and a 200-word one with the same mixture get the same
$\theta$.

The price is that the likelihood is a product over tokens of sums over topics,
which has no closed-form posterior. Everything in this file is a way of paying
that price, and the first three variants differ only in how:

| | what is left to infer | diagnostic | measured |
|:--|:--|:--|:--|
| `collapsed_gibbs` | the token assignments $z$ | $\log p(w,z)$ | exact in the limit, no bound |
| `variational_bayes` | $q(\theta)q(\beta)q(z)$ | the ELBO | monotone, $+4\,270$ nats |
| `online_lda` | the same $q$, one batch at a time | — | 0.996 to the batch fit |
| `seeded_lda` | $z$, with a prior on some words | $\log p(w,z)$ | labelling becomes identity |

On the demo corpus all three unsupervised fitters recover the same three topics
to a cosine of $0.99$ against the parameters that generated the data, and to
$1.00$ against each other (after matching the labelling, which no fit can
determine). The fitter does not change the answer. It changes the cost, the
diagnostic, and the amount of data you need in memory — and `seeded_lda`
changes the *information*, which is a different kind of change.

## Variants

| Variant | What it does | Cost | Where it wins |
|:--|:--|:--|:--|
| `collapsed_gibbs` | sample each token's topic from its full conditional | $O(NK)$ per sweep | exact, no step size, no bound to trust |
| `variational_bayes` | coordinate ascent on a mean-field bound | $O(DKV)$ per iteration | fast, deterministic, has an objective |
| `online_lda` | the same, with a minibatch and a decaying step | $O(BKV)$ per step | data larger than memory, streams |
| `seeded_lda` | collapsed Gibbs with seed words carrying extra prior mass | $O(NK)$ per sweep | labelled topics, faster convergence |

**`collapsed_gibbs`** integrates $\theta$ and $\beta$ out analytically and
samples only the $N$ token assignments. It is exact — the chain's stationary
distribution is the posterior — and it is the variant everyone means by "LDA
with Gibbs". Its weakness is that it has no objective: $\log p(w,z)$ for a
single sample of $z$ is not a bound and not monotone, and the demo's trace shows
it settling into a band about 80 nats wide and staying there forever. That is
correct behaviour, not a bug, and it is the main thing practitioners
misdiagnose as non-convergence.

**`variational_bayes`** replaces each posterior with a factorised $q$ and
ascends the resulting bound. Every step is exact given the others, so the ELBO
increases at every iteration — the demo asserts it, and the smallest observed
step is $0.001$. It is deterministic given a seed, it converges in tens of
iterations here rather than hundreds of sweeps, and at the end you get a number
you can compare across models. What you do not get is the posterior: the
mean-field family is too narrow, and the bound can sit well below the evidence.

**`online_lda`** is `variational_bayes` with two changes: the global parameter
$\lambda$ is updated by a natural-gradient step rather than set, and the local
step runs on a minibatch. The demo makes the step size do visible work. Feed
the batches in topical order — one class of documents at a time, which is what
a stream looks like — and with $\rho_t = (\tau_0+t)^{-0.7}$ the fit lands
$0.996$ cosine from batch VB; with $\kappa = 0$, so $\rho_t = 1$ and the
estimate is *whatever the last batch said*, it lands $0.687$. Nothing about the
model changed. The averaging is the algorithm.

**`seeded_lda`** adds a third Dirichlet parameter: for a token whose word is a
seed for topic $k$, the topic-$k$ concentration is $\alpha + s$ instead of
$\alpha$. It fixes the one quantity the likelihood cannot determine — which
topic is called topic 1 — and it converges faster. On five topics the unseeded
fit returns the topics in the order $(0,2,1,3,4)$ at cosine $0.98$; with seeds,
the order is the identity, and five sweeps reach $0.94$ where the unseeded fit
reaches $0.84$. Seeds are not free, and the demo also measures the bill: at 120
sweeps the seeded fit is $0.95$ against the unseeded $0.98$, because a seed
word is *every occurrence* of that word, and the prior keeps re-asserting them.

## Derivation

### 1. The likelihood, and where the count matrix comes from

With $\beta$ the $K \times V$ topic-word matrix, one document of length $N_d$
has likelihood

$$p(w_d \mid \theta_d, \beta) \;=\; \prod_{n=1}^{N_d} \sum_{k=1}^{K} \theta_{dk}\,\beta_{k w_{dn}}, \qquad (2)$$

a product of *mixtures* — this is exactly the extra sum that LSA does not have.
Grouping tokens by type turns the product over positions into a product over the
vocabulary:

$$p(w_d \mid \theta_d, \beta) \;=\; \prod_{w=1}^{V} \left(\sum_{k} \theta_{dk}\beta_{kw}\right)^{x_{dw}}. \qquad (3)$$

So the count matrix is a sufficient statistic for the whole corpus — this is
why LDA takes the same input as LSA and NMF — but (3) is not a low-rank
approximation of $X$, it is a product of $V$ sums per document. The log of it is
what the perplexity in the demo measures.

### 2. Collapsing: Dirichlet–multinomial conjugacy

The trick that makes Gibbs cheap is that $\theta$ and $\beta$ can be integrated
out in closed form. For one document, with a $\mathrm{Dir}(\alpha)$ prior on
$\theta$,

$$\int p(z_d \mid \theta)\, p(\theta)\, d\theta \;=\; \frac{\Gamma(K\alpha)}{\Gamma(\alpha)^{K}}\, \frac{\prod_k \Gamma(n_{dk}+\alpha)}{\Gamma(N_d + K\alpha)}, \qquad (4)$$

where $n_{dk}$ counts the tokens of document $d$ assigned to topic $k$. That is
the Dirichlet-multinomial, and the same identity with $\beta$ and the topic
counts $n_{kw}$, $n_k$ gives $p(w \mid z)$. Multiplying,

$$p(w, z) \;=\; \underbrace{\prod_d \frac{\Gamma(K\alpha)}{\Gamma(\alpha)^K}\frac{\prod_k \Gamma(n_{dk}+\alpha)}{\Gamma(N_d+K\alpha)}}_{p(z)} \cdot \underbrace{\prod_k \frac{\Gamma(V\beta)}{\Gamma(\beta)^V}\frac{\prod_w \Gamma(n_{kw}+\beta)}{\Gamma(n_k+V\beta)}}_{p(w\mid z)}. \qquad (5)$$

Every quantity in (5) is a count. That is the whole reason the sampler is four
lines of arithmetic: no $\theta$, no $\beta$, no gradient.

The demo computes (5) directly — `_collapsed_loglik` is a transcription of it —
and the resulting trace is the honest picture of MCMC: it starts at $-17\,098$
with a random assignment, jumps to about $-12\,000$ within ten sweeps, and then
fluctuates in a band of roughly $\pm 40$ nats forever. It never becomes
monotone, because each recorded value is one draw from the posterior, not an
average.

### 3. The full conditional

Subtracting token $n$ (currently on topic $k_0$) from the counts and applying
(5) to the two terms that mention it leaves one ratio of Gammas, which collapses
by $\Gamma(x+1) = x\,\Gamma(x)$:

$$p(z_n = k \mid z_{-n}, w) \;\propto\; \underbrace{\frac{n_{dk}^{-n} + \alpha}{N_d - 1 + K\alpha}}_{\text{this document likes } k} \cdot \underbrace{\frac{n_{kw}^{-n} + \beta}{n_k^{-n} + V\beta}}_{\text{topic } k \text{ likes this word}}, \qquad (6)$$

where the superscript $-n$ means the counts with token $n$ removed. Both
factors are what they look like: the first is how much of the document is
already on topic $k$ (with the prior as a pseudo-count), the second is how much
of topic $k$ is already this word.

Two features are worth naming because they are the model, not an implementation
detail.

**The normaliser $n_k + V\beta$ is not cosmetic.** It penalises large topics:
a topic with many tokens pays more for each new one. Without it the sampler
would be a rich-get-richer process with no ceiling, and all tokens would collapse
into one topic. With $\beta$ small the penalty is mild and topics grow; with
$\beta$ large the penalty dominates and every topic stays near-uniform.

**The prior is a pseudo-count, so "small $\alpha$" means "sparse documents".**
$\alpha = 0.1$ puts $0.1$ of mass on each topic before any token is seen; a
30-token document is then a mixture of one or two topics, not an even blend of
three. That is the knob that makes $n_{dk}+\alpha$ behave like a smoothed count
ratio rather than a real one, and choosing it is a modelling decision about how
mixed you believe documents are.

### 4. Seeds as a prior, not a constraint

`seeded_lda` adds $s$ to the concentration of the topic a word seeds:

$$p(z_n = k \mid z_{-n}, w) \;\propto\; \frac{n_{dk}^{-n} + \alpha + s\,\mathbb{1}[k = \mathrm{seed}(w_n)]}{N_d - 1 + K\alpha + s\,\mathbb{1}[\mathrm{seed}(w_n) \ne \varnothing]} \cdot \frac{n_{kw}^{-n}+\beta}{n_k^{-n}+V\beta}. \qquad (7)$$

This is a genuine Dirichlet prior — the model with seeds is LDA with a
word-dependent $\alpha$ — which is why it does not have the pathologies of a
hard clamp. A clamped token cannot move even when the rest of the document
overwhelmingly disagrees, so a wrong seed becomes an unfixable error and the
topic it was pinned to absorbs every occurrence of that word. A seeded token can
move, it just needs more evidence. The demo shows the difference in the numbers:
the strength is $s = 2$ by default, and raising it to $10$ changes the result by
less than $0.002$ on this corpus — the prior saturates, it does not tighten
indefinitely, because once a topic is at $99\%$ the extra mass is irrelevant.

The measure that makes seeds legible is coverage. The demo's five topics have
two seed words each out of fifty, and those ten words account for $1\,168$ of
the $2\,400$ tokens — 49% of the corpus. Seed words are the *frequent* ones, so
"two seed words per topic" is not a small annotation; it is a prior on half the
data. That is why five sweeps improve from $0.84$ to $0.94$, and also why the
converged fit loses $0.03$ of cosine: the repeated evidence is being counted
twice.

### 5. Mean field, and why the updates are exact

Write the variational family as

$$q(\theta, \beta, z) \;=\; \prod_d q(\theta_d) \prod_k q(\beta_k) \prod_{d,n} q(z_{dn}), \qquad (8)$$

with $q(\theta_d) = \mathrm{Dir}(\gamma_d)$, $q(\beta_k) = \mathrm{Dir}(\lambda_k)$
and $q(z_{dn}) = \mathrm{Cat}(\phi_{dn})$. Coordinate ascent on the ELBO gives,
for each coordinate, the exponentiated expectation of the log joint over the
others. For $z$,

$$\phi_{dnk} \;\propto\; \exp\left\{ \mathbb{E}[\log \theta_{dk}] + \mathbb{E}[\log \beta_{kw_{dn}}] \right\}, \qquad (9)$$

and for the two Dirichlets, the same Dirichlet-multinomial form as (4):

$$\gamma_{dk} = \alpha + \sum_n \phi_{dnk}, \qquad \lambda_{kw} = \beta + \sum_d x_{dw}\,\phi_{dkw}. \qquad (10)$$

The expectations are digammas: $\mathbb{E}[\log \theta_{dk}] = \Psi(\gamma_{dk}) - \Psi(\sum_{k'}\gamma_{dk'})$.

The one place an implementation goes wrong is (10)'s second equation. The
statistic for $\lambda$ is $\sum_d x_{dw}\phi_{dkw}$ — a $K \times V$ expected
count of *word-topic* pairs — not the per-document topic totals, which is the
statistic for $\gamma$. Conflating them is tempting because both are sums of
$\phi$, and it is silent: the fit still converges, and the topic-word matrix
comes out at cosine $0.575$ instead of $0.989$ because $\lambda$ was handed
document-level totals with no word information in them at all.

The reason the updates are closed-form at all is the conjugacy of §2: the
Dirichlet is the conjugate prior for the categorical, so the optimal $q$ given
the rest is always the same family with updated counts. This is why LDA's CAVI
is ten lines, and why a model with a logistic-normal prior instead has to be
fitted by gradient methods.

### 6. The ELBO, term by term

$$\mathcal{L} \;=\; \underbrace{\mathbb{E}[\log p(\theta)] + \mathbb{E}[\log p(\beta)]}_{\text{priors}} \;+\; \underbrace{\mathbb{H}[q(\theta)] + \mathbb{H}[q(\beta)] + \mathbb{H}[q(z)]}_{\text{entropies}} \;+\; \underbrace{\mathbb{E}[\log p(w, z \mid \theta, \beta)]}_{\text{data}}, \qquad (11)$$

with $\mathbb{H}[q] = -\mathbb{E}[\log q]$. The middle group is where every
implemented ELBO goes wrong, because it is written with a minus sign in the code
— `- sum((gamma - 1) * Eg)` — and expanded, not left as an entropy.

Written out for the two Dirichlet factors, one document and one topic,

$$\mathbb{E}[\log p(\theta_d)] = (\alpha - 1)\sum_k \mathbb{E}[\log \theta_{dk}] - \log B(\alpha), \qquad (12)$$

$$-\mathbb{E}[\log q(\theta_d)] = -\sum_k (\gamma_{dk}-1)\,\mathbb{E}[\log \theta_{dk}] + \log B(\gamma_d), \qquad (13)$$

with $B$ the Dirichlet normaliser $B(a) = \prod_k \Gamma(a_k) / \Gamma(\sum_k a_k)$.
The signs in (12) and (13) are the part that is easy to get backwards, and
getting them backwards is *not* a crash: the central term flips sign, the
iteration still converges smoothly, and the "bound" decreases monotonically
from $-15\,708$ to $-11\,437$. The demo asserts
`steps.min() > -1e-6`, which is exactly the assertion that catches it — the
symptom of a wrong ELBO is an ELBO that moves the wrong way, and nothing else.

The data term is where $\phi$ earns its keep:

$$\mathbb{E}[\log p(w, z \mid \theta, \beta)] = \sum_{d,w} x_{dw} \sum_k \phi_{dkw}\left(\mathbb{E}[\log \theta_{dk}] + \mathbb{E}[\log \beta_{kw}] - \log \phi_{dkw}\right), \qquad (14)$$

and the $-\log\phi$ inside is the $-\mathbb{E}[\log q(z)]$ term, not a separate
one — the two are the same series written once. That is why the code has one
loop where the derivation has three terms.

### 7. Stochastic VI, and what the step size is for

Batch CAVI touches all $D$ documents per iteration. SVI touches $B$ and treats
the minibatch statistic as an estimate of the full one, rescaled by $D/B$:

$$\lambda \;\leftarrow\; (1-\rho_t)\,\lambda + \rho_t\left(\beta + \frac{D}{B}\sum_{d \in \mathcal{B}} \mathbb{E}[n_{kw}^{(d)}]\right), \qquad \rho_t = (\tau_0 + t)^{-\kappa}. \qquad (15)$$

The rescaling is what makes it unbiased; the step size is what makes it
*converge*. The Robbins–Monro conditions are $\sum_t \rho_t = \infty$ and
$\sum_t \rho_t^2 < \infty$, which $\kappa \in (0.5, 1]$ satisfies and $\kappa = 0$
does not: with $\rho_t = 1$ the recursion is $\lambda \leftarrow$ last batch, so
the estimate has the variance of a single batch no matter how many batches you
have seen. On i.i.d. minibatches that noise is harmless — every batch is a
draw from the same distribution — and the demo's sorted stream is the case where
it is not: the last 16 documents are one topic, and $\lambda$ remembers exactly
that. $0.996$ against $0.687$ is the same algorithm with one constant changed.

The order dependence is a real design decision, not an oversight. `online_lda`
consumes rows in the order you give them and cycles; it does not shuffle. A
caller with i.i.d. data can shuffle first, and a caller with a stream cannot —
and it is the caller who knows which they have.

### 8. What the likelihood cannot tell you

For any permutation $\pi$ of the topics, substituting $\theta_{d\pi(k)}$,
$\beta_{\pi(k)w}$ for the originals leaves (2) unchanged. The posterior
therefore has $K!$ equivalent modes, and a fit returns one of them according to
where the sampler or the initialisation happened to land. This is not a
numerical artefact: it is a symmetry of the model, and no amount of data
removes it.

The demo shows both directions. Permuting `components_` and the matching `lam_`
of a fitted VB model leaves the held-out perplexity bit-identical — measured
difference $0.0$, because everything downstream is a function of
$\theta\beta$ and the product is invariant. Permuting a *Gibbs* fit's
`components_` changes perplexity by $0.05$, which looks like evidence against
invariance and is not: `transform` for Gibbs runs a 20-sweep chain per document
with its own random numbers, so the difference is Monte Carlo noise, $0.24\%$ of
$20.5$. The model is invariant; a sampler is not, bit for bit.

Non-identifiability is why `_match_topics` exists and why every accuracy number
in the demo is computed after a brute-force search over $K!$ permutations. For
$K \le 8$ that is at most $40\,320$ permutations and the honest thing to do; any
greedy matching is a heuristic that silently fails when two topics are similar.
It is also why `seeded_lda` is worth a variant: seeds are the only mechanism in
this file that picks a labelling *by construction* instead of by convention.

### 9. Perplexity, and what the demo's number means

$$\mathrm{perplexity}(X) \;=\; \exp\left\{-\frac{\sum_{d,w} x_{dw} \log \hat{p}_{dw}}{\sum_{d,w} x_{dw}}\right\}, \qquad \hat{p}_{dw} = \sum_k \hat\theta_{dk}\hat\beta_{kw}. \qquad (16)$$

It is the effective vocabulary size the model is choosing between; $20.5$ against
a 40-term vocabulary means the model has narrowed the field by a factor of two.

The demo's fitted model scores $20.52$ and the corpus generator scores $20.57$:
the fit beats the model that produced the data. That is not a bug and not
evidence of a better model — $\hat\theta_d$ is inferred from the same document
it is then scored on, so a 30-token document gives the inference a chance to
overfit itself. The comparison that is meaningful is between models scored the
same way, which is why the doc reports Gibbs $20.52$ against VB $20.27$
(it is a real difference — VB's tighter fit to this corpus) and not against the
oracle.

## Code walkthrough

`LDA(n_topics=3, *, variant="collapsed_gibbs", alpha=0.1, beta=0.01,
n_iter=150, seed=0, tau0=1.0, kappa=0.7, batch_size=16, n_local=5,
seed_words=None, seed_strength=2.0)` with `fit`, `transform`, `fit_transform`,
`perplexity`, and the attributes `components_` $(K, V)$ row-normalised,
`doc_topic_` $(D, K)$ row-normalised, `lam_`, `elbo_curve_`, `loglik_curve_`.

`components_` is `(K, V)` for the same reason `nmf.py`'s is: a row is a topic,
and `theta @ components_` is the fitted rate matrix. `doc_topic_` is the training
mixture from the *last* state of the fit, so for Gibbs it is one sample and for
VB it is the mean of $q(\theta_d)$; both are row-normalised, and the demo
asserts it.

### 1. The count matrix as a token list

```python
def _expand(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Count matrix → the flat token list `(document, term)` it represents."""
    n_docs, n_terms = X.shape
    docs = np.repeat(np.arange(n_docs), X.sum(axis=1).astype(np.int64))
    words = np.repeat(np.tile(np.arange(n_terms), n_docs), X.ravel().astype(np.int64))
    return docs, words
```

The sampler is written in terms of tokens because (6) is a statement about one
token at a time, and because the same code then does not care how the counts are
distributed. `np.repeat` with a counts array is the idiomatic expansion: for each
document, its row of counts says how many times to repeat each term id.
$4\,800$ tokens for the demo corpus, and this function is the entire bridge
between the matrix representation and the generative one.

`X.ravel()` and the two `repeat`s must be in the same order — document-major —
which is why `words` repeats the *tiled* term ids rather than concatenating
documents. Getting this wrong pairs tokens with the wrong document, and the
sampler still runs and still produces topics, just worse ones.

### 2. The Gibbs update

```python
                own = int(owner[n])
                total = 0.0
                p = [0.0] * K
                for k in range(K):
                    a = alpha + (strength if k == own else 0.0)
                    v = (row[k] + a) * (n_kw[k][w] + beta) / (n_k[k] + Vbeta)
                    p[k] = v
                    total += v
                u = uniform() * total
                k1 = K - 1
                for k in range(K):
                    u -= p[k]
                    if u <= 0.0:
                        k1 = k
                        break
```

(6) and (7). The counts were decremented before this block and are incremented
after it, so every factor is the *leave-one-out* count the conditional needs.
Normalising is skipped deliberately: sampling from an unnormalised vector by
scaling one uniform by the total is exact, and it saves a division per token per
topic — with $4\,800$ tokens and 120 sweeps that is $1.7$ million divisions.

`strength` is a local, not `self.seed_strength`, because this loop runs
$1.7$ million times and an attribute lookup inside it is measurable. `own` is
$-1$ for every token when not seeding, so the `k == own` test never fires and
the seeded and unseeded samplers are the same code path — one prior term,
zeroed out.

The cumulative-sum sampling is the standard inverse-CDF trick written as a loop
rather than `np.searchsorted(np.cumsum(p), u)`: at $K=3$ the Python loop is
faster than building two numpy arrays, and at $K=100$ the numpy version wins.
Neither is the bottleneck; the $O(NK)$ sweep is.

### 3. The collapsed log-likelihood

```python
        # p(z | α): one Dirichlet–multinomial normaliser per document ...
        ll = n_docs * (math.lgamma(K * a) - K * math.lgamma(a))
        ll += float(np.sum(_lgamma(np.asarray(n_dk, dtype=float) + a)))
        ll -= float(np.sum(_lgamma(np.asarray([sum(row) for row in n_dk]) + K * a)))
        # ... and p(w | z, β): one per topic.
        ll += K * (math.lgamma(V * b) - V * math.lgamma(b))
        ll += float(np.sum(_lgamma(np.asarray(n_kw, dtype=float) + b)))
        ll -= float(np.sum(_lgamma(np.asarray(n_k, dtype=float) + V * b)))
```

(5), with the two normalisers per level written out. `_lgamma` is applied to the
whole count array at once, which is why the function is vectorised: the corpus
needs $DK + KV$ of them — one per document-topic and topic-word pair — and a
scalar loop over that many `math.lgamma` calls costs more than the sweep it is
measuring.

Two honest caveats live in this function's docstring and in the trace it
produces. It is $\log p(w, z)$, not $\log p(w)$ — the assignments are a
variable, so it is not a bound and it is not comparable to the ELBO. And it is
evaluated every tenth sweep rather than every sweep: $DK + KV = 480$ log-Gammas
is cheap, but not cheap enough to pay for twelve times over when the number it
produces moves by $\pm 40$ nats anyway.

### 4. One mean-field E-step

```python
        Eg = _digamma(gamma) - _digamma(gamma.sum(axis=1, keepdims=True))
        El = _digamma(lam) - _digamma(lam.sum(axis=1, keepdims=True))
        stats = np.zeros((K, V))
        cnt = np.zeros_like(gamma)
        phis = []
        for d in range(X.shape[0]):
            logits = Eg[d][:, None] + El                   # (K, V) per document
            phi = np.exp(logits - _logsumexp(logits, axis=0))
            phis.append(phi)
            per = phi * X[d]
            stats += per
            cnt[d] = per.sum(axis=1)
```

(9) and (10). `Eg[d][:, None] + El` is a $K \times V$ matrix of the two
expected log-probabilities; a softmax *down the columns* gives $\phi_{dkw}$ for
every word in the document at once, which is why the loop is over documents and
not over tokens. `per = phi * X[d]` then multiplies by the counts, so a word
appearing three times contributes three times — the count-matrix shortcut that
makes CAVI $O(DKV)$ rather than $O(NK)$.

`stats` is (10)'s second equation and `cnt` is its first, accumulated in the
same pass. Keeping them separate is the difference between cosine $0.989$ and
$0.575$; the two are easy to confuse because both are sums of $\phi$, and only
one of them knows which word it came from.

The `_logsumexp` shift is not needed at the demo's parameters and belongs in the
code anyway. Since $\Psi(x) \approx -1/x$ for small $x$, an unseen word in a
topic has expected log-probability $\approx -1/\beta$: $-100$ at
$\beta = 0.01$, and $-10\,000$ at $\beta = 10^{-4}$, which is below the double-
precision underflow point of $-745$. When every exponential in a column flushes
to zero, `phi` is $0/0$ and `stats` is NaN, and $\beta$ is exactly the knob a
user is invited to make smaller when they want sparser topics.

### 5. The ELBO

```python
        out = float((a - 1.0) * Eg.sum()) - n_docs * (
            K * math.lgamma(a) - math.lgamma(K * a))
        out += float((b - 1.0) * El.sum()) - K * (
            V * math.lgamma(b) - math.lgamma(V * b))
```

(12), twice — once for $\theta$, once for $\beta$ — with the Dirichlet
normaliser written as $K\log\Gamma(\alpha) - \log\Gamma(K\alpha) = -\log B(\alpha)$
and multiplied by the number of documents (or topics) that have one. Expanding
`math.lgamma` rather than computing $B$ is a small clarity win here: the
subtraction that has to be right is visible.

The prior term is where the sign errors live, and the demo's assertion is the
only defence. Note what is *not* here: no $\log\phi$, no data. Those are the
last group, and they are computed in the loop over documents because they are
the only part that scales with the counts.

### 6. The two closed-form updates

```python
            stats, cnt, phis, Eg, El = self._vb_step(X, gamma, lam)
            self.elbo_curve_.append(self._elbo(X, gamma, lam, phis, Eg, El))
            gamma = self.alpha + cnt
            lam = self.beta + stats
```

(10), and the reason CAVI needs no step size: each coordinate is set to its
exact optimum given the others, so the bound cannot decrease. The order of the
three lines matters for the diagnostic — the ELBO is recorded for the state that
produced `phis`, i.e. *before* the update — and recording it after the update
instead would compare a state against a $q$ that was optimised for a different
one, which shows up as a small negative step.

`lam = self.beta + stats` is not `lam += ...`. It is a full reassignment from
the sufficient statistic, which is what makes the iteration converge to a fixed
point rather than accumulate: the Dirichlet $\lambda$ is the only global
parameter, and (10) says it is a function of the current $q(z)$ with no memory.

### 7. The step size that does the work

```python
            rho = (self.tau0 + t) ** (-self.kappa)
            lam = (1.0 - rho) * lam + rho * (self.beta + (n_docs / B) * stats)
```

(15). The `n_docs / B` rescaling is the unbiasedness; the interpolation is the
variance reduction. `tau0 = 1` shifts the schedule so that $\rho_0 = 1$ — the
first batch is used whole, which is standard and matters because a fresh
$\lambda$ from `rng.random` is a terrible initialisation and should be
overwritten rather than averaged with. `kappa = 0.7` is the paper's default and
satisfies both Robbins–Monro conditions.

Above it, `self.n_local` iterations of the local step run per batch, and each
batch's $\gamma$ starts from $\alpha + N_d/K$ rather than from the previous
batch's — the documents are new, so there is nothing to carry over, and the
`D/B` rescaling in the global step is what supplies the memory. The nested loop
reinitialises `stats` and `cnt` inside the local loop deliberately: only the
last local iteration's statistic corresponds to the $\gamma$ that is exported.

### 8. Inference for a new document

```python
            z = rng.integers(0, K, size=n)
            n_dk = np.bincount(z, minlength=K).astype(float)
            for _ in range(n_iter):
                for i in range(n):
                    n_dk[z[i]] -= 1
                    p = (n_dk + self.alpha) * phi[:, w[i]]
                    p /= p.sum()
                    k = int(np.searchsorted(np.cumsum(p), rng.random()))
                    k = min(k, K - 1)
                    z[i] = k
                    n_dk[k] += 1
```

(6) with $\beta$ frozen: only the first factor is unknown, so the conditional is
`(n_dk + alpha) * phi[:, word]`. The `n_kw` term of the full conditional is
replaced by the fitted `phi`, which is the whole point of having fitted it — the
document no longer has a vote on what a topic means, only on which topics it
uses.

`np.searchsorted(np.cumsum(p), rng.random())` is the inverse-CDF sample in its
numpy form; `min(k, K - 1)` guards the case where the uniform lands above the
last cumulative entry by a rounding hair. And the 20 sweeps are per document, on
30 tokens — cheap, and the reason the demo's Gibbs perplexity is $0.05$ away
from being permutation-invariant: a short chain does not fully mix.

### 9. Digamma and log-gamma, written out

```python
    y = np.asarray(x, dtype=float).copy()
    r = np.zeros_like(y)
    small = y < 6.0
    while small.any():                         # Ψ(x) = Ψ(x+1) − 1/x
        r = np.where(small, r - 1.0 / np.where(small, y, 1.0), r)
        y = np.where(small, y + 1.0, y)
        small = y < 6.0
    f = 1.0 / (y * y)
    return r + np.log(y) - 0.5 / y - f * (1 / 12 - f * (1 / 120 - f * (
        1 / 252 - f * (1 / 240 - f / 132))))
```

The recurrence pushes every argument up to $y \ge 6$, where the asymptotic
series $\Psi(y) \approx \log y - \frac{1}{2y} - \sum_n \frac{B_{2n}}{2n y^{2n}}$
is accurate to about $10^{-10}$ — measured against the true values at
$\Psi(1) = -0.577216$, $\Psi(0.5) = -1.963510$, $\Psi(10) = 2.251753$. The
Bernoulli coefficients in the bracket are $1/12, 1/120, 1/252, 1/240, 1/132$:
five terms, and the fifth is there because $\alpha = 0.1$ makes $\gamma$ close to
$0.1$ often and the series is worse the smaller the argument.

`_lgamma` is the same construction with the recurrence $\Gamma(y) =
\Gamma(y+1)/y$ and Stirling's series, accurate to $2 \times 10^{-14}$ on the
values this file uses. Both are vectorised with `while small.any()` for exactly
one reason: the ELBO calls them on whole count arrays, and a scalar loop over
$KV$ entries would cost more than the rest of the fit.

The `np.where(small, y, 1.0)` idiom is the safe form — it prevents `1/y` from
dividing by zero on entries that have already been shifted out of the "small"
set, which produces a warning and an inf even though `np.where` discards it.

### 10. Matching the labelling

```python
    C = (A @ B.T) / np.outer(np.maximum(na, EPS), np.maximum(nb, EPS))

    def score(p):
        return float(sum(C[p[k], k] for k in range(K)))

    best = max(itertools.permutations(range(K)), key=score)
    return score(best) / K, best
```

§8, as a function. The cosine matrix is computed once, and then every
permutation is a lookup — $K!$ additions rather than $K!$ dot products. The
returned permutation is reported by the demo rather than hidden, because for the
unseeded fit it is the interesting output: `(0, 2, 1, 3, 4)` says the first two
topics came back swapped, which is not an error and not a worse fit.

`np.maximum(., EPS)` guards a topic that received no mass at all; with
$\beta = 0.01$ that cannot happen after a fit, but `_match_topics` is also called
on partial states by the demo.

### 11. What the estimator refuses

```python
        if np.isnan(X).any() or (X < 0.0).any():
            raise ValueError("LDA needs non-negative counts")
        if not np.allclose(X, np.round(X)):
            raise ValueError("LDA samples tokens, so the counts must be integers")
```

The refusal is the model. $x_{dw}$ in (5) enters through $\Gamma(n_{dw} + \beta)$,
and a fractional $x$ would make that a continuous function of a quantity that is
supposed to be a count — the sampler has no token to move. tf-idf'd LSA input
is rejected here, correctly, and the error message says why rather than leaving
a user to discover that their weighted matrix produced a Gamma function of 0.37.

The integer check is `np.allclose` against a round, not `dtype.kind == "i"`,
because a float array of whole numbers is a perfectly good count matrix and a
caller who built it with `astype(float)` should not be punished. What is rejected
is *values* that are not counts.

## Complexity

$D$ documents, $V$ terms, $K$ topics, $N$ tokens, $B$ batch size, $T$ iterations.

| Operation | Time | Memory | Note |
|:--|:--|:--|:--|
| `_expand` | $O(N + DV)$ | $O(N)$ | the token list, once |
| `_fit_gibbs`, per sweep | $O(NK)$ | $O(N + KV)$ | counts are sparse lists |
| `_collapsed_loglik` | $O(DK + KV)$ | $O(KV)$ | every tenth sweep |
| `_vb_step` | $O(DKV)$ | $O(DKV)$ | one $K \times V$ per document |
| `_elbo` | $O(DKV)$ | $O(1)$ extra | reuses the `phis` the E-step made |
| `_fit_online`, per step | $O(BKV)$ | $O(B K + KV)$ | independent of $D$ |
| `_infer_gibbs` | $O(N_d K)$ per document | $O(N_d)$ | 20 sweeps |
| `perplexity` | $O(mKV)$ | $O(mV)$ | over the documents you pass |

The interesting comparison is the second row against the fourth. Gibbs is
$O(NK)$ per sweep with $O(N)$ state, and VB is $O(DKV)$ per iteration with an
$O(DKV)$ `phi`. For a corpus of $30$-token documents, $N = 30D$, so a Gibbs
sweep touches $30K$ operations per document and a VB iteration touches
$KV = 40K$: nearly the same arithmetic, and Gibbs needs *more* passes.

What decides it in practice is vectorisation, and it decides against Gibbs. A
120-sweep Gibbs fit here takes $0.48$ s against $0.13$ s for 80 VB iterations,
a factor of $3.7$, because VB's inner work is three numpy calls on a
$K 	imes V$ array and Gibbs's is a Python loop over $4\,800$ tokens per sweep.
The asymptotics say they are comparable; the constant says use VB. The whole
demo is $1.9$ s because it runs eight fits.

The asymptotics do eventually matter, and they matter in Gibbs's favour on
*long* documents: $N_d$ can be ten thousand tokens, while $KV$ cannot, and
CAVI never revisits a document while Gibbs revisits every token every sweep.

The online row is the one that matters at scale: per step, $B$ documents, so
the cost is independent of $D$ once the data is streamed. `lam` is the only
persistent state — $KV$ floats — which is why online LDA can fit a corpus that
does not fit in memory, and why the demo can hold the whole thing in an array
and pretend.

Memory is where a real implementation diverges from this file. `_vb_step`
allocates a $K \times V$ matrix per document; at $K = 100$, $V = 10^5$ that is
80 MB per document, and no corpus survives it. The standard fix is to keep
`phi` implicitly as `Eg[d] + El` and compute the statistics in the same pass,
which is what the loop does *not* do because the ELBO needs the explicit
`phis`. Trade a factor of two in speed for the bound, or drop the ELBO.

## Failure modes

**1. $K$ is not estimated, and neither diagnostic picks it.** On this corpus the
truth is $K = 3$, and held-out perplexity says $K = 6$ and still falling:
$23.14, 20.27, 20.12, 19.96, 19.72$ for $K = 2, \ldots, 6$. More topics always
help here because $\theta_d$ is inferred from the document it is then scored on,
so the extra topics are being used to fit each document's noise. The training
ELBO does not rescue it either: $-11\,423$ at $K=3$, then $-11\,591$,
$-11\,706$, $-11\,826$. That ordering is not a capacity statement — $\alpha$ is
per topic, so raising $K$ also raises the total prior mass $K\alpha$ and widens
the mean-field gap — and comparing bounds computed in different parameter spaces
is not a model comparison at all. Every "choose $K$" heuristic in the literature
is a proxy: coherence needs a reference corpus, stability needs several seeds,
and held-out perplexity needs a scoring rule that does not let the model fit the
document it is scored on.

**2. Topics are not identified, so any cross-fit comparison is void until you
permute.** §8. The demo's unseeded five-topic fit comes back as
$(0,2,1,3,4)$. A pipeline that compares `components_[0]` across two fits is
comparing arbitrary objects; a pipeline that averages topic-word matrices across
seeds is producing an average of different labelings and will report a topic
that is the mean of two real ones. This is not exotic — it is the default
failure of every topic-model ensemble.

**3. Seeds are evidence, and evidence counts each occurrence.** The demo's
ten seed words cover 49% of the tokens. At strength $2$ that buys faster
convergence ($0.84 \to 0.94$ at five sweeps) and costs converged accuracy
($0.98 \to 0.95$). The mechanism is not subtle: a seed word's every appearance
pushes towards its topic, so a topic seeded with a word that is merely frequent
in the corpus becomes the "frequent word" topic and drags unrelated documents
into it. Choosing seeds is choosing a prior, and it should be as deliberate as
choosing $\alpha$.

**4. Non-integer or weighted input is rejected, and that is correct.** §11. The
temptation is to feed tf-idf or L2-normalised rows because that is what LSA
takes. LDA's likelihood is a product over tokens; there is no token to move in a
weighted row, and a "count" of $0.37$ gives a Gamma function of nothing.

**5. $\alpha$ and $\beta$ decide document sparsity and topic sparsity, and both
are usually left at defaults.** The demo's $\alpha = 0.1$ makes a 30-token
document a mixture of one or two topics; $\alpha = 5$ makes every document a
flat mixture and drops the recovery from $0.99$ to $0.95$. $\beta = 1.0$ spreads
every topic over the whole vocabulary and raises held-out perplexity from $20.3$
to $20.5$. Neither is catastrophic, and neither is a default worth inheriting
unexamined.

**6. The Gibbs diagnostic is not monotone, and it is not a bound.** The
$80$-nat band in §2 is what a correct chain looks like. Users used to the ELBO
read it as failure and run 10 000 sweeps to chase a number that will never stop
moving. The right check is on a quantity you care about — held-out perplexity,
topic stability across seeds, or a downstream metric.

**7. Perplexity is comparable only within a model class, and it can beat its
own oracle.** §9: $20.52$ against $20.57$ because $\theta$ is inferred from the
scored document. LDA's perplexity is also known to favour $\alpha \to 0$ and to
disagree with human-judged topic quality (Chang et al. 2009), so a perplexity
table is not a topic-quality table.

**8. Gibbs `transform` is stochastic, so fits are not reproducible bit for
bit.** The demo's perplexity changes by $0.05$ when the topic order is
permuted — not because the model changed but because `_infer_gibbs` draws its
own random numbers and the chain's trajectory does. `variational_bayes` has no
such issue: its transform is deterministic and permutation-invariant to exactly
$0.0$.

**9. Online LDA with $\kappa = 0$ is not online LDA.** $0.996$ against $0.687$
on a topic-sorted stream. The failure is silent on shuffled data, which is why
it survives testing: i.i.d. minibatches hide the missing averaging, and the
first deployment on a real stream — where order carries information — exposes
it.

**10. Documents with no tokens break the mixture.** $N_d = 0$ makes (6)'s first
factor $0/0$; `_infer_gibbs` returns the uniform mixture instead, which is the
right limit and a reminder that $\theta_d$ is unidentifiable without data. The
same happens to a document whose every term is unseen at fit time.

**11. Everything here is dense and single-machine.** `X` is $D \times V$ dense,
`phi` is $K \times V$ per document, and the sampler is a Python loop. For
$D = 10^6$ and $V = 10^5$ this file is off by several orders of magnitude in
both memory and time; the algorithms survive (sparse counts, sparse `phi`,
minibatches) and the code does not.

## Applications

- **Topic discovery in text.** The original application and still the most
  common: a corpus of abstracts, reviews or news, and the output is $K$ ranked
  word lists plus per-document mixtures. The mixtures are often more useful than
  the topics — they are a low-dimensional document representation that a
  downstream regression can use without ever reading the words.
- **Population genetics.** Pritchard, Stephens & Donnelly (2000) introduced
  this model independently as "admixture" for genotype data, where a document
  is an individual, a word is an allele, and $\theta_d$ is that individual's
  ancestry proportions. `structure` and `admixture` are LDA with a different
  likelihood, and the $K$ problem there is the famous "how many ancestral
  populations" question.
- **Recommendation and collaborative filtering.** Users as documents, items as
  words: the mixture is a taste profile and each topic is a co-purchase
  pattern. It behaves like a non-negative factorisation with a probabilistic
  story, and it inherits the same popularity bias.
- **Image and audio categorisation.** "Words" are visual codewords from a
  quantised feature space (SIFT descriptors, MFCC frames); a "document" is an
  image or a clip. The bag-of-words assumption is even more obviously wrong here
  and it works anyway, which is worth knowing.
- **Seeded and supervised variants.** Seeds (§4) are how LDA is used when some
  labels exist; the same idea with a per-token label likelihood is supervised
  LDA, and the same idea with a time index is a dynamic topic model. The prior
  is the extension point, which is why this file's `seeded_lda` is a variant
  rather than a separate atom.
- **Diachronic and comparative linguistics.** Cognate sets, sound changes and
  language families: a "document" is a word form in a language, the topics are
  sound correspondences. The model's lack of word order is an advantage here,
  because the correspondence is positional rather than linear.
- **As a baseline for embeddings.** A topic model is the cheap, interpretable
  representation: $K$ numbers per document, no GPU, and the weights are readable.
  Modern practice uses it to *describe* an embedding space — fitting LDA to
  clusters of contextual vectors is a standard way to say what a direction in a
  transformer means — which is the same trick in the other direction.

Where it is a trap: choosing $K$ by likelihood; comparing topics across fits
without permuting; treating perplexity as quality; seeding with frequent words
and calling the result semi-supervised; and using LDA on data that is not counts
of exchangeable tokens.

## References

- **Blei, Ng & Jordan (2003),** *Latent Dirichlet allocation* — the paper, with
  the mean-field derivation of §5 and the original variational inference. Note
  that it is variational, not Gibbs; the collapsed sampler came from the next
  entry.
- **Griffiths & Steyvers (2004),** *Finding scientific topics* — the collapsed
  Gibbs sampler of §2–§3, and the demonstration on PNAS abstracts that made LDA
  the default topic model.
- **Pritchard, Stephens & Donnelly (2000),** *Inference of population structure
  using multilocus genotype data* — the same model for genetic admixture, found
  independently, and still cited in that literature under the name `structure`.
- **Hoffman, Blei & Bach (2013),** *Stochastic variational inference* — §7:
  natural-gradient steps, the $(\tau_0 + t)^{-\kappa}$ schedule, and the
  minibatch rescaling. The online variant here is a direct transcription.
- **Jagarlamudi, Daumé III & Udupa (2012),** *Incorporating lexical priors into
  topic models using seeding* — seeded LDA, and the analysis of why seeds should
  be a prior rather than a constraint.
- **Teh, Newman & Welling (2007),** *A collapsed variational Bayesian inference
  algorithm for latent Dirichlet allocation* — the middle ground between §3 and
  §5: collapse $\theta$ and $\beta$ but keep $q(z)$ factorised, which is more
  accurate than mean field and faster than Gibbs.
- **Wallach, Mimno, Dicker & McCallum (2009),** *Rethinking LDA: why priors
  matter* — the $\alpha$ and $\beta$ sensitivity of §5 and of failure mode 5, in
  more detail than a demo can afford.
- **Chang, Boyd-Graber, Gerrish, Wang & Blei (2009),** *Reading tea leaves: how
  humans interpret topic models* — the result that perplexity and human-judged
  topic quality disagree, which is failure mode 7.
- **Blei & Lafferty (2006),** *Dynamic topic models* — what the same generative
  story becomes when $\beta$ is allowed to drift, and the clearest example of
  the prior as the extension point.
