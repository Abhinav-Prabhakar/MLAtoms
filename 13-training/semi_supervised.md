# Semi-supervised learning — seven ways to use unlabelled points

`semi_supervised.py` · Training · Tier 2 · 7 variants

## Intuition

Semi-supervised learning starts from a strange bargain: you have a few
labels, many inputs, and no extra information about what the missing labels
are. Everything it can exploit is already in the input distribution — the
shape of the data, the smoothness you impose on the model, and the guess that
nearby points share a label.

The seven variants here take two routes. Pseudo-labelling methods run the
model on unlabelled inputs and train on its own predictions: self-training
adds the confident ones as hard labels, soft pseudo-labelling keeps the
probabilities, FixMatch adds strong augmentation, MixMatch mixes labelled and
unlabelled batches. Consistency methods compare two views of the same input:
the mean teacher uses an exponential moving average of the weights, and the
ladder reconstructs clean activations from noisy ones. Label propagation
ignores the network and solves a graph problem.

On two moons with eight labels, the supervised network scores 0.690. The
consistency and graph methods are far ahead — mean teacher 0.985, label
propagation 0.998, ladder 0.733 — while every pseudo-labelling method lands
within 0.003 of the baseline. That is the result worth taking seriously: a
model trained on its own predictions has a fixed point, and a bad boundary is
a very stable fixed point.

## Variants

| variant | uses | signal | extra cost |
|:--|:--|:--|:--|
| `self_training` | hard pseudo-labels | confidence threshold | $R$ fits |
| `pseudo_label` | soft pseudo-labels | all unlabelled, weighted | one fit |
| `fixmatch` | strong augmentation | weak view gives the label | one fit |
| `mean_teacher` | EMA teacher | KL between two views | two forwards |
| `mixmatch` | sharpen + MixUp | mixed targets | one fit |
| `label_propagation` | kNN graph | closed-form diffusion | one solve |
| `ladder` | input noise | denoising cost | one noisy forward |

One learner, one labelled set, one unlabelled pool. The variants differ in
what they add to the cross-entropy, not in the network: a 2→32→32→2 tanh MLP
trained with Adam and weight decay.

## Derivation

### The objective

With labelled data $(\mathcal L)$ and unlabelled inputs $(\mathcal U)$, the
target is still classification accuracy, but the loss has two terms:

$$
L(\theta)=\underbrace{\sum_{(x,y)\in\mathcal L}
\ell\!\left(f_\theta(x),y\right)}_{\text{supervised}}
+\lambda\underbrace{\sum_{u\in\mathcal U}
\rho\!\left(f_\theta,u\right)}_{\text{unsupervised}},
\qquad (1)
$$

and every variant below is a different $\rho$. The unlabelled term carries no
label information, so it can only shape the function between the labelled
points. If that shape is wrong, more unlabelled data makes the model more
confidently wrong.

### Hard pseudo-labels

Self-training picks the model's own answer where it is confident:

$$
\hat y(u)=\arg\max_c p_\theta(c\mid u),\qquad
\mathcal U^{+}=\{u:\max_c p_\theta(c\mid u)\ge\tau\},
\qquad (2)
$$

and then retrains on $\mathcal L\cup\{(u,\hat y(u))\}$. Retraining from the
current weights rather than from scratch makes the procedure a fixed-point
iteration. On the demo it reaches 367 of 392 points at $\tau=0.9$ and the
accuracy does not move at all (0.690, the same as the baseline).

The reason is visible in the gradient. Once every accepted point satisfies
$p_\theta(u)\approx e_{\hat y(u)}$, the pseudo-label term contributes

$$
\nabla_W L_{\mathrm{pseudo}}
=\sum_{u\in\mathcal U^{+}}\left(p_\theta(u)-e_{\hat y(u)}\right)\phi(u)^\top
\approx 0,
\qquad (3)
$$

so the boundary is already a stationary point of its own labelling. The
threshold sweep makes the same point four ways: 389, 377, 367 and 320 labels
at thresholds 0.60, 0.80, 0.90 and 0.99, and 0.690 accuracy every time. The
network is overconfident almost everywhere, so the threshold selects a
quantity of points, not a quality of labels.

### Soft pseudo-labels

Instead of committing, keep the distribution and weight it:

$$
L_{\mathrm{soft}}=\sum_{u\in\mathcal U}w(u)\,
H\!\left(p_\theta(u),\,p_{\bar\theta}(u)\right),
\qquad (4)
$$

where $\bar\theta$ is the model that produced the target and $w$ is the
variant's weight — 1.0 here. Soft targets carry the model's uncertainty, but
they carry no more information than the hard ones; the demo's soft variant
also lands at 0.690. Both are useful when the starting model is already good,
which is why they work in distillation-style pipelines and not from scratch.

### FixMatch

FixMatch makes the pseudo-label from a weak view and applies it to a strong
view of the same input:

$$
L_{\mathrm{fix}}=\sum_{u\in\mathcal U}
\mathbf{1}\!\left[\max_c p_\theta(c\mid \alpha_w(u))\ge\tau\right]
H\!\left(\hat y_w(u),\,p_\theta(\alpha_s(u))\right).
\qquad (5)
$$

The weak view here is a small jitter, the strong view a rotation, a rescale
and a larger jitter. The two views force the prediction to be invariant to
transformations that should not change the label, which is information the
model did not have from eight points. FixMatch reaches 0.785, and adding the
strong view is the only change from plain pseudo-labelling.

### Mean teacher

Keep a second copy of the weights and update it as an exponential moving
average of the student:

$$
\bar\theta_t=\gamma\bar\theta_{t-1}+(1-\gamma)\theta_t,\qquad
L_{\mathrm{mt}}=H\!\left(p_{\bar\theta}(\alpha_w(u)),\,
p_\theta(\alpha_s(u))\right).
\qquad (6)
$$

The teacher is an average over the whole training trajectory, so it is
smoother and more stable than the student; the student is pulled towards it on
the strong view. With $\gamma=0.99$ the teacher lags by about a hundred steps.
This is the strongest method in the demo: 0.985 against 0.690, from eight
labels, because the consistency term does not need the labels to be right —
only the invariance to hold.

### MixMatch

MixMatch sharpens the averaged unlabelled guesses and mixes them with the
labelled batch:

$$
\bar q(u)=\mathrm{Sharpen}\!\left(\frac{1}{K}\sum_{k=1}^{K}
p_\theta(u_k),\,T\right),\qquad
\tilde x=\lambda x_i+(1-\lambda)x_j,
\qquad (7)
$$

with $\lambda\sim\mathrm{Beta}(0.75,0.75)$ and the same mix applied to the
targets. Sharpening with $T=0.5$ turns a soft guess into a nearly one-hot
target, and MixUp interpolates both inputs and labels. On the demo it scores
0.693: the mixing regularises, but the targets still come from the same
boundary, so the fixed point survives.

### Label propagation

Build a k-nearest-neighbour graph over the labelled and unlabelled points and
diffuse the labels:

$$
F=(I-\alpha S)^{-1}Y,\qquad
S=D^{-1/2}WD^{-1/2},
\qquad (8)
$$

where $W_{ij}=\exp(-\lVert x_i-x_j\rVert^2/2\sigma^2)$ on the kNN edges, $D$
is its degree matrix, $Y$ holds the one-hot labels on labelled rows and zeros
elsewhere, and $\alpha\in(0,1)$ controls how far the diffusion travels. The
inverse is a closed-form solve, not an iteration, and it is the only method
here that never trains a network. At $k=7$, $\sigma=0.3$ and $\alpha=0.8$ it
scores 0.998: the two moons are two connected components of the graph, and
propagation finds them.

### The ladder

The ladder adds noise at each layer and asks the network to reconstruct the
clean activations. With one decoder at the top, the cost is

$$
L_{\mathrm{ladder}}=L_{\mathrm{sup}}
+\beta\sum_{u\in\mathcal U}
\left\lVert \tilde h_2(u)-h_2(u)\right\rVert^2,
\qquad (9)
$$

where $h_2$ is the clean second hidden layer and $\tilde h_2$ is the same
layer computed from a noisy input. The reconstruction term forces the hidden
representation to be stable under corruption, which is the same invariance
argument as FixMatch, applied one layer down. The full ladder reconstructs
every layer with its own decoder; this atom keeps one to stay readable, and
the demo's 0.733 shows the idea survives the simplification.

### The network

Every variant shares the same two-hidden-layer MLP:

$$
h_1=\tanh(W_1x+b_1),\quad h_2=\tanh(W_2h_1+b_2),\quad
z=W_3h_2+b_3,
\qquad (10)
$$

trained on soft targets $t$ with the weighted cross-entropy

$$
\ell=-\frac{1}{\sum_i w_i}\sum_i w_i\sum_c t_{ic}\log p_{ic}.
\qquad (11)
$$

The soft-target form is what lets the same training step serve hard labels,
soft pseudo-labels, mixed labels and consistency targets: only $t$ and $w$
change.

### What the numbers say

The demo's summary is:

| method | accuracy | pseudo-labels |
|:--|--:|--:|
| supervised only | 0.690 | 0 |
| `self_training` | 0.690 | 367 |
| `pseudo_label` | 0.690 | 367 |
| `fixmatch` | 0.785 | 339 |
| `mean_teacher` | 0.985 | 0 |
| `mixmatch` | 0.693 | 392 |
| `label_propagation` | 0.998 | 392 |
| `ladder` | 0.733 | 0 |

The methods that move are the ones whose loss adds information the model could
not have generated itself: an invariance, a graph, or a smoother teacher. The
methods that do not move are the ones whose targets are a function of the
current boundary.

## Code walkthrough

The augmentation is the only place the two views differ, so it is worth
reading first. The strong view rotates, rescales and jitters; the weak view
only jitters:

```python
def _augment(x: np.ndarray, rng: np.random.Generator,
             strong: bool) -> np.ndarray:
    """Jitter alone is weak; a rotation and a rescale make it strong."""
    values = x.copy()
    if strong:
        angle = rng.uniform(-0.6, 0.6, len(values))
        scale = rng.uniform(0.8, 1.2, len(values))
        cosine, sine = np.cos(angle), np.sin(angle)
        centre = values.mean(axis=0)
        shifted = values - centre
        values = np.column_stack([
            cosine * shifted[:, 0] - sine * shifted[:, 1],
            sine * shifted[:, 0] + cosine * shifted[:, 1]])
        values = values * scale[:, None] + centre
        values += rng.normal(0.0, 0.15, values.shape)
    else:
        values += rng.normal(0.0, 0.05, values.shape)
    return values
```

The network is deliberately small, and the backward pass is written once and
shared by every variant:

```python
def _backward(params: dict, cache, dlogits: np.ndarray) -> dict:
    x, first, second = cache
    del x, first
    grads = {"W3": second.T @ dlogits, "b3": dlogits.sum(axis=0)}
    d_second = (dlogits @ params["W3"].T) * (1.0 - second ** 2)
    grads.update(_backward_second(params, cache, d_second))
    return grads
```

The soft-target cross-entropy is (11) as a gradient, which is the single
function all seven variants call:

```python
def _loss_grad(params: dict, cache, target: np.ndarray,
               weight: np.ndarray) -> np.ndarray:
    """Weighted soft-target cross-entropy, returned as d(loss)/d(logits)."""
    logits = cache[2] @ params["W3"] + params["b3"]
    probs = _softmax(logits)
    return (probs - target) * weight[:, None] / weight.sum()
```

Self-training is the fixed-point loop of (2)–(3): the `added` mask stops a
point from being bought twice, and `n_added` reports the union over rounds:

```python
    def _self_training(self, x_lab, y_lab, x_unlab) -> None:
        added = np.zeros(len(x_unlab), dtype=bool)
        for _ in range(self.rounds):
            self._train(x_lab, y_lab, self.steps)
            probs = self._probs(x_unlab)
            confidence = probs.max(axis=1)
            fresh = (confidence >= self.threshold) & ~added
            keep = np.where(fresh)[0]
            if not keep.size:
                break
            added[keep] = True
            x_lab = np.vstack([x_lab, x_unlab[keep]])
            y_lab = np.concatenate([y_lab, probs[keep].argmax(axis=1)])
        self.n_added = int(added.sum())
```

FixMatch is the mask plus the strong view; note that the pseudo-label is read
from the *weak* probabilities and the gradient flows through the *strong*
ones:

```python
            weak = _augment(x_unlab, self.rng, strong=False)
            strong = _augment(x_unlab, self.rng, strong=True)
            probs = _softmax(_forward(self.params, weak)[0])
            confidence = probs.max(axis=1)
            mask = confidence >= self.threshold
            x = np.vstack([x_lab, strong[mask]])
            y = np.concatenate([y_lab, probs[mask].argmax(axis=1)])
```

The mean teacher computes the consistency gradient as `student - teacher`,
which is the derivative of $\mathrm{KL}(p_{\bar\theta}\,\Vert\,p_\theta)$ with
respect to the student logits, then adds it to the labelled gradient:

```python
            teacher = _softmax(_forward(self.teacher, weak)[0])
            _, cache = _forward(self.params, strong)
            student = _softmax(cache[2] @ self.params["W3"]
                               + self.params["b3"])
            consistency = _backward(self.params, cache, self.lam * (
                student - teacher) / len(strong))
```

Sharpening is a temperature applied to the averaged guesses, and MixUp mixes
both inputs and targets with the same coefficient:

```python
        guesses = np.mean([self._probs(x_unlab) for _ in range(2)], axis=0)
        sharp = guesses ** (1.0 / 0.5)
        sharp = sharp / sharp.sum(axis=1, keepdims=True)
```

```python
            fraction = self.rng.beta(0.75, 0.75, len(x))
            lam = np.maximum(fraction, 1.0 - fraction)[:, None]
            mixed_x = lam * x[order] + (1.0 - lam) * x[shuffle]
            mixed_y = lam * target[order] + (1.0 - lam) * target[shuffle]
```

Label propagation builds a symmetric kNN graph, normalises it, and solves (8)
in one linear solve:

```python
        adjacency = np.maximum(adjacency, adjacency.T)
        degree = adjacency.sum(axis=1) + 1e-12
        inverse = 1.0 / np.sqrt(degree)
        normalised = adjacency * inverse[:, None] * inverse[None, :]
        seed = np.zeros((len(pool), CLASSES))
        seed[np.arange(len(y_lab)), y_lab] = 1.0
        scores = np.linalg.solve(
            np.eye(len(pool)) - self.alpha * normalised, seed)
```

The ladder adds noise to the labelled inputs, computes the noisy hidden layer,
and backpropagates the reconstruction error from that layer alone — the clean
activation is the constant target:

```python
            _, clean_cache = _forward(self.params, x_unlab)
            noisy_x = x_unlab + self.rng.normal(0.0, 0.3, x_unlab.shape)
            _, noisy_cache = _forward(self.params, noisy_x)
            cost = 2.0 * self.lam * (noisy_cache[2] - clean_cache[2]) / len(
                x_unlab)
            denoise = _backward_second(self.params, noisy_cache, cost)
```

Inference for the graph method is a weighted vote of the nearest propagated
points, which is how a transductive method answers a question about a point it
never trained on:

```python
            gap = values[:, None, :] - self.pool_x[None, :, :]
            distance = np.sqrt(np.sum(gap ** 2, axis=-1))
            order = np.argsort(distance, axis=1)[:, :self.neighbours]
            weights = np.exp(-np.take_along_axis(
                distance, order, axis=1) ** 2 / (2.0 * 0.3 ** 2))
            scores = np.take_along_axis(
                self.pool_scores[None, :, :], order[:, :, None], axis=1)
            probed = np.sum(weights[:, :, None] * scores, axis=1)
```

## Complexity

With $n$ labelled points, $m$ unlabelled points, $d=2$ inputs, $h=32$ hidden
units and $R$ rounds:

- **Supervised training:** $O((n+m)h^2)$ per epoch, dominated by the hidden
  matrix; the demo's 400 full-batch steps on 400 points is a few million
  multiply-adds.
- **Self-training:** $R$ supervised runs, so $R$ times the cost, plus $R$
  predict passes over the pool.
- **Soft pseudo-labels, MixMatch:** one extra run over $n+m$ points with soft
  targets; the soft cross-entropy costs the same as the hard one.
- **FixMatch, mean teacher:** one fit, but two augmented forward passes per
  step (weak and strong), so roughly 2× the supervised cost. The teacher adds
  one more forward pass and a copy of the parameters.
- **Label propagation:** an $m\times m$ distance matrix, $O(m^2d)$, plus the
  dense solve $O(m^3)$. For 392 points this is instant; for a million points
  it is the reason people use sparse graph approximations or mini-batch
  diffusion.
- **The demo:** seven variants plus a threshold sweep and an agreement check,
  about three seconds on one core. The network is small enough that the
  augmentation and the linear solve, not the matrix multiplies, dominate.

## Failure modes

**Pseudo-labelling a weak model is a fixed point.** The demo's self-training,
soft pseudo-labels and MixMatch all land within 0.003 of the 0.690 baseline.
The gradient on confident points is already zero (3), so retraining reproduces
the same boundary; a bad boundary is a stable one. Pseudo-labelling helps when
the starting model is strong — a pretrained network, a large teacher — not
when it is the thing you are trying to improve.

**Confidence is not correctness.** The demo's threshold sweep accepts 320 to
389 of 392 points depending on the threshold, and every setting scores the
same. A 32-unit tanh network trained for 400 steps is confidently wrong in the
region it has never seen. Temperature scaling or an entropy penalty is the
usual mitigation; neither is in this file.

**Consistency needs the right invariances.** Mean teacher's 0.985 comes from
rotation and scale being label-preserving for two moons. On digits, rotation
is nearly free; on text, it is meaningless. A strong augmentation that changes
the label turns the consistency loss into a source of error, and the model
will happily learn the wrong invariance.

**The EMA teacher can stall.** With $\gamma$ close to one the teacher is almost
constant, so the student chases a target that never moves. With $\gamma$ small
the teacher is just the student and the term adds noise. The demo's 0.99 is a
starting point, not a rule; the right value depends on the number of steps.

**MixMatch's sharpening is a temperature, and it can overshoot.** $T=0.5$ turns
a 60/40 guess into roughly 69/31, which is mild; $T=0.1$ would turn it into a
near-one-hot target and reintroduce the pseudo-label fixed point. The paper's
value is tuned for image benchmarks.

**Graph methods are transductive and quadratic.** The solve is over the pool,
so a new point requires either a new solve or the kNN vote used here, and the
$O(m^3)$ cost rules out the full pool at scale. The graph also assumes that
nearby points share a label, which fails at class boundaries that pass through
dense regions.

**The graph can leak the test distribution.** The demo builds the graph from
labelled and unlabelled training points only, and classifies test points by a
kNN vote. Including the test inputs in the graph — the standard transductive
setup — would use their features without their labels; that is legal, but it
changes what "test accuracy" means and needs to be stated.

**Few labels means high variance.** The supervised baseline swings between
0.69 and 0.85 across label seeds, and the ranking of the weaker methods moves
with it. The demo fixes one seed; the table is a demonstration, not a
benchmark. A real comparison needs several seeds and error bars.

**Accuracy hides class imbalance.** Both moons are balanced, so overall
accuracy is a fair score. On an imbalanced pool, a model that predicts the
majority class everywhere can look respectable while being useless, and the
consistency losses will happily reinforce it.

**The unlabelled pool is not free.** The demo treats 392 points as available
without cost. In practice unlabelled data has to be collected, cleaned and
stored, and it can be out of distribution relative to the test set. The
regulariser is only as good as the distribution it is computed on.

## Applications

Pseudo-labelling is the workhorse of industrial semi-supervised pipelines: a
strong teacher labels a large corpus, a smaller student trains on the result,
and the loop repeats. Soft pseudo-labels and noisy-student training are the
same idea with a temperature, and they underpin most of the recent progress in
speech and vision with limited labels.

Consistency regularisation — the mean teacher, FixMatch, and their descendants
— is the standard recipe for image classification with a few labels per class,
and the same two-view trick appears in self-supervised pretraining. The ladder
is the historical ancestor of both the denoising autoencoder and the modern
consistency losses.

Label propagation and its relatives are used wherever the data is naturally a
graph: citation networks, social networks, molecules, and the kNN graphs built
over embeddings. It is also the cleanest illustration of the semi-supervised
bargain — no network, no training, just the assumption that the graph
neighbourhood carries the label.

## References

- Chapelle, O., Schölkopf, B., Zien, A. (2006). *Semi-Supervised Learning.*
  MIT Press. The taxonomy and the graph view, (8).
- Lee, D.-H. (2013). *Pseudo-label: the simple and efficient semi-supervised
  learning method for deep neural networks.* ICML Workshop. Hard
  pseudo-labels, (2).
- Tarvainen, A., Valpola, H. (2017). *Mean teachers are better role models.*
  NeurIPS. The EMA teacher and (6).
- Sohn, K., Berthelot, D., Li, C.-L., Zhang, Z., Carlini, N., Cubuk, E. D.,
  Kurakin, A., Zhang, H., Raffel, C. (2020). *FixMatch: simplifying
  semi-supervised learning with consistency and confidence.* NeurIPS. (5).
- Berthelot, D., Carlini, N., Goodfellow, I., Papernot, N., Oliver, A.,
  Raffel, C. (2019). *MixMatch: a holistic approach to semi-supervised
  learning.* NeurIPS. Sharpening and MixUp, (7).
- Berthelot, D., Carlini, N., Cubuk, E. D., Kurakin, A., Sohn, K., Zhang, H.,
  Raffel, C. (2020). *ReMixMatch: semi-supervised learning with distribution
  alignment and augmentation anchoring.* ICLR. The refined recipe.
- Zhu, X., Ghahramani, Z., Lafferty, J. (2003). *Semi-supervised learning
  using Gaussian fields and harmonic functions.* ICML. Label propagation,
  (8).
- Zhou, D., Bousquet, O., Lal, T. N., Weston, J., Schölkopf, B. (2004).
  *Learning with local and global consistency.* NeurIPS. The normalised
  Laplacian form of the same diffusion.
- Rasmus, A., Berglund, M., Honkala, M., Valpola, H., Raiko, T. (2015).
  *Semi-supervised learning with ladder networks.* NeurIPS. The denoising
  cost, (9).
- Oliver, A., Odena, A., Raffel, C., Cubuk, E. D., Goodfellow, I. J. (2018).
  *Realistic evaluation of deep semi-supervised learning algorithms.*
  NeurIPS. Why the comparisons in this area are harder than they look.
