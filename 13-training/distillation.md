# Knowledge distillation — eight ways to move knowledge between models

`distillation.py` · Training · Tier 1 · 8 variants

## Intuition

A trained teacher's logits are richer than its labels. The label says which
class is right; the logits also say which wrong classes the teacher considers
plausible, and by how much. Distillation trains a smaller student to reproduce
that whole distribution rather than the one-hot target, which transfers
information the labels never contained.

The eight variants differ in *what* gets transferred. `response` matches
softened logits — the original Hinton recipe. `feature` matches hidden states
through a learned regressor, as in FitNets. `relation` matches the pairwise
structure of a batch, so the student learns how examples relate rather than
what each one scores. `attention_transfer` matches the token attention maps.
`dkd` splits the logit loss into a target-class term and a non-target term, and
weights them separately.

The remaining three are procedures rather than new losses. `self_distill` uses
a slow exponential moving average of the student as its own teacher.
`born_again` trains a sequence of students where generation $g$ teaches
$g+1$ with no external teacher. `online_distill` trains two networks at once,
each distilling from the other, so no pretrained teacher is needed at all.

The demo trains a teacher on 140 examples, then gives a student only eight
labels. The hard-label student reaches 0.550 on held-out data; the distilled
student reaches 0.617, because it also sees the 132 unlabelled inputs through
the teacher's soft targets.

## Variants

| variant | transferred | extra parameters | needs labels |
|:--|:--|:--|:--|
| `response` | softened logits | none | no |
| `feature` | hidden states through a regressor | $H_tH_s$ | no |
| `relation` | pairwise distances in a batch | none | no |
| `self_distill` | logits of an EMA of itself | none | no |
| `born_again` | logits of the previous generation | none | no |
| `online_distill` | logits of a jointly trained peer | a second network | no |
| `attention_transfer` | token attention maps | none | no |
| `dkd` | target and non-target logit terms | none | yes |

Every variant accepts an optional label vector. When labels are present the
loss is `alpha` cross-entropy plus the transfer term; with `alpha = 1` the model
is trained normally, and with `alpha = 0` it is trained only by the teacher.

## Derivation

### The response loss

With temperature $T$, both networks produce softened distributions

$$
p_s=\mathrm{softmax}(z_s/T),\qquad p_t=\mathrm{softmax}(z_t/T),
\qquad (1)
$$

and the student minimises

$$
L=\alpha\,\mathrm{CE}(y,p_s)+(1-\alpha)\,T^2\,
\mathrm{KL}\bigl(p_t\,\|\,p_s\bigr).
\qquad (2)
$$

The $T^2$ factor exists because the KL gradient carries a $1/T$: differentiating
(1) and the KL gives

$$
\frac{\partial L_{\text{soft}}}{\partial z_s}
=\frac{T}{B}\,(p_s-p_t),
\qquad (3)
$$

so without the $T^2$ the soft term would shrink by $1/T$ as the temperature
rises and $\alpha$ would no longer mean what it says. With it, changing $T$
changes the *shape* of the target without changing the overall gradient scale.

Large $T$ flattens both distributions, which is the point: at $T=1$ the demo's
trained teacher has a top-class gap of 0.978, and at $T=4$ the gap is 0.565 with
entropy rising from 0.052 to 0.736. The non-target probabilities become large
enough to carry a usable signal.

### Feature distillation

FitNets matches an intermediate layer, with a regressor $R$ that maps the
teacher's width to the student's:

$$
L_f=\frac{1}{BTH}\bigl\|H_s-H_tR\bigr\|_F^2,\qquad
\frac{\partial L_f}{\partial R}
=-\frac{2}{BTH}H_t^{\top}\bigl(H_s-H_tR\bigr).
\qquad (4)
$$

$R$ is trained jointly with the student, so the student is free to choose any
linear re-encoding of the teacher's features. In the demo $H_t$ has width 8,
$H_s$ has width 6, and the regressor is $8\times6$; a student that reproduces
$H_tR$ scores exactly zero. This is the "hint" of FitNets, and the regressor is
why the student does not need the same architecture as the teacher.

### Relation distillation

RKD compares the geometry of a batch rather than individual outputs. With
$d_{ij}=\lVert z_i-z_j\rVert$ and $\tilde d=d/\mathrm{mean}(d)$,

$$
L_r=\frac{1}{N}\sum_{i<j}\bigl(\tilde d^{s}_{ij}
-\tilde d^{t}_{ij}\bigr)^2,\qquad N=\frac{B(B-1)}{2}.
\qquad (5)
$$

Normalising by the batch mean makes the loss invariant to a global rescale of
the logits, and summing over unordered pairs makes it invariant to the order of
the batch. The demo checks both: tripling both sets of logits and reversing the
batch leave the loss at 0.2007.

The gradient needs the normalisation differentiated too. With
$r=\tilde d^s-\tilde d^t$ and $c=\mathrm{mean}(r\odot\tilde d^s)$,

$$
\frac{\partial L_r}{\partial z_i}
=\sum_{j\neq i}w_{ij}\,(z_i-z_j),\qquad
w_{ij}=\frac{2\,(r_{ij}-c)}{N\,\mu_s\,d_{ij}},
\qquad (6)
$$

where $\mu_s=\mathrm{mean}(d^s)$. The matrix of weights is symmetric, so the
implementation builds it once and evaluates the sum as a row-sum times $z$
minus the matrix product.

### Attention transfer

For hidden states $H\in\mathbb R^{B\times T\times H}$, the attention map is

$$
A=\mathrm{softmax}\!\left(\frac{HH^{\top}}{\sqrt H}\right),\qquad
\hat A=\frac{A}{\lVert A\rVert_F},
\qquad (7)
$$

with the softmax along the key axis, so every row sums to one. The loss is
$\lVert\hat A_s-\hat A_t\rVert^2$ averaged over the batch. The row-stochastic
property is the useful part: the student learns *where* the teacher looks, not
what it represents. The backward pass is a softmax Jacobian followed by the
normalisation, and then the bilinear form contributes
$(dS+dS^{\top})H/\sqrt H$ because $S$ is symmetric in its two token axes.

### Decoupled knowledge distillation

DKD divides the logit loss into two terms that measure different errors. Let
$p_t$ be the target-class probability and let $q$ be the distribution on the
remaining classes after renormalising:

$$
q_i=\frac{p_i}{1-p_t}\quad (i\neq t),\qquad
L_{\text{TCKD}}=T^2\,\mathrm{KL}\bigl([p_t^{t},1-p_t^{t}]\,
\|\,[p_t^{s},1-p_t^{s}]\bigr),
\qquad (8)
$$

$$
L_{\text{NCKD}}=T^2\,\mathrm{KL}\bigl(q_t\,\|\,q_s\bigr).
\qquad (9)
$$

TCKD is a single binary decision: is the target class likely or not. NCKD is
the shape of the remaining mass, which is where the inter-class similarities
live. Their gradients w.r.t. the student logits are

$$
\frac{\partial L_{\text{TCKD}}}{\partial z_j}
=T\,c\,p_j\,(\delta_{jt}-p_t),\qquad
c=-\frac{p_t^{t}}{p_t^{s}}+\frac{1-p_t^{t}}{1-p_t^{s}},
\qquad (10)
$$

$$
\frac{\partial L_{\text{NCKD}}}{\partial z_j}
=T\,(q_{sj}-q_{tj})\ \ (j\neq t),\qquad 0\ \ (j=t).
\qquad (11)
$$

The demo separates the cases: a student that matches the teacher exactly scores
zero on both; a student that gets the target right but flattens the non-target
mass scores 0.167 TCKD and 0.031 NCKD; a student that gets the target wrong
scores 4.61 and 0.76. The paper's finding is that NCKD carries most of the
benefit, which is why the weights are exposed separately.

### Symmetric KL for online distillation

Deep mutual learning has no pretrained teacher: two networks each minimise
their own cross-entropy plus the KL of the other to themselves. The combined
soft loss is the symmetric divergence

$$
S=\mathrm{KL}(q\|p)+\mathrm{KL}(p\|q),
\qquad (12)
$$

and its gradient is *not* just the difference of the two distributions, because
each KL depends on both arguments:

$$
\frac{\partial S}{\partial z_s}
=\frac1T\Bigl[(p-q)+p\odot\bigl(\log(p/q)-S_p\bigr)\Bigr],
\qquad S_p=\mathrm{KL}(p\|q).
\qquad (13)
$$

Dropping the log-ratio term is a plausible-looking bug: the loss still falls
and the networks still train, but the finite-difference check reports an error
around $2\times10^{-2}$. The demo's version passes at $1.9\times10^{-10}$.

### Self-distillation and born-again

A mean teacher keeps a slow copy of the student and distils from it:

$$
\bar\theta\leftarrow\beta\bar\theta+(1-\beta)\theta,\qquad \beta=0.99.
\qquad (14)
$$

The teacher never receives gradients, so the target drifts slowly and acts as a
temporal ensemble. Born-again instead trains generations sequentially:

$$
\theta_{g+1}=\arg\min_\theta\ \alpha\,\mathrm{CE}(y;\theta)
+(1-\alpha)T^2\,\mathrm{KL}\bigl(p_{\theta_g}\,\|\,p_\theta\bigr),
\qquad (15)
$$

with generation 0 trained on labels alone. The demo's four generations on
noisy labels score 0.583, 0.633, 0.650 and 0.650, against 0.583 for one plain
run on the same data.

## Code walkthrough

Cross-entropy returns its own gradient, which every variant reuses as the hard
term:

```python
def _cross_entropy(logits: np.ndarray, labels: np.ndarray):
    probs = _softmax(logits)
    batch = logits.shape[0]
    loss = -np.mean(np.log(probs[np.arange(batch), labels] + 1e-12))
    grad = probs.copy()
    grad[np.arange(batch), labels] -= 1.0
    return loss, grad / batch
```

The KL helper returns the loss and the $T(p_s-p_t)/B$ gradient together. Keeping
them in one place is what makes the $T^2$ bookkeeping consistent:

```python
    log_probs = _log_softmax(student_logits / temperature)
    loss = temperature ** 2 * float(
        np.mean(np.sum(teacher_probs * (np.log(teacher_probs + 1e-12)
                                        - log_probs), axis=-1)))
    probs = np.exp(log_probs)
    grad = temperature * (probs - teacher_probs) / student_logits.shape[0]
    return loss, grad
```

Both networks are the same three lines, so the student and teacher differ only
in their parameters. The mean over tokens is the only place the two token
positions interact:

```python
    def _mlp(self, prefix: str, x: np.ndarray):
        p = self.parameters
        pre = np.einsum("bti,ih->bth", x, p[f"{prefix}.W1"]) + p[f"{prefix}.b1"]
        hidden = np.tanh(pre)
        logits = hidden.mean(axis=1) @ p[f"{prefix}.W2"] + p[f"{prefix}.b2"]
        return hidden, logits
```

The response branch is then one line, and the same code serves the EMA teacher
and the born-again teacher because both live in `self.parameters["teacher.*"]`:

```python
        if self.variant in ("response", "self_distill", "born_again"):
            probs_t = _softmax(cache["z_t"] / self.temperature)
            _, grad = _kl(cache["z_s"], probs_t, self.temperature)
            dz_s += soft * grad
```

Feature distillation adds a direct gradient to the hidden state, and the
regressor gradient is the outer product of the teacher's features and the
residual. Note the two different index patterns in the same block:

```python
        elif self.variant == "feature":
            difference = (cache["h_s"] - cache["h_t"] @ self.parameters[
                "regressor.W"])
            scale = 2.0 * soft / difference.size
            dh_s += scale * difference
            self.gradients["regressor.W"] += -scale * np.einsum(
                "bth,btd->hd", cache["h_t"], difference)
```

Relation distillation folds the normalisation into a symmetric weight matrix.
The row-sum-times-$z$ form is the vectorised version of
$\sum_j w_{ij}(z_i-z_j)$:

```python
        residual = normalised_s - normalised_t
        count = distance_s.size
        share = float(np.mean(residual * normalised_s))
        weight = 2.0 * (residual - share) / (count * mean_s)
        weight = weight / np.maximum(distance_s, 1e-12)
        spread = np.zeros((z_s.shape[0], z_s.shape[0]))
        spread[upper] = weight
        spread = spread + spread.T
        return spread.sum(axis=1)[:, None] * z_s - spread @ z_s
```

Attention transfer is three backward passes stacked: normalisation, softmax,
and the bilinear score. The last line symmetrises the score gradient because
$S_{ij}$ and $S_{ji}$ both touch $h_i$:

```python
        inner = np.sum(scaled_s * grad_scaled, axis=(1, 2), keepdims=True)
        grad_attention = (grad_scaled - scaled_s * inner) / norm_s
        inner = np.sum(attention_s * grad_attention, axis=-1, keepdims=True)
        grad_scores = attention_s * (grad_attention - inner)
        scale = 1.0 / np.sqrt(h_s.shape[-1])
        return (grad_scores + grad_scores.transpose(0, 2, 1)) @ h_s * scale
```

The DKD target term is the binary KL differentiated through the softmax. The
coefficient is large exactly when the student is confident and wrong:

```python
        coefficient = (-target_t / np.maximum(target_s, 1e-12)
                       + (1.0 - target_t)
                       / np.maximum(1.0 - target_s, 1e-12))
        target_grad = (temperature * coefficient[:, None] * probs_s
                       * (onehot - target_s[:, None]) / z_s.shape[0])
```

The mean-teacher update is the whole of `self_distill`; there is no gradient
path into the teacher at all:

```python
        for name in ("W1", "b1", "W2", "b2"):
            teacher = self.parameters[f"teacher.{name}"]
            student = self.parameters[f"student.{name}"]
            teacher *= decay
            teacher += (1.0 - decay) * student
```

## Complexity

Let $B$ be the batch, $T$ the tokens, $d$ the input width, $H_s,H_t$ the hidden
widths, $C$ the classes and $N=B(B-1)/2$.

- **Two MLP passes:** $O(BTdH+BHC)$ forward and the same backward, twice for
  `online_distill` and once otherwise.
- **Response, self, born-again, online:** $O(BC)$ for the KL and its gradient.
- **Feature:** $O(BTH_tH_s)$ for the regressor and its gradient; the largest
  per-example term when $H$ is wide.
- **Relation:** $O(B^2C)$ for the pairwise distances and $O(B^2)$ for the
  weight matrix, independent of the widths. It is the only variant that
  scales quadratically in the batch.
- **Attention transfer:** $O(BT^2H)$ for the scores and their backward.
- **DKD:** $O(BC)$, plus the two masked renormalisations.
- **The demo:** batches of 60 to 140, widths up to 16, eight variants. Each
  finite-difference loop evaluates one forward pass per parameter entry, and
  the training blocks run 600 steps over 140 examples. Everything is
  milliseconds except the fitting loops.

## Failure modes

**A weak teacher teaches weakness.** Every loss here is a divergence from the
teacher, so a teacher with 0.53 held-out accuracy caps what the student can
learn from soft targets. The demo's student beats its teacher only because the
hard labels anchor it; a student trained with `alpha = 0` would inherit the
teacher's errors directly.

**Temperature interacts with the loss weight.** The $T^2$ factor keeps the
gradient scale fixed, but it also scales the reported loss. Tuning $T$ and
$\alpha$ independently on the loss value rather than held-out accuracy is a
common way to pick a bad pair.

**The feature regressor can hide a bad student.** $R$ is trained at the same
time as the student, so it will happily absorb a mismatch the student should
have fixed — the loss goes to zero while the student's features stay nothing
like the teacher's. FitNets freezes the hint layer or fixes $R$ for this
reason.

**Relation losses are invariant to what matters most.** Because the distances
are normalised by their batch mean, the loss cannot transfer absolute logit
scale or a rotation of the feature space. That is desirable for robustness and
bad if the calibration itself is the knowledge.

**Attention maps with few tokens carry almost nothing.** With two tokens each
row of $A$ has one free parameter, so the transfer is a single number per
example. Attention transfer was designed for convolutional maps with hundreds
of spatial positions; here it demonstrates the mechanism, not its power.

**Self-distillation with a stale teacher drifts.** The EMA teacher lags by
$1/(1-\beta)$ steps. If the student diverges, the teacher keeps pulling it back
toward the divergence for a while. A very large $\beta$ is stable but slow; a
small one turns the teacher into a noisy copy of the student.

**Born-again is not a free win.** Each generation inherits its teacher's bias,
and with a small model on a small dataset the extra rounds can be flat or
slightly worse. The demo's first generation already sees every label; the
gain comes from soft targets averaging label noise, not from the recursion
itself.

**Online distillation needs the symmetric gradient.** Each network's loss
depends on both logits, so differentiating only the plain difference of
probabilities is wrong by the log-ratio term in (13). The loss still decreases,
which is why only a gradient check catches it.

**DKD needs the target index.** TCKD and NCKD are defined relative to the
ground-truth class, so `dkd` cannot run without labels; the constructor allows
`y=None` and the loss raises instead.

**Numerical zeros in KL.** A teacher that assigns probability exactly zero to a
class makes $\log p_t$ infinite. Every log here adds $10^{-12}$, which changes
the gradient slightly but keeps a saturated teacher from producing NaNs.

**Teacher and student must see the same preprocessing.** Soft targets are
computed from the teacher's view of the input. Augmenting the student's input
differently from the teacher's breaks the correspondence silently.

## Applications

Distillation is the standard way to compress a large model into a deployable
one: train the teacher once, then fit a smaller student to its outputs. It is
also the backbone of semi-supervised learning — Noisy Student, self-training
and mean-teacher methods all generate targets for unlabelled data from a model
and train on them. FitNets and attention transfer are used when intermediate
supervision is cheaper than matching outputs, for example when the student has
a different head.

DKD and its relatives are now standard in detection and segmentation, where
the background class dominates and the target/non-target split matters. Deep
mutual learning trains an ensemble without a pretrained teacher, and born-again
networks are a cheap way to squeeze another point from a fixed architecture.
The same recipes scale to language models, where a small model is distilled
from a large one's token distributions instead of its labels.

## References

- Hinton, G., Vinyals, O., Dean, J. (2015). *Distilling the knowledge in a
  neural network.* arXiv:1503.02531. The temperature and the response loss,
  equations (1)–(3).
- Romero, A., Ballas, N., Kahou, S. E., Chassang, A., Gatta, C., Bengio, Y.
  (2015). *FitNets: hints for thin deep nets.* ICLR. Equation (4).
- Park, W., Kim, D., Lu, Y., Cho, M. (2019). *Relational knowledge
  distillation.* CVPR. Distance-wise and angle-wise potentials, (5)–(6).
- Zagoruyko, S., Komodakis, N. (2017). *Paying more attention to attention.*
  ICLR. Attention transfer, equation (7).
- Zhao, B., Cui, Q., Song, R., Qiu, Y., Liang, J. (2022). *Decoupled knowledge
  distillation.* CVPR. TCKD and NCKD, (8)–(11).
- Zhang, Y., Xiang, T., Hospedales, T. M., Lu, H. (2018). *Deep mutual
  learning.* CVPR. The symmetric KL of (12)–(13).
- Tarvainen, A., Valpola, H. (2017). *Mean teachers are better role models.*
  NeurIPS. The EMA teacher of (14).
- Furlanello, T., Lipton, Z. C., Tschannen, M., Anandkumar, A. (2018).
  *Born-again neural networks.* ICML. The generation recursion of (15).
