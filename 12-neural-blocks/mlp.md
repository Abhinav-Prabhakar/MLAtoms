# MLP — the affine layer, and four ways to make it deep

`mlp.py` · Neural blocks · Tier 1 · 6 variants

## Intuition

An MLP is a stack of affine maps with a nonlinearity between them. Everything
else in this chapter — convolution, attention, gating, normalisation — is a
variation on how those two operations are arranged. The interesting question is
not what an MLP is, but what makes a stack of them trainable at depth, and that
is what the variants here measure.

Four ideas cover most of the design space:

- **The skip connection** adds the block's input to its output. It costs no
  parameters, and it gives the gradient an additive path to the input.
- **Pre-normalisation** applies layer norm on the way into the block, so the
  branch always sees a standardised input while the residual stream stays
  untouched.
- **Gating** multiplies one learned projection by a sigmoid of another. The
  output is a product of two functions of the input, which is a richer object
  than a single activation applied to one projection.
- **SwiGLU** replaces the sigmoid gate with a SiLU-activated branch multiplied
  by a linear branch, and projects the product back. It is the feed-forward
  layer of most current language models.

The demo fits all six variants to three interleaved spiral arms — a problem no
affine map can solve, because the classes are not linearly separable. On 300
training points and 60 held out, the affine model lands at $0.333$, exactly
chance. The five nonlinear stacks land between $0.983$ and $1.000$. That is the
easy half. The hard half is the depth study: at twelve layers the plain stack
reaches $0.833$ and the pre-normalised residual reaches $0.983$, but the
ordering flips at a different epoch budget, so the *architecture* claim has to be
made somewhere other than accuracy. It is made on the gradient path, which is
exact.

## Variants

| variant | block | depth | parameters | test accuracy |
|:--|:--|:--:|--:|--:|
| `linear` | affine head | 0 | 9 | 0.333 |
| `mlp` | affine, ReLU | 3 | 3363 | 1.000 |
| `residual_mlp` | affine, ReLU, skip | 3 | 3363 | 0.983 |
| `gated_mlp` | affine, ReLU gate | 3 | 6531 | 1.000 |
| `swiglu_mlp` | SiLU gate, product, projection | 3 | 9699 | 0.983 |
| `deep_mlp` | affine, pre-norm, skip | 6 | 6531 | 0.983 |

**`linear`** is the control. It has nine parameters and cannot separate three
spirals; its accuracy is chance and its role is to prove the dataset needs
nonlinearity.

**`mlp`** is a three-layer ReLU stack between the stem and the logits. It
reaches $1.000$ on the sixty held-out points.

**`residual_mlp`** adds $x$ to every hidden block. Parameter count is identical
to `mlp` — $3363$ against $3363$ — because the skip is free. Accuracy is
$0.983$, one point lower, which is noise.

**`gated_mlp`** multiplies a ReLU branch by a sigmoid gate. It carries $3168$
more parameters than the plain stack, all of them in the gate matrices, and
reaches $1.000$.

**`swiglu_mlp`** uses the SwiGLU form: two projections, one activated by SiLU,
their elementwise product, and a third projection back to the model width. It
is the largest model at $9699$ parameters and reaches $0.983$.

**`deep_mlp`** is the same width and parameter count as `mlp` but with
pre-normalised residual blocks, and the demo runs it at depth six. The depth
study below is the reason it exists.

The accuracy column is nearly saturated, which is deliberate: a spiral is a
solved problem, and the variants are not here to be ranked by a task this small.
The measurements that distinguish them are the parameter counts, the gate
statistics and the gradient path.

## Derivation

### One layer

$$
h=\phi\bigl(xW+b\bigr),\qquad x\in\mathbb R^{n},\ W\in\mathbb R^{n\times d}.
\qquad (1)
$$

The affine map alone cannot do anything a single matrix could not do. Stacking
without $\phi$ composes affines into an affine:

$$
W_2^{\top}(W_1^{\top}x+b_1)+b_2
=\tilde W^{\top}x+\tilde b,
\qquad (2)
$$

so a ten-layer linear network is exactly a one-layer linear network with a
different weight matrix. The demo's `linear` variant is the honest version of
that statement: nine parameters, chance accuracy.

### A stack

$$
h_0=x,\qquad h_\ell=\phi\bigl(h_{\ell-1}W_\ell+b_\ell\bigr),\qquad
\hat y=\mathrm{softmax}\bigl(h_LW_{\mathrm{head}}+b_{\mathrm{head}}\bigr).
\qquad (3)
$$

With ReLU and He initialisation, the forward variance is preserved in
expectation:

$$
\mathrm{Var}(w)=\frac{2}{n}
\quad\Longrightarrow\quad
\mathbb E\bigl[h^{2}\bigr]=\tfrac12 n\,\mathrm{Var}(w)\,\mathbb E\bigl[x^{2}\bigr]
=\mathbb E\bigl[x^{2}\bigr].
\qquad (4)
$$

The factor $1/2$ is the probability that a ReLU unit is active. Without it the
activations halve at every layer, and with He's $2/n$ the standard deviation of
the output is roughly constant across depth. The demo's `_Dense` uses $2/n$ for
ReLU and $1/n$ for the identity stem and head, which is the same rule with the
$1/2$ removed.

### Residual

$$
y=x+F(x),\qquad
\frac{\partial L}{\partial x}
=\frac{\partial L}{\partial y}\Bigl(I+\frac{\partial F}{\partial x}\Bigr).
\qquad (5)
$$

The identity term is the point. Expand the product over $L$ blocks and every
term that picks at least one $I$ survives; only the single term that picks
$\partial F/\partial x$ at every block has no identity in it. A gradient
therefore reaches the input through $2^{L}-1$ paths instead of one, and the
$I$-only path has gain exactly one at every depth.

That is what the demo measures. Its gradient-norm ratio from the logits to the
input is

| depth | 2 | 4 | 8 | 12 |
|:--|--:|--:|--:|--:|
| `mlp` | 1.252 | 1.156 | 0.883 | 1.518 |
| `deep_mlp` | 2.394 | 2.348 | 4.355 | 5.445 |

The plain column hovers around one whatever the depth; the residual column grows
because of the identity term. And when every branch is set to zero the residual
chain passes a gradient through twelve blocks with a norm ratio of exactly
$1.000000$ — not approximately, exactly, because the chain has become the
identity.

### Pre-normalisation

$$
y=x+F\bigl(\mathrm{LN}(x)\bigr),\qquad
\mathrm{LN}(x)=\frac{x-\mu}{\sqrt{\sigma^{2}+\epsilon}}.
\qquad (6)
$$

Normalising the input to the branch, rather than its output, keeps the residual
stream in whatever scale it has accumulated. A post-normalised block renormalises
$x+F(x)$ and therefore changes the stream at every layer; the pre-normalised
version leaves it alone. That is why pre-norm is the default in deep
transformers, and it is why (5) holds exactly here: the $I$ term is untouched by
the normalisation.

The normalisation itself is elementwise over the width, with the mean and
variance computed per row:

$$
\frac{\partial L}{\partial x}
=\frac{1}{\sigma}\Bigl(g-\bar g-\hat x\,\overline{g\hat x}\Bigr),
\qquad (7)
$$

with $g=\partial L/\partial \hat x$ and bars denoting means over the width. The
two subtraction terms are the gradients through $\sigma$ and $\mu$; dropping
either one produces a direction that still decreases the loss, which is exactly
why the error survives code review.

### Gating

$$
y=\phi\bigl(xW+b\bigr)\odot\sigma\bigl(xV+c\bigr).
\qquad (8)
$$

The gate is a second linear function of the input, squashed into $(0,1)$. The
product is not a nonlinearity applied to a linear map; it is a bilinear
interaction between two learned projections, which is why it costs an extra
$n d + d$ parameters per layer. The demo's fitted gate has mean $0.453$ with
$44.3\%$ of units above one half: it is modulating, not saturated.

SwiGLU replaces the sigmoid with a SiLU on the first branch and drops the squash
on the second:

$$
y=\bigl(\mathrm{SiLU}(xW_1+b_1)\odot(xW_3+b_3)\bigr)W_2+b_2,
\qquad (9)
$$

$$
\mathrm{SiLU}(u)=u\,\sigma(u),\qquad
\mathrm{SiLU}'(u)=\sigma(u)\bigl(1+u(1-\sigma(u))\bigr).
\qquad (10)
$$

The second branch is linear, so the product can take either sign and the layer
is not limited to a convex combination of two non-negative quantities. That
extra range is the reason SwiGLU outperformed the sigmoid GLU in the language
modelling comparisons that introduced it, at the cost of a third matrix per
layer.

### Backward pass

Each block stores what its gradient needs. For (1),

$$
\frac{\partial L}{\partial W}=x^{\top}\bigl(g\odot\phi'(z)\bigr),\qquad
\frac{\partial L}{\partial x}
=\bigl(g\odot\phi'(z)\bigr)W^{\top},\qquad z=xW+b,
\qquad (11)
$$

and for (8) the product rule splits the incoming gradient into a branch
gradient and a gate gradient:

$$
\frac{\partial L}{\partial a}=g\odot\sigma(v),\qquad
\frac{\partial L}{\partial v}=g\odot a\odot\sigma(v)\bigl(1-\sigma(v)\bigr),
\qquad (12)
$$

with $a=\phi(xW+b)$ and $v=xV+c$. Because both branches read $x$, the input
gradient is the sum of their contributions.

The loss is softmax cross-entropy over $K$ classes:

$$
L=-\frac1B\sum_{i=1}^{B}\log p_{i,y_i},\qquad
\frac{\partial L}{\partial z_i}=\frac{1}{B}\bigl(p_i-e_{y_i}\bigr),
\qquad (13)
$$

where $e_{y}$ is the one-hot label. The subtraction of the label is what makes
the gradient small when the prediction is right, and the division by $B$ is what
makes the reported loss and the gradient consistent.

### Parameter count

For input width $n$, hidden width $d$, depth $L$ and $K$ classes,

$$
P_{\mathrm{plain}}=nd+d+L(d^{2}+d)+dK+K,
\qquad (14)
$$

with $L(d^{2}+d)$ extra for the gate in (8) and $2L(d^{2}+d)$ extra for SwiGLU in
(9). At $n=2$, $d=32$, $L=3$, $K=3$ that is $96+3168+99=3363$ for `mlp`,
`residual_mlp` and `deep_mlp`, $6531$ for `gated_mlp` and $9699$ for
`swiglu_mlp`. The skip connection and the layer norm contribute nothing, which
is the whole argument for using them.

### Adam

$$
\theta\leftarrow\theta-\eta\,
\frac{m/(1-0.9^{t})}{\sqrt{v/(1-0.999^{t})}+\epsilon},\qquad
m=0.9m+0.1g,\quad v=0.999v+0.001g^{2},
\qquad (15)
$$

with $t$ the global minibatch step, not the epoch. Correcting by the epoch
number instead of the step leaves the first epoch badly scaled and is a common
off-by-one in hand-written loops.

## Code walkthrough

The initialisation is the only place where the activation changes the scale:

```python
        scale = np.sqrt(1.0 / inputs)
        if activation == "relu":
            scale = np.sqrt(2.0 / inputs)
```

Equation (4) is the reason for the branch: a ReLU kills half its inputs in
expectation, so the weight variance has to double to compensate. The identity
stem and head keep the LeCun scale, because they see every unit.

A dense layer keeps only what its backward pass needs:

```python
    def forward(self, x: np.ndarray):
        p = self.parameters
        function, derivative = ACTIVATIONS[self.activation]
        pre = x @ p[f"{self.name}.W"] + p[f"{self.name}.b"]
        return function(pre), (x, pre, derivative(pre))
```

The cache holds the input, the pre-activation and the already-evaluated
derivative. Evaluating $\phi'$ in the forward pass costs one array per layer and
saves recomputing it from the pre-activation in the backward pass — and, more
importantly, guarantees the two use the same value.

The backward pass is (11):

```python
        x, _pre, slope = cache
        p, g = self.parameters, self.gradients
        d_pre = gradient * slope
        g[f"{self.name}.W"][...] = x.T @ d_pre
        g[f"{self.name}.b"][...] = d_pre.sum(axis=0)
        return d_pre @ p[f"{self.name}.W"].T
```

The in-place `[...]` assignment matters: the optimiser holds references to the
same arrays, so rebinding `g[key]` to a new array would silently disconnect the
gradient from the parameter it belongs to.

The gate splits the incoming gradient into its two branches:

```python
        d_active = gradient * gate
        d_logits = gradient * active * gate * (1.0 - gate)
        d_pre = d_active * _relu_grad(pre)
        g[f"{self.name}.W"][...] = x.T @ d_pre
        g[f"{self.name}.b"][...] = d_pre.sum(axis=0)
        g[f"{self.name}.V"][...] = x.T @ d_logits
        g[f"{self.name}.c"][...] = d_logits.sum(axis=0)
        return d_pre @ p[f"{self.name}.W"].T + d_logits @ p[f"{self.name}.V"].T
```

`d_logits` is (12): the gradient into the gate, multiplied by the gate's own
derivative $\sigma(1-\sigma)$. Both branches read `x`, so the input gradient is
the sum of two matrix products.

SwiGLU applies the product rule one level deeper:

```python
        d_product = gradient @ p[f"{self.name}.W2"].T
        d_upper = d_product * value * _silu_grad(upper)
        d_value = d_product * _silu(upper)
```

The gradient splits symmetrically: the SiLU branch receives $d_{\text{product}}
\odot v$ and the linear branch receives $d_{\text{product}}\odot
\mathrm{SiLU}(u)$. Getting the two the wrong way round is the natural bug, and
it still trains — just to a worse optimum.

The residual wrapper is (5) plus a normalisation:

```python
    def backward(self, gradient: np.ndarray, cache):
        norm_cache, block_cache = cache
        d_normed = self.block.backward(gradient, block_cache)
        if self.normalise:
            d_normed = _layer_norm_backward(d_normed, norm_cache)
        return gradient + d_normed
```

The `gradient +` is the identity path, and it is added *after* the branch is
back-propagated through the normaliser. Adding before, then normalising the sum,
is the mistake that was in this file first: it makes the identity term subject
to the layer norm's Jacobian and breaks the exactness the section above relies
on.

Layer norm's backward is (7) written out:

```python
    centred, inverse = cache
    width = gradient.shape[-1]
    scaled = (gradient * centred).sum(axis=-1, keepdims=True)
    total = gradient.sum(axis=-1, keepdims=True)
    return (inverse / width) * (
        width * gradient - total - centred * inverse ** 2 * scaled)
```

`total` removes the gradient through the mean and `scaled` removes it through
the variance. The `inverse / width` prefactor carries the $1/\sigma$ and the
averaging over the width in one place.

The variant dispatch is a single if-chain, so the differences between the six
models are visible in one screen:

```python
                if self.variant == "mlp":
                    blocks.append(_Dense(name, self.width, self.width, rng))
                elif self.variant == "residual_mlp":
                    blocks.append(_Residual(_Dense(name, self.width,
                                                   self.width, rng)))
                elif self.variant == "gated_mlp":
                    blocks.append(_GLU(name, self.width, self.width, rng))
                elif self.variant == "swiglu_mlp":
                    blocks.append(_SwiGLU(name, self.width, self.width, rng))
                else:
                    blocks.append(_Residual(_Dense(name, self.width,
                                                   self.width, rng),
                                            normalise=True))
```

The stem is always a dense layer and the head is always an affine map; only the
hidden blocks change. That is why the parameter counts in the table follow (14)
exactly.

Cross-entropy is (13), with the label subtraction done in place:

```python
    probabilities = _softmax(logits)
    rows = np.arange(labels.size)
    loss = float(-np.mean(np.log(probabilities[rows, labels] + 1e-12)))
    gradient = probabilities.copy()
    gradient[rows, labels] -= 1.0
    return loss, gradient / labels.size
```

The `1e-12` only matters for a numerically zero probability, which Adam can
produce on a confidently wrong example. Without it the loss becomes infinite and
the run dies at the first such batch.

## Complexity

For batch size $B$, input width $n$, hidden width $d$, depth $L$ and $K$
classes, per forward pass:

- **Plain and residual stacks:** $O\bigl(B(nd+Ld^{2}+dK)\bigr)$. The residual
  adds an addition per layer and nothing else.
- **Gated and SwiGLU:** the same with the per-layer term multiplied by two or
  three, matching the parameter counts in (14).
- **Layer norm:** $O(BLd)$ for the means and variances, negligible against the
  matrix products.
- **Backward pass:** the same cost again, plus the transpose products.
- **One epoch:** the above times the number of minibatches. The demo's 300
  points at batch 64 is five steps per epoch, and six models at 300 epochs
  finish in about $2.5$ seconds.
- **Memory:** $O(BLd)$ for the caches, which is what makes the hand-written
  backward pass possible.

The quadratic term in $d$ is why width is expensive and depth is cheap: doubling
$d$ quadruples the per-layer cost, while doubling $L$ only doubles it. Modern
architectures spend their parameters on width inside a block and keep the block
count modest.

## Failure modes

**A stack without a nonlinearity is one matrix.** Equation (2) is exact, and the
demo's `linear` variant is the demonstration: nine parameters, chance accuracy.
Removing the activation from an "MLP" does not make it a weaker model, it makes
it a different model with the same expressive power as logistic regression.

**Sixty test points cannot rank six models.** One point is $1.7$ percentage
points of accuracy. The five nonlinear variants sit within two points of each
other, and `residual_mlp` at $0.983$ is not meaningfully worse than `mlp` at
$1.000$. A benchmark at this scale can show that a mechanism is not broken; it
cannot show that one mechanism beats another.

**Depth is a lottery at this size.** At depth twelve with 300 epochs the plain
stack reaches $0.833$ and the pre-normalised residual $0.983$; at 200 epochs the
same comparison came out the other way, and at depth eight both reach $1.000$.
The gradient path in (5) is a theorem; the accuracy ordering is a draw. Report
the former and treat the latter as an anecdote, which is why the demo asserts
the gradient ratio and only prints the accuracies.

**Dead ReLUs never come back.** A unit whose pre-activation is negative for every
input has zero gradient and zero output forever. At learning rate $0.01$ with
He initialisation this is rare on standardised inputs, but one bad batch — a
feature with a huge scale, or a label imbalance — can kill a large fraction of
the width. The signature is a training loss that stops moving while the weights
are still finite.

**A saturated gate is a dead gate.** The sigmoid in (8) has derivative at most
$0.25$, and at $|v|>6$ it is numerically zero. If the gate drifts to saturation
early, that branch stops learning and the layer degenerates into a masked linear
map. The demo measures the gate distribution for this reason: mean $0.453$ is
healthy, mean $0.02$ would not be.

**Standardisation is part of the model.** `fit` stores the training mean and
spread and `predict` reuses them. Fitting the scaler on the full dataset before
splitting leaks the test distribution into training, and the leak is invisible
in the accuracy because it inflates it. Split first, then fit.

**The epoch budget is a hyperparameter.** Every accuracy in this file is at 300
epochs with learning rate $0.01$. The depth study shows that the ordering can
flip with the budget, and a hand-written loop that never plots the loss will
report whichever number the stopping rule happened to land on.

**A minibatch that is smaller than the class count.** With batch size 64 over
three classes the gradient is well conditioned. At batch size 4 a minibatch can
miss a class entirely, the softmax gradient becomes nearly one-hot, and the
effective learning rate spikes. The last partial batch is the usual culprit
when a run diverges at the end of an epoch.

## Applications

The MLP is the tabular baseline. On most tabular problems a two-layer network
with standardised inputs and a tuned width is within a fraction of a point of
gradient-boosted trees, and it is the model to beat before reaching for anything
more elaborate.

The gated and SwiGLU blocks are the feed-forward half of every transformer. The
attention layer mixes across positions; the gated MLP mixes across features, and
the gate is what lets a feature be switched on or off depending on the context
in the residual stream.

The residual and pre-norm pattern is what makes depth possible at all. Every
modern deep network uses it, and the reason is (5): the identity path is the only
part of the network that is guaranteed to have a well-conditioned gradient at
any depth.

The affine layer alone is the right model when the problem really is linear —
small $n$, few samples, or a strong prior that the effect is additive. The demo's
`linear` variant is a reminder that "neural" is not a synonym for "better", and
that the first thing to check on a new dataset is whether the simple model is
already at the ceiling.

## References

- Rumelhart, D. E., Hinton, G. E., Williams, R. J. (1986). *Learning
  representations by back-propagating errors.* Nature. The backward pass in
  (11).
- Glorot, X., Bengio, Y. (2010). *Understanding the difficulty of training deep
  feedforward neural networks.* AISTATS. Initialisation scale and the
  activation-variance argument behind (4).
- He, K., Zhang, X., Ren, S., Sun, J. (2015). *Delving deep into rectifiers.*
  ICCV. The $2/n$ rule for ReLU.
- Ba, J. L., Kiros, J. R., Hinton, G. E. (2016). *Layer normalization.*
  arXiv:1607.06450. Equation (6) and its gradient.
- He, K., Zhang, X., Ren, S., Sun, J. (2016). *Deep residual learning for image
  recognition.* CVPR. The skip connection of (5).
- Dauphin, Y. N., Fan, A., Auli, M., Grangier, D. (2017). *Language modeling
  with gated convolutional networks.* ICML. The gated linear unit of (8).
- Shazeer, N. (2020). *GLU variants improve transformer.* arXiv:2002.05202.
  SwiGLU, equations (9)–(10).
- Kingma, D. P., Ba, J. (2015). *Adam: a method for stochastic optimization.*
  ICLR. Equation (15).
