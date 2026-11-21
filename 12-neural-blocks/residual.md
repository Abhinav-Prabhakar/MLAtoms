# Residual blocks — the identity path and six ways to dress it up

`residual.py` · Neural blocks · Tier 1 · 7 variants

## Intuition

A residual block computes a correction, not a replacement:

$$
y = x + F(x).
\qquad (1)
$$

The addition is free. It costs no parameters, one array add, and it changes
the gradient from a product of Jacobians into a sum that contains an identity.
That single term is why a network with a hundred layers can still be trained:
the gradient can always travel through $x$ untouched, whatever the branch does.

The variants here are the standard ways of arranging $F$ and the identity:

- **`basic`** is two affine layers with a ReLU between, added back.
- **`bottleneck`** squeezes the width down, works there, and projects back up,
  which makes a block cheaper than two full-width matrices.
- **`preactivation`** normalises the input to the branch, so the residual
  stream stays in whatever scale it has accumulated.
- **`densenet`** concatenates instead of adding, so the identity path is a
  widening buffer that every later block can read.
- **`highway`** learns a per-unit gate that mixes branch and identity.
- **`scaled_residual`** multiplies the branch by one learned scalar,
  initialised at zero, so training starts from a pure identity.
- **`stochastic_depth`** drops whole branches per sample during training, which
  makes the expected network shallower and acts as a depth-wise regulariser.

The demo trains all seven on two concentric rings — 320 points to fit, 80 held
out. Every variant reaches $1.000$ on the held-out set. That is the easy
measurement. The informative ones are structural: a zeroed branch returns its
input exactly for four of the variants; the bottleneck costs $0.32$ of the
basic block at width 16 and depth 12 ($2098$ against $6610$ parameters);
`densenet` grows from width 16 to 40 and every block's first sixteen columns
are its input verbatim; the twelve learned scalars of `scaled_residual` start at
zero and end between $-0.183$ and $0.207$; the first highway gate has mean
$0.488$ and spans $0.174$ to $0.899$, so it is a mixture rather than a switch;
and stochastic depth keeps $0.909$ of block 5's branches per draw while
evaluation uses the full branch.

## Variants

| variant | branch | shortcut | parameters |
|:--|:--|:--|--:|
| `basic` | two affine, ReLU | add | 6610 |
| `bottleneck` | down, inner, up | add | 2098 |
| `preactivation` | normalise, two affine | add | 6610 |
| `densenet` | one affine, ReLU | concatenate | 802 |
| `highway` | two affine, ReLU | learned gate | 9874 |
| `scaled_residual` | two affine, ReLU | add, scaled | 6622 |
| `stochastic_depth` | two affine, ReLU | add, dropped | 6610 |

The counts are for the demo: width 16, depth 12, and for `densenet` growth 2,
so its final width is 40. The stem, head and biases are included.

**`basic`** is the ResNet-18 block and the reference implementation. At width
16 it has $2(16\cdot16+16)=544$ parameters per block; twelve blocks plus the
stem and head give $6610$.

**`bottleneck`** is the ResNet-50 block: project to a quarter width, apply the
expensive transform there, project back. At width 16 the narrow width is 4, and
a block costs $16\cdot4+4\cdot4+4\cdot16+24=168$ parameters, so the whole
network is $2098$ — a third of the basic stack. The saving is real but it
depends on the width being large enough that a quarter is still useful.

**`preactivation`** moves the normalisation before the branch. It has the same
parameters as `basic` because the layer norm here has no learned gain. The
difference is where the norm sits, and that is enough to change how the
residual stream scales with depth.

**`densenet`** replaces the sum with a concatenation. Each block appends two new
channels, so block $i$ reads width $16+2i$ and writes $16+2(i+1)$. The stack's
parameter count is small ($802$) because every transform is narrow, but the
activations grow linearly and the compute grows quadratically in depth.

**`highway`** adds a third matrix that produces a gate in $(0,1)$ per unit. With
$3\cdot(16\cdot16+16)=816$ parameters per block it is the most expensive
variant here ($9874$), and it is the only one where the shortcut itself depends
on the input.

**`scaled_residual`** adds one scalar per block, initialised at zero. At
initialisation the model is exactly its stem and head; the branch is admitted
only as far as the scalar grows. It costs twelve parameters over `basic`.

**`stochastic_depth`** keeps the basic block and randomly zeroes the whole
branch per sample, rescaling by the survival probability so the expectation is
unchanged. At evaluation the branch is always present and unscaled.

## Derivation

### The basic block and its gradient

$$
y=x+W_2\,\phi\bigl(W_1x+b_1\bigr)+b_2 .
\qquad (2)
$$

For upstream gradient $g=\partial L/\partial y$,

$$
\frac{\partial L}{\partial x}
=g+W_1^{\top}\Bigl(\bigl(gW_2^{\top}\bigr)\odot\phi'(z)\Bigr),
\qquad z=W_1x+b_1 .
\qquad (3)
$$

The first term is the identity path and it is added *after* the branch is
back-propagated. That ordering is the whole architecture. Unroll a stack of $L$
blocks:

$$
\frac{\partial L}{\partial x_0}
=g\prod_{\ell=1}^{L}\Bigl(I+\frac{\partial F_\ell}{\partial x_{\ell-1}}\Bigr).
\qquad (4)
$$

Expanding the product gives $2^L$ terms. Exactly one of them is the product of
all the Jacobians, the term that would be the whole gradient in a plain stack;
every other term contains at least one $I$ and therefore does not vanish when
the Jacobians are small. The identity term itself has gain one at every depth,
which is the formal version of "the gradient has a highway home".

### Preactivation

$$
y=x+F\bigl(\mathrm{LN}(x)\bigr),
\qquad
\mathrm{LN}(x)=\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}} .
\qquad (5)
$$

Normalising the branch input leaves the residual stream untouched: the stream
accumulates the branch outputs at whatever scale they arrive with, and only the
branch sees a standardised input. The alternative, normalising $x+F(x)$, changes
the stream at every layer and makes the identity path pass through the
normaliser's Jacobian. The demo's zeroed-branch check still returns $x$ exactly
here, because the norm is applied inside the branch and the branch is zero.

The backward pass has to respect that split. The branch gradient goes through
the layer-norm Jacobian; the identity gradient does not. Adding them before the
normalisation is the natural-looking mistake, and it makes the identity term
depth-dependent.

### Bottleneck

$$
y=x+W_3\,\phi\Bigl(W_2\,\phi\bigl(W_1x+b_1\bigr)+b_2\Bigr)+b_3,
\qquad
W_1\in\mathbb R^{d\times d/4},\ W_2\in\mathbb R^{d/4\times d/4}.
\qquad (6)
$$

The parameter count per block is

$$
d\cdot\frac d4+\frac{d^2}{16}+\frac d4\cdot d
=\frac{9d^2}{16},
\qquad (7)
$$

against $2d^2$ for the basic block, a ratio of $9/32\approx0.28$. The demo's
measured ratio is $0.32$ because the biases and the stem and head do not shrink.
The depth of the bottleneck is what buys the width: at a fixed parameter budget
one can afford many more bottleneck blocks than basic ones.

### Dense connections

$$
y=\bigl[x;\ \phi(Wx+b)\bigr],
\qquad W\in\mathbb R^{d\times g},
\qquad d_{\ell}=d_0+\ell g .
\qquad (8)
$$

The semicolon is concatenation. Block $\ell$ reads all $d_0+\ell g$ channels
produced by every earlier block, so gradients reach early layers through
$L-\ell$ concatenation paths as well as through the transforms. The parameter
cost is linear in depth for a fixed growth $g$, but the activation memory and
the per-block arithmetic are quadratic, because block $\ell$ multiplies a
$d_0+\ell g$ input.

The demo checks the definition directly rather than a proxy: the first sixteen
columns of a block's output equal its input, element for element.

### Highway and gated shortcuts

$$
y=T\odot F(x)+(1-T)\odot x,
\qquad T=\sigma\bigl(xW_t+b_t\bigr).
\qquad (9)
$$

When $T\to1$ the block is a plain residual branch; when $T\to0$ it is the
identity and the branch is switched off. The gate is a function of the input,
so the network can choose per example and per unit how much transformation to
apply. That is the difference from a fixed skip: the shortcut has parameters.

The gradient inherits the mixture:

$$
\frac{\partial L}{\partial T}=g\odot\bigl(F(x)-x\bigr),
\qquad
\frac{\partial T}{\partial z}=T\,(1-T),
\qquad z=xW_t+b_t .
\qquad (10)
$$

The derivative of the sigmoid is at most $0.25$, so a saturated gate learns
slowly in both directions. The demo's fitted gates span $0.174$ to $0.899$ with
mean $0.488$, which is unsaturated and genuinely mixed.

### Scaled residual

$$
y=x+\alpha\,F(x),
\qquad
\frac{\partial y}{\partial x}=I+\alpha\,\frac{\partial F}{\partial x}.
\qquad (11)
$$

With $\alpha=0$ the block is exactly the identity and the gradient to the
branch is zero; only $\alpha$ moves on the first step, then the branch follows.
This is ReZero's trick, and it makes a deep stack trainable without a warm-up
because no branch can inject a large signal before it has learned one. The same
scalar can be initialised at $1/\sqrt{L}$ instead of zero, which keeps the
initial variance roughly constant rather than exactly zero.

### Stochastic depth

For a block with survival probability $p$, draw $m\sim\mathrm{Bernoulli}(p)$
per sample and use

$$
y=x+\frac{m}{p}\,F(x),
\qquad
\mathbb E[y]=x+F(x).
\qquad (12)
$$

Training sees a random subnetwork whose expected depth is $pL$; evaluation sees
the full network with no rescaling, which is why the expectation has to be
preserved during training. The usual schedule decays linearly,

$$
p_\ell=1-\frac{\ell}{L-1}\,(1-p_L),
\qquad (13)
$$

from $p_0=1$ at the first block to $p_L$ at the last. The demo's $p_L=0.8$ over
twelve blocks gives $p_5=1-0.2\cdot5/11=0.909$, the value it prints.

The catch is the variance. Training with branches dropped changes the scale of
the activations relative to evaluation, so a network without normalisation
between blocks can lose accuracy at test time. That is not a hypothetical: with
$p_L=0.5$ this file's stochastic-depth model scored between $0.825$ and $0.925$
at 60 epochs while the other six sat at $1.000$, and raising $p_L$ to $0.8$ was
enough to fix it. Batch or layer normalisation is the usual remedy; a cleaner
demo is a smaller amount of noise.

### Parameter counts

For width $d$, depth $L$, growth $g$ and a stem from $n$ features to $d$ into
$K$ classes:

$$
P_{\mathrm{basic}}=nd+d+L(2d^2+2d)+dK+K,
\qquad (14)
$$

with $L(3d^2+3d)$ for the highway gate,
$L\bigl(9d^2/16+3d/2\bigr)$ for the bottleneck, and
$Ld_0g+g^2L(L-1)/2+Lg$ for a dense stack before the stem and head. At $n=2$,
$d=16$, $L=12$, $K=2$ the basic count is $6610$, matching the demo.

### Optimiser

$$
\theta\leftarrow\theta-\eta\,
\frac{m/(1-0.9^{t})}{\sqrt{v/(1-0.999^{t})}+\epsilon},
\qquad
m=0.9m+0.1g,\quad v=0.999v+0.001g^2,
\qquad (15)
$$

with $t$ the minibatch step. The demo's 320 points at batch 32 is ten steps per
epoch, and the bias correction is what makes the first ten steps usable.

## Code walkthrough

The basic block keeps the input for the skip:

```python
    def forward(self, x, *, training=True, rng=None):
        p = self.parameters
        pre = x @ p[f"{self.name}.first.W"] + p[f"{self.name}.first.b"]
        hidden = _relu(pre)
        branch = hidden @ p[f"{self.name}.second.W"] + p[f"{self.name}.second.b"]
        return x + branch, (x, pre, hidden)
```

The cache is only what the backward pass needs: the input, the pre-activation
and the activated hidden state. The sum is done once, in the return.

```python
        g[f"{self.name}.first.W"][...] = x.T @ d_pre
        g[f"{self.name}.first.b"][...] = d_pre.sum(axis=0)
        return gradient + d_pre @ p[f"{self.name}.first.W"].T
```

Equation (3) in three lines. The `gradient +` is the identity path; `d_pre`
already carries the ReLU mask and the second layer's transpose. In-place
assignment keeps the gradient array connected to the optimiser's reference.

Preactivation moves the norm inside the branch:

```python
        normed, norm_cache = _layer_norm(x)
        pre = normed @ p[f"{self.name}.first.W"] + p[f"{self.name}.first.b"]
        hidden = _relu(pre)
        branch = hidden @ p[f"{self.name}.second.W"] + p[f"{self.name}.second.b"]
        return x + branch, (normed, norm_cache, pre, hidden)
```

The returned value is still `x + branch`, not `normed + branch`. The cache keeps
both the normalised input and its statistics, because the backward pass needs
to undo the normalisation *after* propagating through the two matrices.

The dense block changes only the join:

```python
        pre = x @ p[f"{self.name}.transform.W"] + p[f"{self.name}.transform.b"]
        branch = _relu(pre)
        return np.concatenate([x, branch], axis=1), (x, pre, branch)
```

```python
        width = x.shape[1]
        d_branch = gradient[:, width:] * _relu_grad(pre)
        g[f"{self.name}.transform.W"][...] = x.T @ d_branch
        g[f"{self.name}.transform.b"][...] = d_branch.sum(axis=0)
        return gradient[:, :width] + d_branch @ p[f"{self.name}.transform.W"].T
```

The split of the incoming gradient at `width` is the concatenation's backward
pass: the first part goes straight back to the input, the second goes through
the transform. The returned sum is the two paths meeting again.

The highway gate is a sigmoid and a convex combination:

```python
        gate = _sigmoid(x @ p[f"{self.name}.gate.W"]
                        + p[f"{self.name}.gate.b"])
        mixed = gate * branch + (1.0 - gate) * x
        return mixed, (x, pre, hidden, gate, branch)
```

Storing `branch` in the cache costs one array but saves recomputing it, and the
gate gradient in (10) needs the difference `branch - x`:

```python
        d_branch = gradient * gate
        d_gate = gradient * (branch - x)
        ...
        d_gate_pre = d_gate * gate * (1.0 - gate)
        g[f"{self.name}.gate.W"][...] = x.T @ d_gate_pre
        g[f"{self.name}.gate.b"][...] = d_gate_pre.sum(axis=0)
        d_x = gradient * (1.0 - gate) + d_gate_pre @ p[f"{self.name}.gate.W"].T
```

Both the branch and the gate read $x$, so the input gradient has three terms:
the direct shortcut `gradient * (1 - gate)`, the branch's contribution, and the
gate's contribution through `gate.W`.

The scaled block makes the scalar a parameter like any other:

```python
        alpha = p[f"{self.name}.alpha"]
        return x + alpha * branch, (x, pre, hidden, branch)
```

```python
        g[f"{self.name}.alpha"][...] = float(np.sum(gradient * branch))
        d_branch = gradient * alpha
```

The scalar's gradient is the inner product of the upstream gradient with the
branch, and the branch's own gradient is multiplied by `alpha`. At
initialisation that factor is zero, so `alpha` is the only thing that moves —
which is exactly the intended behaviour, not a bug.

Stochastic depth wraps the basic block and edits the gradient by the same mask:

```python
        keep = (rng.random((x.shape[0], 1)) < self.survival).astype(float)
        return x + keep * branch / self.survival, (cache, keep)
```

```python
    def backward(self, gradient, cache):
        basic_cache, keep = cache
        if keep is None:
            return super().backward(gradient, basic_cache)
        return super().backward(gradient * keep / self.survival, basic_cache)
```

The mask is per sample and broadcast over width, so a whole branch is either
present or absent for a given example. The `keep is None` branch is evaluation:
no mask, no rescale, and the full branch — the deterministic network that (12)
promises in expectation.

The variant dispatch is one constructor per block:

```python
            if self.variant == "basic":
                block = _Basic(name, current, rng)
            elif self.variant == "bottleneck":
                block = _Bottleneck(name, current, rng)
            elif self.variant == "preactivation":
                block = _PreActivation(name, current, rng)
```

and the loop tracks `current = block.out_width`, so only the dense variant
changes the width from one block to the next. The head is built after the stack
from whatever width the last block produced.

## Complexity

For batch $B$, width $d$, depth $L$, growth $g$ and $K$ classes:

- **Basic:** $O(BLd^2)$ per forward pass, the same again backward. This is the
  baseline all the others are measured against.
- **Bottleneck:** $O(BL\cdot\frac{9}{16}d^2)$; about $0.56$ times the basic
  arithmetic, while the demo's parameter ratio is $0.32$ because biases and the
  stem do not shrink.
- **Preactivation:** the same arithmetic plus $O(BLd)$ for the norms.
- **Dense:** block $\ell$ costs $O\bigl(B(d_0+\ell g)g\bigr)$, so the stack is
  $O(BLd_0g + BL^2g^2)$ — quadratic in depth. That is the price of keeping every
  earlier feature map.
- **Highway:** $O(BLd^2)$ with a factor $3/2$ over basic from the gate matrix.
- **Scaled:** one extra multiply and one scalar per block, negligible.
- **Stochastic depth:** the same as basic at training time; the saving comes
  from the blocks whose branch is dropped, which a dense implementation would
  have to skip explicitly rather than multiply by zero.
- **Memory:** $O(BLd)$ for the cached activations, and $O(BL^2g)$ for a dense
  stack because the concatenated buffers are never freed.

## Failure modes

**A residual add needs matching shapes.** When the shortcut changes width — a
stride, a channel change, a dense concatenation — the add is illegal and the
standard fix is a $1\times1$ projection on the shortcut. This file avoids the
issue by keeping width fixed outside `densenet`, which is why no projection
appears; a real ResNet at four stages needs one at every stage boundary.

**Pre-LN grows the residual stream.** Each block adds an unscaled branch to the
stream, so the stream's variance grows with depth and a final normalisation is
required before the head. Post-LN keeps the stream bounded but puts the
normaliser on the identity path, which is the arrangement that needed warm-up
in the original transformer. The two choices are not interchangeable.

**Zero-initialised scaling can stall.** With $\alpha=0$ the branch receives no
gradient at the first step, and if the learning rate is also tiny, $\alpha$
crawls and the block stays effectively dead for many epochs. ReZero works
because the loss gradient with respect to $\alpha$ is large; a schedule that
decays the learning rate before $\alpha$ has grown defeats it.

**A saturated highway gate is a plain network or an identity.** The sigmoid's
derivative peaks at $0.25$ and is numerically zero beyond $|z|\approx6$. Once
the gate saturates, the block stops choosing and the shortcut stops adapting.
The demo measures the gate distribution for exactly this reason; a mean near
$0.02$ or $0.98$ would be the warning sign.

**Stochastic depth changes the train/eval distribution.** Dropping branches
reduces the variance of the training activations; evaluation restores the full
branch and the variance jumps. Without normalisation the mismatch can cost
several points, as the demo's $p_L=0.5$ run showed. The rescaling by
$1/\mathrm{survival}$ fixes the mean but not the variance.

**Dense concatenation is quadratic.** Every block reads every earlier feature
map, so activation memory and arithmetic grow with the square of depth. A
DenseNet with a large growth rate is cheap in parameters and expensive in
FLOPs, which is the opposite of the usual trade.

**Bottlenecks can be too narrow.** At width 16 the quarter width is 4, and a
3-layer transform through width 4 loses information that no amount of depth
recovers. Real bottlenecks are used at widths of 256 and above, where a quarter
is still 64 channels.

**Finite differences fail at ReLU kinks.** A numerical gradient check on a ReLU
network can be wrong by $10^{-2}$ even when the analytic gradient is exact,
because a perturbation of $10^{-6}$ flips a unit whose pre-activation is within
$10^{-6}$ of zero. This file's check passes at $10^{-10}$ only with a smooth
stand-in for the activation; the ReLU version of the same check reported errors
of $0.04$ on gradients that were correct. Always check the smooth version
first, then the real one away from the kinks.

**Weight decay fights the scaling.** Decaying the branch weights pulls $F$ toward
zero and turns the block into an identity, while decaying the shortcut (if it
has one) pulls the identity itself toward zero. Modern practice excludes
normalisation and bias parameters from weight decay; residual scalars deserve
the same treatment.

## Applications

The basic and bottleneck blocks are the ResNet family, which is still the
default vision backbone and the reference architecture for anything with a
skip connection. A bottleneck at widths of 64 to 512 is the standard
image-classification block.

Preactivation is the transformer's block: normalise the input, apply attention
or a feed-forward network, add the result. Every large language model uses it,
and the reason is the one in (5): the residual stream has to stay a clean
highway for gradients across dozens of layers.

Dense connections are used where parameter efficiency matters more than FLOPs:
small medical imaging models, feature reuse in segmentation, and any setting
where the dataset is small enough that $802$ parameters is a feature.

Highway networks are the historical bridge between LSTMs and residual
networks, and the gated-shortcut idea survives in gated mixture-of-experts and
in any layer that learns to interpolate between its input and its output.

Scaled residual is ReZero and its relatives, used to train very deep
transformers without warm-up. Stochastic depth is DropPath in vision
transformers and EfficientNet-style convnets, where it is the standard
regulariser for deep stacks.

## References

- He, K., Zhang, X., Ren, S., Sun, J. (2016). *Deep residual learning for image
  recognition.* CVPR. The basic block and equation (1).
- He, K., Zhang, X., Ren, S., Sun, J. (2016). *Identity mappings in deep
  residual networks.* ECCV. Preactivation and the clean identity path of (5).
- Srivastava, R. K., Greff, K., Schmidhuber, J. (2015). *Highway networks.*
  arXiv:1507.06228. The gated shortcut of (9).
- Huang, G., Liu, Z., van der Maaten, L., Weinberger, K. Q. (2017). *Densely
  connected convolutional networks.* CVPR. The concatenation of (8).
- Huang, G., Sun, Y., Liu, Z., Sedra, D., Weinberger, K. Q. (2016). *Deep
  networks with stochastic depth.* ECCV. Equations (12) and (13).
- Szegedy, C., Ioffe, S., Vanhoucke, V., Alemi, A. (2017). *Inception-v4,
  Inception-ResNet and the impact of residual connections.* AAAI. Scaled
  residual branches.
- Bachlechner, T., Majumder, B. P., Mao, H., Cottrell, G., McAuley, J. (2021).
  *ReZero is all you need.* ICML. Equation (11) with $\alpha=0$.
- Kingma, D. P., Ba, J. (2015). *Adam: a method for stochastic optimization.*
  ICLR. Equation (15).
