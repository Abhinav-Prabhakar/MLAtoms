# Data augmentation — eight policies that make one image many

`augmentation.py` · Training · Tier 2 · 8 variants

## Intuition

A label says what an image *is*, not which pixels it must contain. A horizontal
bar shifted by one row is still the same class; a photograph with different
light is still the same object. Training on the handful of images a dataset
happens to contain teaches a model those exact pixels, and augmentation
declares the larger set the label actually covers.

The eight variants declare different sets. `mixup` and `cutmix` build new
training points *between* two labelled ones, with targets to match. `cutout`
and `random_erasing` hide part of an image so the model cannot rely on a single
region. `randaugment` and `auto_augment` compose several elementary transforms
into a policy with a small number of knobs. `augmix` mixes several random
chains of transforms and adds a consistency term. `color_jitter` changes
brightness, contrast, saturation and hue in a random order.

The demo ends with the sharpest version of the argument. Six labelled images
of a bar, randomly placed. A plain network reaches 0.468 on held-out data —
chance — because with uniformly random shifts the two classes have *identical*
pixel means, so no linear signal exists. Training with random translations on
the same six images reaches 1.000. Mixup and cutmix stay near chance: they
smooth a boundary, they do not invent a missing invariance.

## Variants

| variant | family | image | label | needs labels |
|:--|:--|:--|:--|:--|
| `mixup` | mixing | convex combination of two | mixed one-hot | yes |
| `cutmix` | mixing | a box from another image | mixed one-hot | yes |
| `cutout` | occlusion | one square zeroed | unchanged | no |
| `random_erasing` | occlusion | sampled rectangle | unchanged | no |
| `randaugment` | policy | $n$ ops at severity $m$ | unchanged | no |
| `augmix` | policy | Dirichlet mixture of chains | unchanged | no |
| `color_jitter` | photometric | four ordered factors | unchanged | no |
| `auto_augment` | policy | one sampled sub-policy | unchanged | no |

All eight accept a batch of `(B, H, W, C)` images in $[0,1]$ and return
`(images, targets, lam)`. Targets are always a soft `(B, classes)` matrix —
one-hot unless the variant mixes two rows — so one training loop can consume
every policy. `lam` is the weight of the untouched image and 1.0 for the
policies that do not mix.

## Derivation

### Mixup

Draw $\lambda\sim\mathrm{Beta}(\alpha,\alpha)$ and a permutation $\pi$:

$$
\tilde x=\lambda x_i+(1-\lambda)x_{\pi(i)},\qquad
\tilde y=\lambda e_{y_i}+(1-\lambda)e_{y_{\pi(i)}}.
\qquad (1)
$$

Because cross-entropy is linear in the target, training on $(\tilde x,\tilde y)$
is exactly

$$
L=\lambda\,\mathrm{CE}\!\left(f(\tilde x),y_i\right)
+(1-\lambda)\,\mathrm{CE}\!\left(f(\tilde x),y_{\pi(i)}\right),
\qquad (2)
$$

and the symmetric Beta has $\mathbb E[\lambda]=\tfrac12$: the demo's 400 draws
average 0.492. The pair is a sample from the line segment between two training
points, so the model is asked to interpolate its prediction along that segment.
That is the vicinal risk of (1) replacing the empirical risk of the isolated
points, and it is why mixup behaves like a regulariser rather than a source of
new information.

### CutMix

CutMix replaces a rectangle instead of blending everything:

$$
\tilde x_{i}[r]=\begin{cases}
x_{\pi(i)}[r], & r\in\text{box},\\
x_i[r], & \text{otherwise},
\end{cases}
\qquad
\lambda=1-\frac{h\,w}{H\,W},
\qquad (3)
$$

sampling $h=H\sqrt{1-\lambda_0}$ and $w=W\sqrt{1-\lambda_0}$ from a Beta draw
$\lambda_0$, then *recomputing* $\lambda$ from the clipped box. The recompute
matters: rounding $h,w$ to integers and clamping at the border changes the true
area, and the target must use the area that was actually pasted. The demo's box
is $7\times7$, so the returned weight is $1-49/64=0.234$, not the sampled value.

### Occlusion

Cutout and random erasing apply a mask:

$$
\tilde x=M\odot x,\qquad
\frac{|\{r:M_r=0\}|}{HW}=\text{coverage}.
\qquad (4)
$$

Cutout fixes the coverage and samples the position, which makes the erased
fraction exact — the demo's half-width square zeroes exactly 25% of an $8\times8$
image. Random erasing samples the area from $[s_l,s_h]$ and the aspect ratio
log-uniformly from $[r_l,r_h]$, then the position, and applies the mask with
probability $p$. Both are input-space dropout: they force the evidence for a
class to be redundant across the image.

### Policies over elementary ops

Let $\mathcal T$ be a set of transforms, each taking a severity $m\in[0,1]$.
RandAugment applies $n$ of them:

$$
\tilde x=T_{n}\circ\cdots\circ T_{1}(x),\qquad T_j\sim\mathrm{Unif}(\mathcal T),
\qquad (5)
$$

so the search space is two numbers instead of a table of sub-policies. The
demo's severity 0 is the identity by construction, and solarize flips a growing
share of a ramp as the severity rises: 0.00, 0.28, 0.62, 0.89.

AutoAugment keeps a table of sub-policies and samples one per image:

$$
\tilde x=\prod_{(T,p,m)\in\text{policy}} T^{\,\mathbb 1[u<p]}(x),
\qquad
\mathbb E[\#\text{ops}]=\sum_{(T,p,m)}p.
\qquad (6)
$$

With three two-op policies and every probability set to 1, the demo draws one
policy per image and applies all six ops' worth — 60 policies, 120 ops.

### AugMix

AugMix builds $k$ chains of depth $d$, mixes them with Dirichlet weights, and
adds a consistency term:

$$
\tilde x=\sum_{k}w_k\,b_k,\quad w\sim\mathrm{Dir}(\mathbf 1),\quad
b_k=T_{k,d}\circ\cdots\circ T_{k,1}(x),
\qquad (7)
$$

$$
\mathrm{JSD}(p_0,\dots,p_k)=
\frac{1}{k+1}\sum_{j}\mathrm{KL}\!\left(p_j\,\Big\|\,\bar p\right),\qquad
\bar p=\frac{1}{k+1}\sum_j p_j.
\qquad (8)
$$

The weights are non-negative and sum to one, so every mixed pixel is a convex
combination of the branches — the demo checks that bound pixel by pixel. The
JSD is zero for identical branches and positive otherwise (0.8128 here); in a
real run the model is trained so the prediction on the mixture agrees with the
predictions on the branches, which is what makes the invariance stick.

### Colour jitter

Four affine operations, applied in a random order with factors drawn around
one:

$$
x'=f_b\,x,\qquad
x'=(x-\mu)f_c+\mu,\qquad
x'=(x-g)f_s+g,
\qquad (9)
$$

where $\mu$ is the per-image channel mean and $g$ the per-pixel mean over
channels. Hue is a matrix:

$$
x'=H(a)x,\qquad
H(a)=\begin{pmatrix}
0.213+0.787c-0.213s & \cdots\\
\vdots & \ddots
\end{pmatrix},
\qquad (10)
$$

whose rows sum to one for every $a$: it moves colour around the wheel without
changing a grey pixel. The demo verifies exactly that, to $3.3\times10^{-16}$.
Order matters — contrast after brightness is not brightness after contrast —
which is why the four steps are shuffled rather than fixed.

### Why translation cannot be learned from six images

Take the two classes to be a white bar, horizontal or vertical, shifted
uniformly. For a linear model only the class means matter:

$$
\langle w,\mathbb E[x\mid y=0]\rangle
-\langle w,\mathbb E[x\mid y=1]\rangle
=\left\langle w,\mu_0-\mu_1\right\rangle,
\qquad (11)
$$

and averaging a bar over all shifts makes every pixel equally likely, so
$\mu_0=\mu_1$ and the expression is zero for every $w$: the linear part of any
model carries no signal here, and what is left is second-order structure that
six examples cannot pin down. The demo's shallow network lands at 0.468, which
is chance with sampling noise. Augmentation
changes the training distribution instead: averaging over the shift group
projects the learned feature onto the invariant subspace, and the same network
reaches 1.000 once the shifts are seen. Mixup interpolates *between* classes
along a line, which is a different subspace entirely — hence 0.476.

## Code walkthrough

Mixup is three lines: one Beta draw, one permutation, and the same convex
combination applied to pixels and targets:

```python
    def mixup(self, x: np.ndarray, labels: np.ndarray):
        lam = float(self.rng.beta(self.alpha, self.alpha))
        other = self.rng.permutation(x.shape[0])
        images = lam * x + (1.0 - lam) * x[other]
        target = lam * _onehot(labels, self.classes) + (1.0 - lam) * _onehot(
            labels[other], self.classes)
        return images, target, lam
```

CutMix overwrites a rectangle and then recomputes the weight from the box it
actually pasted — the line that keeps the target honest after integer rounding:

```python
        images = x.copy()
        images[:, top:top + box_h, left:left + box_w] = x[
            other, top:top + box_h, left:left + box_w]
        lam = 1.0 - box_h * box_w / float(height * width)
```

Cutout places one square per image and counts the pixels it removes, which is
how the demo knows the fraction is exactly the coverage:

```python
        for index in range(x.shape[0]):
            top = int(self.rng.integers(0, height - box_h + 1))
            left = int(self.rng.integers(0, width - box_w + 1))
            out[index, top:top + box_h, left:left + box_w] = 0.0
            self.counts["erased"] += box_h * box_w
```

Random erasing samples area and aspect before deriving the box, and the fill
mode decides whether the hole is black, the image mean, or noise:

```python
            area = self.rng.uniform(*self.erase_scale) * height * width
            aspect = np.exp(self.rng.uniform(np.log(self.ratio[0]),
                                             np.log(self.ratio[1])))
            box_h = min(height, max(1, int(round(np.sqrt(area / aspect)))))
            box_w = min(width, max(1, int(round(np.sqrt(area * aspect)))))
```

The severity gate lives in one place, so every policy inherits the rule that
magnitude zero means no change:

```python
    def _apply(self, name: str, image: np.ndarray, magnitude: float,
               rng) -> np.ndarray:
        if magnitude <= 0.0:
            return image
        return OPS[name](image, magnitude, rng)
```

AugMix's whole structure is the weight matrix and the sum it drives; keeping the
branches on `self` is what lets the demo test the convexity bound:

```python
        weights = self.rng.dirichlet(np.ones(self.k), size=x.shape[0])
        branches = []
        for _ in range(self.k):
            branch = x.copy()
            for _ in range(self.depth):
                name = str(self.rng.choice(list(self.ops)))
                magnitude = float(self.rng.random()) * self.m / 10.0
                branch = self._apply(name, branch, magnitude, self.rng)
            branches.append(branch)
        mixed = sum(weights[:, index, None, None, None] * branch
                    for index, branch in enumerate(branches))
```

The JSD is the mean KL to the mean distribution, with a floor so an exact zero
does not produce an infinite logarithm:

```python
        stack = np.stack([np.asarray(p, dtype=float) for p in probs])
        mean = stack.mean(axis=0)
        terms = []
        for member in stack:
            safe = np.maximum(member, 1e-12)
            terms.append(np.sum(safe * (np.log(safe)
                                        - np.log(np.maximum(mean, 1e-12))),
                                axis=-1))
        return np.mean(terms, axis=0)
```

Colour jitter is a shuffled sequence of four affine maps; the shuffle is the
reason the loop is a loop and not four lines in a fixed order:

```python
        for step in self.rng.permutation(4):
            if step == 0:
                factor = self.rng.uniform(1.0 - self.brightness,
                                          1.0 + self.brightness, (batch, 1, 1, 1))
                out = out * factor
```

Hue is the only step that mixes channels, and it does so with a per-image $3
\times3$ matrix applied by `einsum`:

```python
                angle = self.rng.uniform(-self.hue, self.hue, batch)
                matrix = np.stack([_hue_matrix(value) for value in angle])
                out = np.einsum("bij,bhwj->bhwi", matrix, out)
```

The training demo's model is a small MLP whose only unusual feature is that its
target is a distribution, not an index:

```python
    def step(self, x: np.ndarray, target: np.ndarray, lr: float) -> None:
        probs, hidden = self.probabilities(x)
        difference = (probs - target) / x.shape[0]
        grad_pre = (difference @ self.W2.T) * (1.0 - hidden ** 2)
        self.W2 -= lr * hidden.T @ difference
        self.b2 -= lr * difference.sum(axis=0)
        self.W1 -= lr * x.T @ grad_pre
        self.b1 -= lr * grad_pre.sum(axis=0)
```

## Complexity

For a batch of $B$ images of size $H\times W\times C$ with $N=BHWC$ values:

- **Mixing:** mixup is one multiply and one add over $N$, plus $O(BC)$ for the
  targets. CutMix copies at most $N$ values and recomputes one area.
- **Occlusion:** cutout and erasing are $O(Bhw)$ writes, where $hw$ is the box
  area; everything else is a copy of the batch.
- **Policy:** each elementary op is $O(N)$ — `rot90` and `roll` are views or
  copies, posterize and solarize are elementwise. RandAugment costs $O(nN)$,
  AugMix $O(kdN)$, AutoAugment $O(B\,|p|\,N/B)=O(|p|N)$.
- **Photometric:** four passes over $N$; hue adds a $3\times3$ contraction.
- **Memory:** every policy needs one extra batch-sized copy, and AugMix holds
  $k$ of them plus the branches.
- **The demo:** 60 images of $8\times8\times3$ (1920 values each), four recipes
  at 600 steps over five seeds. The transforms are microseconds; the 12,000
  full-batch training steps dominate and still finish in about a second.

## Failure modes

**A geometric op that changes the label is not augmentation.** The demo's
`rotate` turns a horizontal bar into a vertical one, which flips the class; it
is safe only because no demo trains on it. Rotation, shears and crops need
label transforms or a dataset where orientation carries no meaning.

**Mixup's permutation must be per batch, not per sample.** Applying the same
permutation to images and targets is the definition; permuting only the images
produces corrupted labels, and permuting with a fresh draw for the target mixes
the wrong pair.

**The mixing weight must match the pixels.** CutMix's returned $\lambda$ has to
be recomputed after clipping (3); using the sampled $\lambda_0$ leaves the
target inconsistent with the image by the rounding error.

**Alpha at the extremes is not mixing.** As $\alpha\to0$ the Beta concentrates
at 0 and 1 and mixup is the identity; as $\alpha\to\infty$ it concentrates at
$1/2$ and every pair becomes the same 50/50 blend. Both are legitimate settings
and neither is the default one.

**Coverage near one destroys the sample.** Cutout with coverage 0.9 leaves a
sliver of the image; the target survives but the input cannot support it, and
the loss turns into noise. Random erasing's upper area bound exists for the same
reason.

**Colour jitter clips.** Brightness or contrast can push values outside $[0,1]$,
and the clip loses the highlight rather than scaling it. In an 8-bit pipeline
the same clip is a real information loss.

**Order is part of the policy.** Contrast is defined against the image mean, so
applying it after brightness gives a different result from before. AutoAugment
encodes the order in the sub-policy; a shuffled `color_jitter` deliberately
does not.

**AugMix without the consistency term is just a random blend.** The JSD in (8)
is what ties the mixture's prediction to the branches'. Drop it and the policy
is a weaker form of RandAugment with a larger compute bill.

**Augmenting before splitting leaks.** Applying random crops before the
train/test split puts near-duplicates of test images into training. Augment the
training split only, and generate the test set once, unmodified.

**Test-time augmentation changes calibration.** Averaging predictions over
augmented test views usually helps accuracy and shifts the confidence scale;
thresholds tuned without it no longer apply.

**Augmentation cannot invent information.** The demo is the proof: six images
with a missing invariance stay at chance under smoothing and jump to 1.000 only
when the invariance is supplied. Mixing and erasing regularise; they do not
teach a symmetry the training distribution never expresses.

**Seeded policies are not reproducible across frameworks.** Which op a policy
draws depends on the order of the random stream. The demo fixes one seed per
augmenter; a data loader with several workers needs one stream per worker.

## Applications

RandAugment, AutoAugment and AugMix are the standard ImageNet recipes, usually
combined with mixup or CutMix; the four together are worth several points of
top-1 accuracy over plain training. Mixup and CutMix also mix bounding boxes
and masks, which is why they are standard in detection and segmentation rather
than only in classification.

Occlusion policies double as robustness training: a model trained with cutout
degrades less under real occlusion, and random erasing is used in person and
vehicle re-identification where occluders are common. Colour jitter is the
cheap way to buy invariance to lighting and camera white balance, and the CSS
hue matrix of (10) is the same operator used in browser filters.

Outside vision, mixup applies to tabular data and to audio spectrograms, and
augmentation is the engine of semi-supervised learning: FixMatch pairs a weak
and a strong augmentation of the same image and trains the strong view to match
the weak view's pseudo-label, which is the consistency idea of (8) taken to its
conclusion.

## References

- Zhang, H., Cissé, M., Dauphin, Y. N., Lopez-Paz, D. (2018). *mixup: beyond
  empirical risk minimization.* ICLR. Equations (1)–(2).
- Yun, S., Han, D., Oh, S. J., Chun, S., Choe, J., Yoo, Y. (2019). *CutMix:
  regularization strategy to train strong classifiers with localizable
  features.* ICCV. The box and the recomputed weight, (3).
- DeVries, T., Taylor, G. W. (2017). *Improved regularization of convolutional
  neural networks with cutout.* arXiv:1708.04552. Equation (4).
- Zhong, Z., Zheng, L., Kang, G., Li, S., Yang, Y. (2020). *Random erasing
  data augmentation.* AAAI. Area and aspect sampling.
- Cubuk, E. D., Zoph, B., Shlens, J., Le, Q. V. (2020). *RandAugment: practical
  automated data augmentation with a reduced search space.* CVPR. Equation (5).
- Cubuk, E. D., Zoph, B., Mane, D., Vasudevan, V., Le, Q. V. (2019).
  *AutoAugment: learning augmentation strategies from data.* CVPR. Sub-policies
  and (6).
- Hendrycks, D., Mu, N., Cubuk, E. D., Zoph, B., Gilmer, J., Lakshminarayanan,
  B. (2020). *AugMix: a simple data processing method to improve robustness and
  uncertainty.* ICLR. Equations (7)–(8).
- Shorten, C., Khoshgoftaar, T. M. (2019). *A survey on image data augmentation
  for deep learning.* Journal of Big Data. The taxonomy behind the table.
- Beyer, L., Hénaff, O. J., Kolesnikov, A., Zhai, X., van den Oord, A. (2020).
  *Are we done with ImageNet?* arXiv:2006.07159. Label-changing augmentations
  and the cost of treating them as free.
