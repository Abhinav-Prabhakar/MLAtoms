# Class imbalance — eight remedies for a skewed dataset

`class_imbalance.py` · Training · Tier 2 · 8 variants

## Intuition

A classifier trained on 300 majority and 30 minority points will usually
predict the majority class and be right 91% of the time. That number is worse
than useless: it hides the only thing you cared about. Balanced accuracy —
the average of the per-class recalls — is 0.717 for the same model, and the
minority recall is 0.467.

There are three places to intervene. Resample the training set: SMOTE
interpolates new minority points, ADASYN puts them where the minority is most
outnumbered, borderline-SMOTE only near the frontier, Tomek links and edited
nearest neighbours remove ambiguous majority points instead. Change the loss:
inverse-frequency weights give each class the same total mass, and focal loss
down-weights the easy examples that dominate a skewed gradient. Or keep the
model and move the decision threshold afterwards.

On the demo the plain model scores 0.717 balanced and 0.921 plain accuracy.
SMOTE reaches 0.873, class weights 0.872, threshold moving 0.867, focal 0.860,
and two under-samplers do less (0.777 and 0.688). The methods converge on the
same answer because they are all correcting the same prior; the differences
show up in what they cost and what they break.

## Variants

| variant | intervenes | changes | keeps |
|:--|:--|:--|:--|
| `smote` | data | interpolates minority points | all rows |
| `adasyn` | data | interpolation, density-weighted | all rows |
| `borderline_smote` | data | interpolation near the frontier | all rows |
| `tomek` | data | removes linked majority points | 318 of 330 |
| `enn` | data | removes misclassified points | 308 of 330 |
| `focal` | loss | $(1-p_t)^\gamma$ modulation | all rows |
| `class_weight` | loss | inverse-frequency weights | all rows |
| `threshold_moving` | decision | threshold tuned on a split | all rows |

One logistic classifier, eight treatments. The five resamplers run before the
fit, the two losses during it, and threshold moving after it; everything else
is shared.

## Derivation

### The metric

With prior $\pi=\Pr(y=1)$ and predictions $\hat y$, plain accuracy is

$$
\mathrm{acc}=\pi\,\mathrm{TPR}+(1-\pi)\,\mathrm{TNR},
\qquad (1)
$$

so at $\pi=0.091$ a model with $\mathrm{TPR}=0$ and $\mathrm{TNR}=1$ scores
0.909. Balanced accuracy removes the prior:

$$
\mathrm{BA}=\tfrac12\left(\mathrm{TPR}+\mathrm{TNR}\right),
\qquad (2)
$$

and it is the number every table in this atom reports. The demo's plain model
scores 0.921 plain and 0.717 balanced, with 0.467 minority recall.

### The plain model and its gradient

The classifier is a logistic regression on the two input features,
$p=\sigma(w^\top x+b)$, trained by full-batch gradient descent on

$$
L_{\mathrm{CE}}=-\frac1n\sum_i\left[y_i\log p_i
+(1-y_i)\log(1-p_i)\right],\qquad
\nabla_w L=\frac1n\sum_i (p_i-y_i)x_i.
\qquad (3)
$$

Each majority point contributes the same gradient as each minority point, so
the majority class sets the boundary. That is the whole mechanism of the skew,
and every loss-based remedy below is a different way of reweighting (3).

### Class weights

Give each class the same total mass:

$$
w_i=\frac{1}{2\,\mathrm{count}(y_i)},\qquad
\nabla_w L=\sum_i w_i(p_i-y_i)x_i .
\qquad (4)
$$

With 300 and 30 points the minority point carries $300/30=10$ times the weight
of a majority point, in the sense that the two class totals are equal. The
demo's `class_weight` uses exactly this and reaches 0.872, close to SMOTE's
0.873. It costs nothing at training time and nothing at inference time, which
is why it is the default first attempt.

### Focal loss

Focal loss multiplies the cross-entropy by a factor that vanishes for easy
examples:

$$
L_{\mathrm{focal}}=-\alpha_t\,(1-p_t)^{\gamma}\log p_t,
\qquad (5)
$$

with $p_t=p$ for a positive and $1-p$ for a negative. For $\gamma=2$ a point
the model already gets right at $p_t=0.9$ contributes $0.01$ of its loss, and
a point at $p_t=0.5$ contributes $0.25$. The gradient used in the code is the
exact derivative of (5) with respect to the logit; for a positive point,

$$
\frac{\partial L}{\partial z}
=(1-p)^{\gamma}\left[\gamma p\log p-(1-p)\right],
\qquad (6)
$$

and the negative case is the same expression with $p\leftrightarrow 1-p$. The
demo uses $\alpha=0.9$ for the minority, so focal combines the reweighting of
(4) with the hard-example focus of (5); without the $\alpha$ term it reaches
only 0.75, because down-weighting the easy majority is not the same as
up-weighting the rare class.

### Threshold moving

A model trained at prior $\pi$ estimates $p(y=1\mid x)$ under that prior. To
decide as if the classes were balanced, shift the odds:

$$
\frac{p'}{1-p'}=\frac{p}{1-p}\cdot\frac{1-\pi}{\pi},
\qquad (7)
$$

so $p'\ge\frac12$ exactly when $p\ge\pi$. The optimal threshold under balanced
accuracy is therefore

$$
\tau^\star=\pi,
\qquad (8)
$$

which for the demo is 0.091. The code does not use the closed form; it tunes
the threshold on a stratified 25% split of the training set and finds 0.10,
then retrains on everything. The agreement is not a coincidence: the closed
form is the population version of exactly the quantity the split estimates.

### SMOTE

SMOTE synthesises a point on the segment between a minority point and one of
its $k$ minority neighbours:

$$
x_{\mathrm{new}}=x_i+\lambda\left(x_{nn(i)}-x_i\right),\qquad
\lambda\sim U(0,1).
\qquad (9)
$$

The demo adds 270 points, bringing the set to 600 and the minority share to
0.5, and reaches 0.873. The assumption is that the minority class is locally
connected: interpolating between two minority points stays in the minority
region. When the minority is a set of separated clusters, or when the nearest
minority neighbour is across a majority region, (9) manufactures points in
places the data never occupied.

### Borderline-SMOTE and ADASYN

Borderline-SMOTE only interpolates from points that are in danger — minority
points whose $k$ nearest neighbours in the whole set are mostly majority:

$$
\mathcal D=\{i:y_i=1,\ \textstyle\sum_{j\in\mathrm{kNN}(i)}
\mathbf{1}[y_j=0]\ge k/2\}.
\qquad (10)
$$

ADASYN keeps all minority points but allocates synthetic points in proportion
to the same majority fraction:

$$
r_i=\frac{1}{k}\sum_{j\in\mathrm{kNN}(i)}\mathbf{1}[y_j=0],\qquad
n_i=\left\lfloor n_{\mathrm{new}}\frac{r_i}{\sum_j r_j}\right\rfloor .
\qquad (11)
$$

The demo's borderline variant adds 270 points from the danger set and reaches
0.847; ADASYN adds 261 and reaches 0.858. Both are more selective than plain
SMOTE, and both pay for it with fewer new points in the interior where the
class is already easy.

### Tomek links and edited nearest neighbours

A Tomek link is a pair of opposite-class points that are each other's nearest
neighbour:

$$
(i,j)\ \text{is a link}\iff nn(i)=j,\ nn(j)=i,\ y_i\ne y_j .
\qquad (12)
$$

Removing the majority member of each link cleans the boundary without touching
the class balance. Edited nearest neighbours is stronger: drop any point whose
class disagrees with the majority of its $k$ neighbours. The demo removes 12
majority points with Tomek (330 → 318, 0.777) and 22 with ENN (330 → 308,
0.688). Cleaning the frontier does not fix the prior, and ENN makes balanced
accuracy slightly worse by pulling the boundary further into the minority.

## Code walkthrough

SMOTE needs the minority set, its neighbour graph, and a uniform draw per
synthetic point:

```python
    picks = rng.choice(source, majority_count, replace=True)
    peers = order[picks, rng.integers(0, order.shape[1], majority_count)]
    fraction = rng.random((majority_count, 1))
    synthetic = minority[picks] + fraction * (minority[peers]
                                              - minority[picks])
    return (np.vstack([x, synthetic]),
            np.concatenate([y, np.ones(majority_count, dtype=int)]))
```

The borderline variant differs only in which rows become `source`: a point is
in danger when at least half of its full-set neighbours are majority:

```python
    if borderline:
        overall = _neighbours(x, k)
        index = np.where(y == 1)[0]
        danger = np.array([np.sum(y[overall[row]] == 0) * 2 >= k
                           for row in index])
        source = np.where(danger)[0]
        if not len(source):
            source = np.arange(len(minority))
```

ADASYN computes the same fraction as a continuous ratio and uses it as the
allocation:

```python
    overall = _neighbours(x, k)
    index = np.where(y == 1)[0]
    ratio = np.array([np.mean(y[overall[row]] == 0) for row in index])
    if ratio.sum() <= 0.0:
        ratio = np.ones(len(minority))
    share = np.floor(target * ratio / ratio.sum()).astype(int)
```

Tomek links are mutual-nearest-neighbour pairs; the majority member is the one
that goes, and a mutual check is what distinguishes a link from a one-sided
nearest neighbour:

```python
    drop = np.zeros(len(x), dtype=bool)
    for index in range(len(x)):
        partner = nearest[index]
        if y[index] != y[partner] and nearest[partner] == index:
            if y[index] == 0:
                drop[index] = True
            else:
                drop[partner] = True
    return x[~drop], y[~drop]
```

ENN keeps a point only when its neighbours agree with it:

```python
    votes = np.mean(y[nearest], axis=1)
    keep = (votes >= 0.5).astype(int) == y
    return x[keep], y[keep]
```

Class weights are the inverse-frequency rule of (4), normalised so the two
class masses are equal before the batch mean:

```python
    def _sample_weights(self, y: np.ndarray) -> np.ndarray:
        """Inverse frequency: each class contributes the same total mass."""
        counts = np.bincount(y, minlength=2).astype(float)
        return np.where(y == 1, 0.5 / max(counts[1], 1.0),
                        0.5 / max(counts[0], 1.0)) * len(y)
```

The focal gradient is written out rather than autodiffed, because the
modulation factor is the whole algorithm and hiding it behind a library would
hide the derivation:

```python
        if gamma == 0.0:
            derivative = probs - y
        else:
            true = np.where(y == 1, probs, 1.0 - probs)
            base = (1.0 - true) ** gamma
            positive = base * (gamma * probs * np.log(probs + 1e-12)
                               - (1.0 - probs))
            negative = probs * base * (1.0 - gamma * np.log(
                1.0 - probs + 1e-12))
            derivative = np.where(y == 1, positive, negative)
```

Threshold moving carves a stratified validation split, scans the grid, keeps
the best balanced accuracy, and refits on everything:

```python
        self.w, self.b = self._train(x[keep], y[keep])
        probs = self._probs(x[validation])
        best, chosen = -1.0, 0.5
        for threshold in np.linspace(0.02, 0.98, 25):
            score = _balanced(y[validation], probs >= threshold)
            if score > best:
                best, chosen = score, float(threshold)
        self.threshold_ = chosen
        self.w, self.b = self._train(x, y)
```

## Complexity

With $n$ training points, $k$ neighbours and $n_{\mathrm{new}}$ synthetic
points:

- **SMOTE, borderline-SMOTE, ADASYN:** one $n\times n$ distance matrix,
  $O(n^2)$, then $O(k)$ per synthetic point. The demo's 330 points make the
  matrix trivial; at a million points the exact kNN graph is the bottleneck
  and is usually replaced by an approximate index.
- **Tomek, ENN:** the same distance matrix and a scan, $O(n^2)$.
- **Class weights, focal:** no extra work — the weights are folded into the
  existing gradient, and the fit is $O(n)$ per step.
- **Threshold moving:** one extra fit on 75% of the data and a scan over 25
  candidate thresholds, $O(\text{steps})$ on top of one fit.
- **The demo:** eight treatments, a threshold sweep and a resampling table, all
  under a second. The logistic model is two parameters; the distance matrices
  are the only superlinear part, and at 330 points they are negligible.

## Failure modes

**Accuracy hides the failure.** The demo's plain model is 0.921 accurate and
0.717 balanced, with less than half the minority found. Any report that leads
with accuracy on a skewed problem is a report that has not measured the thing
that matters. Balanced accuracy, per-class recall and the confusion matrix are
the minimum.

**SMOTE interpolates through the wrong regions.** Equation (9) assumes the
minority class is locally connected. If a minority point's nearest minority
neighbour is on the other side of a majority cluster, the synthetic point
lands in majority territory and adds noise. In high dimensions the neighbour
graph is also much sparser, which is why SMOTE's gains shrink as the dimension
grows.

**Duplicate minority points make SMOTE a no-op.** If a minority point's
nearest neighbour is itself at distance zero, every interpolated point is that
point again. Real implementations deduplicate first; this demo's data is
continuous, so it does not arise.

**Under-sampling throws away the majority class.** Tomek removes 12 points and
reaches 0.777; ENN removes 22 and reaches 0.688, below the plain model. Both
clean the boundary, neither rebalances the prior, and ENN's extra removals
move the boundary in the wrong direction. Under-sampling is a tool for label
noise, not a remedy for skew.

**Focal loss without $\alpha$ is not reweighting.** The pure $(1-p_t)^\gamma$
factor down-weights easy examples for both classes, and with a 10:1 prior the
majority still dominates. The demo's unweighted focal run reaches 0.75; the
$\alpha=0.9$ version reaches 0.860. The `alpha` in (5) is what does the
rebalancing, and it is the parameter people forget.

**The threshold is tuned on data you may not have.** Equation (8) needs an
estimate of $\pi$; the demo estimates it from a split and finds 0.10 against
the true 0.091. With a small validation split that estimate is noisy, and with
a test set that has a different prior the tuned threshold is simply wrong.
Threshold moving is a prior correction, so it needs the deployment prior.

**The methods are not additive for free.** SMOTE inside a cross-validation
loop must be fitted on the training fold only; synthesising before the split
leaks minority structure into validation. The same is true of threshold
tuning. The demo's split is inside `fit`, which is the right place.

**Resampling changes calibration.** A model trained on a balanced set outputs
probabilities for that prior, not the deployment one. If you need calibrated
probabilities — for a cost-sensitive decision, say — resampling must be
followed by a prior correction, or the threshold moving must be redone.

**The minority class may be heterogeneous.** SMOTE and ADASYN treat the
minority as one blob; if it is three subpopulations, interpolation between
them creates points in the gaps. Cluster the minority first, or use a method
that conditions on the local density.

**A balanced test set is not a deployment set.** The demo's test set has the
same 10:1 skew as training, and balanced accuracy is invariant to the test
prior, which is why the numbers are comparable. If you report plain accuracy
on a rebalanced test set, you have changed the question.

**Class weights can destabilise a deep model.** In a neural network the
reweighted gradient is larger for rare batches, and with batch norm or a large
learning rate it can oscillate. The logistic model here has neither problem;
the same weights in a transformer need a warmup and a sanity check.

## Applications

Credit scoring, fraud detection, medical diagnosis and defect detection are
the canonical skewed problems, and class weighting plus threshold moving is
the standard first pass because both are one-line changes to an existing
pipeline. Focal loss is the default in dense object detection, where the
background anchors outnumber the objects by orders of magnitude, and in
segmentation, where the same imbalance appears per pixel.

SMOTE and its descendants are used when the minority is genuinely
under-sampled and the feature space is low-dimensional or well-behaved: tabular
credit data, sensor faults, rare-event logs. ADASYN and borderline-SMOTE are
the standard upgrades when the minority sits on a boundary. Tomek links and
ENN are used as cleaning steps in undersampling pipelines and as a
preprocessing step before SMOTE in SMOTE-ENN and SMOTE-Tomek.

The threshold is the cheapest lever of all, and it is also the one most often
forgotten. A model that outputs a probability and a separate decision rule can
be re-tuned for any prior without retraining, which is why the separation
between score and threshold matters in production.

## References

- Chawla, N. V., Bowyer, K. W., Hall, L. O., Kegelmeyer, W. P. (2002).
  *SMOTE: synthetic minority over-sampling technique.* JAIR. Equation (9).
- Han, H., Wang, W.-Y., Mao, B.-H. (2005). *Borderline-SMOTE: a new
  over-sampling method in imbalanced data sets learning.* ICIC. (10).
- He, H., Bai, Y., Garcia, E. A., Li, S. (2008). *ADASYN: adaptive synthetic
  sampling approach for imbalanced learning.* IJCNN. (11).
- Tomek, I. (1976). *Two modifications of CNN.* IEEE Transactions on Systems,
  Man, and Cybernetics. Tomek links, (12).
- Wilson, D. L. (1972). *Asymptotic properties of nearest neighbor rules using
  edited data.* IEEE Transactions on Systems, Man, and Cybernetics. ENN.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., Dollár, P. (2017). *Focal loss
  for dense object detection.* ICCV. Equations (5)–(6).
- King, G., Zeng, L. (2001). *Logistic regression in rare events data.*
  Political Analysis. Prior correction and the threshold shift, (7)–(8).
- Saito, T., Rehmsmeier, M. (2015). *The precision-recall plot is more
  informative than the ROC plot when evaluating binary classifiers on
  imbalanced datasets.* PLoS ONE. Why accuracy and ROC mislead.
- Johnson, J. M., Khoshgoftaar, T. M. (2019). *Survey on deep learning with
  class imbalance.* Journal of Big Data. The loss-level remedies at scale.
- He, H., Garcia, E. A. (2009). *Learning from imbalanced data.* IEEE
  Transactions on Knowledge and Data Engineering. The resampling taxonomy.
