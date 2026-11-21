# 03 · Kernels & instance-based

> Similarity is a model: margins, kernels, prototypes and neighbours.

Kernel methods turn inner products into hypothesis spaces, and instance-based methods turn the training set into the model. Together they explain why deep learning works (neural tangent kernels) and give you baselines that are hard to beat on small data.

<!-- catalog:begin -->
**13 atoms · 74 variants · 9 implemented**

| Atom | File | Tier | Variants | What it does |
|:--|:--|:--:|--:|:--|
| ✓ Support vector machine | [`svm.py`](svm.py) | 1 | 12 | Max-margin classification from the primal to a working SMO solver. |
| ✓ Kernel ridge | [`kernel_ridge.py`](kernel_ridge.py) | 1 | 6 | Representer theorem in closed form: the kernel least-squares solution. |
| ✓ k-nearest neighbours | [`knn.py`](knn.py) | 1 | 9 | Voting and regression with trees, weights and dimensionality caveats. |
| ✓ Support vector regression | [`svr.py`](svr.py) | 2 | 5 | Epsilon-insensitive support vector regression. |
| ✓ RBF networks | [`rbf_network.py`](rbf_network.py) | 2 | 5 | Radial basis units with k-means centres and least-squares readout. |
| ✓ Random features | [`random_features.py`](random_features.py) | 2 | 6 | Random kitchen sinks: Monte Carlo kernel approximation at scale. |
| ✓ One-class models | [`one_class.py`](one_class.py) | 2 | 4 | One-class SVM and SVDD: a boundary around normal data. |
| ✓ Nearest centroid | [`nearest_centroid.py`](nearest_centroid.py) | 2 | 4 | Shrunken centroids and prototypes as a strong, fast baseline. |
| ✓ Nystrom approximation | [`nystrom.py`](nystrom.py) | 3 | 4 | Low-rank kernel approximation from a landmark subset. |
| · Metric learning | [`metric_learning.py`](metric_learning.py) | 3 | 5 | Learn the distance instead of assuming it. |
| · Multiple kernel learning | [`multiple_kernel.py`](multiple_kernel.py) | 3 | 4 | Combine kernels and learn the weights. |
| · MMD & HSIC | [`mmd.py`](mmd.py) | 3 | 5 | Kernel two-sample and independence tests with permutation calibration. |
| · Learning vector quantisation | [`lvq.py`](lvq.py) | 3 | 5 | Prototype-based classifiers with attraction/repulsion updates. |

**Variant map** — every variant below is implemented in its atom's file and derived in its companion `.md`.

- **Support vector machine** ([`svm.py`](svm.py) · [svm.md](svm.md)) — `hard_margin` · `soft_margin` · `dual` · `smo` · `linear` · `polynomial` · `rbf` · `sigmoid` · `nu_svm` · `class_weight` · `ovo` · `ovr`
- **Kernel ridge** ([`kernel_ridge.py`](kernel_ridge.py) · [kernel_ridge.md](kernel_ridge.md)) — `linear` · `rbf` · `laplacian` · `polynomial` · `nystrom` · `leave_one_out`
- **k-nearest neighbours** ([`knn.py`](knn.py) · [knn.md](knn.md)) — `classification` · `regression` · `uniform` · `distance_weighted` · `kd_tree` · `ball_tree` · `cosine` · `edited_knn` · `condensed_knn`
- **Support vector regression** ([`svr.py`](svr.py) · [svr.md](svr.md)) — `epsilon_svr` · `nu_svr` · `linear` · `rbf` · `smo`
- **RBF networks** ([`rbf_network.py`](rbf_network.py) · [rbf_network.md](rbf_network.md)) — `exact_interpolation` · `kmeans_centers` · `ols_readout` · `normalized_rbf` · `grnn`
- **Random features** ([`random_features.py`](random_features.py) · [random_features.md](random_features.md)) — `rff` · `orf` · `laplacian` · `polynomial` · `sparse_jlt` · `nystrom`
- **One-class models** ([`one_class.py`](one_class.py) · [one_class.md](one_class.md)) — `ocsvm` · `svdd` · `kernel_ocsvm` · `nu_one_class`
- **Nearest centroid** ([`nearest_centroid.py`](nearest_centroid.py) · [nearest_centroid.md](nearest_centroid.md)) — `standard` · `shrunken` · `nearest_subclass` · `cosine`
- **Nystrom approximation** ([`nystrom.py`](nystrom.py) · [nystrom.md](nystrom.md)) — `standard` · `ensemble` · `streaming` · `with_kernel_ridge`
- **Metric learning** ([`metric_learning.py`](metric_learning.py) · [metric_learning.md](metric_learning.md)) — `lmnn` · `nca` · `itml` · `mvp` · `deep_metric`
- **Multiple kernel learning** ([`multiple_kernel.py`](multiple_kernel.py) · [multiple_kernel.md](multiple_kernel.md)) — `average` · `alignment` · `mkl_svm` · `easy_mkl`
- **MMD & HSIC** ([`mmd.py`](mmd.py) · [mmd.md](mmd.md)) — `quadratic` · `linear_time` · `mmd_rbf` · `hsic` · `permutation_test`
- **Learning vector quantisation** ([`lvq.py`](lvq.py) · [lvq.md](lvq.md)) — `lvq1` · `lvq2` · `lvq3` · `glvq` · `grlvq`
<!-- catalog:end -->

---

## How to read an atom

Each atom is two files with the same stem: `name.py` is the implementation,
`name.md` is the companion document (intuition, derivation, variant map,
complexity, failure modes, applications). Start with the `.md`.

See [`docs/conventions.md`](../docs/conventions.md) for the file contract and
[`docs/tiers.md`](../docs/tiers.md) for what the tier column means.
