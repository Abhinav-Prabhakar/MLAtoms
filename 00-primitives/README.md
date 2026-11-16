# 00 · Primitives

> Functions, losses, metrics, distances and numerics that every later folder composes.

Nothing here is a model. These are the atoms models are made of: activations, objectives, evaluation scores, initialisation, normalisation and the linear algebra that makes everything else a few lines long. Read this folder first; the rest of the repository assumes it.

<!-- catalog:begin -->
**12 atoms · 142 variants · 8 implemented**

| Atom | File | Tier | Variants | What it does |
|:--|:--|:--:|--:|:--|
| ✓ Activations | [`activations.py`](activations.py) | 1 | 17 | Pointwise nonlinearities with stable forward and exact backward passes. |
| ✓ Losses | [`losses.py`](losses.py) | 1 | 20 | Regression, classification, ranking and contrastive objectives in one place. |
| ✓ Metrics | [`metrics.py`](metrics.py) | 1 | 18 | Thresholded and threshold-free scores, including ranking and clustering. |
| ✓ Initialisation | [`init.py`](init.py) | 1 | 10 | Variance-preserving starts that decide whether deep nets train at all. |
| ✓ Normalisation | [`normalization.py`](normalization.py) | 1 | 8 | Batch, layer, group, instance and RMS normalisation plus weight standardisation. |
| ✓ Distances | [`distances.py`](distances.py) | 1 | 13 | Metric and non-metric dissimilarities used across the repository. |
| ✓ Linear algebra | [`linalg.py`](linalg.py) | 1 | 11 | The factorisations every other file is secretly using. |
| ✓ Sampling | [`sampling.py`](sampling.py) | 1 | 10 | Reservoir, alias, stratified, weighted and Gumbel-top-k sampling. |
| · Tensor ops | [`tensor_ops.py`](tensor_ops.py) | 2 | 10 | einsum patterns, im2col, gather/scatter, segment sums, log-sum-exp. |
| · Embedding encodings | [`embeddings.py`](embeddings.py) | 2 | 9 | Discrete-to-vector encodings, from one-hot to rotary positions. |
| · Similarity graphs | [`similarity_graphs.py`](similarity_graphs.py) | 2 | 8 | Turning points into edges — the first step of every graph and manifold method. |
| · Random streams | [`random.py`](random.py) | 3 | 8 | Reproducible randomness and variance reduction for Monte Carlo estimators. |

**Variant map** — every variant below is implemented in its atom's file and derived in its companion `.md`.

- **Activations** ([`activations.py`](activations.py) · [activations.md](activations.md)) — `relu` · `leaky_relu` · `prelu` · `elu` · `selu` · `gelu_tanh` · `gelu_erf` · `silu` · `mish` · `softplus` · `glu` · `swiglu` · `geglu` · `softmax` · `log_softmax` · `sparsemax` · `entmax`
- **Losses** ([`losses.py`](losses.py) · [losses.md](losses.md)) — `mse` · `mae` · `huber` · `log_cosh` · `quantile` · `bce` · `cross_entropy` · `focal` · `dice` · `tversky` · `hinge` · `squared_hinge` · `triplet` · `contrastive` · `infonce` · `kl` · `js` · `wasserstein` · `cosine_embedding` · `label_smoothing`
- **Metrics** ([`metrics.py`](metrics.py) · [metrics.md](metrics.md)) — `accuracy` · `balanced_accuracy` · `precision_recall_f1` · `micro_macro_weighted` · `confusion` · `roc_auc` · `pr_auc` · `average_precision` · `log_loss` · `brier` · `mcc` · `cohen_kappa` · `ndcg` · `map` · `mrr` · `hit_rate` · `silhouette` · `adjusted_rand`
- **Initialisation** ([`init.py`](init.py) · [init.md](init.md)) — `zeros` · `xavier_uniform` · `xavier_normal` · `he_uniform` · `he_normal` · `orthogonal` · `lsuv` · `fixup` · `truncated_normal` · `scaled_variance`
- **Normalisation** ([`normalization.py`](normalization.py) · [normalization.md](normalization.md)) — `batch` · `layer` · `group` · `instance` · `rms` · `weight_std` · `batch_renorm` · `power_norm`
- **Distances** ([`distances.py`](distances.py) · [distances.md](distances.md)) — `euclidean` · `squared_euclidean` · `manhattan` · `chebyshev` · `cosine` · `correlation` · `mahalanobis` · `jaccard` · `dice` · `hamming` · `levenshtein` · `dtw` · `haversine`
- **Linear algebra** ([`linalg.py`](linalg.py) · [linalg.md](linalg.md)) — `lu` · `qr` · `cholesky` · `svd` · `truncated_svd` · `randomized_svd` · `eigh` · `power_iteration` · `woodbury` · `solve_triangular` · `condition_number`
- **Sampling** ([`sampling.py`](sampling.py) · [sampling.md](sampling.md)) — `uniform` · `weighted_alias` · `reservoir` · `stratified` · `systematic` · `gumbel_top_k` · `top_p` · `negative_sampling` · `importance_sampling` · `rejection`
- **Tensor ops** ([`tensor_ops.py`](tensor_ops.py) · [tensor_ops.md](tensor_ops.md)) — `einsum` · `im2col` · `col2im` · `gather` · `scatter_add` · `segment_sum` · `logsumexp` · `softmax_stable` · `one_hot` · `pad_unfold`
- **Embedding encodings** ([`embeddings.py`](embeddings.py) · [embeddings.md](embeddings.md)) — `one_hot` · `hashing` · `sinusoidal` · `learned` · `rope` · `alibi` · `relative_bias` · `factorized` · `binary`
- **Similarity graphs** ([`similarity_graphs.py`](similarity_graphs.py) · [similarity_graphs.md](similarity_graphs.md)) — `knn` · `epsilon_ball` · `mutual_knn` · `heat_kernel` · `cosine` · `mutual_reachability` · `nystrom` · `anchor`
- **Random streams** ([`random.py`](random.py) · [random.md](random.md)) — `pcg64` · `philox` · `sobol` · `halton` · `latin_hypercube` · `antithetic` · `control_variates` · `common_random_numbers`
<!-- catalog:end -->

---

## How to read an atom

Each atom is two files with the same stem: `name.py` is the implementation,
`name.md` is the companion document (intuition, derivation, variant map,
complexity, failure modes, applications). Start with the `.md`.

See [`docs/conventions.md`](../docs/conventions.md) for the file contract and
[`docs/tiers.md`](../docs/tiers.md) for what the tier column means.
