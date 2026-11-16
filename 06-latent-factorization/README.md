# 06 · Latent structure & factorisation

> Low-rank structure: components, topics, codes and factors.

Most high-dimensional data is low-dimensional in disguise. This folder is about finding that structure: linear projections, independent components, non-negative parts, sparse codes and topic mixtures.

<!-- catalog:begin -->
**12 atoms · 58 variants · 0 implemented**

| Atom | File | Tier | Variants | What it does |
|:--|:--|:--:|--:|:--|
| · PCA | [`pca.py`](pca.py) | 1 | 7 | Six ways to compute the same projection. |
| · Non-negative matrix factorisation | [`nmf.py`](nmf.py) | 1 | 7 | Parts-based factorisation with two divergence choices. |
| · Probabilistic PCA | [`probabilistic_pca.py`](probabilistic_pca.py) | 2 | 4 | PCA as a latent-variable model, with missing data and noise. |
| · Factor analysis | [`factor_analysis.py`](factor_analysis.py) | 2 | 5 | Unique variances, rotation and interpretable factors. |
| · Independent component analysis | [`ica.py`](ica.py) | 2 | 6 | Unmixing sources that are statistically independent. |
| · Latent semantic analysis | [`lsa.py`](lsa.py) | 2 | 4 | Truncated SVD of term-document matrices. |
| · Latent Dirichlet allocation | [`lda_topic.py`](lda_topic.py) | 2 | 4 | Topic models by collapsed Gibbs sampling. |
| · Canonical correlation | [`cca.py`](cca.py) | 2 | 5 | Shared structure between two views. |
| · Sparse coding | [`sparse_coding.py`](sparse_coding.py) | 3 | 4 | L1-regularised codes over an overcomplete dictionary. |
| · Dictionary learning | [`dictionary_learning.py`](dictionary_learning.py) | 3 | 4 | Learning the basis instead of assuming it. |
| · Tensor decomposition | [`tensor_decomposition.py`](tensor_decomposition.py) | 3 | 5 | Low-rank structure in more than two modes. |
| · Probabilistic LSA | [`plsa.py`](plsa.py) | 3 | 3 | Aspect model fitted by EM; LDA's non-Bayesian ancestor. |

**Variant map** — every variant below is implemented in its atom's file and derived in its companion `.md`.

- **PCA** ([`pca.py`](pca.py) · [pca.md](pca.md)) — `full_svd` · `randomized` · `kernel_pca` · `incremental` · `robust_pca` · `sparse_pca` · `whitening`
- **Non-negative matrix factorisation** ([`nmf.py`](nmf.py) · [nmf.md](nmf.md)) — `frobenius_mu` · `kl_mu` · `als` · `hals` · `sparse_nmf` · `online_nmf` · `semi_nmf`
- **Probabilistic PCA** ([`probabilistic_pca.py`](probabilistic_pca.py) · [probabilistic_pca.md](probabilistic_pca.md)) — `ppca_em` · `ppca_closed` · `missing_data` · `mixture_ppca`
- **Factor analysis** ([`factor_analysis.py`](factor_analysis.py) · [factor_analysis.md](factor_analysis.md)) — `ml_fa` · `pca_fa` · `varimax` · `promax` · `minres`
- **Independent component analysis** ([`ica.py`](ica.py) · [ica.md](ica.md)) — `fastica_deflation` · `fastica_symmetric` · `infomax` · `jade` · `kernel_ica` · `whitening`
- **Latent semantic analysis** ([`lsa.py`](lsa.py) · [lsa.md](lsa.md)) — `truncated_svd` · `randomized` · `log_entropy` · `tfidf_lsa`
- **Latent Dirichlet allocation** ([`lda_topic.py`](lda_topic.py) · [lda_topic.md](lda_topic.md)) — `collapsed_gibbs` · `variational_bayes` · `online_lda` · `seeded_lda`
- **Canonical correlation** ([`cca.py`](cca.py) · [cca.md](cca.md)) — `cca` · `kernel_cca` · `regularized_cca` · `sparse_cca` · `multi_view`
- **Sparse coding** ([`sparse_coding.py`](sparse_coding.py) · [sparse_coding.md](sparse_coding.md)) — `omp` · `lasso_sc` · `fista_sc` · `group_sc`
- **Dictionary learning** ([`dictionary_learning.py`](dictionary_learning.py) · [dictionary_learning.md](dictionary_learning.md)) — `ksvd` · `mod` · `online_dl` · `sparse_dict`
- **Tensor decomposition** ([`tensor_decomposition.py`](tensor_decomposition.py) · [tensor_decomposition.md](tensor_decomposition.md)) — `cp_als` · `cp_als_ls` · `tucker_hooi` · `hosvd` · `parafac2`
- **Probabilistic LSA** ([`plsa.py`](plsa.py) · [plsa.md](plsa.md)) — `em` · `tempered_em` · `map_plsa`
<!-- catalog:end -->

---

## How to read an atom

Each atom is two files with the same stem: `name.py` is the implementation,
`name.md` is the companion document (intuition, derivation, variant map,
complexity, failure modes, applications). Start with the `.md`.

See [`docs/conventions.md`](../docs/conventions.md) for the file contract and
[`docs/tiers.md`](../docs/tiers.md) for what the tier column means.
