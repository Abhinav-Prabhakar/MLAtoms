# 17 · Recommenders & ranking

> Ranking what comes next: collaborative filtering, sequence models and learned rankers.

Recommendation is retrieval plus ranking, and both halves have their own algorithms. This folder goes from item co-occurrence to factorisation machines to sequential transformers, plus the learning-to-rank losses that score the final list.

<!-- catalog:begin -->
**14 atoms · 53 variants · 0 implemented**

| Atom | File | Tier | Variants | What it does |
|:--|:--|:--:|--:|:--|
| · Popularity baseline | [`popularity.py`](popularity.py) | 1 | 4 | Non-personalised baselines with temporal decay. |
| · Item-item CF | [`item_cf.py`](item_cf.py) | 1 | 5 | Co-occurrence similarity, normalisation and shrinkage. |
| · Matrix factorisation | [`matrix_factorization.py`](matrix_factorization.py) | 1 | 5 | Biased MF trained by SGD or ALS. |
| · Implicit ALS | [`als.py`](als.py) | 2 | 3 | Confidence-weighted factorisation for implicit feedback. |
| · BPR | [`bpr.py`](bpr.py) | 2 | 3 | Bayesian personalised ranking from pairwise preferences. |
| · Factorization machines | [`fm.py`](fm.py) | 2 | 4 | Second-order interactions over sparse features. |
| · Two-tower retrieval | [`two_tower.py`](two_tower.py) | 2 | 4 | Dual encoders, in-batch negatives and ANN serving. |
| · Sequential recommenders | [`sequential.py`](sequential.py) | 2 | 4 | SASRec and GRU4Rec for session-based prediction. |
| · Learning to rank | [`ranking_losses.py`](ranking_losses.py) | 2 | 6 | RankNet, LambdaRank, ListNet and LambdaMART. |
| · SVD++ | [`svd_plusplus.py`](svd_plusplus.py) | 3 | 3 | Biased factorisation with implicit-feedback factors. |
| · Field-aware FM | [`ffm.py`](ffm.py) | 3 | 2 | Per-field interaction vectors. |
| · DeepFM | [`deepfm.py`](deepfm.py) | 3 | 4 | Wide linear part plus FM plus MLP. |
| · Neural collaborative filtering | [`ncf.py`](ncf.py) | 3 | 3 | GMF and MLP towers replacing the dot product. |
| · Cold start | [`cold_start.py`](cold_start.py) | 3 | 3 | Content features, hybrid blending and exploration. |

**Variant map** — every variant below is implemented in its atom's file and derived in its companion `.md`.

- **Popularity baseline** ([`popularity.py`](popularity.py) · [popularity.md](popularity.md)) — `global` · `time_decay` · `bayesian_smoothing` · `recent`
- **Item-item CF** ([`item_cf.py`](item_cf.py) · [item_cf.md](item_cf.md)) — `cosine` · `adjusted_cosine` · `jaccard` · `cooccurrence` · `shrinkage`
- **Matrix factorisation** ([`matrix_factorization.py`](matrix_factorization.py) · [matrix_factorization.md](matrix_factorization.md)) — `biased_mf` · `sgd` · `als` · `svd` · `regularized`
- **Implicit ALS** ([`als.py`](als.py) · [als.md](als.md)) — `implicit_als` · `weighted_als` · `bm25_weight`
- **BPR** ([`bpr.py`](bpr.py) · [bpr.md](bpr.md)) — `bpr` · `negative_sampling` · `wbpr`
- **Factorization machines** ([`fm.py`](fm.py) · [fm.md](fm.md)) — `fm` · `fm_sgd` · `fm_als` · `second_order`
- **Two-tower retrieval** ([`two_tower.py`](two_tower.py) · [two_tower.md](two_tower.md)) — `dual_encoder` · `in_batch_negatives` · `hard_negatives` · `ann_serve`
- **Sequential recommenders** ([`sequential.py`](sequential.py) · [sequential.md](sequential.md)) — `sasrec` · `gru4rec` · `bert4rec` · `caser`
- **Learning to rank** ([`ranking_losses.py`](ranking_losses.py) · [ranking_losses.md](ranking_losses.md)) — `ranknet` · `lambdarank` · `listnet` · `listmle` · `lambdamart` · `approx_ndcg`
- **SVD++** ([`svd_plusplus.py`](svd_plusplus.py) · [svd_plusplus.md](svd_plusplus.md)) — `svdpp` · `asymmetric` · `implicit_factors`
- **Field-aware FM** ([`ffm.py`](ffm.py) · [ffm.md](ffm.md)) — `ffm` · `ffm_sgd`
- **DeepFM** ([`deepfm.py`](deepfm.py) · [deepfm.md](deepfm.md)) — `wide_deep` · `deepfm` · `dcn` · `autoint`
- **Neural collaborative filtering** ([`ncf.py`](ncf.py) · [ncf.md](ncf.md)) — `gmf` · `mlp` · `neumf`
- **Cold start** ([`cold_start.py`](cold_start.py) · [cold_start.md](cold_start.md)) — `content_based` · `hybrid` · `exploration`
<!-- catalog:end -->

---

## How to read an atom

Each atom is two files with the same stem: `name.py` is the implementation,
`name.md` is the companion document (intuition, derivation, variant map,
complexity, failure modes, applications). Start with the `.md`.

See [`docs/conventions.md`](../docs/conventions.md) for the file contract and
[`docs/tiers.md`](../docs/tiers.md) for what the tier column means.
