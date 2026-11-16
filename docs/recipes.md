# Recipes

The repository is organised by algorithm. Work arrives organised by task. This
page is the translation layer: start from the problem, get the three atoms that
solve it, in the order you should try them.

Every row follows the same shape — **default → upgrade → trap**.

---

### "I have a table, a target column, and no idea where to start"

`04-trees-and-ensembles/xgboost.py` → `19-tabular/tabm.py` → `19-tabular/tabpfn.py`

Boosted trees first, always. If rows are few (< 10k) and columns are numeric,
`tabpfn` can beat a tuned booster in one call. The trap is deep learning on
tabular data with fewer than ~50k rows: `ft_transformer` rarely pays for its
tuning cost there.

### "My table has 40 categorical columns and half the cells are missing"

`19-tabular/feature_engineering.py` → `04-trees-and-ensembles/lightgbm.py` → `04-trees-and-ensembles/catboost.py`

Target encoding with out-of-fold statistics, missingness indicators, then a
booster that handles categoricals natively. The trap is one-hot encoding a
high-cardinality column: it is a memory bomb that also destroys split quality.

### "Predict a probability, and I need it to mean something"

`02-linear-models/logistic_regression.py` → `20-evaluation/calibration.py` → `20-evaluation/conformal.py`

A model's `predict_proba` is a ranking score until you calibrate it. Isotonic
for large n, Platt for small. If the downstream decision has a cost, conformal
gives you a set with a guarantee instead of a number with a hope.

### "Nobody labelled the data"

`07-clustering/hdbscan.py` → `16-representation/contrastive.py` → `13-training/active_learning.py`

Clusters to find structure, contrastive pretraining to get a usable embedding,
active learning to label the 500 points that matter. The trap is k-means on
non-spherical data, then concluding "the data has no structure".

### "Find the weird ones"

`09-density-and-anomaly/isolation_forest.py` → `09-density-and-anomaly/lof.py` → `09-density-and-anomaly/autoencoder_anomaly.py`

Isolation forest is the default because it needs no distance and no tuning. LOF
when anomalies are *locally* odd. Reconstruction error when the data is
high-dimensional and structured (images, logs, sensor traces). The trap is
evaluating any of them on unlabelled data — inject known anomalies and measure.

### "Forecast next quarter"

`11-time-series/baselines.py` → `11-time-series/ets.py` → `11-time-series/arima.py`

Seasonal naive is the bar. ETS handles trend plus seasonality without
differencing. ARIMA when the autocorrelation structure is the story. The trap is
skipping the baseline: on most business series a well-specified ETS beats a
transformer and takes a second to fit.

### "Forecast 10,000 series at once"

`11-time-series/dlinear.py` → `11-time-series/patchtst.py` → `11-time-series/deepar.py`

DLinear for speed and a sanity check, PatchTST for long horizons, DeepAR when
you need a distribution rather than a point. The trap is one model per series:
global models with an ID embedding are both cheaper and better.

### "Detect the moment behaviour changed"

`09-density-and-anomaly/change_point.py` → `20-evaluation/drift_detection.py`

PELT for offline segmentation, BOCPD for online. ADWIN/DDM when you are
monitoring a metric stream, not a raw series. The trap is a fixed threshold on a
non-stationary statistic.

### "Rank 1,000 candidates for one user"

`17-recommender/two_tower.py` → `17-recommender/ranking_losses.py` → `17-recommender/bpr.py`

Two-tower retrieval for the candidate set, a learned ranker on the shortlist,
BPR if all you have is implicit feedback. The trap is optimising RMSE on ratings
when the product is a top-10 list — use NDCG.

### "I have text and a small label budget"

`16-representation/tfidf.py` → `02-linear-models/logistic_regression.py` → `16-representation/contrastive.py`

TF-IDF plus a linear model is a genuinely strong baseline and takes seconds.
Move to embeddings when vocabulary mismatch is the failure. The trap is jumping
to fine-tuning a large model before measuring the TF-IDF baseline.

### "Cluster documents or users"

`16-representation/tfidf.py` → `08-manifold/umap.py` → `07-clustering/hdbscan.py`

Reduce, then cluster — not the other way round. The trap is trusting a 2-D
picture: UMAP distances between clusters are not meaningful, only within.

### "Explain why the model said no"

`20-evaluation/interpretability.py` → `19-tabular/ebm.py` → `04-trees-and-ensembles/cart.py`

Permutation importance for a global view, SHAP for per-row attribution, an EBM
when you need the model itself to be the explanation. The trap is reading SHAP
values as causal effects. They are not; see `21-causality/`.

### "Decide whether shipping this feature helped"

`20-evaluation/ab_testing.py` → `21-causality/potential_outcomes.py` → `21-causality/doubly_robust.py`

Randomised test first, with power analysis and CUPED. When randomisation is
impossible, potential outcomes and a doubly robust estimator. The trap is
comparing before/after and calling it a treatment effect.

### "Learn a policy from logged decisions"

`18-reinforcement-learning/offline_rl.py` → `18-reinforcement-learning/imitation.py` → `18-reinforcement-learning/reward_modeling.py`

CQL/IQL when you have logged actions and rewards, behaviour cloning when you
have an expert, DPO when you have preferences. The trap is off-policy evaluation
without support: if the logged policy never took the action, your estimate is
fantasy.

### "Serve a model on a phone"

`13-training/pruning.py` → `13-training/quantization.py` → `16-representation/product_quantization.py`

Prune, then quantise, then measure — in that order. Int8 PTQ usually costs
nothing; int4 needs per-group scales. The trap is quantising before pruning and
losing both.

### "Fine-tune something huge on one GPU"

`13-training/lora.py` → `13-training/checkpointing.py` → `01-optimization/adafactor.py`

LoRA (or QLoRA) for the parameter budget, checkpointing for the activation
budget, a factored optimiser for the state budget. The trap is full fine-tuning
with Adam on a 7B model — the optimiser state is bigger than the weights.

### "My training loss goes down and validation does not"

`20-evaluation/resampling.py` → `13-training/early_stopping.py` → `01-optimization/weight_averaging.py`

Check the split first (leakage explains most of these), then regularise, then
average checkpoints. The trap is adding dropout when the real problem is a
duplicated row across train and test.

### "I need uncertainty, not just a prediction"

`20-evaluation/uncertainty.py` → `20-evaluation/conformal.py` → `05-probabilistic/gaussian_process.py`

Deep ensembles for a cheap epistemic estimate, conformal for a distribution-free
interval, a GP when n is small and you want the posterior itself. The trap is
MC dropout on a model trained without it — that is not a posterior sample.

### "Find the drivers, not the predictors"

`21-causality/do_calculus.py` → `21-causality/propensity.py` → `21-causality/doubly_robust.py`

Write the graph, find the adjustment set, estimate. The trap is a feature
importance plot presented as a cause-and-effect story.

### "Detect near-duplicates at scale"

`16-representation/minhash.py` → `16-representation/simhash.py` → `16-representation/product_quantization.py`

MinHash for sets (Jaccard), SimHash for vectors (cosine), PQ for dense
embeddings. The trap is pairwise comparison on 10M items — that is 5·10¹³ pairs.

### "My features are 50,000 columns and 300 rows"

`02-linear-models/ridge.py` → `03-kernels-and-instance/svm.py` → `06-latent-factorization/pca.py`

Ridge with the kernel trick, then a kernel SVM, then reduce dimensions.
The trap is any method that inverts `XᵀX` — it does not exist.

### "I want to understand the data before modelling it"

`08-manifold/umap.py` → `07-clustering/cluster_validity.py` → `20-evaluation/interpretability.py`

Embed, quantify the cluster structure, then look at what separates the groups.
The trap is a beautiful embedding with no numbers attached.
