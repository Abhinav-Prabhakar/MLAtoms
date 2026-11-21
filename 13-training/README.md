# 13 · Training

> The craft: how models are actually fitted, compressed and adapted.

Between a model definition and a working product sits training. These files are the procedures rather than the models: autodiff, distillation, adapters, pruning, quantisation, active learning, imbalance handling.

<!-- catalog:begin -->
**18 atoms · 126 variants · 9 implemented**

| Atom | File | Tier | Variants | What it does |
|:--|:--|:--:|--:|:--|
| ✓ Backpropagation | [`backprop.py`](backprop.py) | 1 | 6 | Reverse-mode autodiff on a tiny scalar engine, then vectorised. |
| ✓ Knowledge distillation | [`distillation.py`](distillation.py) | 1 | 8 | Response, feature and relation distillation with temperature. |
| ✓ Gradient checkpointing | [`checkpointing.py`](checkpointing.py) | 2 | 5 | Recompute activations to trade compute for memory. |
| ✓ Mixed precision | [`mixed_precision.py`](mixed_precision.py) | 2 | 6 | FP16/BF16, loss scaling and master weights. |
| ✓ Augmentation | [`augmentation.py`](augmentation.py) | 2 | 8 | Mixup, CutMix, random erasing and RandAugment. |
| ✓ LoRA & PEFT | [`lora.py`](lora.py) | 2 | 9 | Low-rank adapters, QLoRA and the variants that improved them. |
| ✓ Pruning | [`pruning.py`](pruning.py) | 2 | 8 | Magnitude, structured and movement pruning; lottery tickets. |
| ✓ Quantisation | [`quantization.py`](quantization.py) | 2 | 10 | Fewer bits, mostly without losing accuracy. |
| ✓ Hyperparameter search | [`hyperparameter_search.py`](hyperparameter_search.py) | 2 | 8 | Random, TPE, BOHB and successive halving. |
| · Active learning | [`active_learning.py`](active_learning.py) | 2 | 8 | Label the points that teach the most. |
| · Semi-supervised learning | [`semi_supervised.py`](semi_supervised.py) | 2 | 7 | Self-training, pseudo-labels and consistency regularisation. |
| · Class imbalance | [`class_imbalance.py`](class_imbalance.py) | 2 | 8 | SMOTE, ADASYN, focal loss and threshold moving. |
| · Neural architecture search | [`nas.py`](nas.py) | 3 | 6 | Searching the architecture instead of hand-designing it. |
| · Meta-learning | [`meta_learning.py`](meta_learning.py) | 3 | 7 | Learning to learn from a handful of examples. |
| · Continual learning | [`continual_learning.py`](continual_learning.py) | 3 | 8 | New tasks without forgetting the old ones. |
| · Label smoothing | [`label_smoothing.py`](label_smoothing.py) | 3 | 4 | Soft targets, regularisation and calibration effects. |
| · Early stopping | [`early_stopping.py`](early_stopping.py) | 3 | 4 | Patience, restoration and the generalisation gap. |
| · Federated averaging | [`federated.py`](federated.py) | 3 | 6 | Training across clients without moving their data. |

**Variant map** — every variant below is implemented in its atom's file and derived in its companion `.md`.

- **Backpropagation** ([`backprop.py`](backprop.py) · [backprop.md](backprop.md)) — `scalar_engine` · `reverse_mode` · `forward_mode` · `jacobian` · `hessian_vector` · `gradient_check`
- **Knowledge distillation** ([`distillation.py`](distillation.py) · [distillation.md](distillation.md)) — `response` · `feature` · `relation` · `self_distill` · `born_again` · `online_distill` · `attention_transfer` · `dkd`
- **Gradient checkpointing** ([`checkpointing.py`](checkpointing.py) · [checkpointing.md](checkpointing.md)) — `full` · `selective` · `sqrt` · `offload` · `reversible`
- **Mixed precision** ([`mixed_precision.py`](mixed_precision.py) · [mixed_precision.md](mixed_precision.md)) — `fp16` · `bf16` · `fp8` · `loss_scaling` · `dynamic_scaling` · `master_weights`
- **Augmentation** ([`augmentation.py`](augmentation.py) · [augmentation.md](augmentation.md)) — `mixup` · `cutmix` · `cutout` · `random_erasing` · `randaugment` · `augmix` · `color_jitter` · `auto_augment`
- **LoRA & PEFT** ([`lora.py`](lora.py) · [lora.md](lora.md)) — `lora` · `qlora` · `dora` · `adalora` · `rslora` · `loha` · `lokr` · `prefix_tuning` · `prompt_tuning`
- **Pruning** ([`pruning.py`](pruning.py) · [pruning.md](pruning.md)) — `magnitude` · `unstructured` · `structured` · `movement` · `gradual` · `one_shot` · `lottery_ticket` · `svd_prune`
- **Quantisation** ([`quantization.py`](quantization.py) · [quantization.md](quantization.md)) — `absmax` · `zero_point` · `per_channel` · `per_group` · `ptq` · `qat` · `gptq` · `awq` · `int4` · `int8`
- **Hyperparameter search** ([`hyperparameter_search.py`](hyperparameter_search.py) · [hyperparameter_search.md](hyperparameter_search.md)) — `grid` · `random` · `tpe` · `gp_bo` · `bohb` · `hyperband` · `asha` · `pruner`
- **Active learning** ([`active_learning.py`](active_learning.py) · [active_learning.md](active_learning.md)) — `uncertainty` · `margin` · `entropy` · `diversity` · `qbc` · `bald` · `core_set` · `expected_model_change`
- **Semi-supervised learning** ([`semi_supervised.py`](semi_supervised.py) · [semi_supervised.md](semi_supervised.md)) — `self_training` · `pseudo_label` · `fixmatch` · `mean_teacher` · `mixmatch` · `label_propagation` · `ladder`
- **Class imbalance** ([`class_imbalance.py`](class_imbalance.py) · [class_imbalance.md](class_imbalance.md)) — `smote` · `adasyn` · `borderline_smote` · `tomek` · `enn` · `focal` · `class_weight` · `threshold_moving`
- **Neural architecture search** ([`nas.py`](nas.py) · [nas.md](nas.md)) — `darts` · `enas` · `amoebanet` · `regularized_evolution` · `once_for_all` · `supernet`
- **Meta-learning** ([`meta_learning.py`](meta_learning.py) · [meta_learning.md](meta_learning.md)) — `maml` · `first_order_maml` · `reptile` · `prototypical` · `matching` · `relation` · `meta_sgd`
- **Continual learning** ([`continual_learning.py`](continual_learning.py) · [continual_learning.md](continual_learning.md)) — `ewc` · `si` · `mas` · `replay` · `gem` · `lwf` · `packnet` · `lora_merge`
- **Label smoothing** ([`label_smoothing.py`](label_smoothing.py) · [label_smoothing.md](label_smoothing.md)) — `uniform` · `per_class` · `adaptive` · `mixup_smoothing`
- **Early stopping** ([`early_stopping.py`](early_stopping.py) · [early_stopping.md](early_stopping.md)) — `patience` · `min_delta` · `restore_best` · `plateau_decay`
- **Federated averaging** ([`federated.py`](federated.py) · [federated.md](federated.md)) — `fedavg` · `fedprox` · `scaffold` · `secure_agg` · `dp_fedavg` · `personalization`
<!-- catalog:end -->

---

## How to read an atom

Each atom is two files with the same stem: `name.py` is the implementation,
`name.md` is the companion document (intuition, derivation, variant map,
complexity, failure modes, applications). Start with the `.md`.

See [`docs/conventions.md`](../docs/conventions.md) for the file contract and
[`docs/tiers.md`](../docs/tiers.md) for what the tier column means.
