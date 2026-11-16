# 01 · Optimization

> How a gradient becomes an update: first-order, adaptive, matrix-preconditioned and schedule design.

Optimisation is where most practical failures hide. This folder walks the ladder from SGD to matrix-orthogonalised updates, then covers the classical machinery (Newton, EM, proximal methods) that modern deep learning quietly reuses.

<!-- catalog:begin -->
**16 atoms · 97 variants · 5 implemented**

| Atom | File | Tier | Variants | What it does |
|:--|:--|:--:|--:|:--|
| ✓ SGD | [`sgd.py`](sgd.py) | 1 | 8 | The base update and the three ideas that made it work. |
| ✓ Adam / AdamW | [`adam.py`](adam.py) | 1 | 8 | Bias-corrected moments with decoupled weight decay. |
| ✓ Adaptive methods | [`adaptive.py`](adaptive.py) | 1 | 7 | Per-coordinate step sizes and their pathologies. |
| ✓ Schedules | [`schedules.py`](schedules.py) | 1 | 13 | Where to spend the learning rate over a training run. |
| ✓ EM / MM | [`em.py`](em.py) | 1 | 6 | The monotone algorithm hiding inside k-means, GMMs, HMMs and VAEs. |
| · Second-order | [`second_order.py`](second_order.py) | 2 | 8 | Curvature, quasi-Newton and trust regions. |
| · Proximal methods | [`proximal.py`](proximal.py) | 2 | 8 | Optimisation when the objective has a nonsmooth part. |
| · Weight averaging | [`weight_averaging.py`](weight_averaging.py) | 2 | 7 | Free accuracy by averaging the trajectory instead of the gradients. |
| · Lion | [`lion.py`](lion.py) | 2 | 4 | Sign-based momentum from a single EMA; memory-light and strong on transformers. |
| · LAMB | [`lamb.py`](lamb.py) | 2 | 3 | Layerwise adaptive moments for very large batches. |
| · Muon | [`muon.py`](muon.py) | 3 | 4 | Orthogonalised momentum via Newton-Schulz; spectral-norm constrained updates. |
| · SOAP | [`soap.py`](soap.py) | 3 | 4 | Adam run inside the Shampoo preconditioner's eigenbasis. |
| · Natural gradient | [`natural_gradient.py`](natural_gradient.py) | 3 | 5 | Step in distribution space, not parameter space. |
| · SAM | [`sam.py`](sam.py) | 3 | 5 | Sharpness-aware minimisation: seek flat minima, generalise better. |
| · Zeroth-order | [`zeroth_order.py`](zeroth_order.py) | 3 | 4 | Optimise when gradients are unavailable or untrustworthy. |
| · Lookahead | [`lookahead.py`](lookahead.py) | 3 | 3 | Slow and fast weights: a wrapper that stabilises any optimiser. |

**Variant map** — every variant below is implemented in its atom's file and derived in its companion `.md`.

- **SGD** ([`sgd.py`](sgd.py) · [sgd.md](sgd.md)) — `vanilla` · `momentum` · `nesterov` · `dampening` · `weight_decay_l2` · `decoupled_wd` · `clipping` · `nesterov_restart`
- **Adam / AdamW** ([`adam.py`](adam.py) · [adam.md](adam.md)) — `adam` · `adamw` · `amsgrad` · `nadam` · `adamax` · `radam` · `adan` · `eps_inside_sqrt`
- **Adaptive methods** ([`adaptive.py`](adaptive.py) · [adaptive.md](adaptive.md)) — `adagrad` · `adadelta` · `rmsprop` · `rmsprop_centered` · `yogi` · `signsgd` · `sm3`
- **Schedules** ([`schedules.py`](schedules.py) · [schedules.md](schedules.md)) — `constant` · `step` · `multistep` · `exponential` · `cosine` · `cosine_restarts` · `linear_warmup` · `one_cycle` · `cyclical` · `plateau` · `polynomial` · `inv_sqrt` · `wsd`
- **EM / MM** ([`em.py`](em.py) · [em.md](em.md)) — `em` · `mm` · `map_em` · `gmm_em` · `incremental_em` · `hard_em`
- **Second-order** ([`second_order.py`](second_order.py) · [second_order.md](second_order.md)) — `newton` · `damped_newton` · `gauss_newton` · `levenberg_marquardt` · `bfgs` · `lbfgs` · `nonlinear_cg` · `trust_region`
- **Proximal methods** ([`proximal.py`](proximal.py) · [proximal.md](proximal.md)) — `ista` · `fista` · `proximal_gd` · `projected_gd` · `admm` · `backtracking` · `split_bregman` · `proximal_newton`
- **Weight averaging** ([`weight_averaging.py`](weight_averaging.py) · [weight_averaging.md](weight_averaging.md)) — `ema` · `swa` · `swa_gaussian` · `model_soup` · `greedy_soup` · `checkpoint_avg` · `polyak`
- **Lion** ([`lion.py`](lion.py) · [lion.md](lion.md)) — `lion` · `signum` · `signsgd_momentum` · `lion_tiny`
- **LAMB** ([`lamb.py`](lamb.py) · [lamb.md](lamb.md)) — `lamb` · `lans` · `lamb_no_bias`
- **Muon** ([`muon.py`](muon.py) · [muon.md](muon.md)) — `muon` · `nesterov_muon` · `newton_schulz` · `muon_hybrid`
- **SOAP** ([`soap.py`](soap.py) · [soap.md](soap.md)) — `soap` · `shampoo` · `soap_muon` · `grafted`
- **Natural gradient** ([`natural_gradient.py`](natural_gradient.py) · [natural_gradient.md](natural_gradient.md)) — `ngd` · `fisher_diag` · `kfac` · `ekfac` · `tnt`
- **SAM** ([`sam.py`](sam.py) · [sam.md](sam.md)) — `sam` · `asan` · `friendly_sam` · `gsam` · `look_sam`
- **Zeroth-order** ([`zeroth_order.py`](zeroth_order.py) · [zeroth_order.md](zeroth_order.md)) — `spsa` · `finite_differences` · `evolution_strategies` · `ars`
- **Lookahead** ([`lookahead.py`](lookahead.py) · [lookahead.md](lookahead.md)) — `lookahead_k` · `lookahead_radam` · `slow_fast`
<!-- catalog:end -->

---

## How to read an atom

Each atom is two files with the same stem: `name.py` is the implementation,
`name.md` is the companion document (intuition, derivation, variant map,
complexity, failure modes, applications). Start with the `.md`.

See [`docs/conventions.md`](../docs/conventions.md) for the file contract and
[`docs/tiers.md`](../docs/tiers.md) for what the tier column means.
