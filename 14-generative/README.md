# 14 · Generative models

> Models that produce data: latent, adversarial, flow and diffusion.

Four families, one question: how do you represent and sample a distribution? Latent-variable models give you a decoder, GANs give you sharpness, flows give you exact likelihood, diffusion gives you both quality and control.

<!-- catalog:begin -->
**13 atoms · 64 variants · 8 implemented**

| Atom | File | Tier | Variants | What it does |
|:--|:--|:--:|--:|:--|
| ✓ Variational autoencoder | [`vae.py`](vae.py) | 1 | 5 | ELBO, reparameterisation and posterior collapse. |
| ✓ GAN | [`gan.py`](gan.py) | 1 | 7 | Minimax training, loss variants and mode collapse. |
| ✓ Diffusion (DDPM) | [`diffusion.py`](diffusion.py) | 1 | 6 | Forward/reverse processes, noise schedules and the ELBO. |
| ✓ Wasserstein GAN | [`wgan.py`](wgan.py) | 2 | 4 | Wasserstein distance, Lipschitz constraints and gradient penalty. |
| ✓ DDIM | [`ddim.py`](ddim.py) | 2 | 4 | Deterministic, fast sampling and the eta knob. |
| ✓ Guidance | [`guidance.py`](guidance.py) | 2 | 4 | Trading diversity for fidelity at sampling time. |
| ✓ Normalising flows | [`flows.py`](flows.py) | 2 | 6 | Exact likelihoods through invertible transformations. |
| ✓ VQ-VAE | [`vq_vae.py`](vq_vae.py) | 2 | 6 | Discrete codebooks, straight-through gradients, codebook collapse. |
| · Beta-VAE | [`beta_vae.py`](beta_vae.py) | 3 | 4 | Disentanglement pressure and its decomposition. |
| · Score-based models | [`score_sde.py`](score_sde.py) | 3 | 5 | Score matching, probability-flow ODEs, predictors and correctors. |
| · Autoregressive generation | [`autoregressive_gen.py`](autoregressive_gen.py) | 3 | 4 | PixelCNN-style masked likelihoods. |
| · Energy-based models | [`energy_based.py`](energy_based.py) | 3 | 5 | Energy functions, contrastive divergence and Langevin sampling. |
| · Flow matching | [`flow_matching.py`](flow_matching.py) | 3 | 4 | Optimal-transport paths and rectified flow. |

**Variant map** — every variant below is implemented in its atom's file and derived in its companion `.md`.

- **Variational autoencoder** ([`vae.py`](vae.py) · [vae.md](vae.md)) — `standard` · `conditional` · `iwae` · `beta` · `collapse_fix`
- **GAN** ([`gan.py`](gan.py) · [gan.md](gan.md)) — `vanilla` · `dcgan` · `lsgan` · `hinge` · `non_saturating` · `conditional` · `infogan`
- **Diffusion (DDPM)** ([`diffusion.py`](diffusion.py) · [diffusion.md](diffusion.md)) — `ddpm` · `linear_schedule` · `cosine_schedule` · `v_prediction` · `epsilon_prediction` · `latent_diffusion`
- **Wasserstein GAN** ([`wgan.py`](wgan.py) · [wgan.md](wgan.md)) — `wgan` · `wgan_gp` · `wgan_lp` · `spectral_norm`
- **DDIM** ([`ddim.py`](ddim.py) · [ddim.md](ddim.md)) — `ddim` · `eta_sampling` · `ddim_inversion` · `few_step`
- **Guidance** ([`guidance.py`](guidance.py) · [guidance.md](guidance.md)) — `classifier` · `classifier_free` · `negative_prompt` · `rescale_cfg`
- **Normalising flows** ([`flows.py`](flows.py) · [flows.md](flows.md)) — `realnvp` · `glow` · `coupling` · `autoregressive_flow` · `spline_flow` · `continuous_flow`
- **VQ-VAE** ([`vq_vae.py`](vq_vae.py) · [vq_vae.md](vq_vae.md)) — `vq_vae` · `vq_vae2` · `ema_codebook` · `commitment_loss` · `straight_through` · `residual_vq`
- **Beta-VAE** ([`beta_vae.py`](beta_vae.py) · [beta_vae.md](beta_vae.md)) — `beta` · `beta_tc` · `factor_vae` · `dip_vae`
- **Score-based models** ([`score_sde.py`](score_sde.py) · [score_sde.md](score_sde.md)) — `score_matching` · `denoising_score` · `predictor_corrector` · `ode_solver` · `annealed_langevin`
- **Autoregressive generation** ([`autoregressive_gen.py`](autoregressive_gen.py) · [autoregressive_gen.md](autoregressive_gen.md)) — `pixelcnn` · `gated_pixelcnn` · `pixelcnn_pp` · `masked_attention`
- **Energy-based models** ([`energy_based.py`](energy_based.py) · [energy_based.md](energy_based.md)) — `ebm` · `contrastive_divergence` · `langevin` · `jem` · `noise_contrastive`
- **Flow matching** ([`flow_matching.py`](flow_matching.py) · [flow_matching.md](flow_matching.md)) — `flow_matching` · `rectified_flow` · `stochastic_interpolant` · `optimal_transport`
<!-- catalog:end -->

---

## How to read an atom

Each atom is two files with the same stem: `name.py` is the implementation,
`name.md` is the companion document (intuition, derivation, variant map,
complexity, failure modes, applications). Start with the `.md`.

See [`docs/conventions.md`](../docs/conventions.md) for the file contract and
[`docs/tiers.md`](../docs/tiers.md) for what the tier column means.
