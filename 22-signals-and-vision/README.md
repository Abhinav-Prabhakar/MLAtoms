# 22 · Signals & vision

> Classical feature extraction that still runs on the edge.

Not every vision problem needs a transformer. These are the transforms and descriptors that remain in production because they are fast, interpretable and need no training data.

<!-- catalog:begin -->
**9 atoms · 45 variants · 0 implemented**

| Atom | File | Tier | Variants | What it does |
|:--|:--|:--:|--:|:--|
| · Fourier features | [`fft_features.py`](fft_features.py) | 2 | 6 | DFT, periodogram, spectral centroid and filter banks. |
| · Spectrograms | [`spectrogram.py`](spectrogram.py) | 2 | 4 | STFT, windowing and log-mel representations. |
| · Wavelets | [`wavelets.py`](wavelets.py) | 2 | 6 | DWT, CWT, denoising and multiresolution features. |
| · MFCC | [`mfcc.py`](mfcc.py) | 2 | 5 | Mel filterbanks and cepstral coefficients for audio. |
| · Image descriptors | [`image_features.py`](image_features.py) | 2 | 5 | HOG, LBP, Gabor and colour histograms. |
| · Edge detection | [`edge_detection.py`](edge_detection.py) | 2 | 6 | Sobel, Canny and non-maximum suppression. |
| · SIFT | [`sift.py`](sift.py) | 3 | 4 | Difference-of-Gaussians keypoints and orientation descriptors. |
| · Optical flow | [`optical_flow.py`](optical_flow.py) | 3 | 4 | Lucas-Kanade and Horn-Schunck motion estimation. |
| · Sampling theory | [`sampling_theory.py`](sampling_theory.py) | 3 | 5 | Nyquist limits, resampling and anti-aliasing filters. |

**Variant map** — every variant below is implemented in its atom's file and derived in its companion `.md`.

- **Fourier features** ([`fft_features.py`](fft_features.py) · [fft_features.md](fft_features.md)) — `dft` · `fft` · `periodogram` · `spectral_centroid` · `filter_bank` · `stft`
- **Spectrograms** ([`spectrogram.py`](spectrogram.py) · [spectrogram.md](spectrogram.md)) — `stft` · `mel` · `log_mel` · `windowing`
- **Wavelets** ([`wavelets.py`](wavelets.py) · [wavelets.md](wavelets.md)) — `dwt` · `cwt` · `haar` · `daubechies` · `denoising` · `wavelet_packet`
- **MFCC** ([`mfcc.py`](mfcc.py) · [mfcc.md](mfcc.md)) — `mel_filterbank` · `dct` · `delta` · `delta_delta` · `log_mel`
- **Image descriptors** ([`image_features.py`](image_features.py) · [image_features.md](image_features.md)) — `hog` · `lbp` · `gabor` · `color_hist` · `haralick`
- **Edge detection** ([`edge_detection.py`](edge_detection.py) · [edge_detection.md](edge_detection.md)) — `sobel` · `prewitt` · `scharr` · `canny` · `laplacian` · `nms`
- **SIFT** ([`sift.py`](sift.py) · [sift.md](sift.md)) — `dog` · `keypoint` · `orientation` · `descriptor`
- **Optical flow** ([`optical_flow.py`](optical_flow.py) · [optical_flow.md](optical_flow.md)) — `lucas_kanade` · `pyramidal_lk` · `horn_schunck` · `farneback`
- **Sampling theory** ([`sampling_theory.py`](sampling_theory.py) · [sampling_theory.md](sampling_theory.md)) — `nyquist` · `downsampling` · `upsampling` · `anti_aliasing` · `resampling`
<!-- catalog:end -->

---

## How to read an atom

Each atom is two files with the same stem: `name.py` is the implementation,
`name.md` is the companion document (intuition, derivation, variant map,
complexity, failure modes, applications). Start with the `.md`.

See [`docs/conventions.md`](../docs/conventions.md) for the file contract and
[`docs/tiers.md`](../docs/tiers.md) for what the tier column means.
