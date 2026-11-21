"""Spectral analysis — periodogram, Welch, multitaper, Lomb-Scargle and STFT.

Chapter:   11-time-series
Companion: spectral_analysis.md
Tier 2 · 5 variants: periodogram, welch, multitaper, lomb_scargle,
stft_features
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VARIANTS = ("periodogram", "welch", "multitaper", "lomb_scargle",
            "stft_features")


# --------------------------------------------------------------------------- #
# Result object and helpers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Spectrum:
    """A frequency-domain summary returned by `spectrum`.

    Attributes
    ----------
    frequencies : ndarray
        Frequency grid in cycles per unit time, shape `(n_freq,)`.
    power : ndarray
        Spectral density or power, shape `(n_freq,)`. Periodogram, Welch and
        multitaper are normalised so that `sum(power) * df` estimates the
        variance of the detrended signal.
    variant, fs, resolution : str and floats
        What produced it, the sampling rate and the frequency spacing.
    segments : int
        Number of segments or tapers averaged.
    times : ndarray or None
        Frame centres for `stft_features`, otherwise `None`.
    spectrogram : ndarray or None
        Time-by-frequency power matrix for `stft_features`.
    features : dict or None
        Spectral centroid, bandwidth, rolloff and flatness for `stft_features`.
    tapers : ndarray or None
        The DPSS tapers used by `multitaper`, shape `(n_tapers, n)`.
    """

    frequencies: np.ndarray
    power: np.ndarray
    variant: str
    fs: float
    resolution: float
    segments: int = 1
    times: np.ndarray | None = None
    spectrogram: np.ndarray | None = None
    features: dict | None = None
    tapers: np.ndarray | None = None

    def peak(self) -> tuple[float, float]:
        """Frequency and power of the largest bin."""
        if self.power.size == 0:
            raise ValueError("empty spectrum")
        index = int(np.argmax(self.power))
        return float(self.frequencies[index]), float(self.power[index])


def _check_signal(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be one-dimensional")
    if x.size < 8:
        raise ValueError("x needs at least eight samples")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must be finite")
    return x


def _check_fs(fs: float) -> float:
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError("fs must be positive and finite")
    return float(fs)


def _window(name: str, n: int) -> np.ndarray:
    if n < 3:
        raise ValueError("the window needs at least three samples")
    grid = np.arange(n)
    if name in ("boxcar", "rectangular", "none"):
        return np.ones(n)
    if name == "hann":
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * grid / n)
    if name == "hamming":
        return 0.54 - 0.46 * np.cos(2.0 * np.pi * grid / n)
    if name == "blackman":
        return (0.42 - 0.5 * np.cos(2.0 * np.pi * grid / n)
                + 0.08 * np.cos(4.0 * np.pi * grid / n))
    raise ValueError(f"unknown window {name!r}")


def _density(x: np.ndarray, fs: float, name: str,
             nfft: int) -> tuple[np.ndarray, np.ndarray]:
    """One-sided windowed periodogram of a single segment."""
    x = x - x.mean()
    window = _window(name, x.size)
    transform = np.fft.rfft(x * window, n=nfft)
    power = np.abs(transform) ** 2 / (fs * float(window @ window))
    if nfft % 2 == 0:
        power[1:-1] *= 2.0
    else:
        power[1:] *= 2.0
    return np.fft.rfftfreq(nfft, 1.0 / fs), power


def _frames(x: np.ndarray, size: int, hop: int) -> np.ndarray:
    if size < 3 or size > x.size:
        raise ValueError("frame size must be between three and the signal")
    if hop < 1:
        raise ValueError("hop must be a positive integer")
    return np.lib.stride_tricks.sliding_window_view(x, size)[::hop]


def _dpss(n: int, nw: float) -> np.ndarray:
    """Discrete prolate spheroidal sequences from a tridiagonal eigenproblem."""
    if not np.isfinite(nw) or nw <= 0.5:
        raise ValueError("nw must be finite and greater than one half")
    half = 0.5 * (n - 1)
    index = np.arange(n, dtype=float)
    diagonal = (half - index) ** 2 * np.cos(2.0 * np.pi * nw / n)
    off = index[1:] * (n - index[1:]) / 2.0
    matrix = np.diag(diagonal) + np.diag(off, 1) + np.diag(off, -1)
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    count = max(1, int(np.floor(2.0 * nw - 1.0)))
    count = min(count, n)
    tapers = np.empty((count, n))
    for row, column in enumerate(order[:count]):
        taper = vectors[:, column]
        if taper[np.argmax(np.abs(taper))] < 0.0:
            taper = -taper
        tapers[row] = taper / np.linalg.norm(taper)
    return tapers


# --------------------------------------------------------------------------- #
# The public entry point
# --------------------------------------------------------------------------- #
def spectrum(x: np.ndarray, variant: str = "periodogram", *, fs: float = 1.0,
             times: np.ndarray | None = None, window: str = "hann",
             nperseg: int | None = None, noverlap: int | None = None,
             nfft: int | None = None, nw: float = 4.0, tapers: int | None = None,
             fmax: float | None = None, oversample: int = 4,
             hop: int | None = None) -> Spectrum:
    """Estimate the spectrum of `x` with the chosen variant.

    Parameters
    ----------
    x : ndarray
        Signal, shape `(n,)`.
    variant : str
        One of `VARIANTS`.
    fs : float
        Sampling rate in samples per unit time.
    times : ndarray, optional
        Sample times for `lomb_scargle`; defaults to `arange(n) / fs` and must
        be strictly increasing when supplied.
    window : str
        `hann`, `hamming`, `blackman`, `boxcar` or `none`.
    nperseg, noverlap, nfft : int, optional
        Segment length, overlap and FFT length. Defaults are variant-specific.
    nw, tapers : float and int, optional
        Time-bandwidth product and taper count for `multitaper`.
    fmax : float, optional
        Upper frequency limit for `lomb_scargle`; at most `fs / 2`.
    oversample : int
        Frequency-grid oversampling factor for `lomb_scargle`.
    hop : int, optional
        Frame hop for `stft_features`; defaults to half a frame.

    Returns
    -------
    Spectrum
        Frequencies, power and variant-specific extras.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    x = _check_signal(x)
    fs = _check_fs(fs)
    if not isinstance(oversample, (int, np.integer)) or oversample < 1:
        raise ValueError("oversample must be a positive integer")

    if variant == "periodogram":
        size = x.size if nperseg is None else int(nperseg)
        if size < 3 or size > x.size:
            raise ValueError("nperseg must lie between three and the signal")
        length = size if nfft is None else int(nfft)
        if length < size:
            raise ValueError("nfft must be at least nperseg")
        frequencies, power = _density(x[:size], fs, window, length)
        return Spectrum(frequencies, power, variant, fs, fs / size)

    if variant == "welch":
        size = (min(x.size, max(8, x.size // 4)) if nperseg is None
                else int(nperseg))
        if size < 3 or size > x.size:
            raise ValueError("nperseg must lie between three and the signal")
        overlap = size // 2 if noverlap is None else int(noverlap)
        if overlap < 0 or overlap >= size:
            raise ValueError("noverlap must lie in [0, nperseg)")
        frames = _frames(x, size, size - overlap)
        length = size if nfft is None else int(nfft)
        if length < size:
            raise ValueError("nfft must be at least nperseg")
        stack = np.empty((frames.shape[0], length // 2 + 1))
        for row, frame in enumerate(frames):
            frequencies, stack[row] = _density(frame, fs, window, length)
        return Spectrum(frequencies, stack.mean(axis=0), variant, fs, fs / size,
                        segments=int(frames.shape[0]))

    if variant == "multitaper":
        length = x.size if nfft is None else int(nfft)
        if length < x.size:
            raise ValueError("nfft must be at least the signal length")
        sequences = _dpss(x.size, float(nw))
        if tapers is not None:
            if not isinstance(tapers, (int, np.integer)) or tapers < 1:
                raise ValueError("tapers must be a positive integer")
            sequences = sequences[:min(int(tapers), sequences.shape[0])]
        stack = np.empty((sequences.shape[0], length // 2 + 1))
        for row, taper in enumerate(sequences):
            transform = np.fft.rfft((x - x.mean()) * taper, n=length)
            power = np.abs(transform) ** 2 / (fs * float(taper @ taper))
            if length % 2 == 0:
                power[1:-1] *= 2.0
            else:
                power[1:] *= 2.0
            stack[row] = power
        frequencies = np.fft.rfftfreq(length, 1.0 / fs)
        return Spectrum(frequencies, stack.mean(axis=0), variant, fs, fs / x.size,
                        segments=int(sequences.shape[0]), tapers=sequences)

    if variant == "stft_features":
        size = (min(x.size, max(16, x.size // 8)) if nperseg is None
                else int(nperseg))
        step = max(1, size // 2) if hop is None else int(hop)
        if step < 1:
            raise ValueError("hop must be a positive integer")
        frames = _frames(x, size, step)
        length = size if nfft is None else int(nfft)
        if length < size:
            raise ValueError("nfft must be at least nperseg")
        window_values = _window(window, size)
        transform = np.fft.rfft(frames * window_values, n=length, axis=1)
        block = np.abs(transform) ** 2 / (fs * float(window_values @ window_values))
        if length % 2 == 0:
            block[:, 1:-1] *= 2.0
        else:
            block[:, 1:] *= 2.0
        frequencies = np.fft.rfftfreq(length, 1.0 / fs)
        total = block.sum(axis=1)
        total = np.where(total > 0.0, total, 1.0)
        centroid = block @ frequencies / total
        bandwidth = np.sqrt(
            np.maximum(block @ (frequencies ** 2) / total - centroid ** 2, 0.0))
        cumulative = np.cumsum(block, axis=1)
        rolloff = np.empty(block.shape[0])
        for row in range(block.shape[0]):
            threshold = 0.85 * cumulative[row, -1]
            rolloff[row] = frequencies[int(np.searchsorted(cumulative[row],
                                                           threshold))]
        epsilon = 1e-18
        flatness = np.exp(np.mean(np.log(block + epsilon), axis=1)
                          ) / np.maximum(block.mean(axis=1), epsilon)
        times_out = (np.arange(block.shape[0]) * step + size / 2.0) / fs
        features = {"centroid": centroid, "bandwidth": bandwidth,
                    "rolloff": rolloff, "flatness": flatness,
                    "centroid_mean": float(centroid.mean()),
                    "bandwidth_mean": float(bandwidth.mean()),
                    "rolloff_mean": float(rolloff.mean()),
                    "flatness_mean": float(flatness.mean())}
        return Spectrum(frequencies, block.mean(axis=0), variant, fs, fs / size,
                        segments=int(block.shape[0]), times=times_out,
                        spectrogram=block, features=features)

    # Lomb-Scargle
    if times is None:
        times = np.arange(x.size, dtype=float) / fs
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size != x.size:
        raise ValueError("times must be one-dimensional and match x")
    if not np.all(np.isfinite(times)):
        raise ValueError("times must be finite")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times must be strictly increasing")
    nyquist = 0.5 / float(np.median(np.diff(times)))
    limit = nyquist if fmax is None else float(fmax)
    if not np.isfinite(limit) or limit <= 0.0:
        raise ValueError("fmax must be positive and finite")
    span = float(times[-1] - times[0])
    if span <= 0.0:
        raise ValueError("times must span a positive interval")
    spacing = 1.0 / (oversample * span)
    count = max(1, int(np.floor(limit / spacing)))
    frequencies = np.arange(1, count + 1) * spacing
    centred = x - x.mean()
    variance = float(np.var(centred))
    if variance <= 0.0:
        raise ValueError("x must not be constant")
    power = np.empty(frequencies.size)
    for row, frequency in enumerate(frequencies):
        angle = 2.0 * np.pi * frequency * times
        phase = np.arctan2(np.sin(angle).sum(), np.cos(angle).sum()) / (
            2.0 * np.pi * frequency)
        cosine = np.cos(2.0 * np.pi * frequency * (times - phase))
        sine = np.sin(2.0 * np.pi * frequency * (times - phase))
        left = float((centred @ cosine) ** 2 / (cosine @ cosine))
        right = float((centred @ sine) ** 2 / (sine @ sine))
        power[row] = 0.5 * (left + right) / (0.5 * x.size * variance)
    return Spectrum(frequencies, power, variant, fs, spacing)


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
def _two_tone(n: int, rng: np.random.Generator, fs: float = 1.0
              ) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n) / fs
    signal = (2.0 * np.sin(2.0 * np.pi * 0.1 * t)
              + 0.5 * np.cos(2.0 * np.pi * 0.23 * t)
              + 0.5 * rng.normal(0.0, 1.0, n))
    return t, signal


def demo() -> None:
    rng = np.random.default_rng(0)
    fs = 1.0
    n = 1024
    t, x = _two_tone(n, rng, fs)
    print("periodogram. 1024 samples, tones at 0.10 and 0.23 cycles/sample:")
    model = spectrum(x, "periodogram", fs=fs)
    peak, height = model.peak()
    spacing = float(model.frequencies[1] - model.frequencies[0])
    integral = float(model.power.sum() * spacing)
    variance = float(np.var(x))
    assert abs(peak - 0.1) <= 2.0 * model.resolution
    assert abs(integral / variance - 1.0) < 0.05
    print(f"  peak {peak:.4f} at height {height:.2f}, resolution "
          f"{model.resolution:.5f}")
    print(f"  integral {integral:.4f} against the signal variance "
          f"{variance:.4f}")

    n = 8192
    t, x = _two_tone(n, rng, fs)
    print("welch. the same shape on 8192 samples, averaged over segments:")
    model = spectrum(x, "welch", fs=fs, nperseg=512)
    peak, _ = model.peak()
    spacing = float(model.frequencies[1] - model.frequencies[0])
    integral = float(model.power.sum() * spacing)
    assert model.segments >= 20
    assert abs(peak - 0.1) <= 2.0 * model.resolution
    assert abs(integral / np.var(x) - 1.0) < 0.05
    print(f"  {model.segments} segments of 512, peak {peak:.4f}, "
          f"resolution {model.resolution:.5f}")
    print(f"  integral {integral:.4f} against the signal variance "
          f"{np.var(x):.4f}")

    n = 512
    t, x = _two_tone(n, rng, fs)
    print("multitaper. 512 samples, seven DPSS tapers with nw = 4:")
    model = spectrum(x, "multitaper", fs=fs, nw=4.0)
    peak, _ = model.peak()
    tapers = model.tapers
    assert model.segments == 7
    assert np.allclose(tapers @ tapers.T, np.eye(7), atol=1e-8)
    first = tapers[0]
    transform = np.abs(np.fft.fft(first)) ** 2
    frequency = np.fft.fftfreq(n)
    concentration = float(transform[np.abs(frequency) < 4.0 / n].sum()
                          / transform.sum())
    assert concentration > 0.99
    spacing = float(model.frequencies[1] - model.frequencies[0])
    integral = float(model.power.sum() * spacing)
    assert abs(peak - 0.1) <= 3.0 * model.resolution
    print(f"  peak {peak:.4f}, tapers orthonormal, first-taper concentration "
          f"{concentration:.6f}")
    print(f"  integral {integral:.4f} against the signal variance "
          f"{np.var(x):.4f}")

    print("  leakage from a tone ten times stronger, at 0.01:")
    strong = (10.0 * np.sin(2.0 * np.pi * 0.01 * t)
              + 0.5 * np.sin(2.0 * np.pi * 0.2 * t)
              + 0.2 * rng.normal(0.0, 1.0, n))
    plain = spectrum(strong, "periodogram", fs=fs)
    tapered = spectrum(strong, "multitaper", fs=fs, nw=4.0)
    index = int(np.argmin(np.abs(plain.frequencies - 0.2)))
    band = (plain.frequencies > 0.15) & (plain.frequencies < 0.18)
    plain_ratio = float(plain.power[index] / np.median(plain.power[band]))
    tapered_ratio = float(tapered.power[index]
                          / np.median(tapered.power[band]))
    assert tapered_ratio < plain_ratio
    print(f"  power at 0.2 over the 0.15-0.18 floor: {plain_ratio:.1f} for the "
          f"periodogram, {tapered_ratio:.1f} for the multitaper")

    rng = np.random.default_rng(1)
    print("lomb_scargle. 300 irregular samples over 60 seconds, tone at 1.2:")
    times = np.sort(rng.uniform(0.0, 60.0, 300))
    values = (2.0 * np.sin(2.0 * np.pi * 1.2 * times)
              + 0.5 * rng.normal(0.0, 1.0, times.size))
    model = spectrum(values, "lomb_scargle", times=times, fmax=4.0,
                     oversample=8)
    peak, height = model.peak()
    assert abs(peak - 1.2) < 0.02
    print(f"  peak {peak:.4f} at height {height:.4f}, spacing "
          f"{model.resolution:.5f}, {model.frequencies.size} frequencies")

    fs = 50.0
    t = np.arange(0.0, 10.0, 1.0 / fs)
    chirp = np.sin(2.0 * np.pi * (2.0 * t + 0.25 * t * t))
    print("stft_features. a chirp from 2 Hz to 7 Hz, 128-sample frames:")
    model = spectrum(chirp, "stft_features", fs=fs, nperseg=128, hop=32)
    features = model.features
    assert model.spectrogram.shape == (12, 65)
    assert features["centroid"][-1] > features["centroid"][0] + 2.0
    assert features["rolloff"][-1] > features["rolloff"][0]
    print(f"  spectrogram {model.spectrogram.shape}, centroid "
          f"{features['centroid'][0]:.3f} to {features['centroid'][-1]:.3f} Hz")
    print(f"  rolloff {features['rolloff'][0]:.3f} to "
          f"{features['rolloff'][-1]:.3f} Hz, mean flatness "
          f"{features['flatness_mean']:.5f}")

    bad = [
        lambda: spectrum(x, "spectral_lite"),
        lambda: spectrum(np.zeros((32, 2))),
        lambda: spectrum(np.arange(6.0)),
        lambda: spectrum(np.r_[np.zeros(7), np.nan]),
        lambda: spectrum(x, fs=0.0),
        lambda: spectrum(x, fs=np.nan),
        lambda: spectrum(x, oversample=0),
        lambda: spectrum(x, window="triangular"),
        lambda: spectrum(x, "periodogram", nperseg=2),
        lambda: spectrum(x, "periodogram", nperseg=x.size + 1),
        lambda: spectrum(x, "periodogram", nperseg=32, nfft=16),
        lambda: spectrum(x, "welch", nperseg=32, noverlap=32),
        lambda: spectrum(x, "welch", nperseg=32, noverlap=-1),
        lambda: spectrum(x, "welch", nperseg=x.size + 1),
        lambda: spectrum(x, "multitaper", nw=0.5),
        lambda: spectrum(x, "multitaper", nw=np.nan),
        lambda: spectrum(x, "multitaper", tapers=0),
        lambda: spectrum(x, "multitaper", nfft=x.size - 1),
        lambda: spectrum(x, "stft_features", nperseg=2),
        lambda: spectrum(x, "stft_features", nperseg=x.size + 1),
        lambda: spectrum(x, "stft_features", hop=0),
        lambda: spectrum(x, "stft_features", nperseg=32, nfft=16),
        lambda: spectrum(x, "lomb_scargle", times=np.arange(10.0)),
        lambda: spectrum(x, "lomb_scargle", times=np.arange(x.size)[::-1]),
        lambda: spectrum(x, "lomb_scargle",
                         times=np.r_[np.arange(x.size - 1.0), np.nan]),
        lambda: spectrum(x, "lomb_scargle", fmax=0.0),
        lambda: spectrum(x, "lomb_scargle", fmax=-1.0),
        lambda: spectrum(np.ones(32), "lomb_scargle"),
    ]
    refused = 0
    for call in bad:
        try:
            call()
        except (ValueError, RuntimeError):
            refused += 1
    print(f"{refused} malformed calls refused out of {len(bad)}: an unknown")
    print("  variant, a 2-d, short or non-finite signal, a non-positive or")
    print("  non-finite sampling rate, a non-positive oversampling factor, an")
    print("  unknown window, a segment below three or above the signal, an FFT")
    print("  shorter than the segment, an impossible Welch overlap, a")
    print("  multitaper bandwidth at or below one half, a non-finite nw, a zero")
    print("  taper count, a short multitaper FFT, a short STFT frame or zero")
    print("  hop, Lomb-Scargle times of the wrong length, unordered or")
    print("  non-finite times, a non-positive fmax, and a constant signal")
    print("spectral analysis demo passed")


if __name__ == "__main__":
    demo()
