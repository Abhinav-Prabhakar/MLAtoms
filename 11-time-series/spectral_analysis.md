# Spectral analysis — five ways to ask what a signal is made of

`spectral_analysis.py` · Time series · Tier 2 · 5 variants

## Intuition

A time series is a sum of oscillations. Spectral analysis turns that sentence
into arithmetic: project the signal onto sinusoids of every frequency and see
how much energy each one carries. The projection is the discrete Fourier
transform, and its squared magnitude is the periodogram.

The periodogram is the right idea with a practical defect. Each frequency bin
is a single complex number, so its squared magnitude is an exponentially
distributed random variable: noisy, and no less noisy as the series grows
longer. Adding data adds bins, not precision. Every other variant in this file
is a way of buying variance reduction without giving up too much frequency
resolution:

- **`periodogram`** is the raw estimate. Best resolution, worst variance, and
  heavy leakage from strong low-frequency components.
- **`welch`** cuts the series into overlapping segments, computes a periodogram
  of each, and averages. Variance falls roughly as $1/K$; resolution falls to
  $f_s/L$ for segment length $L$.
- **`multitaper`** keeps the full length but multiplies it by several
  orthogonal tapers and averages the results. It reduces leakage rather than
  resolution, which is why it finds a weak tone next to a strong one.
- **`lomb_scargle`** fits sinusoids directly at the observation times, so it
  works when samples are unevenly spaced and the FFT grid does not exist.
- **`stft_features`** keeps time as well as frequency: it transforms short
  frames and summarises each frame by centroid, bandwidth, rolloff and
  flatness, which is what a downstream model actually consumes.

The demo checks the two properties that separate a working implementation from
a plausible-looking one: the integral of the spectrum must recover the signal
variance, and the peak must land within a bin or two of the planted frequency.
It then measures the leakage that the multitaper is supposed to fix.

## Variants

| variant | estimator | demo result |
|:--|:--|:--|
| `periodogram` | one windowed transform | peak 0.0996, integral 2.4251 |
| `welch` | average of segments | 31 segments, integral 2.3773 |
| `multitaper` | average of DPSS tapers | 7 tapers, leakage ratio 79.2 |
| `lomb_scargle` | sinusoid fit at sample times | peak 1.1998 against 1.2 |
| `stft_features` | framed transform plus features | centroid 2.640 → 6.160 Hz |

**`periodogram`** on 1 024 samples with tones at $0.10$ and $0.23$ cycles per
sample puts its maximum at $0.0996$, one bin below the truth, with resolution
$0.00098$. The integral of the one-sided density is $2.4251$ against a signal
variance of $2.3556$ — a 2.9% overshoot from the finite-sample noise and the
DC/Nyquist half-bins. That is the estimator working correctly, and also the
estimator being noisy.

**`welch`** uses 31 half-overlapping segments of 512 samples from an 8 192-point
series. Averaging cuts the bin-to-bin scatter enough that the peak is clean at
$0.0996$, and the integral $2.3773$ against $2.3708$ is within 0.3%. The price
is resolution: $0.00195$, twice the periodogram's, so two tones closer than
about $0.002$ cycles per sample would merge into one hump.

**`multitaper`** builds seven discrete prolate spheroidal sequences at
time-bandwidth product $NW=4$. The first taper concentrates $1.000000$ of its
power inside $|f|<W$, and the tapers are orthonormal to machine precision. On a
signal with a tone ten times stronger at $0.01$, the power at $0.2$ over the
local floor is $838.0$ for the periodogram and $79.2$ for the multitaper: the
leakage from the strong tone has been suppressed by a factor of ten.

**`lomb_scargle`** receives 300 samples at irregular times over 60 seconds and
finds the planted $1.2$ Hz tone at $1.1998$, with a grid spacing of $0.00210$
and 1 906 candidate frequencies. The peak height $0.8551$ is not one because
the normalisation is against total variance, part of which is noise; the
location, not the height, is the estimate of interest.

**`stft_features`** tracks a chirp that sweeps from 2 Hz to 7 Hz. With
128-sample frames at a 50 Hz sampling rate it produces a $12\times65$
spectrogram, a centroid that rises from $2.640$ to $6.160$ Hz, and an 85%
rolloff that rises from $2.734$ to $6.250$ Hz. The mean flatness prints as
$0.00000$: a clean chirp is tonal, so its geometric mean is many orders of
magnitude below its arithmetic mean.

## Derivation

### The discrete Fourier transform

For samples $x_0,\dots,x_{n-1}$ and a window $w_t$, the windowed transform is

$$
X_k=\sum_{t=0}^{n-1}w_t\,x_t\,e^{-i2\pi kt/n},
\qquad k=0,\dots,n-1 .
\qquad (1)
$$

The frequency attached to bin $k$ is $f_k=kf_s/n$, so neighbouring bins are
$\Delta f=f_s/n$ apart and the highest representable frequency is $f_s/2$.
The transform is a change of basis, not a model: it loses nothing and assumes
nothing beyond finite sampling.

### The periodogram

The periodogram estimates the spectral density by

$$
\hat S(f_k)=\frac{\lvert X_k\rvert^2}{f_s\sum_{t}w_t^2},
\qquad (2)
$$

with the non-DC, non-Nyquist bins doubled to fold negative frequencies onto the
one-sided axis. The denominator is the window energy, which is $n$ for a
rectangular window and roughly $3n/8$ for a Hann window. That normalisation is
what makes the integral recover the variance:

$$
\int_0^{f_s/2}\hat S(f)\,df\;\approx\;\frac1n\sum_t (x_t-\bar x)^2 .
\qquad (3)
$$

The demo checks (3) numerically. It is the first thing to test, because a
factor-of-two error in the one-sided folding is invisible in a plot and wrong
by 3 dB in every downstream number.

The defect is in the sampling distribution. For a Gaussian process,

$$
\mathrm{Var}\bigl(\hat S(f)\bigr)\approx S(f)^2
\qquad (4)
$$

at every frequency, independent of $n$. The estimate is asymptotically
unbiased but not consistent: more data gives a finer grid of equally noisy
points. Smoothing over neighbouring bins, averaging over segments or averaging
over tapers are the three ways to reduce (4).

A finite record also convolves the true spectrum with the window's spectral
window. A rectangular window has sidelobes that decay slowly, so a tone at
$0.01$ with amplitude ten leaks across the whole band; a Hann window trades a
wider main lobe for sidelobes below $-31$ dB. This is the leakage the demo
measures.

### Welch averaging

Split the series into $K$ segments of length $L$, overlapping by $L/2$, and
average their periodograms:

$$
\hat S_W(f)=\frac1K\sum_{k=1}^{K}\hat S_k(f).
\qquad (5)
$$

If the segments were independent, (4) would give
$\mathrm{Var}(\hat S_W)\approx S^2/K$. Overlapping segments are correlated, so
the gain is smaller than $1/K$; half overlap is the usual compromise because it
doubles the segment count for a modest correlation penalty. Resolution is now
$f_s/L$, not $f_s/n$: averaging buys variance with bias, and the bias is a
blurred peak.

### Multitaper estimation

Instead of one window, use $K$ orthogonal windows $v_t^{(j)}$ and average their
periodograms. The tapers are chosen to maximise the fraction of their energy
inside a bandwidth $[-W,W]$:

$$
\lambda=\frac{\int_{-W}^{W}\lvert V(f)\rvert^2\,df}
{\int_{-1/2}^{1/2}\lvert V(f)\rvert^2\,df}.
\qquad (6)
$$

The maximiser solves a symmetric tridiagonal eigenproblem. With
$W=NW/n$ and $t=0,\dots,n-1$,

$$
A_{tt}=\Bigl(\frac{n-1}{2}-t\Bigr)^2\cos(2\pi W),\qquad
A_{t,t+1}=A_{t+1,t}=\frac{t(n-t)}{2},
\qquad (7)
$$

and the leading $2NW-1$ eigenvectors are the discrete prolate spheroidal
sequences. Each has most of its energy in the band, so each produces a
periodogram with small leakage, and the estimates are approximately
independent because the tapers are orthogonal. The multitaper estimate is

$$
\hat S_M(f)=\frac1K\sum_{j=1}^{K}
\frac{\lvert\sum_t v_t^{(j)}x_t e^{-i2\pi ft}\rvert^2}
{f_s\sum_t (v_t^{(j)})^2}.
\qquad (8)
$$

The demo checks both halves of the claim: the tapers are orthonormal, the first
one concentrates essentially all its power in the band, and the leakage floor
next to a strong tone is an order of magnitude lower than the periodogram's.

### Lomb-Scargle

When the observation times $t_i$ are irregular, there is no uniform grid and no
FFT. Fit a sinusoid at frequency $f$ by least squares instead, allowing a phase
shift $\tau$ chosen so the sine and cosine bases are orthogonal:

$$
\hat x(t)=a\cos\bigl(2\pi f(t-\tau)\bigr)+b\sin\bigl(2\pi f(t-\tau)\bigr),
\qquad
\tan(4\pi f\tau)=\frac{\sum_i\sin(4\pi ft_i)}{\sum_i\cos(4\pi ft_i)} .
\qquad (9)
$$

For fixed $f$ the two coefficients separate, and the normalised power is

$$
P(f)=\frac12\left[
\frac{\bigl(\sum_i x_i\cos 2\pi f(t_i-\tau)\bigr)^2}
{\sum_i\cos^2 2\pi f(t_i-\tau)}
+
\frac{\bigl(\sum_i x_i\sin 2\pi f(t_i-\tau)\bigr)^2}
{\sum_i\sin^2 2\pi f(t_i-\tau)}
\right]\Big/\Bigl(\tfrac12 n\hat\sigma^2\Bigr).
\qquad (10)
$$

The code evaluates (10) on a grid with spacing $1/(\text{oversample}\cdot
\text{span})$, which is the natural resolution of a record of that length. The
default upper limit is the Nyquist frequency implied by the median spacing; for
irregular sampling that is a guide, not a hard boundary, and aliases can appear
above it.

### Short-time features

The STFT applies (1) to short frames:

$$
X_{m,k}=\sum_{t=0}^{L-1}w_t\,x_{mH+t}\,e^{-i2\pi kt/L},
\qquad
P_{m,k}=\frac{\lvert X_{m,k}\rvert^2}{f_s\sum_t w_t^2},
\qquad (11)
$$

where $H$ is the hop. Frame length is the time-frequency trade-off in its
purest form: a long frame resolves close frequencies but smears events in time,
a short frame does the reverse. The demo uses 128 samples at 50 Hz — 2.56 s per
frame — and a 32-sample hop.

Each frame is then reduced to four numbers. The spectral centroid is the
power-weighted mean frequency,

$$
c_m=\frac{\sum_k f_k P_{m,k}}{\sum_k P_{m,k}},
\qquad (12)
$$

and the bandwidth is the matching standard deviation,

$$
b_m=\sqrt{\frac{\sum_k (f_k-c_m)^2P_{m,k}}{\sum_k P_{m,k}}} .
\qquad (13)
$$

The rolloff is the smallest $f_k$ whose cumulative power reaches 85% of the
frame total, a robust alternative to the maximum. Flatness is the ratio of the
geometric to the arithmetic mean,

$$
\phi_m=\frac{\exp\bigl(\frac1K\sum_k\log(P_{m,k}+\epsilon)\bigr)}
{\frac1K\sum_k P_{m,k}},
\qquad (14)
$$

which is near one for white noise and near zero for a tone. The $\epsilon$
guards the logarithm of an empty bin; without it a silent frame produces
$-\infty$ and the whole feature vector becomes unusable.

## Code walkthrough

All five variants share one normalisation. The window function is where the
leakage trade-off is made, and it is deliberately a small table rather than a
general window factory:

```python
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
```

The three cosine windows differ only in how much weight they put on the ends.
Hann reaches zero at both ends, which removes the discontinuity that causes
leakage; Blackman adds a second harmonic of the cosine to push the sidelobes
lower still, at the cost of a wider main lobe.

The shared density calculation implements (2), including the one-sided
doubling. The `nfft % 2` branch is the detail that is easy to get wrong: for an
even transform length the Nyquist bin is its own mirror image and must not be
doubled, while for an odd length every bin above DC has a partner:

```python
    x = x - x.mean()
    window = _window(name, x.size)
    transform = np.fft.rfft(x * window, n=nfft)
    power = np.abs(transform) ** 2 / (fs * float(window @ window))
    if nfft % 2 == 0:
        power[1:-1] *= 2.0
    else:
        power[1:] *= 2.0
    return np.fft.rfftfreq(nfft, 1.0 / fs), power
```

The mean is removed before the transform, so the DC bin carries only the
residual and the integral in (3) is the variance rather than the second moment.
Note that `nfft` may exceed the segment length: zero padding interpolates the
spectrum onto a finer grid, but it does not add resolution.

The DPSS construction is (7) written as three diagonals. Building the full
matrix is wasteful for large $n$, but it keeps the correspondence to the
eigenproblem explicit, and the taper count is what the demo inspects:

```python
    half = 0.5 * (n - 1)
    index = np.arange(n, dtype=float)
    diagonal = (half - index) ** 2 * np.cos(2.0 * np.pi * nw / n)
    off = index[1:] * (n - index[1:]) / 2.0
    matrix = np.diag(diagonal) + np.diag(off, 1) + np.diag(off, -1)
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    count = max(1, int(np.floor(2.0 * nw - 1.0)))
```

`nw` is the time-bandwidth product $NW$, so the bandwidth is $NW/n$ and the
number of usable tapers is $2NW-1$. The eigenvectors are sorted descending
because `eigh` returns ascending eigenvalues and the leading taper is the one
with the largest concentration. The sign flip makes the largest coefficient
positive, which matters only for reproducibility of plots.

Welch is a loop over a view, not a copy. `sliding_window_view` creates the
frames by stride manipulation, and the `[::hop]` slice selects the overlapping
ones:

```python
        frames = _frames(x, size, size - overlap)
        length = size if nfft is None else int(nfft)
        if length < size:
            raise ValueError("nfft must be at least nperseg")
        stack = np.empty((frames.shape[0], length // 2 + 1))
        for row, frame in enumerate(frames):
            frequencies, stack[row] = _density(frame, fs, window, length)
        return Spectrum(frequencies, stack.mean(axis=0), variant, fs, fs / size,
                        segments=int(frames.shape[0]))
```

The reported resolution is `fs / size`, the segment length, not the full record
length. That is the honest number: Welch has thrown away the extra resolution
in exchange for the variance reduction.

The multitaper loop is the same averaging idea with tapers instead of segments.
Each taper gets its own normalisation by its energy, because the tapers have
different shapes even though they are all unit norm:

```python
        stack = np.empty((sequences.shape[0], length // 2 + 1))
        for row, taper in enumerate(sequences):
            transform = np.fft.rfft((x - x.mean()) * taper, n=length)
            power = np.abs(transform) ** 2 / (fs * float(taper @ taper))
            if length % 2 == 0:
                power[1:-1] *= 2.0
            else:
                power[1:] *= 2.0
            stack[row] = power
```

Because the tapers are orthogonal, the $K$ estimates are approximately
uncorrelated, which is what makes the average worth doing. The mean is removed
once, outside the loop, so every taper sees the same centred signal.

Lomb-Scargle is the only variant that is a direct transcription of a
least-squares problem. The phase is computed from the data at each frequency,
then the cosine and sine projections are normalised separately:

```python
    for row, frequency in enumerate(frequencies):
        angle = 2.0 * np.pi * frequency * times
        phase = np.arctan2(np.sin(angle).sum(), np.cos(angle).sum()) / (
            2.0 * np.pi * frequency)
        cosine = np.cos(2.0 * np.pi * frequency * (times - phase))
        sine = np.sin(2.0 * np.pi * frequency * (times - phase))
        left = float((centred @ cosine) ** 2 / (cosine @ cosine))
        right = float((centred @ sine) ** 2 / (sine @ sine))
        power[row] = 0.5 * (left + right) / (0.5 * x.size * variance)
```

`arctan2` gives the phase in $(-1/2,1/2]$ cycles, and dividing by the frequency
converts it to a time shift. Computing the phase this way is equivalent to the
closed form in (9) and avoids a separate arctangent identity. The final
division by $n\hat\sigma^2/2$ is the normalisation that makes a pure sinusoid
of amplitude $A$ peak near $A^2/(2\hat\sigma^2)$.

The STFT features are vectorised across frames. Each row of `block` is one
frame's power spectrum, so the centroid and bandwidth are matrix-vector
products against the frequency axis:

```python
        total = block.sum(axis=1)
        total = np.where(total > 0.0, total, 1.0)
        centroid = block @ frequencies / total
        bandwidth = np.sqrt(
            np.maximum(block @ (frequencies ** 2) / total - centroid ** 2, 0.0))
        cumulative = np.cumsum(block, axis=1)
```

The `maximum(..., 0.0)` is not decoration: the variance formula subtracts a
squared mean from a mean square, and floating-point cancellation can make the
difference a tiny negative number, whose square root would be `nan`.

Rolloff and flatness finish the feature dictionary. Rolloff searches the
cumulative sum; flatness is the exponential of the mean log, with an epsilon
that keeps silent bins finite:

```python
        rolloff = np.empty(block.shape[0])
        for row in range(block.shape[0]):
            threshold = 0.85 * cumulative[row, -1]
            rolloff[row] = frequencies[int(np.searchsorted(cumulative[row],
                                                           threshold))]
        epsilon = 1e-18
        flatness = np.exp(np.mean(np.log(block + epsilon), axis=1)
                          ) / np.maximum(block.mean(axis=1), epsilon)
```

`searchsorted` returns the first index whose cumulative value reaches the
threshold, which is exactly the definition of the rolloff. The frame time
stored alongside is the centre, `(row * hop + size / 2) / fs`, so that a feature
at time $t$ refers to the middle of the window that produced it rather than its
start.

## Complexity

With $n$ samples, $K$ segments or tapers of length $L$, $F$ candidate
frequencies and $M$ frames:

- **One FFT:** $O(L\log L)$ time and $O(L)$ memory.
- **Periodogram:** one FFT of length $n$, so $O(n\log n)$.
- **Welch:** $K$ FFTs of length $L$, so $O(KL\log L)$; memory $O(KL)$ for the
  stacked spectra if they are kept, $O(L)$ if averaged streaming.
- **Multitaper:** $K$ FFTs plus the DPSS eigenproblem. The code calls
  `eigh` on a dense $n\times n$ matrix, so the construction costs $O(n^3)$ and
  dominates for large $n$; a production implementation exploits the
  tridiagonal structure for $O(n^2)$ or uses a specialized routine.
- **Lomb-Scargle:** $O(nF)$ with no FFT, where
  $F\approx\text{oversample}\cdot\text{span}\cdot f_{\max}$.
- **STFT:** $O(M L\log L)$ for the transforms plus $O(ML)$ for the features.
- **Peak search and integration:** $O(n)$ extra.

The demo's slowest call is the 512-point multitaper fit, at about 0.03 s; the
8 192-point Welch estimate and the 1 024-point periodogram are effectively
instantaneous by comparison. The Lomb-Scargle loop over 1 906 frequencies on
300 points is the other visible cost, and it is linear in both.

## Failure modes

**The periodogram is not a consistent estimator.** Its variance does not shrink
with $n$; it only looks smoother because the bins get closer together. A plot
of a periodogram of a long white-noise series is a forest of spikes whose height
is meaningless. Smooth it, average segments, or use tapers before reading a
peak height.

**Leakage masquerades as signal.** A strong trend or a large low-frequency
component spreads power across every bin. In the demo the periodogram's floor
next to a strong tone is ten times the multitaper's. Detrend before
transforming, and use a window with low sidelobes. A peak that moves when you
change the window is leakage, not a resonance.

**Welch resolution is set by the segment, not the record.** Users often expect
more data to resolve closer frequencies; with Welch it does not. If two tones
are closer than $f_s/L$, choose a longer segment and accept more variance, or
switch to a multitaper estimate with a smaller bandwidth.

**The time-bandwidth product is a real trade-off.** Increasing $NW$ gives more
tapers, hence less variance, but each taper occupies a wider band, so nearby
features blur. The demo's $NW=4$ yields seven tapers and a main lobe about
$8/n$ wide. There is no setting that gives both a narrow lobe and many
independent estimates.

**Irregular sampling has aliases too.** Lomb-Scargle handles gaps but not
aliasing: a sampling pattern with a strong periodicity, such as nightly
observations, imprints that periodicity on the spectrum. The default frequency
limit is the Nyquist rate of the median spacing, which is a heuristic. Check
that a peak survives a random subsample and a different frequency grid before
believing it.

**Normalisations differ between implementations.** The Lomb-Scargle power in
(10) is dimensionless and bounded by roughly the squared amplitude over the
variance; other packages return power spectral density, or an unnormalised
periodogram. Peak heights from two libraries are not comparable, and a
"significant" peak under one normalisation may be ordinary under another.

**Frame length hides events.** A 128-sample STFT frame at 50 Hz spans 2.56 s. A
100 ms transient is averaged into a frame and its spectral signature is diluted.
Shorten the frame for transients, lengthen it for pitch, and always report the
hop and frame length with the features, because a centroid computed at a
different frame size is a different number.

**Flatness needs a floor.** Without the epsilon, a frame with an exact zero bin
gives $\log 0=-\infty$ and the geometric mean collapses to zero even when the
rest of the frame is loud. The demo's chirp is genuinely tonal, so its flatness
is below the printed precision; do not read that as a bug.

## Applications

Spectral analysis is the default first look at any sensor signal. In audio it
separates pitch from timbre, and the STFT features are the input to speech
recognisers, music taggers and anomaly detectors. In electroencephalography,
Welch averaging over two-second epochs is the standard way to estimate band
power in the delta, theta, alpha and beta ranges, and the multitaper is
preferred when the recording is short and leakage from the strong alpha rhythm
would contaminate the higher bands.

Astronomy is the home of Lomb-Scargle: light curves are sampled when the
telescope is open and the weather allows, so the time grid is irregular by
construction. The same estimator appears in paleoclimate work on varved
sediments and in genomics on unevenly spaced methylation measurements. In
vibration monitoring, the multitaper is used to detect a failing bearing tone
against the broad noise of a machine, and the STFT tracks how that tone drifts
as the fault develops.

The method is the wrong tool when the signal is non-stationary on the scale of
the record and the question is about time localisation: a single periodogram
averages the whole record, and the STFT only localises to a frame. For
time-varying spectra with better resolution, wavelets or the empirical mode
decomposition are the natural next step. It is also the wrong tool for very
short records, where the frequency grid is coarser than the features of
interest; there, a parametric model such as an AR process estimated in the time
domain gives sharper peaks from the same data.

## References

- Schuster, A. (1898). *On the investigation of hidden periodicities with
  application to a supposed 26 day period of meteorological phenomena.*
  Terrestrial Magnetism 3(1), 13–41. The periodogram.
- Welch, P. D. (1967). *The use of fast Fourier transform for the estimation of
  power spectra: a method based on time averaging over short, modified
  periodograms.* IEEE Transactions on Audio and Electroacoustics 15(2), 70–73.
  The segment-averaging estimator of (5).
- Thomson, D. J. (1982). *Spectrum estimation and harmonic analysis.*
  Proceedings of the IEEE 70(9), 1055–1096. Multitaper estimation, the DPSS
  construction and the leakage argument.
- Slepian, D. (1978). *Prolate spheroidal wave functions, Fourier analysis, and
  uncertainty V: the discrete case.* Bell System Technical Journal 57(5),
  1371–1430. The discrete sequences behind (7).
- Lomb, N. R. (1976). *Least-squares frequency analysis of unequally spaced
  data.* Astrophysics and Space Science 39(2), 447–462. The original
  least-squares periodogram.
- Scargle, J. D. (1982). *Studies in astronomical time series analysis II:
  statistical aspects of spectral analysis of unevenly spaced data.*
  Astrophysical Journal 263, 835–853. The normalisation and phase convention of
  (9)–(10).
- Harris, F. J. (1978). *On the use of windows for harmonic analysis with the
  discrete Fourier transform.* Proceedings of the IEEE 66(1), 51–83. The
  window comparison behind the leakage discussion.
- Percival, D. B., Walden, A. T. (1993). *Spectral Analysis for Physical
  Applications.* Cambridge University Press. The tridiagonal eigenproblem and
  practical multitaper guidance.
