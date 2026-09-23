# Mathematics

## Fourier Series

For a periodic signal with period `T`, the project uses the trigonometric Fourier Series:

```text
x(t) = a0/2 + sum[n=1..infinity] [a_n cos(n w0 t) + b_n sin(n w0 t)]
```

`a0/2` is the DC or average value. `n` is the harmonic number. The fundamental frequency is `f0 = 1/T`, and the fundamental angular frequency is `w0 = 2*pi/T`. Harmonic `n` oscillates at `n*f0`.

## Fourier Coefficients

The coefficients are:

```text
a0  = (2/T) integral over one period of x(t) dt

a_n = (2/T) integral over one period of x(t) cos(n w0 t) dt

b_n = (2/T) integral over one period of x(t) sin(n w0 t) dt
```

`FourierAnalyzer` approximates these integrals with the trapezoidal rule on uniformly sampled data. It supports an endpoint-inclusive representation `[t0, t0 + T]` and removes the repeated periodic endpoint before integration. It also supports endpoint-exclusive samples and infers the period from the sample spacing.

## Fourier Synthesis

With the first `N` harmonics, the synthesizer computes:

```text
x_N(t) = a0/2 + sum[n=1..N]
         [a_n cos(n w0 t) + b_n sin(n w0 t)]
```

`N` controls the approximation detail. `N=0` contains only the DC component. Smooth signals can be reconstructed accurately with few terms; discontinuous signals usually need more terms.

## Amplitude and Phase Form

Each coefficient pair can be written as one rotating harmonic:

```text
A_n = sqrt(a_n^2 + b_n^2)
phi_n = atan2(-b_n, a_n)
```

Therefore:

```text
a_n cos(n w0 t) + b_n sin(n w0 t)
= A_n cos(n w0 t + phi_n)
```

`atan2` is used because it preserves the correct quadrant and handles zero or negative cosine coefficients safely.

## Reconstruction Error

The point-wise error is:

```text
e(t) = x(t) - x_N(t)
```

The project reports several summaries:

```text
MSE = mean(e(t)^2)
RMSE = sqrt(MSE)
MAE = mean(abs(e(t)))
Maximum Error = max(abs(e(t)))
```

MSE emphasizes larger errors because it squares them. RMSE returns to signal units. MAE gives the average absolute deviation. Maximum error shows the worst local deviation.

## Gibbs Phenomenon

A jump discontinuity is a sudden change in signal value. Fourier partial sums oscillate near such a jump. Increasing `N` generally narrows the oscillatory region and improves the signal away from the jump, but the characteristic overshoot does not simply disappear. For an ideal normalized jump, the limiting overshoot reference is approximately 9 percent of the jump magnitude.

The project estimates discontinuities from unusually large adjacent differences, measures local overshoot and undershoot in a configurable window, and keeps measured values separate from the theoretical reference.

## FFT

The DFT represents a finite sampled signal with complex frequency bins. The FFT is an efficient algorithm for computing that DFT. NumPy returns complex `X[k]` values, not directly the real `a_n` and `b_n` arrays used by the educational analyzer. The project normalizes the real-signal `rfft` output by the sample count and doubles positive-frequency bins except DC and the even-length Nyquist bin.

Direct numerical coefficient calculation is approximately `O(N^2)` when many harmonics are evaluated. FFT is approximately `O(N log N)`. Timings in this project are educational measurements and depend on the machine and NumPy implementation.

## Epicycles

A harmonic becomes a rotating vector. Its radius is `A_n`, its phase is `phi_n`, and its angular velocity is `n*w0`. Vectors are chained head-to-tail. The x-coordinate of the final endpoint equals the Fourier reconstruction because each vector contributes `A_n*cos(n*w0*t + phi_n)` to that coordinate. The y-coordinate makes the rotating-vector geometry visible.

## 2D Closed Curve Fourier Analysis

A closed curve in the 2D plane can be represented as a complex-valued periodic function of a parameter `t`:

```text
z(t) = x(t) + j y(t),  t in [0, 1)
```

### Arc-Length Parameterization

To avoid clustering distortion from variable drawing speeds, the polygonal curve is parameterized by its normalized cumulative arc length:

```text
s_0 = 0
s_i = s_{i-1} + ||p_i - p_{i-1}||_2,  for i = 1..M-1
L = s_{M-1} + ||p_0 - p_{M-1}||_2
t_i = s_i / L
```

The curve is then resampled at uniform intervals `t_m = m / num_samples` for `m = 0..num_samples - 1`.

### Complex Fourier Series

The complex Fourier coefficients for the coordinates are obtained via the discrete Fourier transform:

```text
c_k = (1 / M) sum_{m=0}^{M-1} z(t_m) exp(-j 2 pi k m / M) = X_k + j Y_k
```

where:
- `k = 0` is the DC component representing the centroid `(mean(x), mean(y))`.
- For real coordinate signals `x(t)` and `y(t)`, the coefficients satisfy Hermitian symmetry: `X_{-k} = X_k^*` and `Y_{-k} = Y_k^*`.

### Symmetric Harmonic Reconstruction

Reconstructing with harmonic count `N` sums symmetric positive and negative harmonics:

```text
z_N(t) = sum_{k=-N}^{N} c_k exp(j 2 pi k t) = x_N(t) + j y_N(t)
```

Because of Hermitian symmetry, the real and imaginary parts of `z_N(t)` correspond exactly to the independent trigonometric reconstructions of `x(t)` and `y(t)`:

```text
x_N(t) = Re( sum_{k=-N}^{N} X_k exp(j 2 pi k t) )
y_N(t) = Re( sum_{k=-N}^{N} Y_k exp(j 2 pi k t) )
```

### 2D Geometric Error Metrics

For each sampled parameter point `t_m`, the pointwise Euclidean distance error is:

```text
e_m = sqrt( (x[m] - x_N[m])^2 + (y[m] - y_N[m])^2 )
```

Aggregate metrics are computed over all `M` samples:

```text
MSE = (1 / M) sum_{m=0}^{M-1} e_m^2
RMSE = sqrt(MSE)
Mean Error = (1 / M) sum_{m=0}^{M-1} e_m
Max Error = max_{m} e_m
```

### 2D Complex Epicycles

In 2D epicycles, each harmonic term `c_k exp(j 2 pi k t)` is a rotating planar vector with radius `|c_k|` and instantaneous phase angle. Vectors for `k in {-N, ..., N}` are chained head-to-tail starting from the origin. The final tip of the chain traces out the reconstructed curve `z_N(t)` directly in the Cartesian plane.
