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
