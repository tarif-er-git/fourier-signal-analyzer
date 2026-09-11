# Presentation Script

## Slide 1: Title

**Signal Sketch and Decompose**

An interactive educational Fourier Series tool.

## Slide 2: Motivation

Periodic signals can be represented as sums of simple harmonics. The project makes that representation visible instead of showing only formulas or final plots.

## Slide 3: Objectives

- Generate or draw a signal.
- Calculate Fourier coefficients.
- Reconstruct with a selectable `N`.
- Visualize error and spectra.
- Demonstrate Gibbs phenomenon.
- Compare direct Fourier calculation with FFT.
- Show the same harmonics as epicycles.

## Slide 4: Architecture

```text
Signal source
    -> sampling/preprocessing
    -> Fourier analysis
    -> Fourier synthesis
    -> metrics
    -> visualization and GUI
```

Emphasize that the GUI orchestrates modules rather than duplicating formulas.

## Slide 5: Fourier Mathematics

Show:

```text
x(t) = a0/2 + sum[a_n cos(n w0 t) + b_n sin(n w0 t)]
```

Explain DC, fundamental frequency, harmonics, and the role of `N`.

## Slide 6: Sine Demonstration

1. Generate Sine.
2. Reconstruct with `N=1`.
3. Show the nearly exact reconstruction and small metrics.
4. Open the magnitude/phase or FFT comparison view.

Explain that one harmonic is enough for a pure sine wave.

## Slide 7: Square Wave and Gibbs

Use `N=1`, `5`, `10`, `20`, and `50`.

Point out:

- More harmonics improve the waveform away from jumps.
- Oscillations remain near the jump.
- The Gibbs panel measures local behavior.

## Slide 8: Error Analysis

Show MSE, RMSE, MAE, maximum error, and the error-versus-N graph. Explain that MSE generally improves but need not decrease strictly for every individual N.

## Slide 9: FFT Comparison

Show the Fourier Series and normalized FFT magnitude curves. Explain that FFT produces complex DFT bins, while the educational analyzer computes `a_n` and `b_n` numerically. Mention the illustrative timing comparison.

## Slide 10: Epicycles

Open the epicycle view with a sine at `N=1`, then a square wave at `N=5`. Explain radius, phase, angular velocity, vector chaining, and the endpoint reconstruction.

## Conclusion

The project demonstrates Fourier decomposition interactively, shows reconstruction improvement, makes Gibbs behavior measurable, compares direct calculation with FFT, and connects harmonic equations to rotating vectors.

## Suggested Timing

- Motivation and objectives: 1 minute
- Architecture and mathematics: 2 minutes
- Sine and square demonstrations: 3 minutes
- Error, Gibbs, and FFT comparison: 2 minutes
- Epicycles and conclusion: 1 to 2 minutes
