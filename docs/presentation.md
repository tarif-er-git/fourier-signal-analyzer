# Presentation Script

## Slide 1: Title

**Signal Sketch and Decompose**

An interactive educational Fourier Series tool for 1D signals and 2D closed curves.

## Slide 2: Motivation

Periodic signals and closed shapes can be represented as sums of simple harmonic components. The project makes that representation visible instead of showing only formulas or final plots, bridging the gap between math and visualization.

## Slide 3: Objectives

- Generate or draw a 1D signal.
- Draw freehand 2D closed curves.
- Calculate numerical and complex Fourier coefficients.
- Reconstruct with a selectable `N`.
- Visualize error, convergence, and spectra.
- Demonstrate Gibbs phenomenon in 1D.
- Compare direct Fourier calculation with FFT.
- Show the harmonics dynamically as 1D and 2D rotating epicycles.

## Slide 4: Architecture

```text
Signal / Curve source
    -> sampling/preprocessing (arc-length normalization for 2D)
    -> Fourier analysis (trigonometric for 1D, complex for 2D)
    -> Fourier synthesis
    -> metrics (Gibbs, Euclidean distance, convergence)
    -> visualization and GUI
```

Emphasize that the GUI orchestrates modules rather than duplicating formulas, and the 1D/2D pipelines are strictly isolated.

## Slide 5: 1D Fourier Mathematics

Show:

```text
x(t) = a0/2 + sum[a_n cos(n w0 t) + b_n sin(n w0 t)]
```

Explain DC, fundamental frequency, harmonics, and the role of `N`.

## Slide 6: 2D Complex Fourier Mathematics

Show:

```text
z(t) = sum_{k=-N}^{N} C_k e^{j 2 \pi k t}
```

Explain parameterization by arc-length `t`, complex coefficients $C_k = C_x + jC_y$, and the need for symmetric negative and positive harmonics for 2D motion.

---

# Live Demonstration Script

## Demo 1: Sine Wave (1D)

1. Generate Sine.
2. Reconstruct with `N=1`.
3. Show the nearly exact reconstruction and small metrics.
4. Open the FFT comparison view.
5. Explain that one harmonic is enough for a pure sine wave.

## Demo 2: Square Wave and Gibbs (1D)

Use `N=1`, `5`, `10`, `20`, and `50`.

Point out:

- More harmonics improve the waveform away from jumps.
- Oscillations remain near the jump (Gibbs Phenomenon).
- The Gibbs panel measures local behavior.

## Demo 3: Error Analysis (1D)

Show MSE, RMSE, MAE, maximum error, and the error-versus-N graph. Explain that MSE generally improves but need not decrease strictly for every individual N.

## Demo 4: 1D Epicycles

Open the 1D epicycle view with a square wave at `N=5`. Explain radius, phase, angular velocity, vector chaining, and how the vertical projection forms the reconstructed signal.

## Demo 5: Drawing a 2D Curve

1. Click `Draw 2D Curve`.
2. Draw a distinct, asymmetric shape (e.g., a cursive letter or a star).
3. Click `Finish / Close Curve`.
4. Click `Analyze Curve`.
5. Explain the uniform resampling by arc-length so drawing speed doesn't matter.

## Demo 6: 2D Reconstruction and Error

1. Slide `2D Harmonics N` from `1` to `20`.
2. Watch the dashed reconstructed curve conform to the drawn shape.
3. Click `2D Error & Convergence`.
4. Show the pointwise error segments and the Error vs. $N$ plot.

## Demo 7: 2D Spectrum

1. Click `2D Harmonic Spectrum`.
2. Sort the table by Combined Magnitude to find the dominant frequencies (the largest circles).
3. Explain that $k=0$ is the geometric center (DC).

## Demo 8: 2D Epicycles (The Finale)

1. Click `2D Epicycle View`.
2. Press Play.
3. Show how the chained rotating vectors, both positive (counter-clockwise) and negative (clockwise), trace the shape exactly.
4. Adjust the `Harmonics` slider in the window to see how fewer circles produce a smoother, less detailed trace.

---

## Conclusion

The project demonstrates Fourier decomposition interactively, makes abstract math measurable, and connects trigonometric and complex exponential equations to beautiful, tangible geometric animations.

## Suggested Timing

- Motivation, objectives, math: 2 minutes
- 1D Demonstrations (Sine, Square, Gibbs, FFT): 3 minutes
- 2D Demonstrations (Drawing, Error, Spectrum): 3 minutes
- Epicycles (1D and 2D) and conclusion: 2 minutes
