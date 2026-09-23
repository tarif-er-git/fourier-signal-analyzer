# Architecture

## Core Data Flow

### 1D Signal Workflow

```text
Preset or mouse-drawn signal
        |
        v
Sampling and preprocessing
        |
        v
FourierAnalyzer (calculates a0, a_n, b_n)
        |
        v
FourierSynthesizer (reconstructs DC + N harmonics)
        |
        v
Error, Gibbs, convergence metrics
        |
        v
Matplotlib visualization and PySide6 GUI
```

### 2D Closed Curve Workflow

```text
User-drawn or loaded 2D curve
        |
        v
Cleaning and normalization (Curve2D)
        |
        v
Normalized arc-length resampling
        |
        v
2D Fourier analysis (CurveFourierResult: signed complex Cx, Cy)
        |
        v
Reconstruction with symmetric harmonics (-N..+N)
        |
        v
Pointwise Euclidean error and convergence metrics
        |
        v
Overlay plot, 2D spectrum, 2D error, and 2D epicycle animation
```

## Performance & Caching Architecture

To guarantee interactive responsiveness when dragging the harmonic count slider:

```text
Raw curve / signal
        |
        v
Preprocessing
        |
        v
Fourier coefficients & convergence (computed ONCE and cached)
        |
        v
Interactive reconstruction (slider changes reuse cached coefficients)
```

- In 1D mode, `FourierAnalyzer`, `FourierSynthesizer`, and `ConvergenceResult` are computed once when the signal is generated, drawn, or loaded. When the harmonic slider moves, only `synthesizer.reconstruct(N)`, pointwise error, Gibbs detection, and plot updates are performed.
- In 2D mode, `CurveFourierResult` and `CurveConvergenceResult` are computed once during `Analyze Curve`. Slider updates only re-evaluate reconstruction for harmonic count `N` and look up cached convergence metrics.
- Caches are automatically invalidated and rebuilt whenever a new signal/curve is generated, drawn, or loaded.

## Modules

### `signal/`

- `generator.py`: NumPy-only waveform generators (sine, square, triangle, sawtooth).
- `sampler.py`: Uniform grid creation, point sorting, duplicate time averaging, and interpolation.
- `preprocessor.py`: Array validation, non-finite sample removal, amplitude normalization, and DC preservation.
- `curve2d.py`: `Curve2D` data model for capturing, cleaning, bounding-box calculation, centering, and normalizing closed 2D curves.

### `presets/`

- `signals.py`: Maps preset names to waveform generators.

### `fourier/`

- `analysis.py`: Numerical trigonometric Fourier Series coefficients using trapezoidal integration (`FourierAnalyzer`).
- `synthesis.py`: Synthesizes signals from `a0`, `a_n`, and `b_n` (`FourierSynthesizer`).
- `spectrum.py`: Converts trigonometric coefficient pairs to amplitude and phase (`FourierSpectrum`).
- `fft_comparison.py`: NumPy FFT comparison and educational performance benchmarks.
- `curve2d_analysis.py`: Arc-length parameterization, uniform resampling, and signed complex Fourier analysis/synthesis of 2D curves (`CurveFourierResult`, `CurveReconstruction`).
- `curve2d_error.py`: Pointwise Euclidean distance error and cached harmonic convergence analysis (`CurveErrorMetrics`, `CurveConvergenceResult`).

### `metrics/`

- `error.py`: Signed error, MSE, RMSE, MAE, and maximum absolute error.
- `gibbs.py`: Jump detection, local overshoot/undershoot, and oscillation width (`GibbsResult`).
- `convergence.py`: Evaluates error metrics across multiple harmonic counts (`ConvergenceResult`).

### `visualization/`

- `signal_plot.py`: Plots original, reconstructed, and individual harmonic signals.
- `spectrum_plot.py`: Magnitude and phase stem plots; Fourier Series vs. FFT comparison plots.
- `error_plot.py`: Pointwise error and error vs. harmonic count plots.
- `epicycle.py`: Rotating vector-chain geometry and Matplotlib `FuncAnimation` for 1D Fourier Series.
- `curve2d_spectrum.py`: Structured row models and dominant harmonic extraction for 2D curve spectra.
- `curve2d_epicycle.py`: Vector-chain geometry for planar complex 2D curve epicycles.

### `gui/`

- `main_window.py`: Top-level application controller managing state, modes, menus, and plots.
- `controls.py`: Control panel housing preset selectors, action buttons, harmonic sliders, and metric tables.
- `canvas.py`: Embedded Matplotlib figure canvas supporting 1D signal plotting, 1D drawing, 2D curve drawing, and 2D overlays.
- `drawing.py`: Normalizes mouse gestures into uniform signals.
- `epicycle_window.py`: Dedicated animation window for 1D epicycles.
- `curve2d_epicycle_window.py`: Dedicated window with interactive controls for 2D epicycles.
- `curve2d_spectrum_window.py`: Dedicated window showing 2D harmonic bar charts, phase plots, and sortable tables.
- `curve2d_error_window.py`: Dedicated window showing 2D curve error segments and convergence curves.

### File I/O

- `signal_io.py`: JSON serialization for 1D signals and 2D closed curves, plus CSV exports for reconstructions, spectra, errors, and curve coordinates. Provides clear user-friendly guidance when attempting to cross-load 1D and 2D files.
- `io/signal_io.py`: Compatibility re-export module mirroring all `signal_io` capabilities.

## State Management

- Modes are strictly separated: 1D signal workflows and 2D closed-curve workflows do not share conflicting mutable state.
- Transitioning between modes or resetting clears active signals, caches, derived metrics, and subwindows.
- Controls are dynamically enabled and disabled to prevent invalid actions (e.g. reconstructing before analysis or exporting before data exists).
