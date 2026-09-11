# Architecture

## Core Data Flow

```text
Preset or mouse-drawn signal
        |
        v
Sampling and preprocessing
        |
        v
FourierAnalyzer
        |
        v
FourierSynthesizer
        |
        v
Error, Gibbs, convergence metrics
        |
        v
Matplotlib visualization and PySide6 GUI
```

## Modules

### `signal/`

`generator.py` contains NumPy-only sine, square, triangle, and sawtooth generators. `sampler.py` creates uniform grids, sorts arbitrary points, averages duplicate times, and interpolates. `preprocessor.py` validates finite arrays, optionally removes non-finite samples, normalizes amplitude, and preserves the DC component by default.

### `presets/`

`presets/signals.py` maps user-facing preset names to signal generators.

### `fourier/`

`analysis.py` implements numerical trigonometric Fourier coefficients using trapezoidal integration. `synthesis.py` reconstructs a signal from `a0`, `a_n`, and `b_n`. `spectrum.py` converts coefficient pairs to magnitude and phase. `fft_comparison.py` is a separate NumPy FFT comparison path and does not replace the educational analyzer.

### `metrics/`

`error.py` calculates the signed error, MSE, RMSE, MAE, and maximum absolute error. `gibbs.py` detects likely jumps and measures local overshoot, undershoot, and oscillation width. `convergence.py` evaluates error metrics over several harmonic counts.

### `visualization/`

The plotting modules receive already-computed numerical data and return Matplotlib figures and axes. `epicycle.py` calculates rotating-vector geometry and creates a `FuncAnimation`; it does not own signal analysis.

### `gui/`

`main_window.py` is the application controller. It owns current signal state, calls the mathematical modules, updates labels, and coordinates menus. `controls.py` contains reusable Qt controls. `canvas.py` embeds Matplotlib and handles mouse drawing. `drawing.py` converts raw mouse points into a uniform signal. `epicycle_window.py` hosts the dedicated epicycle animation.

The GUI contains orchestration only. Fourier formulas remain in `fourier/`, metrics remain in `metrics/`, and plotting remains in `visualization/`.

### File I/O

`signal_io.py` stores validated signals as JSON and exports reconstruction, spectrum, error, and report data. `io/signal_io.py` is a compatibility re-export. Application code imports `signal_io` because Python's standard-library `io` module conflicts with a project package named `io`.

## State Management

A new or loaded signal clears coefficients, reconstruction, error, Gibbs data, FFT comparison data, epicycle windows, and displayed metrics. Reconstruction creates fresh derived results. Button and menu enablement follows the current state so normal invalid actions are disabled before they can fail.
