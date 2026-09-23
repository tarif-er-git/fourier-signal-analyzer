# Fourier Craft

## Overview

Fourier Craft is an educational Python application for exploring periodic signals and Fourier Series. Users can generate preset waveforms, draw a custom one-period signal, inspect Fourier harmonics, reconstruct the signal with a selectable number of terms, and study reconstruction error.

The project also demonstrates the Gibbs phenomenon, convergence as the number of harmonics changes, Fourier Series versus NumPy FFT, and rotating epicycles. Signals and numerical results can be saved or exported for further study.

## Features

- Sine, square, triangle, and sawtooth presets.
- Mouse drawing of a custom one-period signal.
- Mouse drawing of a normalized 2D closed curve.
- Fourier coefficient analysis and reconstruction of closed 2D curves.
- Interactive 2D Fourier epicycle visualization.
- 2D curve save/load and CSV data export.
- 2D coordinate magnitude, phase, and combined harmonic spectrum.
- Uniform sampling and preprocessing.
- Numerical Fourier Series coefficients `a0`, `an`, and `bn`.
- Reconstruction with selectable harmonic count `N`.
- Individual harmonic and DC components.
- MSE, RMSE, MAE, and maximum error.
- Magnitude and phase spectrum.
- Gibbs discontinuity detection and local overshoot measurements.
- Error convergence versus harmonic count.
- Educational Fourier Series versus NumPy FFT comparison.
- Interactive epicycle visualization with play, pause, reset, and N control.
- Interactive 1D Signal Convolution Simulation with step-by-step continuous-time animation.
- Dual input methods for both convolution inputs x(t) and h(t): presets or custom 1D drawing.
- JSON signal save/load.
- CSV reconstruction, spectrum, and error export.
- JSON analysis report export.

## Mathematical Background

The application uses the trigonometric Fourier Series convention:

```text
x(t) = a0/2 + sum[a_n cos(n w0 t) + b_n sin(n w0 t)]
```

Here `a0/2` is the DC or average value. `a_n` and `b_n` describe the cosine and sine contribution at harmonic `n`, and `w0 = 2*pi/T` is the fundamental angular frequency for period `T`.

The analyzer estimates these coefficients with numerical trapezoidal integration. The synthesizer adds the DC component and the first `N` harmonics. The error signal is `original - reconstructed`; MSE, RMSE, MAE, and maximum error summarize that signal.

For a discontinuous signal, Fourier partial sums can show Gibbs oscillations near a jump. More harmonics generally improve the signal away from the jump, but the local overshoot does not vanish in the usual pointwise sense.

The FFT comparison is separate from the educational coefficient calculation. NumPy FFT produces complex DFT bins, which are normalized and converted to a single-sided amplitude spectrum before comparison with Fourier harmonic magnitudes.

## Installation

Use Python 3.14 or another supported Python 3.x installation. Create a virtual environment and install the packages listed in `requirements.txt`:

```text
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Required packages:

- NumPy
- Matplotlib
- PySide6
- pytest for the test suite

## Run

From the project directory:

```powershell
python main.py
```

The VS Code project environment can be run explicitly with:

```powershell
.\.venv\Scripts\python.exe main.py
```

## Usage

1. Select a preset and click `Generate`, or choose `Draw Custom Signal`.
2. For a custom signal, draw while holding the left mouse button and click `Finish Drawing`.
3. To draw a 2D curve, choose `Draw 2D Curve`, drag with the left mouse button, and release. The stroke is shown as a candidate closed curve; use `Finish / Close Curve` to finalize it or `Clear Curve` to start again.
4. The curve keeps its captured points, removes consecutive duplicates and invalid points from its processed representation, and stores centered normalized coordinates for later mathematical processing.
5. Set the harmonic count `N` with the slider.
6. Click `Reconstruct` or move the slider after reconstruction for a live update.
7. Inspect the original signal, reconstruction, error, metrics, Gibbs result, and error-versus-N graph.
8. Use `Compare with FFT` for the spectrum and timing comparison.
9. Use `Show Epicycles` for the rotating-vector demonstration.
10. Use the `File` menu to save, load, or export numerical results.

After `Analyze Curve`, click `2D Epicycle View` to watch the selected symmetric Fourier harmonics rotate and trace the reconstructed curve. Use `Play`, `Pause`, `Reset`, and the 2D harmonic slider to explore the reconstruction.

For a finalized 2D curve, click `Analyze Curve`. The curve is parameterized by normalized cumulative arc length so uneven mouse speed does not change the sampling density. X and Y are analyzed independently as complex Fourier series:

```text
x(t) = sum_k Cx[k] exp(j*2*pi*k*t)
y(t) = sum_k Cy[k] exp(j*2*pi*k*t)
```

The DC coefficients `Cx[0]` and `Cy[0]` preserve the curve's average position. Positive and negative harmonics are stored with their magnitudes and phases. Reconstruction keeps the DC term and includes symmetric harmonics from `-N` through `N`. The reported MSE, RMSE, and maximum error use the Euclidean distance between original and reconstructed X/Y samples.

### Signal Convolution Simulation

Click `Convolution Simulation` in the Advanced Visualization group of the control panel to open the interactive convolution window.

The animation visually illustrates continuous-time convolution:

```text
y(t) = (x * h)(t) = ∫ x(τ) h(t - τ) dτ
```

Numerically, this integral is evaluated over a uniform sampling grid with spacing $\Delta\tau$:

```text
y[k] = ∑ x[n] · h_shifted[n; k] · Δτ
```

#### Dual Input Options (Presets vs. Custom Draw)
Users can independently choose the input source for both signals:
- **Signal x(t):** Choose a preset (Rectangular Pulse, Triangle Pulse, Sine Burst, Sine) or choose **Custom Draw** to draw a custom 1D waveform.
- **Kernel h(t):** Choose a preset (Rectangular Pulse, Exponential Decay, Triangle Pulse, Sine Burst) or choose **Custom Draw** to draw a custom kernel.

All four combinations are fully supported:
1. `x(t) = Preset`, `h(t) = Preset`
2. `x(t) = Custom Draw`, `h(t) = Preset`
3. `x(t) = Preset`, `h(t) = Custom Draw`
4. `x(t) = Custom Draw`, `h(t) = Custom Draw`

Example:
```text
x(t) → Custom Draw
h(t) → Preset
        ↓
Convolution Simulation
```

#### Custom 1D Drawing Workflow & Uniform Resampling
1. Select **Custom Draw** from the Source dropdown for $x(t)$ or $h(t)$.
2. Click **✏️ Draw x(t)** or **✏️ Draw h(t)** to switch to the dedicated 1D drawing canvas.
3. Drag with the left mouse button to shape the signal over the time domain $[-1.0, 1.0]$.
4. Click **Use This Signal**. Raw mouse strokes are filtered for duplicates and converted into a uniformly sampled numerical array ($N=256$, $\Delta\tau = 2/255$).
5. A confirmation preview is displayed: `Custom ✓ (256 samples, range [-1.0, 1.0])`.
6. Click **Prepare Convolution** to compute the overlap integral and reference output.
7. Use **Play**, **Pause**, **Reset**, or manually scrub the **Shift Slider** to observe the reversed and shifted kernel $h(t-\tau)$ passing across $x(\tau)$, with the green shaded overlap region building the output waveform $y(t)$ frame by frame.

## Testing

For the complete suite, use an offscreen Qt platform when no desktop display is available:

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m pytest -q
Remove-Item Env:QT_QPA_PLATFORM
```

The suite covers signal generation, sampling, Fourier analysis and synthesis, spectrum, metrics, Gibbs analysis, convergence, FFT comparison, epicycle geometry, persistence, exports, visualization, and headless GUI workflows.

## Course Documents

- [Mathematics](docs/mathematics.md)
- [Architecture](docs/architecture.md)
- [User Guide](docs/user_guide.md)
- [Viva Questions](docs/viva_questions.md)
- [Presentation Script](docs/presentation.md)

## Project Structure

```text
main.py                 Application entry point
signal/                 Signal generation, sampling, preprocessing
						 and 2D curve data modeling
presets/                Named preset waveforms
fourier/                Fourier analysis, synthesis, spectrum, FFT comparison
metrics/                Error metrics, Gibbs, convergence
visualization/          Reusable Matplotlib plots and epicycle geometry
gui/                    PySide6 controls, canvas, windows, and orchestration
signal_io.py            Importable persistence and export implementation
io/                     Compatibility location for signal I/O helpers
tests/                  Automated unit, integration, and GUI smoke tests
```

## Educational Purpose

This project is designed to make the relationship between a signal, its harmonics, and its reconstruction visible. It separates numerical algorithms from plotting, GUI orchestration, and file I/O so that each part can be read, tested, and extended by a CSE student. The inclusion of both 1D signal analysis and 2D closed-curve complex epicycles provides a comprehensive interactive learning experience for Fourier mathematics.

## Known Limitations

- Automatic Gibbs detection is intentionally simple and works best for square-wave-like signals; it may struggle with highly complex hand-drawn waves.
- 1D mouse drawing represents one period and closes the two boundaries by averaging their values.
- FFT and direct Fourier timings are illustrative measurements, not rigorous benchmarks.
- The application performs calculations synchronously on the main thread; very large 2D curves or extremely high sample rates could theoretically cause brief UI stutters, though the current limits and capping (e.g., 480 points for epicycle traces) maintain responsiveness.
- File export currently targets JSON and CSV, not PDF or image reports.
- Self-intersecting 2D curves will report a mathematical area that may not match the visual enclosed area (due to the standard polygon area formula summing signed areas).

## Future Work

- **Background Threading:** Move the heavy Fourier integrations and FFT computations to background worker threads to ensure the UI remains perfectly fluid even with massive datasets.
- **Audio Output:** Add a feature to play the 1D reconstructed waveform as sound, allowing users to *hear* the effect of adding higher-frequency harmonics.
- **Image Tracing:** Implement computer vision (e.g., using OpenCV) to automatically extract a 2D contour from an imported image, rather than relying solely on freehand mouse drawing.
- **PDF Report Generation:** Add the ability to export a comprehensive, formatted PDF report containing the plots, metrics, and analysis summary for easy submission.
- **Continuous 3D Epicycles:** Extend the visualization to 3D, showing time/arc-length on a Z-axis, which can further clarify the nature of complex exponential spirals.
