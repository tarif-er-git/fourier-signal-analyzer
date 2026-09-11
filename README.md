# Signal Sketch and Decompose

## Overview

Signal Sketch and Decompose is an educational Python application for exploring periodic signals and Fourier Series. Users can generate preset waveforms, draw a custom one-period signal, inspect Fourier harmonics, reconstruct the signal with a selectable number of terms, and study reconstruction error.

The project also demonstrates the Gibbs phenomenon, convergence as the number of harmonics changes, Fourier Series versus NumPy FFT, and rotating epicycles. Signals and numerical results can be saved or exported for further study.

## Features

- Sine, square, triangle, and sawtooth presets.
- Mouse drawing of a custom one-period signal.
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
3. Set the harmonic count `N` with the slider.
4. Click `Reconstruct` or move the slider after reconstruction for a live update.
5. Inspect the original signal, reconstruction, error, metrics, Gibbs result, and error-versus-N graph.
6. Use `Compare with FFT` for the spectrum and timing comparison.
7. Use `Show Epicycles` for the rotating-vector demonstration.
8. Use the `File` menu to save, load, or export numerical results.

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

This project is designed to make the relationship between a signal, its harmonics, and its reconstruction visible. It separates numerical algorithms from plotting, GUI orchestration, and file I/O so that each part can be read, tested, and extended by a CSE student.

## Known Limitations

- Automatic Gibbs detection is intentionally simple and works best for square-wave-like signals.
- Mouse drawing represents one period and closes the two boundaries by averaging their values.
- FFT and direct Fourier timings are illustrative measurements, not rigorous benchmarks.
- The application performs calculations synchronously; the current project sizes are small enough for this to remain responsive.
- File export currently targets JSON and CSV, not PDF or image reports.
