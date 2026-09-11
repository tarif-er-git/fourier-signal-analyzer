# User Guide

## Start the Application

Create or activate the project environment, install dependencies, and run:

```powershell
python main.py
```

The main window contains signal input controls, Fourier controls, result labels, and Matplotlib visualization panels.

## Preset Signal

1. Select Sine, Square, Triangle, or Sawtooth.
2. Click `Generate`.
3. Choose a harmonic count with the `Harmonics N` slider.
4. Click `Reconstruct`.
5. Inspect the original signal, reconstruction, error, metrics, Gibbs status, and convergence graph.

Changing the slider after reconstruction updates the reconstruction and metrics live.

## Custom Signal

1. Click `Draw Custom Signal`.
2. Hold the left mouse button in the drawing panel.
3. Draw approximately one period from left to right.
4. Release the mouse and click `Finish Drawing`.
5. The raw points are sorted, duplicate times are averaged, and the result is interpolated to a uniform grid.
6. Choose `N` and click `Reconstruct`.

The left and right edges represent the same periodic boundary. The two endpoint values are averaged before analysis so the custom signal does not gain an arbitrary boundary jump.

## Spectrum

After reconstruction, the Fourier coefficient spectrum is available through the analysis state and FFT comparison view. Magnitude shows the strength of each harmonic. Phase shows the phase angle in radians.

## Gibbs Demonstration

Use the Square preset and compare:

```text
N = 3, 5, 10, 20, 50
```

Observe that the reconstruction improves away from the jump while oscillations remain near the discontinuity. The Gibbs panel reports measured overshoot, undershoot, and a theoretical reference when a significant jump is detected. Smooth sine waves report that no significant discontinuity was detected.

## FFT Comparison

1. Generate or load a signal.
2. Click `Compare with FFT`.
3. Compare the Fourier Series and normalized FFT magnitude lines.
4. Read the illustrative direct-Fourier and FFT timings in the control panel.

The FFT comparison is separate from the educational Fourier reconstruction.

## Epicycles

1. Generate or load a signal.
2. Set the desired harmonic count.
3. Click `Show Epicycles`.
4. Use Play, Pause, Reset, and the epicycle window's N slider.

Each rotating vector represents one amplitude-phase harmonic. The final chain endpoint's horizontal coordinate is the reconstructed signal value.

## Save, Load, and Export

Use the `File` menu:

- `Save Signal`: stores time, signal, and metadata in JSON.
- `Load Signal`: validates and displays a saved JSON signal.
- `Export Reconstruction`: writes `time,reconstructed` CSV data.
- `Export Spectrum`: writes `harmonic,magnitude,phase` CSV data.
- `Export Error`: writes `time,error` CSV data.
- `Export Analysis Report`: writes a JSON summary of current results.

Loading a signal clears old coefficients and derived results. Reconstruct it again to calculate fresh analysis.
