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

## 2D Closed Curve

1. Click `Draw 2D Curve`.
2. Hold the left mouse button and draw an arbitrary curve in the equal-aspect plot.
3. Release the mouse to preview the candidate as a closed curve.
4. Click `Finish / Close Curve` to finalize it, or `Clear Curve` to start again.

The curve stores the original captured points and a cleaned representation with invalid and consecutive duplicate points removed. Its bounding box and centered normalized coordinates are also retained. The first point is connected to the last point for display without duplicating the first vertex in the stored curve.

Click `Analyze Curve` after finalizing a curve. The analysis uses normalized cumulative arc length as the periodic parameter, resamples X and Y uniformly, and calculates separate complex Fourier coefficients for both coordinates. The DC coefficients preserve the average X/Y position. Reconstruction includes DC plus symmetric positive and negative harmonics. Reported MSE, RMSE, and maximum error are based on pointwise Euclidean X/Y differences.

## 2D Fourier Reconstruction

1. Draw and finalize a 2D closed curve.
2. Click `Analyze Curve` to calculate its X/Y coefficients once.
3. Adjust `2D Harmonics N` to reconstruct the curve with DC plus symmetric harmonics from `-N` through `N`.
4. Compare the original and dashed reconstructed curves in the equal-aspect overlay.
5. Observe the MSE, RMSE, and maximum error update for the selected harmonic count.

Increasing the harmonic count generally makes more geometric detail available, although the error need not decrease monotonically for every curve or sampling shape. The coefficient calculation is reused while the harmonic control changes.

## 2D Reconstruction Error and Convergence

Click `2D Error & Convergence` after analyzing a curve. The view compares the original and reconstructed curves, draws pointwise error segments, plots error against normalized curve parameter `t`, and plots MSE/RMSE/mean/max error against harmonic count. The current harmonic count is marked on the convergence plot.

For corresponding samples, pointwise Euclidean error is:

```text
e_i = sqrt((x_i - x_i(reconstructed))^2 + (y_i - y_i(reconstructed))^2)
```

The view reports MSE, RMSE, mean error, maximum error, and the normalized parameter location of the largest sampled error. Convergence values are calculated once from the stored Fourier coefficients and reused when the harmonic slider changes. More harmonics generally provide more degrees of freedom, but the actual computed convergence results should be used to interpret each curve.

## 2D Fourier Harmonic Spectrum

After `Analyze Curve`, click `2D Harmonic Spectrum`. The view reads the existing coordinate coefficients without recalculating them. For each signed harmonic index `k`, it shows:

- X magnitude `|Cx[k]|` and phase `arg(Cx[k])`.
- Y magnitude `|Cy[k]|` and phase `arg(Cy[k])`.
- Combined 2D magnitude `sqrt(|Cx[k]|^2 + |Cy[k]|^2)`, a coordinate-contribution ranking rather than a physical power spectrum.

The plots include positive and negative indices and the DC coefficient `k=0`. The table can be sorted by harmonic index or combined magnitude, and selecting a row shows the coefficient details. Harmonic index refers to the normalized curve parameter, not a physical frequency in hertz.

## Saving a 2D Curve

Use `File > Save 2D Curve` to save the numerical curve data as versioned JSON. The file preserves raw points, cleaned points, normalized points, closed status, and metadata; it is not an image or screenshot. Use `File > Load 2D Curve` to restore the curve. Loaded curves require fresh Fourier analysis before reconstruction-dependent actions are enabled.

## Exporting Data

- `Export 2D Curve CSV` writes cleaned mathematical vertices as `index,x,y`.
- `Export 2D Reconstruction CSV` writes original samples, currently selected-harmonic reconstruction samples, and pointwise Euclidean error.
- `Export 2D Fourier Coefficients` writes signed harmonic indices and real, imaginary, magnitude, and phase columns for both X and Y coefficients.

## Analysis Summary

The 2D analysis summary reports the closed-curve type, sampled point count, arc-length parameterization, X/Y ranges, approximate perimeter, estimated polygonal area, available harmonics, DC position, and selected reconstruction errors. The perimeter and area are estimates from the sampled polygon. Area is not interpreted as a physical enclosed area for self-intersecting curves.

## 2D Fourier Epicycle Visualization

After analyzing a 2D curve, click `2D Epicycle View`. Each rotating vector represents one planar Fourier harmonic. For the existing X/Y coefficient convention, the planar coefficient is `C[k] = Cx[k] + j Cy[k]`, and the endpoint is calculated as:

```text
z(t) = sum_k C[k] exp(j*2*pi*k*t)
```

The vector radius is `|C[k]|`, its initial orientation is `arg(C[k])`, and its rotation speed is proportional to the signed harmonic index `k`. The chain starts with the DC/average position, and the final endpoint leaves a trace of the reconstructed curve. `Play`, `Pause`, and `Reset` control the animation; changing `Harmonics N` resets the trace and uses the same symmetric `-N ... +N` harmonic set as reconstruction.

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
