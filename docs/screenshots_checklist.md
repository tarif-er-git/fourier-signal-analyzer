# Screenshots Checklist for Final Report

When compiling the final project report, ensure you capture screenshots of the following states to comprehensively demonstrate the application's capabilities.

## 1D Signal Workflows

- [ ] **Main Window - Initial State:** Showing the UI layout and empty plots before generation.
- [ ] **Preset Generation (e.g., Square Wave):** `N=10` showing original and reconstructed signals, error plot, and populated metrics table.
- [ ] **Gibbs Phenomenon:** Focus on the Gibbs panel and the overshoot visible in the plot for a square wave at `N=20`.
- [ ] **Convergence Graph (1D):** The `Error vs N` window showing how MSE and Max Error decay as $N$ increases.
- [ ] **Custom Signal Drawing:** A screenshot capturing a hand-drawn 1D signal with its corresponding reconstruction.
- [ ] **1D Spectrum (Magnitude & Phase):** The stem plots showing the harmonic coefficients.
- [ ] **FFT Comparison:** The overlay of the educational Fourier Series magnitude against the NumPy FFT magnitude.
- [ ] **1D Epicycles:** A frame of the 1D epicycle animation showing chained vectors reconstructing a waveform.

## 2D Closed Curve Workflows

- [ ] **2D Drawing Canvas:** A hand-drawn complex shape (e.g., a star or letter) immediately after closing the curve, showing the bounding box or original points.
- [ ] **2D Reconstruction Overlay:** The main canvas showing the solid original curve and the dashed reconstructed curve with a moderate $N$ (e.g., $N=5$ or $N=10$).
- [ ] **2D Error & Convergence:** The dedicated window showing the pointwise error segments and the $N$ vs. Error plot.
- [ ] **2D Harmonic Spectrum Table:** The window showing the sorted list of $C_x$ and $C_y$ coefficients and their combined magnitudes.
- [ ] **2D Epicycles (Action Shot):** A dynamic screenshot of the 2D epicycle view showing the chain of circles tracing the drawn shape.

## Error States & Edge Cases

- [ ] **Validation Warning:** A screenshot of a warning message (e.g., trying to reconstruct before analyzing, or loading a 1D file while in 2D mode).
