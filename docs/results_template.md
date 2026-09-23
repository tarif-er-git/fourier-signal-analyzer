# Results Section Template

*Use this template to structure the "Results and Analysis" section of your final project report. Fill in the bracketed placeholders with your actual observations and exported data.*

## 1. 1D Signal Reconstruction Accuracy

**Objective:** To quantify how the number of harmonics ($N$) affects the reconstruction accuracy of standard waveforms.

**Methodology:**
A square wave and a sine wave were generated. Reconstruction was performed for $N \in \{1, 5, 10, 50\}$. The Mean Squared Error (MSE) and Maximum Error were recorded for each case.

**Results Data:**
*Insert a table here summarizing the metrics for the Square wave and Sine wave at different N values. You can get this data from the main window metrics panel or by exporting the Analysis Report.*

**Analysis:**
As expected, the pure sine wave achieved near-zero error with just $N=1$. In contrast, the square wave's MSE decreased as $N$ increased, but the Maximum Error remained relatively high due to the sharp discontinuities. This mathematically demonstrates the difficulty of approximating non-smooth signals with smooth basis functions.

## 2. Observation of the Gibbs Phenomenon

**Objective:** To empirically verify the theoretical overshoot percentage of the Gibbs phenomenon.

**Methodology:**
A square wave was analyzed at $N=50$. The automated Gibbs detection tool was used to measure the local overshoot at the discontinuity.

**Results Data:**
- Theoretical Overshoot: ~8.95%
- Measured Overshoot: [Insert measured % from the GUI]
- Oscillation Width: [Insert measured width from the GUI]

**Analysis:**
The measured overshoot closely matched the theoretical prediction. Increasing $N$ (e.g., from 10 to 50) visibly compressed the ringing closer to the jump boundary (reduced oscillation width) but did not eliminate the amplitude of the overshoot, confirming the persistent nature of the Gibbs phenomenon.

## 3. 2D Closed Curve Fourier Descriptors

**Objective:** To demonstrate the application of complex Fourier series to 2D shape representation and reconstruction.

**Methodology:**
A custom, asymmetric shape (e.g., a hand-drawn star) was drawn and analyzed. The reconstruction was evaluated qualitatively (visual match) and quantitatively (Euclidean distance error) at $N=2, 10, \text{and } 50$.

**Results Data:**
*Insert a table showing N, MSE, and Max Error for the 2D curve. Insert screenshots of the 2D curve at N=2, N=10, and N=50.*

**Analysis:**
At low $N$, only the low-frequency geometric features (basic size and broad curves) were captured, effectively acting as a low-pass spatial filter. As $N$ increased, higher-frequency details (sharp corners and small deviations) were resolved. The DC coefficient ($k=0$) correctly identified the spatial centroid of the drawn shape.

## 4. Algorithmic Comparison: Direct Fourier Series vs. FFT

**Objective:** To compare the educational direct integration method with the highly optimized Fast Fourier Transform (FFT).

**Methodology:**
A signal with $M$ samples was analyzed up to $N=100$ harmonics. The execution time for the numerical trapezoidal integration was compared to NumPy's `rfft`.

**Results Data:**
- Direct Calculation Time: [Insert time from GUI] ms
- FFT Calculation Time: [Insert time from GUI] ms
- Frequency Match: [Did the spectrums align visually in the comparison plot?]

**Analysis:**
While both methods produced matching magnitude spectra, the FFT was orders of magnitude faster. This confirms the $O(N \log N)$ efficiency of the FFT compared to the $O(N^2)$ scaling of calculating many harmonics via direct numerical integration, validating why FFT is the industry standard for spectral analysis.
