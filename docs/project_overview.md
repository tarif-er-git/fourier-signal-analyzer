# Project Overview: Fourier Craft

## Project Title
Fourier Craft: An Interactive 1D Signal and 2D Closed-Curve Fourier Analysis Educational Tool

## Problem Statement
Understanding the Fourier Series and its practical applications is often challenging for students due to the abstract nature of the underlying mathematics. Traditional learning methods often lack interactive visual feedback, making it difficult to intuitively grasp concepts such as harmonic synthesis, the Gibbs phenomenon, error convergence, and complex Fourier descriptors for 2D shapes. There is a need for a unified, interactive tool that bridges the gap between mathematical theory and visual, programmatic realization for both 1D signals and 2D contours.

## Motivation
The primary motivation is to create a robust educational platform that demystifies Fourier analysis. By allowing users to visually draw signals or 2D shapes and instantly see the effect of changing the number of reconstruction harmonics, the mathematical formulas become tangible. Extending the tool from 1D periodic signals to 2D closed curves (using epicycles) provides a comprehensive view of how Fourier transforms can decompose complex structures into simple circular motions, an elegant concept fundamental to signal processing and computer graphics.

## Main Objectives
1.  **Interactive Visualization:** Provide a graphical interface for real-time Fourier analysis and synthesis of both 1D time-domain signals and 2D spatial curves.
2.  **Educational Demonstration:** Illustrate key signal processing concepts including harmonic addition, the Gibbs phenomenon, error metrics, and frequency spectra.
3.  **Algorithmic Transparency:** Separate core numerical algorithms from GUI and visualization code, allowing students to inspect, test, and understand the raw mathematical implementation.
4.  **Robust Architecture:** Maintain strict isolation between the 1D signal pipeline and the 2D curve pipeline to ensure stability and state consistency.
5.  **Data Persistence:** Allow saving and loading of custom signals and curves, as well as exporting numerical results and analysis reports.

## Technologies Used
*   **Language:** Python 3
*   **GUI Framework:** PySide6 (Qt for Python)
*   **Numerical Computation:** NumPy
*   **Visualization:** Matplotlib (integrated within the PySide6 application)
*   **Testing:** pytest
*   **Serialization:** Standard Library `json` and `csv` for file I/O

## Main Features

### 1D Signal Features
*   **Signal Presets:** Generation of standard waveforms (Sine, Square, Triangle, Sawtooth).
*   **Custom Signal Drawing:** Interactive mouse drawing of a custom one-period time-domain signal.
*   **Fourier Analysis & Synthesis:** Calculation of DC component ($a_0/2$), and $a_n, b_n$ coefficients.
*   **Reconstruction Control:** Interactive slider to adjust the number of harmonics ($N$) used in the reconstruction.
*   **Error Metrics:** Calculation of Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and Maximum Error.
*   **Gibbs Phenomenon Detection:** Automated detection of discontinuities and measurement of local overshoots.
*   **Convergence Analysis:** Generation of an Error vs. $N$ plot to visualize convergence rates.
*   **FFT Comparison:** Educational comparison between the analytical Fourier Series calculation and NumPy's Fast Fourier Transform (FFT), including timing benchmarks.
*   **Spectrum Visualization:** Plotting of the magnitude and phase spectra of the 1D signal.
*   **Data Persistence:** Saving/loading of 1D signal arrays and JSON export of analysis reports.

### 2D Closed-Curve Features
*   **Custom 2D Curve Drawing:** Interactive canvas to draw any freehand closed 2D shape.
*   **Curve Processing:** Automatic normalization, duplicate point removal, and resampling by normalized arc length for uniform sampling density.
*   **Complex Fourier Analysis:** Independent analysis of X and Y coordinates to compute complex Fourier descriptors ($C_x[k], C_y[k]$).
*   **2D Reconstruction:** Synthesis of the curve using symmetric positive and negative harmonics ($-N$ to $+N$).
*   **Epicycle Visualization:** An interactive, animated "Epicycle View" that demonstrates the rotating vectors (phasors) tracing the reconstructed shape, with playback controls (Play, Pause, Reset, Harmonic slider).
*   **Safeguards & Caching:** Phase-zeroing for near-zero magnitude coefficients to prevent numerical instability, trace length capping for performance, and result caching for smooth UI updates.
*   **Data Persistence:** Saving/loading of normalized 2D curve coordinate data.

## Mathematical Concepts Used
*   **Trigonometric Fourier Series:** Representing periodic 1D signals as sums of sine and cosine waves.
*   **Complex Fourier Series:** Representing 2D spatial coordinates $(x,y)$ as complex functions of a parameter $t$ (arc length) and decomposing them into complex exponential terms.
*   **Numerical Integration:** Using the Trapezoidal rule for computing Fourier coefficients from discrete sample points.
*   **Phasors and Epicycles:** Visualizing complex coefficients as rotating vectors with a specific magnitude, phase, and angular velocity.
*   **Error Metrics:** Utilizing MSE, RMSE, MAE, and $L_\infty$ (Maximum) norms to quantify reconstruction accuracy.
*   **Discrete Fourier Transform (DFT):** Understanding the relationship and differences between analytical harmonic series and discrete transforms.

## Expected Educational Value
Students utilizing this project will transition from passively reading equations to actively manipulating parameters. They will empirically observe that sharp corners require high-frequency components, see the Gibbs ringing at discontinuities, and intuitively understand complex phasors through the epicycle animation. The clean separation of the project's codebase also serves as a practical template for structuring scientific and mathematical software applications.
