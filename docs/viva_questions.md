# Viva Questions and Answers

## 1D Fourier Series

1. **What is a Fourier Series?**  It represents a periodic signal as a DC value plus sine and cosine harmonics.
2. **How is it different from a Fourier Transform?**  A Fourier Series describes periodic signals with discrete harmonics; a Fourier Transform describes frequency content more generally.
3. **What are `a0`, `a_n`, and `b_n`?**  `a0/2` is the DC value; `a_n` and `b_n` are cosine and sine coefficients.
4. **Why is `a0` divided by 2 during reconstruction?**  The coefficient convention defines the constant term as `a0/2`.
5. **What is a harmonic?**  Harmonic `n` is a sinusoid at `n` times the fundamental frequency.
6. **What does `N` represent?**  The number of harmonics included in a partial reconstruction.
7. **What happens when `N` increases?**  Approximation detail generally improves, especially away from discontinuities.
8. **Why does a square wave require many harmonics?**  Its sharp jumps require many sinusoidal components to approximate.
9. **What is Gibbs phenomenon?**  Persistent oscillation and overshoot near a jump in a Fourier partial sum.
10. **Why does Gibbs phenomenon occur?**  Smooth sinusoidal components approximate a discontinuous transition only through oscillatory interference.
11. **Why does increasing `N` not remove overshoot completely?**  It narrows the oscillatory region, but the limiting local overshoot remains.
12. **What is reconstruction error?**  The point-wise difference `original - reconstructed`.
13. **What is MSE?**  The mean of squared point-wise errors.
14. **Why use RMSE?**  It has the same units as the signal while still emphasizing larger errors.
15. **What is the difference between MAE and MSE?**  MAE averages absolute errors; MSE squares errors and penalizes large deviations more strongly.
16. **What is the FFT?**  An efficient algorithm for computing the discrete Fourier transform.
17. **Why is FFT faster than direct DFT?**  It reuses intermediate calculations instead of evaluating every frequency-time pair independently.
18. **What is the direct DFT complexity?**  Approximately `O(N^2)`.
19. **What is the FFT complexity?**  Approximately `O(N log N)`.
20. **Why are odd harmonics dominant in a symmetric square wave?**  The square wave's symmetry cancels the even harmonic contributions.
21. **What does the phase spectrum show?**  The phase shift associated with each harmonic's cosine/sine pair.
22. **How do 1D epicycles relate to Fourier Series?**  Each amplitude-phase harmonic is drawn as a rotating vector; their chained endpoint gives the reconstruction on the y-axis.
23. **How is mouse drawing converted to a signal?**  Pixel coordinates become time/amplitude points, then sorting and interpolation create a uniform numerical grid.
24. **Why must the drawn signal be uniformly sampled?**  The numerical integrations and FFT frequency mapping assume a regular sample interval.
25. **Why separate the GUI from mathematics?**  It keeps formulas testable and reusable, while the GUI only coordinates inputs, results, and display.
26. **Why is the repeated endpoint removed?**  In `[0,T]`, the first and last samples represent the same periodic point and should not be counted twice.
27. **How are FFT amplitudes normalized?**  The rFFT magnitude is divided by the sample count and positive non-DC bins are doubled, with a Nyquist exception.
28. **How are invalid files handled?**  Required fields, shapes, finiteness, ordering, and sample count are validated before loading.
29. **Why preserve the DC component?**  The average value is part of the Fourier Series and should not be removed by default.

## 2D Closed Curves and Epicycles

30. **How is a 2D closed curve parameterized for Fourier analysis?** It is parameterized by normalized cumulative arc length, meaning the parameter `t` goes from 0 to 1 along the curve's perimeter, independent of drawing speed.
31. **Why do we resample the curve by arc length?** If we used raw mouse coordinates, parts drawn slowly would have more points, artificially increasing their weight in the Fourier analysis and distorting the frequency content.
32. **How are the X and Y coordinates analyzed?** They are treated as independent, real-valued 1D periodic functions of the arc-length parameter `t` and analyzed using a complex discrete Fourier transform to produce complex coefficients `C_x` and `C_y`.
33. **What is a complex Fourier descriptor?** It's a complex coefficient `C_k` representing the amplitude and phase of a harmonic rotating at frequency `k`. For 2D shape representation, we often use `C_k = C_x[k] + j*C_y[k]`.
34. **Why do 2D reconstructions require negative harmonics?** Real-valued signals can be built from cosines/sines (positive frequencies). Complex epicycle motion in a 2D plane requires both counter-clockwise (positive `k`) and clockwise (negative `k`) rotating vectors to trace arbitrary paths (not just circles).
35. **What do the `k=0` (DC) coefficients represent in a 2D curve?** `C_x[0]` and `C_y[0]` represent the geometric centroid (average position) of the curve.
36. **How does 2D epicycle visualization work?** Each harmonic `k` (from `-N` to `N`) acts as a vector rotating at speed `k`. Placing these vectors head-to-tail dynamically traces the reconstructed 2D curve at the tip.
37. **How is the 2D reconstruction error calculated?** Using the pointwise Euclidean distance: $e = \sqrt{(x - x_{rec})^2 + (y - y_{rec})^2}$.
38. **Why do we zero out the phase of extremely small magnitudes?** Small magnitudes (e.g., $10^{-15}$) are often numerical noise. Their computed phase angles (via `atan2`) can be wildly arbitrary, causing unstable rotations in the epicycle view that don't affect the shape but look confusing.
39. **What is the main educational purpose of this tool?**  To connect signal shape, harmonic coefficients, reconstruction error, Gibbs behavior, FFT output, and 2D complex epicycle geometry in one interactive, testable application.

## Signal Convolution Simulation

40. **How can the user provide the two convolution input signals?**  The user can independently choose either a built-in preset signal (e.g. Rectangular Pulse, Triangle Pulse, Exponential Decay, Sine Burst) or draw a custom 1D continuous waveform using the interactive mouse drawing canvas for both $x(t)$ and $h(t)$.
41. **How does the application convert a custom-drawn signal into numerical samples?**  Raw mouse stroke points are sorted by horizontal coordinate and duplicate times are averaged via `prepare_signal_points`. The stroke is then interpolated onto a uniform, monotonic time grid spanning $[-1.0, 1.0]$ with 256 samples, producing continuous numerical arrays $(t, x)$.
42. **Can x(t) and h(t) use different input methods?**  Yes. All four combinations are fully supported: Preset/Preset, Custom/Preset, Preset/Custom, and Custom/Custom.
43. **Why can't raw mouse coordinates be directly used for convolution?**  Raw mouse events arrive with non-uniform time spacing, jitter, and duplicate positions depending on hand speed and polling rate. Numerical continuous-time convolution $\int x(\tau) h(t-\tau) d\tau \approx \sum x[\tau_n] h[t-\tau_n] \Delta\tau$ requires a strictly monotonic, uniformly spaced time grid with a known constant $\Delta\tau$ to scale the integral properly.
44. **How did you prevent the two custom signals from overwriting each other?**  By maintaining completely independent state variables `_custom_x` and `_custom_h` in `ConvolutionWindow`, routing drawing outcomes based on an active target flag (`'x'` vs `'h'`), and ensuring that clearing, redrawing, or switching one signal never mutates the stored state of the other.
