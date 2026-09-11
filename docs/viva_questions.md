# Viva Questions and Answers

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
22. **How do epicycles relate to Fourier Series?**  Each amplitude-phase harmonic is drawn as a rotating vector; their chained endpoint gives the reconstruction.
23. **How is mouse drawing converted to a signal?**  Pixel coordinates become time/amplitude points, then sorting and interpolation create a uniform numerical grid.
24. **Why must the drawn signal be uniformly sampled?**  The numerical integrations and FFT frequency mapping assume a regular sample interval.
25. **Why separate the GUI from mathematics?**  It keeps formulas testable and reusable, while the GUI only coordinates inputs, results, and display.
26. **Why is the repeated endpoint removed?**  In `[0,T]`, the first and last samples represent the same periodic point and should not be counted twice.
27. **How are FFT amplitudes normalized?**  The rFFT magnitude is divided by the sample count and positive non-DC bins are doubled, with a Nyquist exception.
28. **How are invalid files handled?**  Required fields, shapes, finiteness, ordering, and sample count are validated before loading.
29. **Why preserve the DC component?**  The average value is part of the Fourier Series and should not be removed by default.
30. **What is the main educational purpose?**  To connect signal shape, harmonic coefficients, reconstruction error, Gibbs behavior, FFT output, and epicycle geometry in one testable application.
