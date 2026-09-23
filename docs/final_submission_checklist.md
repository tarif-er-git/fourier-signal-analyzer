# Final Submission Checklist

Before submitting your final project archive for grading, ensure you have completed all the items on this checklist.

## 1. Codebase Verification
- [ ] Code runs without any errors on the target Python version (Python 3.14 recommended).
- [ ] All required packages are correctly listed in `requirements.txt`.
- [ ] The `pytest` test suite passes (run `python -m pytest -q` in terminal).
- [ ] Unused or leftover debugging code/print statements have been removed.
- [ ] No hardcoded absolute file paths exist in the code (ensure `io` modules use relative paths).

## 2. Documentation Completeness
- [ ] `README.md` is updated with your name/details (if required by your instructor) and accurately reflects the final features.
- [ ] `docs/project_overview.md` outlines the problem, motivation, and main features clearly.
- [ ] `docs/user_guide.md` provides accurate step-by-step instructions for all features (including 2D curves).
- [ ] `docs/mathematics.md` contains the required mathematical derivations for both 1D and 2D Fourier Series.
- [ ] `docs/architecture.md` accurately describes the system modules and data flow.
- [ ] `docs/viva_questions.md` is reviewed and understood for the oral examination.

## 3. Final Report Preparation
- [ ] Use `docs/results_template.md` to structure the results section of your report.
- [ ] Use `docs/screenshots_checklist.md` to ensure all necessary visual evidence is captured.
- [ ] Ensure all exported data (CSV/JSON) used in your report matches the screenshots provided.
- [ ] Ensure the "Limitations and Future Work" section is included in your written report, referencing the points in the README.

## 4. Presentation & Demo Readiness
- [ ] Review `docs/presentation.md` and practice the live demonstration script.
- [ ] Verify that the application does not crash during the specific sequences outlined in the demo script.
- [ ] Save a "perfect" 1D JSON signal and 2D JSON curve to load quickly during the live demo (to save time if drawing takes too long).

## 5. Archive Creation
- [ ] Exclude the `.venv` directory from your final zip file.
- [ ] Exclude the `__pycache__` and `.pytest_cache` directories.
- [ ] Exclude any temporary `scratch/` files or system generated folders (`.git`).
- [ ] Zip the project directory and verify that it extracts and runs cleanly on a fresh environment.
