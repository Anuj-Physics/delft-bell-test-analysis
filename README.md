# Delft Bell Test Reanalysis (ψ⁻ State Only)

This repository contains the Python code used for the statistical reanalysis of the publicly released 2015 Delft Bell Test data, as described in the paper:

**"Analysis of Fragile Quantum Correlations: No CHSH Violation in Combined ψ⁻ Dataset from Delft Bell Test"**

## 📄 Description

This study re-examines the CHSH inequality using only ψ⁻-entangled trials from the Delft dataset. While individual subsets exhibit violations, the combined dataset fails to surpass the classical bound ($S \leq 2$), suggesting statistical fragility in observed nonlocal correlations.

## 📁 Files

- `bell_analysis.py` — Main analysis code (CHSH $S$, p-values, Z-scores)
- `bootstrap_analysis.py` — Bootstrap resampling for $S$ distribution

## 📊 Requirements

- Python 3.8+
- NumPy
- Matplotlib
- SciPy

```bash
pip install numpy matplotlib scipy
