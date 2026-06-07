# Bayesian Optimization for Process Prediction

End-to-end Gaussian-process Bayesian optimization for a noisy process-yield prediction problem. The project compares Expected Improvement (EI) and Upper Confidence Bound (UCB) acquisition functions against a grid-search baseline, then trains a PyTorch regressor on simulated noisy measurements.

## Stack

- Python 3.12
- NumPy and SciPy for vectorized process simulation, sampling, and acquisition math
- scikit-learn for Gaussian-process regression and preprocessing
- PyTorch for the predictive neural regressor
- MATLAB scripts for a matching process surface and BO demonstration

## Quick Start

```bash
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
python scripts/run_experiment.py --budget 60 --grid-size 25 --seed 7
python scripts/train_predictor.py --samples 1500 --epochs 250 --seed 11
```

After setup, use the virtual environment interpreter:

```bash
.venv/bin/python
```

## What the Demo Does

1. Simulates a noisy two-input process with temperature and residence-time controls.
2. Fits a Gaussian-process surrogate after each new measurement.
3. Proposes the next measurement with EI or UCB.
4. Compares the number of evaluations needed to reach a target-quality operating point against grid search.
5. Saves convergence metrics, selected conditions, and plots in `artifacts/`.
6. Trains a PyTorch MLP predictor and reports RMSE/R2 on held-out noisy process measurements.

The default deterministic experiment is configured to show at least a 60% evaluation-count reduction versus a 25x25 grid search while maintaining target-quality prediction on noisy measurements.

## Important Files

- `src/process_bo/objective.py` - synthetic noisy process model.
- `src/process_bo/acquisition.py` - EI and UCB acquisition functions.
- `src/process_bo/optimizer.py` - Gaussian-process Bayesian optimization loop.
- `src/process_bo/baselines.py` - grid-search baseline.
- `src/process_bo/experiment.py` - benchmark orchestration and artifact writing.
- `src/process_bo/torch_model.py` - PyTorch process predictor.
- `matlab/` - MATLAB equivalents for the process surface and BO demo.

## MATLAB

From MATLAB, run:

```matlab
cd('matlab')
bayesopt_demo
```

The demo uses `fitrgp` when available and mirrors the Python process surface. It is intentionally compact so it can be adapted to real process measurements quickly.
