Overview
- This repository provides reproduction code for the paper “L2-Regularized Empirical Risk Minimization Guarantees Small Smooth Calibration Error” (https://arxiv.org/abs/2510.13450). We study calibration behavior of kernel-based ERM (kernel ridge regression, kernel logistic regression) on synthetic and real datasets, reporting smooth calibration errors (LinECE), kernel calibration error (KCE), binned ECE, and accuracy.

Key contents
- Evaluate KRR (squared loss) and KLR (logistic loss) with Gaussian and Laplace kernels
- Sweep training set size and regularization strength; report calibration metrics and accuracy
- Provide both synthetic experiments (`Synthetic/`) and real-data benchmarks (`realdata/`)

Repository layout
- `Synthetic/`: Synthetic data generation, kernels, training loops, metrics
  - `run_experiments.py`: Main entry point with two modes:
    - `n_asymptotic` (default): sweep training set size on a log scale and aggregate across seeds
    - `reg_sensitivity`: fix training size and sweep regularization values (log-spaced)
  - `data.py`: d-dimensional balanced binary Gaussian mixture generator (default: 1D)
  - `kernels.py`: Gaussian and Laplace kernels
  - `model.py`: KRR (closed-form) and KLR (GD/L-BFGS); GD uses learning rates from spectral radius estimates
  - `metrics.py`: Classification error, KCE, LinECE (convex optimization), binned ECE
  - `utils.py`: Median-heuristic bandwidth selection and spectral-norm estimators
- `realdata/`: Real-data experiments on OpenML datasets
  - `binary_exp.py`: Training-size sweeps (KRR/KLR, RBF/Laplacian) with dataset-wise median heuristic bandwidths
  - `binary_exp_reg.py`: Regularization sweeps at fixed sample size (`alpha` grid)
- `pycalib/`: Third-party implementation of “Non-Parametric Calibration for Classification” (Wenger et al., AISTATS 2020). See `pycalib/README_Wenger_et_al..md`. Not required to run the synthetic/real experiments in this repo.

Environment
- Python 3.9+
- Core deps: `numpy`, `pandas`, `scipy`, `scikit-learn`, `torch`
- For calibration optimization: `cvxpy`, `ecos` (optionally `scs` as a fallback)

Quick setup
```bash
# From repository root
python -m pip install --upgrade pip
pip install numpy pandas scipy scikit-learn torch cvxpy ecos
```

Synthetic experiments
Models and settings
- Models: KRR (squared loss), KLR (logistic loss)
- Kernels: Gaussian, Laplace
- Bandwidth (σ): median heuristic on training pairwise distances
- Regularization:
  - Baseline: `reg=1e-2` (used in `n_asymptotic` sweeps)
  - Theoretical schedules (for reference): KRR with Gaussian uses `1/√n`; Laplace uses `1/n^(1/3)`
- KLR optimization (GD): learning rate from an upper bound on the Lipschitz constant via `λ_max(K)` estimated by power method or randomized SVD

How to run
```bash
cd Synthetic

# 1) Asymptotics over training size (seeds × log-spaced n)
python run_experiments.py --num_candidates_sample 20 --sample_size_low 100 --sample_size_high 10000 --sample_size_test 10000

# 2) Regularization sensitivity at fixed n
python run_experiments.py --exp_name reg_sensitivity --sample_size_high 10000 --sample_size_test 10000 --num_candidates_reg 20
```

Outputs
- Saved under `Synthetic/results/` when run from the `Synthetic/` directory:
  - `experiment_results_n.csv`: raw rows (seed × train_size × model × kernel)
  - `experiment_results_grouped_n.csv`: means and stds grouped by `(train_size, model, kernel)`
  - `experiment_results_reg.csv`: raw rows for `reg_sensitivity` (includes `reg`)
  - `experiment_results_grouped_reg.csv`: grouped summary for `reg_sensitivity`

Real-data benchmarks
Datasets
- OpenML dataset names in code: `kr-vs-kp`, `spambase`, `sick`, `churn`, `Satellite`
- Preprocessing: `ColumnTransformer` (median-impute and standardize numerics, one-hot encode categoricals). Stratified 80/20 train/test split

Modeling choices
- Bandwidths via median heuristic: `gamma_rbf = 1/(2σ^2)`, `gamma_lap = 1/σ`
- KLR (`KernelLogisticRegressionExact`): batch GD; regularization scaled as `lambda_reg = α / n` with `α=0.1` for size sweeps
- KRR (`sklearn.kernel_ridge.KernelRidge`): `alpha = 1/√n` for RBF, `alpha = 1/n^(1/3)` for Laplacian in size sweeps

Metrics (per test set)
- Log loss (`ent`), accuracy (`acc`), predicted entropy (`shannon`), mean absolute error of probabilities (`grad`), binned ECE (`binned`), smooth ECE / LinECE (`smooth`), MMCE (`mmce`)

How to run
```bash
cd realdata

# 1) Training-size sweeps (roughly 50–2000)
python binary_exp.py --dataset_name "kr-vs-kp,spambase" --n_seeds 5 --n_sample_candidates 10

# 2) Regularization sweeps (e.g., n=2000)
python binary_exp_reg.py --dataset_name "kr-vs-kp" --n_seeds 5 --n_sample 2000 --num_candidates_reg 10
```

Outputs
- Saved under `realdata/results/`:
  - Size sweeps: `binary_benchmark_raw_results_<datasets>.csv`, `binary_benchmark_summary_<datasets>.csv`
  - Regularization sweeps: `raw_alpha_variation_<datasets>.csv`, `summary_alpha_variation_<datasets>.csv`

Reproducibility
- Synthetic: seeds fixed in `run_experiments.py` (e.g., `[123, 456, …]`, 10 seeds)
- Real data: seeds controlled via `--n_seeds`; stratified sampling per sampled `n`
- Solvers: LinECE in `Synthetic/metrics.py` falls back to SCS if ECOS fails; `realdata/` uses ECOS by default

Performance notes
- Complexity dominated by Gram matrix operations (time/memory O(n^2)); large `n` can be slow
- GPU is not required. Installing `cvxpy`/`ecos` may require system toolchains

Citation
- If you use this repository, please cite:
  - “L2-Regularized Empirical Risk Minimization Guarantees Small Smooth Calibration Error”, arXiv, 2025. [https://arxiv.org/abs/2510.13450]
- If you use the bundled third-party code, please also cite:
  - Jonathan Wenger, Hedvig Kjellström, Rudolph Triebel. “Non-Parametric Calibration for Classification”, AISTATS 2020. See `pycalib/`

License
- This project is licensed under the MIT License. See `LICENSE` for details
- This repository includes third-party code under their respective licenses (see `pycalib/`) and you must comply with those licenses as well
