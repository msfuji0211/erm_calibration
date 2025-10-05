ERM Calibration Experiments

Overview
- This repository contains reproducible experiments studying calibration behavior of empirical risk minimization with kernel methods on both synthetic and real datasets.
- Synthetic experiments evaluate Kernel Ridge Regression (KRR, squared loss) and Kernel Logistic Regression (KLR) with Gaussian and Laplace kernels under varying training sizes and regularization strengths.
- Real-data benchmarks evaluate the same models across multiple OpenML datasets and report a comprehensive set of calibration and accuracy metrics.

Repository layout
- `Synthetic/`: Synthetic data generation, kernels, training loops, and metrics.
  - `run_experiments.py`: Main entry point. Two modes:
    - `n_asymptotic` (default): sweep training set size on a log scale, aggregate across seeds.
    - `reg_sensitivity`: sweep regularization values (log-spaced) at a fixed training size.
  - `data.py`: Balanced binary 2D mixture of Gaussians with identity covariance; labels `y∈{0,1}` with P(y=1)=0.5; class-conditional means `[-1,-1]` and `[1,1]`.
  - `kernels.py`: Gaussian and Laplace kernel implementations.
  - `model.py`: KRR closed-form solver and KLR solvers (GD and L-BFGS); adaptive step-size for GD using estimated Lipschitz constant via power method or randomized SVD for large n.
  - `metrics.py`: Classification error, Kernel Calibration Error (KCE), LinECE (linear ECE via convex optimization), binned ECE.
  - `utils.py`: Median-heuristic bandwidth selection and spectral-norm (λ_max) estimators.
- `realdata/`: Real-data benchmarking on OpenML datasets.
  - `binary_exp.py`: Sweep training size; KLR (RBF/Laplacian) and KRR (RBF/Laplacian) with dataset-wise median heuristic bandwidths; saves raw and aggregated CSVs.
  - `binary_exp_reg.py`: Fix training size and sweep regularization strength `alpha` on a log scale; saves raw and aggregated CSVs.
- `pycalib/`: Third-party code and benchmarks from "Non-Parametric Calibration for Classification" (Wenger et al., AISTATS 2020). See `pycalib/README_Wenger_et_al..md` for details. Not required for running the synthetic/real experiments in this repo.

Environment
- Recommended: Python 3.9+.
- Common dependencies (install before running):
  - Core: `numpy`, `pandas`, `scipy`, `scikit-learn`, `torch`
  - Optimization for metrics: `cvxpy`, `ecos` (SCS can be used as a fallback where implemented)

Quick setup
```bash
# From repository root
python -m pip install --upgrade pip
pip install numpy pandas scipy scikit-learn torch cvxpy ecos
```

Synthetic experiments
Models and kernels
- Models: KRR (squared loss) and KLR (logistic loss).
- Kernels: Gaussian and Laplace.
- Bandwidths (σ): Chosen by the median heuristic on pairwise distances of training features.
- Regularization:
  - Fixed baseline: `reg=1e-2` (used in `n_asymptotic` sweeps).
  - Theoretical schedules (when reported): Gaussian KRR uses `1/√n`; Laplace KRR uses `1/n^(1/3)`.
- Optimization (KLR, GD): Step-size set from an upper bound of the Lipschitz constant using `λ_max(K)` estimated by power method (small n) or randomized SVD (n ≥ 3000).

Metrics (reported on both train and test)
- Classification error: fraction of `(p≥0.5)` misclassifications.
- Kernel Calibration Error (KCE): computed with Gaussian and Laplace kernels in the probability space; the bandwidth for KCE is the median distance between predicted probabilities.
- LinECE (a smooth/linear ECE variant): solved via convex optimization in `cvxpy`.
- Binned ECE: equal-width binning with `num_bins ≈ n^(1/3)`.

How to run
```bash
cd Synthetic

# 1) Asymptotics over training size (default: seeds×logspaced n)
python run_experiments.py --num_candidates_sample 20 --sample_size_low 100 --sample_size_high 10000 --sample_size_test 10000

# 2) Regularization sensitivity at fixed n
python run_experiments.py --exp_name reg_sensitivity --sample_size_high 10000 --sample_size_test 10000 --num_candidates_reg 20
```

Outputs
- Results are saved under `Synthetic/results/` when invoked from the `Synthetic/` directory.
  - `experiment_results_n.csv`: raw per-run rows (seed × train_size × model × kernel).
  - `experiment_results_grouped_n.csv`: means and stds grouped by `(train_size, model, kernel)`.
  - `experiment_results_reg.csv`: raw per-run rows for the `reg_sensitivity` sweep (includes `reg`).
  - `experiment_results_grouped_reg.csv`: grouped summary for `reg_sensitivity`.

Real-data benchmarks
Datasets
- OpenML names supported in code: `kr-vs-kp`, `spambase`, `sick`, `churn`, `Satellite`.
- Features are preprocessed via `ColumnTransformer` (median-impute numerics, one-hot encode categoricals, standardize numerics). Train/test split is stratified (80/20).

Modeling choices
- Bandwidths from median heuristic on training pairwise distances: `gamma_rbf = 1/(2σ^2)`, `gamma_lap = 1/σ`.
- KLR (`KernelLogisticRegressionExact`): batch gradient descent; regularization scaled as `lambda_reg = α / n` with `α=0.1` for size sweeps.
- KRR (scikit-learn `KernelRidge`): `alpha = 1/√n` for RBF, `alpha = 1/n^(1/3)` for Laplacian in size sweeps.

Metrics (per test set)
- Log loss (`ent`), accuracy (`acc`), predicted entropy (`shannon`), mean absolute error of probabilities (`grad`), binned ECE (`binned`), smooth ECE / LinECE (`smooth`), MMCE (`mmce`).

How to run
```bash
cd realdata

# 1) Training-size sweeps (log-spaced from ~50 to 2000) for multiple datasets
python binary_exp.py --dataset_name "kr-vs-kp,spambase" --n_seeds 5 --n_sample_candidates 10

# 2) Regularization sweeps at fixed n (e.g., n=2000)
python binary_exp_reg.py --dataset_name "kr-vs-kp" --n_seeds 5 --n_sample 2000 --num_candidates_reg 10
```

Outputs
- Saved under `realdata/results/`:
  - `binary_benchmark_raw_results_<datasets>.csv` and `binary_benchmark_summary_<datasets>.csv` for size sweeps.
  - `raw_alpha_variation_<datasets>.csv` and `summary_alpha_variation_<datasets>.csv` for regularization sweeps.

Reproducibility
- Synthetic: seeds are fixed in `run_experiments.py` (`[123, 456, …]`, 10 seeds).
- Real data: seeds are specified via `--n_seeds`; stratified sampling ensures class balance at each sampled `n`.
- Optimization solvers: LinECE in `Synthetic/metrics.py` falls back to SCS if ECOS fails; `realdata/` LinECE implementation uses ECOS by default.

Performance notes
- Complexity is dominated by Gram matrix operations (O(n^2) memory/time). Large `n` may be slow; KLR adapts its learning rate using spectral estimates to improve stability.
- GPU is not required; all experiments can be run on CPU. Installing `cvxpy` and `ecos` may require system toolchains.

Citation
- If you use the `pycalib` code bundled here, please cite:
  - Jonathan Wenger, Hedvig Kjellström, Rudolph Triebel. "Non-Parametric Calibration for Classification", AISTATS 2020.

License
- This repository includes third-party code under their respective licenses (see `pycalib/`).

