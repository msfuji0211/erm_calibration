#!/usr/bin/env python
# binary_exp.py
import argparse, os, math, warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import log_loss, accuracy_score
from sklearn.metrics.pairwise import laplacian_kernel
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.kernel_approximation import RBFSampler
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.utils.validation import check_random_state
from sklearn.metrics.pairwise import laplacian_kernel, pairwise_kernels
from sklearn.metrics import pairwise_distances
from scipy import sparse

import cvxpy as cp


class KernelLogisticRegressionExact(BaseEstimator, ClassifierMixin):
    """
    Exact Kernel Logistic Regression (Batch GD)
    Loss: (1/n)∑ log(1+exp(-y_i f_i)) + λ ||w||^2
    Kernel: 'rbf' or 'laplacian', etc.
    gamma: kernel width
    lambda_reg: regularization parameter λ
    lr: learning rate
    max_iter, tol, verbose: GD settings
    """
    def __init__(self, kernel='rbf', gamma=None,
                 lambda_reg=1.0, lr=1e-3,
                 max_iter=1000, tol=1e-6, verbose=False):
        self.kernel     = kernel
        self.gamma      = gamma
        self.lambda_reg = lambda_reg
        self.lr         = lr
        self.max_iter   = max_iter
        self.tol        = tol
        self.verbose    = verbose

    def fit(self, X, y):
        # Exchange y to {+1, -1}
        self.y_train_ = np.where(y==1, 1, -1)
        n = X.shape[0]
        self.X_train_ = X

        # Gram matrix K
        self.K_train_ = pairwise_kernels(
            X, X, metric=self.kernel, gamma=self.gamma)

        # Initialize dual variable α
        self.alpha_ = np.zeros(n, dtype=float)

        for it in range(1, self.max_iter+1):
            f     = self.K_train_.dot(self.alpha_)
            sigma = 1/(1+np.exp(-f))
            y_pos = (self.y_train_ + 1)/2  # {0,1}

            # Gradient: (1/n) K (σ - y_pos) + 2λ K α
            grad = (self.K_train_.dot(sigma - y_pos))/n
            grad += 2 * self.lambda_reg * (self.K_train_.dot(self.alpha_))

            alpha_old = self.alpha_.copy()
            self.alpha_ -= self.lr * grad

            if np.linalg.norm(self.alpha_ - alpha_old) < self.tol:
                break
            if self.verbose and it%100==0:
                loss = (np.mean(np.log(1+np.exp(-self.y_train_*f)))
                        + self.lambda_reg*(alpha_old.dot(self.K_train_).dot(alpha_old)))
                print(f"Iter {it}: loss={loss:.6f}")

        return self

    def decision_function(self, X):
        K_te = pairwise_kernels(
            X, self.X_train_, metric=self.kernel, gamma=self.gamma)
        return K_te.dot(self.alpha_)

    def predict_proba(self, X):
        f = self.decision_function(X)
        p = 1/(1+np.exp(-f))
        return np.vstack([1-p, p]).T

    def predict(self, X):
        return (self.predict_proba(X)[:,1] >= 0.5).astype(int)

# --- 1) LaplaceSampler  ----------------------------------
class LaplaceSampler(BaseEstimator, TransformerMixin):
    """
    exp(-gamma * |x-y|_1) is approximated by Cauchy distribution
    """
    def __init__(self, gamma=1.0, n_components=500, random_state=None):
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        rng = check_random_state(self.random_state)
        # Sample weights from Cauchy(0, scale)
        # Adjust scale to match gamma (Sriperumbudur+ 2017)
        self.random_weights_ = rng.standard_cauchy(
            size=(X.shape[1], self.n_components)
        ) * self.gamma
        self.random_offset_ = rng.uniform(0, 2*np.pi, size=self.n_components)
        return self

    def transform(self, X):
        projection = X @ self.random_weights_ + self.random_offset_
        return np.sqrt(2.0 / self.n_components) * np.cos(projection)

def stratified_sample(X, y, n_samples, seed):
    """Stratified sampling from X, y to return n_samples"""
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")
    # split()'s train_size is 0~1 or integer
    sss = StratifiedShuffleSplit(
        n_splits=1, train_size=n_samples, random_state=seed
    )
    idx, _ = next(sss.split(X, y))   # 1 time split
    return X[idx], y[idx]

# ===============  General utilities =============== #
def compute_binned_ece(probs, y_true, num_bins=10):
    bins = np.linspace(0, 1, num_bins + 1)
    idx = np.digitize(probs, bins, right=True) - 1
    ece, n = 0.0, len(probs)
    for i in range(num_bins):
        m = idx == i
        if m.any():
            ece += (m.sum() / n) * abs(y_true[m].mean() - probs[m].mean())
    return ece

def compute_smooth_ece(probs, y_true, bw=0.1):
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    n, se = len(probs), 0.0
    for i in range(n):
        w = np.exp(-((probs - probs[i]) ** 2) / (2 * bw**2)); w /= w.sum()
        se += abs((w * y_true).sum() - (w * probs).sum())
    return se / n

def compute_mmce(probs, y_true, gamma=1.0):
    K = laplacian_kernel(probs[:, None], probs[:, None], gamma=gamma)
    err = (probs - y_true)[:, None] * (probs - y_true)[None, :] * K
    return math.sqrt(err.sum() / (len(probs) ** 2))

def LinECE_fast(v, y):
    n, v, y = len(v), np.asarray(v), np.asarray(y)
    idx = np.argsort(v); v, y = v[idx], y[idx]
    z = cp.Variable(n)
    constraints = [z >= -1, z <= 1,
                   cp.abs(z[:-1] - z[1:]) <= np.abs(np.diff(v)) / 4]
    obj = cp.Maximize((1/n) * cp.sum((y - v) * z))
    cp.Problem(obj, constraints).solve(solver=cp.ECOS, verbose=False)
    return obj.value

# ===============  データセットローダー =============== #
def load_openml_binary(name, version=1):
    """OpenML 2-class classification → (X_np, y_np) with preprocessing."""
    X_df, y = fetch_openml(name=name, version=version, as_frame=True,
                           parser='auto', return_X_y=True)
    y = y.astype("category").cat.codes.to_numpy()
    assert set(np.unique(y)) <= {0, 1}, f"{name}: not 2-class"

    num_cols = X_df.select_dtypes(exclude=["object", "category"]).columns
    cat_cols = X_df.select_dtypes(include=["object", "category"]).columns

    num_pipe = make_pipeline(SimpleImputer(strategy="median"),
                             StandardScaler())
    cat_pipe = make_pipeline(SimpleImputer(strategy="most_frequent"),
                             OneHotEncoder(handle_unknown="ignore"))

    ct = ColumnTransformer(
        [("num", num_pipe, num_cols),
         ("cat", cat_pipe, cat_cols)],
        remainder="drop"
    )

    X_processed = ct.fit_transform(X_df)
    X_np = X_processed.toarray() if sparse.issparse(X_processed) else X_processed
    return X_np.astype(float), y

DATASETS = {
    "kr-vs-kp":         lambda s: load_openml_binary("kr-vs-kp"),
    "spambase":         lambda s: load_openml_binary("spambase"),
    "sick":             lambda s: load_openml_binary("sick"),
    "churn":            lambda s: load_openml_binary("churn"),
    "Satellite":        lambda s: load_openml_binary("Satellite"),
}

# ===============  ★ コマンドライン引数 =============== #
parser = argparse.ArgumentParser(description="Binary benchmark exp.")
parser.add_argument("--dataset_name", type=str, default="breast_cancer",
                    help="dataset (multiple are comma-separated)")
parser.add_argument("--n_seeds", type=int, default=5,
                    help="number of seeds")
parser.add_argument("--n_sample_candidates", type=int, default=10,
                    help="number of sample candidates (log-spaced from 50 to 2000)")
args = parser.parse_args()

dataset_names = [d.strip() for d in args.dataset_name.split(",")]
for d in dataset_names:
    if d not in DATASETS:
        parser.error(f"--dataset_name '{d}' is not registered.")

seeds = list(range(args.n_seeds))                    # ★ seeds are variable
sample_sizes = np.unique(
    np.logspace(np.log10(50), np.log10(2000),
                num=args.n_sample_candidates, dtype=int)   # ★ number of candidates is variable
).tolist()
alpha_fixed = 0.1

# ===============  Experiment loop =============== #
raw_rows = []
for dname in dataset_names:
    loader = DATASETS[dname]
    for seed in seeds:
        X, y = loader(seed)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y)
        scaler = StandardScaler().fit(X_tr)
        X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

        # --- Estimate γ using median heuristic ----
        dists = pairwise_distances(X_tr, metric="euclidean")
        sigma = np.median(dists)
        gamma_rbf = 1.0 / (2 * sigma**2)
        gamma_lap = 1.0 / sigma

        for n in sample_sizes:
            # ---- KLR ----
            X_sub, y_sub = stratified_sample(X_tr, y_tr, n, seed)
            klr_exact = KernelLogisticRegressionExact(
                kernel='rbf', gamma=gamma_rbf,
                lambda_reg=alpha_fixed/n, lr=1e-2, max_iter=1000, tol=1e-6)
            klr_exact.fit(X_sub, y_sub)
            p_klr = klr_exact.predict_proba(X_te)[:, 1]

            raw_rows.append(
                dict(dataset=dname, seed=seed, model="KLR",
                     n_samples=n,
                     ent=log_loss(y_te, p_klr),
                     acc=accuracy_score(y_te, p_klr > 0.5),
                     shannon=np.mean(-p_klr*np.log(p_klr) - (1-p_klr)*np.log(1-p_klr)),
                     grad=np.mean(np.abs(y_te - p_klr)),
                     binned=compute_binned_ece(p_klr, y_te),
                     smooth=LinECE_fast(p_klr, y_te),
                     mmce=compute_mmce(p_klr, y_te))
            )

            # ---- KLR with Laplace kernel ------------------
            klr_lap_exact = KernelLogisticRegressionExact(
                kernel='laplacian', gamma=gamma_lap,
                lambda_reg=alpha_fixed/n, lr=1e-2, max_iter=1000, tol=1e-6)
            klr_lap_exact.fit(X_sub, y_sub)
            p_klr_lap = klr_lap_exact.predict_proba(X_te)[:, 1]

            raw_rows.append(
                dict(dataset=dname, seed=seed, model="KLR_Laplace",
                     n_samples=n,
                     ent=log_loss(y_te, p_klr_lap),
                     acc=accuracy_score(y_te, p_klr_lap > 0.5),
                     shannon=np.mean(-p_klr_lap*np.log(p_klr_lap) -
                                     (1-p_klr_lap)*np.log(1-p_klr_lap)),
                     grad=np.mean(np.abs(y_te - p_klr_lap)),
                     binned=compute_binned_ece(p_klr_lap, y_te),
                     smooth=LinECE_fast(p_klr_lap, y_te),
                     mmce=compute_mmce(p_klr_lap, y_te))
            )

            # ---- KRR ----
            reg_n = 1/(n ** 0.5)
            krr = KernelRidge(alpha=reg_n, kernel="rbf", gamma=gamma_rbf)
            krr.fit(X_sub, y_sub)
            p_krr = np.clip(krr.predict(X_te), 1e-6, 1-1e-6)
            raw_rows.append(
                dict(dataset=dname, seed=seed, model="KRR",
                     n_samples=n,
                     ent=log_loss(y_te, p_krr),
                     acc=accuracy_score(y_te, p_krr > 0.5),
                     shannon=np.mean(-p_krr*np.log(p_krr) - (1-p_krr)*np.log(1-p_krr)),
                     grad=np.mean(np.abs(y_te - p_krr)),
                     binned=compute_binned_ece(p_krr, y_te),
                     smooth=compute_smooth_ece(p_krr, y_te),
                     mmce=compute_mmce(p_krr, y_te))
            )

            # ---- KRR with Laplace kernel ------------------
            reg_n = 1/(n ** (1/3))
            krr_lap = KernelRidge(alpha=reg_n,
                      kernel="laplacian",
                      gamma=gamma_lap)
            krr_lap.fit(X_sub, y_sub)
            p_krr_lap = np.clip(krr_lap.predict(X_te), 1e-6, 1-1e-6)

            raw_rows.append(
                dict(dataset=dname, seed=seed, model="KRR_Laplace",
                     n_samples=n,
                     ent=log_loss(y_te, p_krr_lap),
                     acc=accuracy_score(y_te, p_krr_lap > 0.5),
                     shannon=np.mean(-p_krr_lap*np.log(p_krr_lap) -
                                     (1-p_krr_lap)*np.log(1-p_krr_lap)),
                     grad=np.mean(np.abs(y_te - p_krr_lap)),
                     binned=compute_binned_ece(p_krr_lap, y_te),
                     smooth=compute_smooth_ece(p_krr_lap, y_te),
                     mmce=compute_mmce(p_krr_lap, y_te))
            )

# ===============  CSV save (results folder) =============== #
out_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(out_dir, exist_ok=True)

# ★ dataset name is reflected in the file name
dataset_tag = "_".join(dataset_names).replace(",", "_")  # adult_spambase …
raw_path = os.path.join(out_dir,
            f"binary_benchmark_raw_results_{dataset_tag}.csv")
sum_path = os.path.join(out_dir,
            f"binary_benchmark_summary_{dataset_tag}.csv")

raw_df = pd.DataFrame(raw_rows)
raw_df.to_csv(raw_path, index=False)

agg = raw_df.groupby(["dataset", "model", "n_samples"]).agg(["mean", "std"])
agg.columns = ["_".join(c) for c in agg.columns]
agg.reset_index().to_csv(sum_path, index=False)

print(f"✓ Finished: {raw_path}\n        {sum_path} saved.")