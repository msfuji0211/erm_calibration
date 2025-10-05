import numpy as np
import torch

# ------------------------------
# Lipschitz constant estimation
# ------------------------------

def estimate_lambda_max(K, num_iter=100):
    K = np.asarray(K, dtype=np.float32)
    n = K.shape[0]
    b = np.random.rand(n).astype(np.float32)
    b /= np.linalg.norm(b)
    for _ in range(num_iter):
        b = K @ b
        b_norm = np.linalg.norm(b)
        if b_norm == 0:
            break
        b /= b_norm
    lambda_max = (b.T @ (K @ b)).item()
    return lambda_max

def estimate_lambda_max_randomized(K, l=10, q=2):
    '''
    Approximate lambda_max estimation via randomized SVD.
    '''
    n = K.shape[0]
    Omega = np.random.randn(n, l).astype(np.float32)
    Y = K @ Omega
    for _ in range(q):
        Y = K @ Y
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ K @ Q
    eigenvalues = np.linalg.eigvalsh(B)
    lambda_max = eigenvalues[-1]
    return lambda_max

# ------------------------------
# sigma selection function (2D case)
# ------------------------------
def select_sigma_from_data(X, subsample_size=1000):
    n = X.shape[0]
    if n > subsample_size:
        idx = torch.randperm(n)[:subsample_size]
        X_sample = X[idx]
    else:
        X_sample = X
    # median of Euclidean distance in 2D space
    diff = torch.cdist(X_sample, X_sample, p=2)
    median_val = diff.view(-1).median().item()
    return median_val

def select_sigma_from_vector(v, subsample_size=1000, min_sigma=1e-3):
    n = v.shape[0]
    if n > subsample_size:
        idx = torch.randperm(n)[:subsample_size]
        v_sample = v[idx]
    else:
        v_sample = v
    diff = torch.abs(v_sample.unsqueeze(1) - v_sample.unsqueeze(0))
    median_val = diff.view(-1).median().item()
    return max(median_val, min_sigma)