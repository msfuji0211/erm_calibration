import torch

# ------------------------------
# Kernel functions (Gaussian, Laplace)
# ------------------------------
def gaussian_kernel(X, Y, sigma):
    # X: [n, d], Y: [m, d]
    X_exp = X.unsqueeze(1)   # [n,1,d]
    Y_exp = Y.unsqueeze(0)   # [1,m,d]
    dists_sq = ((X_exp - Y_exp)**2).sum(dim=2)
    return torch.exp(- dists_sq / (2 * sigma**2))

def laplace_kernel(X, Y, sigma):
    X_exp = X.unsqueeze(1)   # [n,1,d]
    Y_exp = Y.unsqueeze(0)   # [1,m,d]
    dists = torch.abs(X_exp - Y_exp).sum(dim=2)
    return torch.exp(- dists / sigma)