import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import estimate_lambda_max, estimate_lambda_max_randomized

# ------------------------------
# Prediction Model
# ------------------------------
class PredictionModel(nn.Module):
    def __init__(self, beta0, beta1):
        """
        beta0: scalar bias
        beta1: 2D weight vector (given as list, np.array, or torch.Tensor)
        """
        super(PredictionModel, self).__init__()
        self.beta0 = nn.Parameter(torch.tensor(beta0, dtype=torch.float32))
        self.beta1 = nn.Parameter(torch.tensor(beta1, dtype=torch.float32))  # shape: [2]
    
    def forward(self, x):
        # x: [n, 2]
        linear = self.beta0 + x @ self.beta1  # inner product calculation
        z1 = torch.sigmoid(linear)
        z2 = 1 - z1
        return z1, z2


# ------------------------------
# Kernel Ridge Regression, Kernel Logistic Regression
# ------------------------------
def kernel_ridge_regression(X_train, y_train, kernel_func, sigma, reg):
    """
    min_{f} \sum_{i}^{n} (l(x_{i},y_{i})) + reg * |f|_{H}^{2}
    """
    K = kernel_func(X_train, X_train, sigma)
    n = X_train.shape[0]
    alpha = torch.linalg.solve(K + reg * torch.eye(n), y_train.float())
    return alpha

def kernel_logistic_regression(X_train, y_train, kernel_func, sigma, reg, max_iter=100, lr=0.01, tol=1e-6, method='gd'):
    if method == 'gd':
        return kernel_logistic_regression_gd(X_train, y_train, kernel_func, sigma, reg, max_iter, lr, tol)
    elif method == 'lbfgs':
        return kernel_logistic_regression_lbfgs(X_train, y_train, kernel_func, sigma, reg, max_iter, lr, tol)
    else:
        raise ValueError("Unknown method. Use 'gd' or 'lbfgs'.")

# ------------------------------
# Solver for Kernel Logistic Regression
# ------------------------------
def kernel_logistic_regression_gd(X_train, y_train, kernel_func, sigma, reg, 
                                  max_iter=100, init_lr=0.01, tol=1e-6, clip_value=1.0):
    """
    min_{f} (1/n) {\sum_{i}^{n} (l(x_{i},y_{i})) + reg * |f|_{H}^{2}}
    """
    n = X_train.shape[0]
    K = kernel_func(X_train, X_train, sigma)
    
    if isinstance(K, np.ndarray):
        K_np = K.astype(np.float32)
        K_tensor = torch.from_numpy(K_np)
    else:
        K_tensor = K.float()
        K_np = K_tensor.detach().cpu().numpy().astype(np.float32)
    
    approx_threshold = 3000
    if n >= approx_threshold:
        lambda_max = estimate_lambda_max_randomized(K_np, l=10, q=2)
        print("Using approximate lambda_max estimation (Randomized SVD).")
    else:
        lambda_max = estimate_lambda_max(K_np)
        print("Using exact lambda_max estimation (Power Method).")
        
    L = (0.25 + reg) * lambda_max
    lr = 0.5 / L if L > 0 else init_lr
    print(f"Estimated lambda_max: {lambda_max:.4f}, L: {L:.4f}, using learning rate: {lr:.6f}")
    
    alpha = torch.zeros(n, requires_grad=True, dtype=torch.float32)
    optimizer = optim.SGD([alpha], lr=lr)
    prev_loss = float('inf')
    
    for i in range(max_iter):
        optimizer.zero_grad()
        f = K_tensor @ alpha
        loss = F.binary_cross_entropy_with_logits(f, y_train.float(), reduction='mean') \
               + reg * (alpha @ (K_tensor @ alpha)) / n
        loss.backward()
        torch.nn.utils.clip_grad_norm_([alpha], clip_value)
        optimizer.step()
        
        loss_val = loss.item()
        improvement = abs(prev_loss - loss_val)
        if i % 10 == 0:
            print(f"GD Iteration {i}, loss: {loss_val:.6f}, improvement: {improvement:.6e}")
        if improvement < tol:
            print(f"Early stopping at iteration {i} with improvement {improvement:.6e}")
            break
        prev_loss = loss_val
        
    return alpha.detach()

def kernel_logistic_regression_lbfgs(X_train, y_train, kernel_func, sigma, reg, max_iter=100, lr=0.1, tol=1e-6):
    n = X_train.shape[0]
    K = kernel_func(X_train, X_train, sigma)
    alpha = torch.zeros(n, requires_grad=True)
    optimizer = optim.LBFGS([alpha], lr=lr, max_iter=1, history_size=10)
    prev_loss = float('inf')
    
    for i in range(max_iter):
        def closure():
            optimizer.zero_grad()
            f = K @ alpha
            loss = F.binary_cross_entropy_with_logits(f, y_train.float(), reduction='mean') \
                   + 0.5 * reg * (alpha @ (K @ alpha))
            loss.backward()
            return loss
        loss = optimizer.step(closure)
        loss_val = loss.item()
        improvement = abs(prev_loss - loss_val)
        if i % 10 == 0:
            print(f"Iteration {i}, loss: {loss_val:.6f}, improvement: {improvement:.6e}")
        if improvement < tol:
            print(f"Early stopping at iteration {i} with improvement {improvement:.6e}")
            break
        prev_loss = loss_val
    return alpha.detach()