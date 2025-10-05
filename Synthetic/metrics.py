import numpy as np
import torch
import cvxpy as cp

# ------------------------------
# Evaluation metrics
# ------------------------------
def classification_error(y_true, y_prob):
    y_pred = (y_prob >= 0.5).float()
    return (y_pred != y_true.float()).float().mean().item()

def kernel_calibration_error(y_true, y_prob, sigma=0.1, kernel_type='gaussian'):
    n = y_true.shape[0]
    r = y_true.float() - y_prob
    Yp = y_prob.view(-1, 1)
    diff = Yp - Yp.t()
    
    if kernel_type == 'gaussian':
        K_mat = torch.exp(- (diff**2) / (2 * sigma**2))
    elif kernel_type == 'laplace':
        K_mat = torch.exp(- torch.abs(diff) / sigma)
    else:
        raise ValueError("kernel_type must be 'gaussian' or 'laplace'")
    
    kce_sq = (r.view(-1, 1) * r.view(1, -1) * K_mat).sum() / (n**2)
    return torch.sqrt(torch.clamp(kce_sq, min=0.0))

def compute_LinECE(y_true, y_prob):
    v = y_prob.detach().cpu().numpy().flatten()
    y = y_true.detach().cpu().numpy().flatten()
    n = len(v)
    sorted_indices = np.argsort(v)
    v_sorted = v[sorted_indices]
    y_sorted = y[sorted_indices]
    z = cp.Variable(n)
    objective = cp.Maximize((1/n) * cp.sum((y_sorted - v_sorted) * z))
    constraints = [z >= -1, z <= 1]
    v_diff = np.abs(np.diff(v_sorted))
    z_diff = cp.abs(z[:-1] - z[1:])
    constraints.append(z_diff <= v_diff)
    prob = cp.Problem(objective, constraints)
    
    try:
        prob.solve(solver=cp.ECOS, verbose=False)
    except cp.SolverError as e:
        print("ECOS failed, trying SCS instead...")
        prob.solve(solver=cp.SCS, verbose=False)
    return prob.value

def expected_calibration_error(y_true, y_prob, num_bins):
    y_true_np = y_true.detach().cpu().numpy()
    y_prob_np = y_prob.detach().cpu().numpy()

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(y_prob_np, bin_boundaries, right=True) - 1

    n = len(y_prob_np)
    ece = 0.0
    for i in range(num_bins):
        bin_mask = bin_indices == i
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_acc = np.mean(y_true_np[bin_mask])
            bin_conf = np.mean(y_prob_np[bin_mask])
            bin_weight = bin_size / n
            ece += bin_weight * np.abs(bin_acc - bin_conf)
    
    return ece