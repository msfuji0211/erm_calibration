import torch

# ------------------------------
# Generating dataset
# ------------------------------
def generate_data(n_samples=1000000, input_dim=1):
    # 0/1 binary label
    y = torch.distributions.Bernoulli(0.5).sample((n_samples,)).long()
    # d-dimensional case: label==1 -> mean=-1, label==0 -> mean=1
    mean_y1 = torch.full((input_dim,), -1.0)
    mean_y0 = torch.full((input_dim,),  1.0)
    cov = torch.eye(input_dim)  # covariance matrix is identity matrix
    mvn_y1 = torch.distributions.MultivariateNormal(mean_y1, cov)
    mvn_y0 = torch.distributions.MultivariateNormal(mean_y0, cov)
    x_given_y1 = mvn_y1.sample((n_samples,))
    x_given_y0 = mvn_y0.sample((n_samples,))
    # choose x according to label (y is [n_samples], so unsqueeze to [n_samples, 1])
    x = torch.where(y.unsqueeze(1) == 1, x_given_y1, x_given_y0)
    return x, y