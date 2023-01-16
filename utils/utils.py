import numpy as np
import scipy
import torch
from torch.distributions import MultivariateNormal

def count_params(model):
    """
    Counts two types of parameters:

    - Total no. of parameters in the model (including trainable parameters)
    - Number of trainable parameters (i.e. parameters whose gradients will be computed)

    """
    total_num_params = sum(p.numel() for p in model.parameters())
    total_num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad == True)
    return total_num_params, total_num_trainable_params

def get_mvnpdf(mean, cov):

    distr = MultivariateNormal(loc=mean, covariance_matrix=cov)
    return distr

def sample_from_pdf(distr, N_samples=100):

    samples = distr.sample((N_samples,))    
    return samples

def log_prob_normal(X, mean, cov):

    Lambda_cov, U_cov = torch.linalg.eig(cov)
    logprob_normal_fn = lambda X : torch.real(- 0.5 * X.shape[1] * torch.log(torch.Tensor([2*torch.pi])) - \
        0.5 * torch.sum(torch.log(Lambda_cov)) - \
        torch.diag(0.5 * (U_cov.H @ (X.H - mean).type(torch.cfloat)).H \
            @ torch.diag(1 / (Lambda_cov + 1e-16)) \
            @ (U_cov.H @ (X.H - mean).type(torch.cfloat))))

    pX = logprob_normal_fn(X)

    return pX

def check_psd_cov(C):
    C = C.detach().cpu().numpy()
    return np.all(np.linalg.eigvals(C) >= 0)

def create_diag(x):
    return torch.diag(x)








