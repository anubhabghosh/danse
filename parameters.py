# This function is used to define the parameters of the model
import numpy as np
import math
import torch
from utils.utils import dB_to_lin
from ssm_models import LinearSSM
import torch
from torch.autograd.functional import jacobian

torch.manual_seed(10)

def A_fn(z):
    return np.array([
                    [-10, 10, 0],
                    [28, -1, -z],
                    [0, z, -8.0/3]
                ])

def h_fn(z):
    return z

def f_sinssm_fn(z, alpha=0.9, beta=1.1, phi=0.1*math.pi, delta=0.01):
    return alpha * (beta * z + phi) + delta

def h_sinssm_fn(z, a=1, b=1, c=0):
    return a * (b * z + c)

def get_H_DANSE(type_, n_states, n_obs):
    if type_ == "LinearSSM":
        return LinearSSM(n_states=n_states, n_obs=n_obs).construct_H()
    elif type_ == "LorenzSSM":
        return np.eye(n_obs, n_states)
    elif type_ == "SinusoidalSSM":
        return jacobian(h_sinssm_fn, torch.randn(n_states,)).numpy()

def get_parameters(N=1000, T=100, n_states=5, n_obs=5, q=1.0, r=1.0, 
    inverse_r2_dB=40, nu_dB=0, device='cpu'):

    #H_DANSE = np.eye(n_obs, n_states) # Lorenz attractor model
    #H_DANSE = LinearSSM(n_states=n_states, n_obs=n_obs).construct_H() # Linear SSM
    H_DANSE = None

    r2 = 1.0 / dB_to_lin(inverse_r2_dB)
    q2 = dB_to_lin(nu_dB - inverse_r2_dB)

    ssm_parameters_dict = {
        # Parameters of the linear model 
        "LinearSSM":{
            "n_states":n_states,
            "n_obs":n_obs,
            "F":None,
            "G":np.zeros((n_states,1)),
            "H":None,
            "mu_e":np.zeros((n_states,)),
            "mu_w":np.zeros((n_obs,)),
            "inverse_r2_dB":inverse_r2_dB,
            "nu_dB":nu_dB,
            "q":q,
            "r":r,
            "N":N,
            "T":T,
            "Q":None,
            "R":None

        },
        # Parameters of the Lorenz Attractor model
        "LorenzSSM":{
            "n_states":n_states,
            "n_obs":n_obs,
            "J":5,
            "delta":0.02,
            "A_fn":A_fn,
            "h_fn":h_fn,
            "delta_d":0.02,
            "decimate":False,
            "mu_e":np.zeros((n_states,)),
            "mu_w":np.zeros((n_obs,)),
            "inverse_r2_dB":inverse_r2_dB,
            "nu_dB":nu_dB,
            "use_Taylor":True
        },
        # Parameters of the Sinusoidal SSM
        "SinusoidalSSM":{
            "n_states":n_states,
            "alpha":0.9,
            "beta":1.1,
            "phi":0.1*math.pi,
            "delta":0.01,
            "a":1.0,
            "b":1.0,
            "c":0.0,
            "decimate":False,
            "mu_e":np.zeros((n_states,)),
            "mu_w":np.zeros((n_obs,)),
            "inverse_r2_dB":inverse_r2_dB,
            "nu_dB":nu_dB,
            "use_Taylor":False
        },
    }

    estimators_dict={
        # Parameters of the DANSE estimator
        "danse":{
            "n_states":n_states,
            "n_obs":n_obs,
            "mu_w":np.zeros((n_obs,)),
            "C_w":np.eye(n_obs,n_obs)*r2,
            "H":H_DANSE,
            "mu_x0":np.zeros((n_states,)),
            "C_x0":np.eye(n_states,n_states),
            "batch_size":64,
            "rnn_type":"gru",
            "device":device,
            "rnn_params_dict":{
                "gru":{
                    "model_type":"gru",
                    "input_size":n_obs,
                    "output_size":n_states,
                    "n_hidden":40,
                    "n_layers":2,
                    "lr":1e-2,
                    "num_epochs":2000,
                    "min_delta":5e-2,
                    "device":device
                },
                "rnn":{
                    "model_type":"gru",
                    "input_size":n_obs,
                    "output_size":n_states,
                    "n_hidden":40,
                    "n_layers":2,
                    "lr":1e-3,
                    "num_epochs":300,
                    "min_delta":1e-3,
                    "device":device
                },
                "lstm":{
                    "model_type":"lstm",
                    "input_size":n_obs,
                    "output_size":n_states,
                    "n_hidden":50,
                    "n_layers":2,
                    "lr":1e-3,
                    "num_epochs":300,
                    "min_delta":1e-3,
                    "device":device
                }
            }
        },
        # Parameters of the Model-based filters - KF, EKF, UKF
        "KF":{
            "n_states":n_states,
            "n_obs":n_obs
        },
        "EKF":{
            "n_states":n_states,
            "n_obs":n_obs
        },
        "UKF":{
            "n_states":n_states,
            "n_obs":n_obs,
            "n_sigma":n_states*2,
            "kappa":0.0,
            "alpha":1e-3
        }
    }

    return ssm_parameters_dict, estimators_dict
