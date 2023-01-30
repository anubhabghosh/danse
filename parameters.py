# This function is used to define the parameters of the model
import numpy as np
import torch
from utils.utils import dB_to_lin
from ssm_models import LinearSSM

torch.manual_seed(10)

def A_fn(z):
    return np.array([
                    [-10, 10, 0],
                    [28, -1, -z],
                    [0, z, -8.0/3]
                ])

def h_fn(z):
    return z

def get_parameters(N=1000, T=100, n_states=5, n_obs=5, q=1.0, r=1.0, 
    inverse_r2_dB=40, nu_dB=0):

    #H_DANSE = np.eye(n_obs, n_states) # Lorenz attractor model
    H_DANSE = LinearSSM(n_states=n_states, n_obs=n_obs).construct_H() # Linear SSM

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
            "J":20,
            "delta":0.01,
            "A_fn":A_fn,
            "h_fn":h_fn,
            "delta_d":0.02,
            "decimate":False,
            "mu_e":np.zeros((n_states,)),
            "mu_w":np.zeros((n_obs,)),
            "inverse_r2_dB":inverse_r2_dB,
            "nu_dB":-20
        },
    }


    estimators_dict={
        # Parameters of the DANSE estimator
        "danse":{
            "n_states":n_states,
            "n_obs":n_obs,
            "mu_w":np.zeros((n_obs,1)),
            "C_w":np.eye(n_obs,n_obs)*r2,
            "H":H_DANSE,
            "mu_x0":np.zeros((n_states,1)),
            "C_x0":np.eye(n_states,n_states),
            "batch_size":64,
            "rnn_type":"gru",
            "rnn_params_dict":{
                "gru":{
                    "model_type":"gru",
                    "input_size":n_obs,
                    "output_size":n_states,
                    "n_hidden":40,
                    "n_layers":2,
                    "lr":1e-3,
                    "num_epochs":200,
                    "min_delta":1e-3
                },
                "rnn":{
                    "model_type":"gru",
                    "input_size":n_obs,
                    "output_size":n_states,
                    "n_hidden":40,
                    "n_layers":2,
                    "lr":1e-3,
                    "num_epochs":200,
                    "min_delta":1e-3
                },
                "lstm":{
                    "model_type":"lstm",
                    "input_size":n_obs,
                    "output_size":n_states,
                    "n_hidden":50,
                    "n_layers":2,
                    "lr":1e-3,
                    "num_epochs":200,
                    "min_delta":1e-3
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