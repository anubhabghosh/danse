# This function is used to define the parameters of the model
import numpy as np
import torch

torch.manual_seed(10)

def get_parameters(N=1000, T=100, n_states=5, n_obs=5, q=1.0, r=1.0, 
    inverse_r2_dB=40, nu_dB=0):

    H_DANSE = torch.randn(n_obs, n_states)

    ssm_parameters_dict = {
        # Parameters of the linear model 
        "LinearSSM":{
            "n_states":n_states,
            "n_obs":n_obs,
            "F":None,
            "G":np.zeros((n_states,1)),
            "H":None,
            "mu_e":np.zeros((n_states,1)),
            "mu_w":np.zeros((n_obs,1)),
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
            "delta":0.01,
            "A_fn":lambda z: np.array([
                    [-10, 10, 0],
                    [28, -1, -z],
                    [0, z, -8.0/3]
                ]),
            "h_fn":lambda x: x,
            "delta_d":0.02,
            "decimate":False,
            "inverse_r2_dB":inverse_r2_dB,
            "nu_dB":-20
        },
    }

    estimators_dict={
        # Parameters of the DANSE estimator
        "danse":{
            "n_states":n_states,
            "n_obs":n_obs,
            "mu_w":torch.zeros(n_obs,1),
            "C_w":torch.eye(n_obs,n_obs),
            "H":H_DANSE,
            "mu_x0":torch.zeros(n_states,1),
            "C_x0":torch.zeros(n_states,n_states),
            "batch_size":64,
            "rnn_type":"gru",
            "rnn_params_dict":{
                "gru":{
                    "model_type":"gru",
                    "input_size":n_obs,
                    "output_size":2*n_states,
                    "n_hidden":40,
                    "n_layers":2,
                    "lr":5e-4,
                    "num_epochs":3000
                },
                "rnn":{
                    "model_type":"gru",
                    "input_size":n_obs,
                    "output_size":2*n_states,
                    "n_hidden":40,
                    "n_layers":2,
                    "lr":5e-4,
                    "num_epochs":3000
                },
                "lstm":{
                    "model_type":"lstm",
                    "input_size":n_obs,
                    "output_size":2*n_states,
                    "n_hidden":50,
                    "n_layers":2,
                    "lr":1e-3,
                    "num_epochs":4000
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