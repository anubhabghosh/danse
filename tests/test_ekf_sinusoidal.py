import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import sys
import math
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.autograd.functional import jacobian
from parse import parse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.plot_functions import *
from utils.utils import generate_normal, dB_to_lin, lin_to_dB, mse_loss
from parameters import get_parameters, h_sinssm_fn, f_sinssm_fn
from generate_data import SinusoidalSSM, generate_SSM_data
from src.ekf import EKF
from src.danse import DANSE, push_model

def test_danse_sinusoidal(danse_model, saved_model_file, Y, device='cpu'):

    danse_model.load_state_dict(torch.load(saved_model_file, map_location=device))
    danse_model = push_model(nets=danse_model, device=device)
    danse_model.eval()

    with torch.no_grad():

        Y_test_batch = Variable(Y, requires_grad=False).type(torch.FloatTensor).to(device)
        X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered = danse_model.compute_predictions(Y_test_batch)
    
    return X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered

def test_ekf_sinusoidal(X, Y, ekf_model):

    X_estimated_ekf, Pk_estimated_ekf, mse_arr_ekf = ekf_model.run_mb_filter(X, Y)
    return X_estimated_ekf, Pk_estimated_ekf, mse_arr_ekf

def get_test_sequence(n_states, T, alpha, beta, phi, delta, a, b, c, inverse_r2_dB, nu_dB, decimate, use_Taylor):

    sin_model = SinusoidalSSM(n_states, alpha, beta, phi, delta, a, b, c,
                            decimate=decimate, mu_e=np.zeros((n_states,)), mu_w=np.zeros((n_states,)),
                            use_Taylor=use_Taylor)

    x_sin, y_sin = sin_model.generate_single_sequence(T=T, 
                                                    inverse_r2_dB=inverse_r2_dB,
                                                    nu_dB=nu_dB)

    plot_state_trajectory(x_sin)
    plot_measurement_data(y_sin)

    if decimate == True:
        print("Decimated ...")
        print(x_sin.shape, y_sin.shape)
        plot_state_trajectory(x_sin)
        plot_measurement_data(y_sin)

    plot_state_trajectory_axes(x_sin)
    plot_measurement_data_axes(y_sin)

    return x_sin, y_sin, sin_model

def test_sinusoidal(device='cpu', model_file_saved=None):

    #_, rnn_type, m, n, T, _, inverse_r2_dB, nu_dB = parse("{}_danse_{}_m_{:d}_n_{:d}_T_{:d}_N_{:d}_{:f}dB_{:f}dB", model_file_saved.split('/')[-2])
    n_states = 2
    T = 100
    alpha = 0.9
    beta = 1.1
    phi = math.pi * 0.1
    delta = 0.01
    a = 1.0
    b = 1.0 
    c = 0.0
    inverse_r2_dB = 20
    nu_dB = -20
    decimate=False
    use_Taylor=False
    #A_fn = lambda z: torch.Tensor([
    #        [-10, 10, 0],
    #        [28, -1, -z[0]],
    #        [0, z[0], -8.0/3]
    #   ]).type(torch.FloatTensor)

    x_sin, y_sin, sin_model = get_test_sequence(n_states=n_states, T=T, alpha=alpha, beta=beta, phi=phi, delta=delta, a=a, b=b, c=c, 
                                        inverse_r2_dB=inverse_r2_dB, nu_dB=nu_dB, decimate=decimate, use_Taylor=use_Taylor)

    Ty, dy = y_sin.shape
    Tx, dx = x_sin.shape
    Y = torch.Tensor(y_sin.reshape((1, Ty, dy))).type(torch.FloatTensor)
    X = torch.Tensor(x_sin.reshape((1, Tx, dx))).type(torch.FloatTensor)

    # Initialize the Kalman filter model in PyTorch
    ekf_model = EKF(
        n_states=sin_model.n_states,
        n_obs=sin_model.n_obs,
        J=3,
        f=f_sinssm_fn,
        h=h_sinssm_fn,
        Q=None,
        R=None,
        inverse_r2_dB=inverse_r2_dB,
        nu_dB=nu_dB,
        device=device,
        use_Taylor=sin_model.use_Taylor
    )

    # Get the estimates using an extended Kalman filter model
    X_estimated_ekf = None
    Pk_estimated_ekf = None

    X_estimated_ekf, Pk_estimated_ekf, mse_arr_ekf = test_ekf_sinusoidal(X=X, Y=Y, ekf_model=ekf_model)

    plot_state_trajectory_axes(X=torch.squeeze(X,0), X_est_EKF=torch.squeeze(X_estimated_ekf,0))
    plot_state_trajectory(X=torch.squeeze(X,0), X_est_EKF=torch.squeeze(X_estimated_ekf,0))
    
    '''
    # Initialize the DANSE model in PyTorch

    ssm_dict, est_dict = get_parameters(N=1, T=Ty, n_states=sin_model.n_states,
                                        n_obs=sin_model.n_obs, 
                                        inverse_r2_dB=inverse_r2_dB, 
                                        nu_dB=nu_dB)

    # Initialize the DANSE model in PyTorch
    danse_model = DANSE(
        n_states=sin_model.n_states,
        n_obs=sin_model.n_obs,
        mu_w=sin_model.mu_w,
        C_w=sin_model.R,
        batch_size=1,
        H=jacobian(h_sinssm_fn, torch.randn(sin_model.n_states,)).numpy(),
        mu_x0=np.zeros((sin_model.n_states,)),
        C_x0=np.eye(sin_model.n_states),
        rnn_type=rnn_type,
        rnn_params_dict=est_dict['danse']['rnn_params_dict'],
        device=device
    )

    X_estimated_pred = None
    X_estimated_filtered = None
    Pk_estimated_ekf = None
    Pk_estimated_filtered = None

    X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered = test_danse_sinusoidal(danse_model=danse_model, 
                                                                                                saved_model_file=model_file_saved,
                                                                                                Y=Y,
                                                                                                device=device)
    
    # Plot the result
    plot_state_trajectory_axes(X=torch.squeeze(X,0), X_est_EKF=torch.squeeze(X_estimated_ekf,0), X_est_DANSE=torch.squeeze(X_estimated_filtered,0))
    plot_state_trajectory(X=torch.squeeze(X,0), X_est_EKF=torch.squeeze(X_estimated_ekf,0), X_est_DANSE=torch.squeeze(X_estimated_filtered,0))
    '''
    plt.show()
    return None

if __name__ == "__main__":
    device = 'cpu'
    model_file_saved = None
    test_sinusoidal(device=device, model_file_saved=model_file_saved)

