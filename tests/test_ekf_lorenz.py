import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import sys
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.plot_functions import *
from utils.utils import generate_normal, dB_to_lin, lin_to_dB, mse_loss
from parameters import get_parameters, A_fn, h_fn
from generate_data import LorenzAttractorModel, generate_SSM_data
from src.ekf import EKF
from src.danse import DANSE, push_model

def test_danse_lorenz(danse_model, saved_model_file, Y, device='cpu'):

    danse_model.load_state_dict(torch.load(saved_model_file))
    danse_model = push_model(nets=danse_model, device=device)
    danse_model.eval()

    with torch.no_grad():

        Y_test_batch = Variable(Y, requires_grad=False).type(torch.FloatTensor).to(device)
        X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered = danse_model.compute_predictions(Y_test_batch)
    
    return X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered

def test_ekf_lorenz(X, Y, ekf_model):

    _, Ty, dy = Y.shape
    _, Tx, dx = X.shape
    
    assert Tx == Ty, "State and obervation sequence lengths are mismatching !"
    X_estimated_ekf = torch.zeros_like(X).type(torch.FloatTensor)
    Pk_estimated_ekf = torch.zeros((1, Tx, dx, dx)).type(torch.FloatTensor)
    mse_arr_ekf = torch.zeros((1)).type(torch.FloatTensor)

    # Running the Kalman filter on the given data
    for j in range(0, 1):

        for k in range(0, Tx):

            # Kalman prediction 
            x_rec_hat_neg_k, Pk_neg = ekf_model.predict_estimate(Pk_pos_prev=ekf_model.Pk_pos, Q_k_prev=ekf_model.Q_k)

            # Kalman filtering 
            x_rec_hat_pos_k, Pk_pos = ekf_model.filtered_estimate(y_k=Y[j,k].view(-1,1))

            # Save filtered state estimates
            X_estimated_ekf[j,k,:] = x_rec_hat_pos_k.view(-1,)

            # Also save covariances
            Pk_estimated_ekf[j,k,:,:] = Pk_pos

        mse_arr_ekf[j] = mse_loss(X_estimated_ekf[j], X[j])  # Calculate the squared error across the length of a single sequence
        #print("batch: {}, sequence: {}, mse_loss: {}".format(j+1, mse_arr[j]), file=orig_stdout)
        print("batch: {}, mse_loss: {}".format(j+1, mse_arr_ekf[j]))
    
    return X_estimated_ekf, Pk_estimated_ekf, mse_arr_ekf

def get_test_sequence(d, J, T, delta, delta_d, inverse_r2_dB, nu_dB, decimate):

    lorenz_model = LorenzAttractorModel(d=d, J=J, delta=delta, delta_d=delta_d,
                                    A_fn=A_fn, h_fn=h_fn, decimate=decimate)

    x_lorenz, y_lorenz = lorenz_model.generate_single_sequence(T=T, 
                                                    inverse_r2_dB=inverse_r2_dB,
                                                    nu_dB=nu_dB)

    plot_state_trajectory(x_lorenz)
    plot_measurement_data(y_lorenz)

    if decimate == True:
        print("Decimated ...")
        print(x_lorenz.shape, y_lorenz.shape)
        plot_state_trajectory(x_lorenz)
        plot_measurement_data(y_lorenz)

    plot_state_trajectory_axes(x_lorenz)
    plot_measurement_data_axes(y_lorenz)

    return x_lorenz, y_lorenz, lorenz_model

def test_lorenz(device='cpu', model_file_saved=None):

    d = 3
    T = 6_000
    nu_dB = 0
    delta = 0.01
    delta_d = 0.02
    J = 20
    inverse_r2_dB = 0
    decimate=False
    #A_fn = lambda z: torch.Tensor([
    #        [-10, 10, 0],
    #        [28, -1, -z[0]],
    #        [0, z[0], -8.0/3]
    #   ]).type(torch.FloatTensor)
    A_fn = A_fn
    h_fn = h_fn

    x_lorenz, y_lorenz, lorenz_model = get_test_sequence(d=d, T=T, nu_dB=nu_dB, delta=delta, delta_d=delta_d, J=J, 
                                        inverse_r2_dB=inverse_r2_dB, decimate=decimate)

    Ty, dy = y_lorenz.shape
    Tx, dx = x_lorenz.shape
    Y = torch.Tensor(y_lorenz.reshape((1, Ty, dy))).type(torch.FloatTensor)
    X = torch.Tensor(x_lorenz.reshape((1, Tx, dx))).type(torch.FloatTensor)

    # Initialize the Kalman filter model in PyTorch
    ekf_model = EKF(
        n_states=lorenz_model.d,
        n_obs=lorenz_model.d,
        J=3,
        f=lorenz_model.A_fn,
        h=lorenz_model.h_fn,
        Q=None,
        R=None,
        inverse_r2_dB=inverse_r2_dB,
        nu_dB=nu_dB,
        device=device
    )

    # Get the estimates using an extended Kalman filter model
    X_estimated_ekf = None
    Pk_estimated_ekf = None

    X_estimated_ekf, Pk_estimated_ekf, mse_arr_ekf = test_ekf_lorenz(X=X, Y=Y, ekf_model=ekf_model)

    # Initialize the DANSE model in PyTorch

    ssm_dict, est_dict = get_parameters(N=1, T=Ty, n_states=lorenz_model.n_states,
                                            n_obs=lorenz_model.n_obs, inverse_r2_dB=inverse_r2_dB, )

    # Initialize the DANSE model in PyTorch
    danse_model = DANSE(
        n_states=lorenz_model.d,
        n_obs=lorenz_model.d,
        mu_w=lorenz_model.mu_w,
        C_w=lorenz_model.R,
        batch_size=1,
        mu_x0=np.zeros((lorenz_model.n_states,)),
        C_x0=np.eye(lorenz_model.n_states),
        rnn_type=est_dict['danse']['rnn_type'],
        rnn_params_dict=est_dict['danse']['rnn_params_dict'][est_dict['danse']['rnn_type']],
        device=device
    )

    X_estimated_pred = None
    X_estimated_filtered = None
    Pk_estimated_ekf = None
    Pk_estimated_filtered = None

    X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered = test_danse_lorenz(danse_model=danse_model, 
                                                                                                saved_model_file=model_file_saved,
                                                                                                Y=Y,
                                                                                                device=device)
    
    # Plot the result
    plot_state_trajectory_axes(X=torch.squeeze(X,0), X_est_EKF=torch.squeeze(X_estimated_ekf,0), X_est_DANSE=torch.squeeze(X_estimated_filtered,0))
    plot_state_trajectory(X=torch.squeeze(X,0), X_est_EKF=torch.squeeze(X_estimated_ekf,0), X_est_DANSE=torch.squeeze(X_estimated_filtered,0))
    
    plt.show()
    return None

if __name__ == "__main__":
    test_lorenz()

