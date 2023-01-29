import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import sys
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.plot_functions import *
from utils.utils import generate_normal, dB_to_lin, lin_to_dB, mse_loss
from parameters import get_parameters
from generate_data import LinearSSM, LorenzAttractorModel, generate_SSM_data
from src.ekf import EKF
    
def test_kf_lorenz():

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
    A_fn = lambda z: torch.Tensor([
            [-10, 10, 0],
            [28, -1, -z],
            [0, z, -8.0/3]]).type(torch.FloatTensor)
    h_fn = lambda x: x

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
        device='cpu'
    )

    Ty, dy = y_lorenz.shape
    Tx, dx = x_lorenz.shape
    Y = torch.Tensor(y_lorenz.reshape((1, Ty, dy))).type(torch.FloatTensor)
    X = torch.Tensor(x_lorenz.reshape((1, Tx, dx))).type(torch.FloatTensor)

    X_estimated = torch.zeros_like(X).type(torch.FloatTensor)
    Pk_estimated = torch.zeros((1, Tx, dx, dx)).type(torch.FloatTensor)
    mse_arr = torch.zeros((1)).type(torch.FloatTensor)

    # Running the Kalman filter on the given data
    for j in range(0, 1):

        for k in range(0, T):

            # Kalman prediction 
            x_rec_hat_neg_k, Pk_neg = ekf_model.predict_estimate(Pk_pos_prev=ekf_model.Pk_pos, Q_k_prev=ekf_model.Q_k)

            # Kalman filtering 
            x_rec_hat_pos_k, Pk_pos = ekf_model.filtered_estimate(y_k=Y[j,k].view(-1,1))

            # Save filtered state estimates
            X_estimated[j,k,:] = x_rec_hat_pos_k.view(-1,)

            # Also save covariances
            Pk_estimated[j,k,:,:] = Pk_pos

        mse_arr[j] = mse_loss(X_estimated[j], X[j])  # Calculate the squared error across the length of a single sequence
        #print("batch: {}, sequence: {}, mse_loss: {}".format(j+1, mse_arr[j]), file=orig_stdout)
        print("batch: {}, mse_loss: {}".format(j+1, mse_arr[j]))

    # Plot the result
    plot_state_trajectory_axes(X=torch.squeeze(X,0), X_est=torch.squeeze(X_estimated,0))
    plot_state_trajectory(X=torch.squeeze(X,0), X_est=torch.squeeze(X_estimated,0))
    
    plt.show()
    return None

if __name__ == "__main__":
    test_kf_lorenz()

