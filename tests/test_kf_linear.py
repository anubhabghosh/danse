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
from src.kf import KF
    
def test_kf_linear():

    m = 10
    n = 10
    q = 0.1
    r = 0.1
    T = 6_000
    nu_dB = 0
    inverse_r2_dB = 40
    linear_ssm = LinearSSM(n_states=m, n_obs=n, F=None, G=np.zeros((m,1)), H=None, 
                        mu_e=np.zeros((m,1)), mu_w=np.zeros((n,1)), q=q, r=r, 
                        Q=None, R=None)
    x_lin, y_lin = linear_ssm.generate_single_sequence(T=T, inverse_r2_dB=inverse_r2_dB, nu_dB=nu_dB)

    # Plotting the trajectories in 3d
    plot_state_trajectory(x_lin)
    plot_measurement_data(y_lin)

    # Plotting the same axis-wise
    plot_state_trajectory_axes(x_lin)
    plot_measurement_data_axes(y_lin)

    # Initialize the Kalman filter model in PyTorch
    kf_model = KF(n_states=linear_ssm.n_states,
                        n_obs=linear_ssm.n_obs,
                        F=linear_ssm.F,
                        G=linear_ssm.G,
                        H=linear_ssm.H,
                        Q=linear_ssm.Q,
                        R=linear_ssm.R,
                        inverse_r2_dB=inverse_r2_dB,
                        nu_dB=nu_dB,
                        device='cpu')

    Ty, dy = y_lin.shape
    Tx, dx = x_lin.shape
    Y = torch.Tensor(y_lin.reshape((1, Ty, dy))).type(torch.FloatTensor)
    X = torch.Tensor(x_lin.reshape((1, Tx, dx))).type(torch.FloatTensor)

    X_estimated = torch.zeros_like(X).type(torch.FloatTensor)
    Pk_estimated = torch.zeros((1, Tx, dx, dx)).type(torch.FloatTensor)
    mse_arr = torch.zeros((1)).type(torch.FloatTensor)

    # Running the Kalman filter on the given data
    for j in range(0, 1):

        for k in range(0, T):

            # Kalman prediction 
            x_rec_hat_neg_k, Pk_neg = kf_model.predict_estimate(F_k_prev=kf_model.F_k, Pk_pos_prev=kf_model.Pk_pos, Q_k_prev=kf_model.Q_k)

            # Kalman filtering 
            x_rec_hat_pos_k, Pk_pos = kf_model.filtered_estimate(y_k=Y[j,k].view(-1,1))

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
    test_kf_linear()
