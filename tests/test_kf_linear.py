import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import sys
import os
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.plot_functions import *
from utils.utils import generate_normal, dB_to_lin, lin_to_dB, mse_loss
from parameters import get_parameters
from generate_data import LinearSSM, generate_SSM_data
from src.kf import KF
from src.danse import DANSE, push_model
from parse import parse

def test_kf_linear(X, Y, kf_model):

    N, Ty, dy = Y.shape
    N, Tx, dx = X.shape

    X_estimated_KF = torch.zeros_like(X).type(torch.FloatTensor)
    Pk_estimated_KF = torch.zeros((N, Tx, dx, dx)).type(torch.FloatTensor)
    mse_arr_KF = torch.zeros((N,)).type(torch.FloatTensor)

    # Running the Kalman filter on the given data
    for j in range(0, N):

        for k in range(0, Ty):

            # Kalman prediction 
            x_rec_hat_neg_k, Pk_neg = kf_model.predict_estimate(F_k_prev=kf_model.F_k, Pk_pos_prev=kf_model.Pk_pos, Q_k_prev=kf_model.Q_k)

            # Kalman filtering 
            x_rec_hat_pos_k, Pk_pos = kf_model.filtered_estimate(y_k=Y[j,k].view(-1,1))

            # Save filtered state estimates
            X_estimated_KF[j,k+1,:] = x_rec_hat_pos_k.view(-1,)

            # Also save covariances
            Pk_estimated_KF[j,k+1,:,:] = Pk_pos

        mse_arr_KF[j] = mse_loss(X_estimated_KF[j], X[j])  # Calculate the squared error across the length of a single sequence
        #print("batch: {}, sequence: {}, mse_loss: {}".format(j+1, mse_arr[j]), file=orig_stdout)
        print("batch: {}, mse_loss: {}".format(j+1, mse_arr_KF[j]))

    return X_estimated_KF, Pk_estimated_KF, mse_arr_KF

def test_danse_linear(danse_model, saved_model_file, Y, device='cpu'):

    danse_model.load_state_dict(torch.load(saved_model_file, map_location=device))
    danse_model = push_model(nets=danse_model, device=device)
    danse_model.eval()

    with torch.no_grad():

        Y_test_batch = Variable(Y, requires_grad=False).type(torch.FloatTensor).to(device)
        X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered = danse_model.compute_predictions(Y_test_batch)
    
    return X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered

def test_linear(device='cpu', model_file_saved=None):

    _, rnn_type, m, n, T, _, inverse_r2_dB, nu_dB = parse("{}_danse_{}_m_{:d}_n_{:d}_T_{:d}_N_{:d}_{:f}dB_{:f}dB", model_file_saved.split('/')[-2])
    
    #m = 10
    #n = 10
    q = 1.0
    r = 1.0
    #T = 6_000
    #nu_dB = 0
    #inverse_r2_dB = 40
    
    # Initialize a Linear SSM with the extracted parameters
    linear_ssm = LinearSSM(n_states=m, n_obs=n, F=None, G=np.zeros((m,1)), H=None, 
                        mu_e=np.zeros((m,)), mu_w=np.zeros((n,)), q=q, r=r, 
                        Q=None, R=None)

    x_lin, y_lin = linear_ssm.generate_single_sequence(T=T, inverse_r2_dB=inverse_r2_dB, nu_dB=nu_dB)

    # Plotting the trajectories in 3d
    plot_state_trajectory(x_lin)
    plot_measurement_data(y_lin)

    # Plotting the same axis-wise
    plot_state_trajectory_axes(x_lin)
    plot_measurement_data_axes(y_lin)

    Ty, dy = y_lin.shape
    Tx, dx = x_lin.shape
    Y = torch.Tensor(y_lin.reshape((1, Ty, dy))).type(torch.FloatTensor)
    X = torch.Tensor(x_lin.reshape((1, Tx, dx))).type(torch.FloatTensor)

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

    # Get the estimates using an extended Kalman filter model
    X_estimated_kf = None
    Pk_estimated_kf = None
    
    X_estimated_kf, Pk_estimated_kf, mse_arr_kf = test_kf_linear(X=X, Y=Y, kf_model=kf_model)
    


    # Initialize the DANSE model in PyTorch
    ssm_dict, est_dict = get_parameters(N=1, T=Ty, n_states=linear_ssm.n_states,
                                        n_obs=linear_ssm.n_obs, 
                                        inverse_r2_dB=inverse_r2_dB,
                                        nu_dB=nu_dB)

    # Initialize the DANSE model in PyTorch
    danse_model = DANSE(
        n_states=linear_ssm.n_states,
        n_obs=linear_ssm.n_obs,
        mu_w=linear_ssm.mu_w,
        C_w=linear_ssm.R,
        batch_size=1,
        H=linear_ssm.H,
        mu_x0=np.zeros((linear_ssm.n_states,)),
        C_x0=np.eye(linear_ssm.n_states),
        rnn_type=rnn_type,
        rnn_params_dict=est_dict['danse']['rnn_params_dict'],
        device=device
    )

    X_estimated_pred = None
    X_estimated_filtered = None
    Pk_estimated_ekf = None
    Pk_estimated_filtered = None

    X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered = test_danse_linear(danse_model=danse_model, 
                                                                                                        saved_model_file=model_file_saved,
                                                                                                        Y=Y,
                                                                                                        device=device)
    
  
    # Plot the result
    plot_state_trajectory_axes(X=torch.squeeze(X,0), X_est_KF=torch.squeeze(X_estimated_kf, 0), X_est_DANSE=torch.squeeze(X_estimated_filtered, 0))
    plot_state_trajectory(X=torch.squeeze(X,0), X_est_KF=torch.squeeze(X_estimated_kf, 0), X_est_DANSE=torch.squeeze(X_estimated_filtered, 0))
    
    plt.show()
    return None

if __name__ == "__main__":
    device = 'cpu'
    model_file_saved = './models/LinearSSM_danse_gru_m_5_n_5_T_500_N_500_20.0dB_0.0dB/danse_gru_ckpt_epoch_300_best.pt'
    test_linear(device=device, model_file_saved=model_file_saved)
