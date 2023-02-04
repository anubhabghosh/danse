import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import sys
import os
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.autograd.functional import jacobian
from parse import parse
from timeit import default_timer as timer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.plot_functions import *
from utils.utils import generate_normal, dB_to_lin, lin_to_dB, mse_loss, nmse_loss
from parameters import get_parameters, A_fn, h_fn
from generate_data import LorenzAttractorModel, generate_SSM_data
from src.ekf import EKF
from src.danse import DANSE, push_model

def test_danse_lorenz(danse_model, saved_model_file, Y, device='cpu'):

    danse_model.load_state_dict(torch.load(saved_model_file, map_location=device))
    danse_model = push_model(nets=danse_model, device=device)
    danse_model.eval()

    with torch.no_grad():

        Y_test_batch = Variable(Y, requires_grad=False).type(torch.FloatTensor).to(device)
        X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered = danse_model.compute_predictions(Y_test_batch)
    
    return X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered

def test_ekf_lorenz(X, Y, ekf_model):

    X_estimated_ekf, Pk_estimated_ekf, mse_arr_ekf = ekf_model.run_mb_filter(X, Y)
    return X_estimated_ekf, Pk_estimated_ekf, mse_arr_ekf

def get_test_sequence(lorenz_model, T, inverse_r2_dB, nu_dB):

    x_lorenz, y_lorenz = lorenz_model.generate_single_sequence(T=T, 
                                                    inverse_r2_dB=inverse_r2_dB,
                                                    nu_dB=nu_dB)

    #plot_state_trajectory(x_lorenz)
    #plot_measurement_data(y_lorenz)

    #if decimate == True:
    #    print("Decimated ...")
    #    print(x_lorenz.shape, y_lorenz.shape)
    #    plot_state_trajectory(x_lorenz)
    #    plot_measurement_data(y_lorenz)

    #plot_state_trajectory_axes(x_lorenz)
    #plot_measurement_data_axes(y_lorenz)

    return x_lorenz, y_lorenz

def test_lorenz(device='cpu', model_file_saved=None):

    _, rnn_type, m, n, T, _, inverse_r2_dB, nu_dB = parse("{}_danse_{}_m_{:d}_n_{:d}_T_{:d}_N_{:d}_{:f}dB_{:f}dB", model_file_saved.split('/')[-2])
    print("*"*100)
    print("1/r2: {}dB, nu: {}dB".format(inverse_r2_dB, nu_dB))
    d = m
    #T = 6_000
    #nu_dB = 0.0
    delta = 0.01
    delta_d = 0.02
    J = 20
    #inverse_r2_dB = 0
    decimate=False
    use_Taylor=True
    #A_fn = lambda z: torch.Tensor([
    #        [-10, 10, 0],
    #        [28, -1, -z[0]],
    #        [0, z[0], -8.0/3]
    #   ]).type(torch.FloatTensor)

    N_test = 1 # No. of trajectories at test time / evaluation
    X = torch.zeros((N_test, T+1, m))
    Y = torch.zeros((N_test, T, n))

    lorenz_model = LorenzAttractorModel(d=d, J=J, delta=delta, delta_d=delta_d,
                                    A_fn=A_fn, h_fn=h_fn, decimate=decimate,
                                    mu_e=np.zeros((d,)), mu_w=np.zeros((d,)),
                                    use_Taylor=use_Taylor)

    for i in range(N_test):

        x_lorenz_i, y_lorenz_i  = get_test_sequence(lorenz_model=lorenz_model, T=T, nu_dB=nu_dB,
                                        inverse_r2_dB=inverse_r2_dB)

        X[i, :, :] = torch.from_numpy(x_lorenz_i).type(torch.FloatTensor)
        Y[i, :, :] = torch.from_numpy(y_lorenz_i).type(torch.FloatTensor)

    _, Ty, dy = Y.shape
    _, Tx, dx = X.shape
    
    #Ty, dy = y_lorenz.shape
    #Tx, dx = x_lorenz.shape
    #Y = torch.Tensor(y_lorenz.reshape((1, Ty, dy))).type(torch.FloatTensor)
    #X = torch.Tensor(x_lorenz.reshape((1, Tx, dx))).type(torch.FloatTensor)

    # Initialize the Kalman filter model in PyTorch
    ekf_model = EKF(
        n_states=lorenz_model.n_states,
        n_obs=lorenz_model.n_obs,
        J=J,
        f=lorenz_model.A_fn,
        h=lorenz_model.h_fn,
        Q=None,
        R=None,
        inverse_r2_dB=inverse_r2_dB,
        nu_dB=nu_dB,
        device=device,
        use_Taylor=use_Taylor
    )

    # Get the estimates using an extended Kalman filter model
    X_estimated_ekf = None
    Pk_estimated_ekf = None

    start_time_ekf = timer()
    X_estimated_ekf, Pk_estimated_ekf, mse_arr_ekf = test_ekf_lorenz(X=X, Y=Y, ekf_model=ekf_model)
    time_elapsed_ekf = timer() - start_time_ekf

    # Initialize the DANSE model in PyTorch

    ssm_dict, est_dict = get_parameters(N=1, T=Ty, n_states=lorenz_model.n_states,
                                        n_obs=lorenz_model.n_obs, 
                                        inverse_r2_dB=inverse_r2_dB, 
                                        nu_dB=nu_dB)

    # Initialize the DANSE model in PyTorch
    danse_model = DANSE(
        n_states=lorenz_model.n_states,
        n_obs=lorenz_model.n_obs,
        mu_w=lorenz_model.mu_w,
        C_w=lorenz_model.R,
        batch_size=1,
        H=jacobian(h_fn, torch.randn(lorenz_model.n_states,)).numpy(),
        mu_x0=np.zeros((lorenz_model.n_states,)),
        C_x0=np.eye(lorenz_model.n_states),
        rnn_type=rnn_type,
        rnn_params_dict=est_dict['danse']['rnn_params_dict'],
        device=device
    )

    X_estimated_pred = None
    X_estimated_filtered = None
    Pk_estimated_ekf = None
    Pk_estimated_filtered = None
    start_time_danse = timer()
    X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered = test_danse_lorenz(danse_model=danse_model, 
                                                                                                saved_model_file=model_file_saved,
                                                                                                Y=Y,
                                                                                                device=device)
    time_elapsed_danse = timer() - start_time_danse
    nmse_ekf = nmse_loss(X[:,1:,:], X_estimated_ekf[:,1:,:])
    nmse_danse = nmse_loss(X[:,1:,:], X_estimated_filtered[:,0:,:])
    print("ekf, batch: {}, mse over {} samples : {}, time: {} secs".format(1, N_test, nmse_ekf, time_elapsed_ekf))
    print("danse, batch: {}, mse over {} samples : {}, time: {} secs".format(1, N_test, nmse_danse, time_elapsed_danse))

    # Plot the result for a single trajcetory
    plot_state_trajectory_axes(X=torch.squeeze(X[0],0), X_est_EKF=torch.squeeze(X_estimated_ekf[0],0), X_est_DANSE=torch.squeeze(X_estimated_filtered[0],0), savefig=True, savefig_name="./figs/Lorenz_3d_r2{}dB_nu_{}dB.pdf".format(inverse_r2_dB, nu_dB))
    plot_state_trajectory(X=torch.squeeze(X[0],0), X_est_EKF=torch.squeeze(X_estimated_ekf[0],0), X_est_DANSE=torch.squeeze(X_estimated_filtered[0],0), savefig=True, savefig_name="./figs/Lorenz_3d_axeswise_r2{}dB_nu_{}dB.pdf".format(inverse_r2_dB, nu_dB))
    
    return nmse_ekf, nmse_danse, time_elapsed_ekf, time_elapsed_danse

if __name__ == "__main__":
    device = 'cpu'
    inverse_r2_dB_arr = np.array([-10.0, 0.0, 10.0, 20.0, 30.0])
    nmse_ekf_arr = np.zeros((len(inverse_r2_dB_arr,)))
    nmse_danse_arr = np.zeros((len(inverse_r2_dB_arr,)))
    t_ekf_arr = np.zeros((len(inverse_r2_dB_arr,)))
    t_danse_arr = np.zeros((len(inverse_r2_dB_arr,)))

    for i, inverse_r2_dB in enumerate(inverse_r2_dB_arr):
        model_file_saved = 'models/LorenzSSM_danse_gru_m_3_n_3_T_1000_N_500_{}dB_-20.0dB/danse_gru_ckpt_epoch_300_best.pt'.format(inverse_r2_dB)
        nmse_ekf_i, nmse_danse_i, time_elapsed_ekf_i, time_elapsed_danse_i = test_lorenz(device=device, model_file_saved=model_file_saved)
        nmse_ekf_arr[i] = nmse_ekf_i.numpy().item()
        nmse_danse_arr[i] = nmse_danse_i.numpy().item()
        t_ekf_arr[i] = time_elapsed_ekf_i
        t_danse_arr[i] = time_elapsed_danse_i

    # Plotting the NMSE Curve
    plt.figure()
    plt.plot(inverse_r2_dB_arr, nmse_ekf_arr, 'rd--', linewidth=1.5, label="NMSE-EKF")
    plt.plot(inverse_r2_dB_arr, nmse_danse_arr, 'bo-', linewidth=2.0, label="NMSE-DANSE")
    plt.xlabel('$\\frac{1}{r^2}$ (in dB)')
    plt.ylabel('NMSE (in dB)')
    plt.grid(True)
    plt.legend()

    # Plotting the Time-elapsed Curve
    plt.figure()
    plt.plot(inverse_r2_dB_arr, t_ekf_arr, 'rd--', linewidth=1.5, label="Inference time-EKF")
    plt.plot(inverse_r2_dB_arr, t_danse_arr, 'bo-', linewidth=2.0, label="Inference time-DANSE")
    plt.xlabel('$\\frac{1}{r^2}$ (in dB)')
    plt.ylabel('Time (in s)')
    plt.grid(True)
    plt.legend()

    plt.show()
