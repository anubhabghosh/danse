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
from utils.utils import generate_normal, dB_to_lin, lin_to_dB, mse_loss, nmse_loss, mse_loss_dB
from parameters import get_parameters, A_fn, h_fn, f_lorenz, f_lorenz_danse
from generate_data import LorenzAttractorModel, generate_SSM_data
from src.ekf import EKF
from src.ukf import UKF
from src.ukf_aliter import UKF_Aliter
from src.danse import DANSE, push_model

# traj_resultName = ['traj_lor_KNetFull_rq1030_T2000_NT100.pt']#,'partial_lor_r4.pt','partial_lor_r5.pt','partial_lor_r6.pt']

# KalmanNet data
DatafolderName = '../KalmanNet_TSP/Simulations/Lorenz_Atractor/data/T2000_NT100/' + '/'
dataFileName = ['data_lor_v20_rq020_T2000.pt','data_lor_v20_rq1030_T2000.pt','data_lor_v20_rq2040_T2000.pt','data_lor_v20_rq3050_T2000.pt','data_lor_v20_rq4060_T2000.pt']# for T=2000
r2 = torch.tensor([1,0.1,0.01,1e-3,1e-4])
# r2 = torch.tensor([100, 10, 1, 0.1, 0.01])
r = torch.sqrt(r2)
vdB = -20 # ratio v=q2/r2
v = 10**(vdB/10)

q2 = torch.mul(v,r2)
q = torch.sqrt(q2)

### q and r searched for filters
r2searchdB = torch.tensor([-5,0,5])
rrsearch = torch.sqrt(10**(-r2searchdB/10.0))
q2searchdB = torch.tensor([20,15,10])
qsearch = torch.sqrt(10**(-q2searchdB/10.0))

### q and r optimized for filters
r2optdB = torch.tensor([3.0103])
ropt = torch.sqrt(10**(-r2optdB/10))

r2optdB_partial = torch.tensor([3.0103])
ropt_partial = torch.sqrt(10**(-r2optdB_partial/10))

q2optdB = torch.tensor([18.2391,28.2391,38.2391,48,55])
qopt = torch.sqrt(10**(-q2optdB/10))

q2optdB_partial = torch.tensor([18.2391,28.2391,38.2391,48,55])
qopt_partial = torch.sqrt(10**(-q2optdB_partial/10))

def test_danse_lorenz(danse_model, saved_model_file, Y, device='cpu'):

    danse_model.load_state_dict(torch.load(saved_model_file, map_location=device))
    danse_model = push_model(nets=danse_model, device=device)
    danse_model.eval()

    with torch.no_grad():

        Y_test_batch = Variable(Y, requires_grad=False).type(torch.FloatTensor).to(device)
        X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered = danse_model.compute_predictions(Y_test_batch)
    
    return X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered


def test_ukf_lorenz(X, Y, ukf_model):

    X_estimated_ukf, Pk_estimated_ukf, mse_arr_uk_lin, mse_arr_ukf = ukf_model.run_mb_filter(X, Y)
    return X_estimated_ukf, Pk_estimated_ukf, mse_arr_uk_lin, mse_arr_ukf

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

def test_lorenz(device='cpu', model_file_saved=None, X=None, Y=None, q_available=None, r_available=None):

    _, rnn_type, m, n, T, _, inverse_r2_dB, nu_dB = parse("{}_danse_{}_m_{:d}_n_{:d}_T_{:d}_N_{:d}_{:f}dB_{:f}dB", model_file_saved.split('/')[-2])
    print("*"*100)
    print("1/r2: {}dB, nu: {}dB".format(inverse_r2_dB, nu_dB))
    d = m
    #d = 3
    T = 2_000
    #nu_dB = 0.0
    delta = 0.02 # If decimate is True, then set this delta to 1e-5 and run it for long time
    delta_d = 0.02
    J = 5
    #inverse_r2_dB = 20
    decimate=False
    use_Taylor = False 
    #A_fn = lambda z: torch.Tensor([
    #        [-10, 10, 0],
    #        [28, -1, -z[0]],
    #        [0, z[0], -8.0/3]
    #   ]).type(torch.FloatTensor)
    lorenz_model = LorenzAttractorModel(d=d, J=J, delta=delta, delta_d=delta_d,
                                    A_fn=A_fn, h_fn=h_fn, decimate=decimate,
                                    mu_e=np.zeros((d,)), mu_w=np.zeros((d,)),
                                    use_Taylor=use_Taylor)
    
    # My own data generation scheme
    N_test = 5 # No. of trajectories at test time / evaluation
    X = torch.zeros((N_test, T+1, m))
    Y = torch.zeros((N_test, T, n))

    lorenz_model = LorenzAttractorModel(d=d, J=J, delta=delta, delta_d=delta_d,
                                    A_fn=A_fn, h_fn=h_fn, decimate=decimate,
                                    mu_e=np.zeros((d,)), mu_w=np.zeros((d,)),
                                    use_Taylor=use_Taylor)

    print("Test data generated using r2: {} dB, nu: {} dB".format(inverse_r2_dB, nu_dB))
    for i in range(N_test):

        x_lorenz_i, y_lorenz_i  = get_test_sequence(lorenz_model=lorenz_model, T=T, nu_dB=nu_dB,
                                        inverse_r2_dB=inverse_r2_dB)

        X[i, :, :] = torch.from_numpy(x_lorenz_i).type(torch.FloatTensor)
        Y[i, :, :] = torch.from_numpy(y_lorenz_i).type(torch.FloatTensor)

    N_test, Ty, dy = Y.shape
    X = torch.cat((torch.zeros(N_test,1,lorenz_model.n_states),X),dim=1) # For KalmanNet data
    N_test, Tx, dx = X.shape

    # Get the estimate using the baseline
    H_tensor = torch.from_numpy(jacobian(h_fn, torch.randn(lorenz_model.n_states,)).numpy()).type(torch.FloatTensor)
    H_tensor = torch.repeat_interleave(H_tensor.unsqueeze(0),N_test,dim=0)
    X_LS = torch.einsum('ijj,ikj->ikj',H_tensor,Y)
    '''
    # Initialize the Kalman filter model in PyTorch
    ekf_model = EKF(
        n_states=lorenz_model.n_states,
        n_obs=lorenz_model.n_obs,
        J=J,
        f=f_lorenz,#lorenz_model.A_fn, f_lorenz for KalmanNet paper, f_lorenz_danse for our work
        h=lorenz_model.h_fn,
        Q=q_available*np.eye(lorenz_model.n_states), #For KalmanNet
        R=r_available*np.eye(lorenz_model.n_obs), # For KalmanNet
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
    '''
    # Initialize the Kalman filter model in PyTorch
    ukf_model = UKF_Aliter(
        n_states=lorenz_model.n_states,
        n_obs=lorenz_model.n_obs,
        f=None,
        h=lorenz_model.h_fn,
        Q=q_available*np.eye(lorenz_model.n_states), # For KalmanNet, else None
        R=r_available*np.eye(lorenz_model.n_obs), # For KalmanNet, else None,
        kappa=-1, # Usually kept 0
        alpha=0.1, # Usually small 1e-3
        delta_t=lorenz_model.delta,
        beta=2,
        n_sigma=2*lorenz_model.n_states+1,
        inverse_r2_dB=inverse_r2_dB,
        nu_dB=nu_dB,
        device=device
    )

    # Get the estimates using an extended Kalman filter model
    X_estimated_ukf = None
    Pk_estimated_ukf = None
    start_time_ukf = timer()
    X_estimated_ukf, Pk_estimated_ukf, mse_arr_ukf_lin, mse_arr_ukf = test_ukf_lorenz(X=X, Y=Y, ukf_model=ukf_model)
    time_elapsed_ukf = timer() - start_time_ukf
    ''''''
    # Initialize the DANSE model in PyTorch
    
    ssm_dict, est_dict = get_parameters(N=1, T=Ty, n_states=lorenz_model.n_states,
                                        n_obs=lorenz_model.n_obs, 
                                        inverse_r2_dB=inverse_r2_dB, 
                                        nu_dB=nu_dB)
    

    # Initialize the DANSE model in PyTorch
    C_w = r_available*np.eye(lorenz_model.n_obs)
    danse_model = DANSE(
        n_states=lorenz_model.n_states,
        n_obs=lorenz_model.n_obs,
        mu_w=lorenz_model.mu_w,
        C_w=C_w,
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
    
    nmse_ls = mse_loss_dB(X[:,1:,:], X_LS[:,0:,:])
    #nmse_ekf = nmse_loss(X[:,1:,:], X_estimated_ekf[:,1:,:])
    #nmse_ukf = nmse_loss(X[:,1:,:], X_estimated_ukf[:,1:,:])
    #nmse_danse = nmse_loss(X[:,1:,:], X_estimated_filtered[:,0:,:])
    #nmse_ekf = mse_loss_dB(X[:,1:,:], X_estimated_ekf[:,1:,:])
    nmse_ukf = mse_loss_dB(X[:,1:,:], X_estimated_ukf[:,1:,:])
    nmse_danse = mse_loss_dB(X[:,1:,:], X_estimated_filtered[:,0:,:])
    snr = mse_loss(X[:,1:,:], torch.zeros_like(X[:,1:,:])) * dB_to_lin(inverse_r2_dB)

    print("LS, batch: {}, mse over {} samples : {}".format(1, N_test, nmse_ls))
    print("ekf, batch: {}, mse over {} samples : {}, time: {} secs".format(1, N_test, nmse_ekf, time_elapsed_ekf))
    print("ukf, batch: {}, mse over {} samples : {}, time: {} secs".format(1, N_test, nmse_ukf, time_elapsed_ukf))
    print("danse, batch: {}, mse over {} samples : {}, time: {} secs".format(1, N_test, nmse_danse, time_elapsed_danse))

    # Plot the result
    plot_state_trajectory_axes(X=torch.squeeze(X[0,1:,:],0), X_est_EKF=torch.squeeze(X_estimated_ekf[0,1:,:],0), X_est_UKF=torch.squeeze(X_estimated_ukf[0,1:,:],0), X_est_DANSE=torch.squeeze(X_estimated_filtered[0],0))
    plot_state_trajectory(X=torch.squeeze(X[0,1:,:],0), X_est_EKF=torch.squeeze(X_estimated_ekf[0,1:,:],0), X_est_UKF=torch.squeeze(X_estimated_ukf[0,1:,:],0), X_est_DANSE=torch.squeeze(X_estimated_filtered[0],0))
    #plot_state_trajectory_axes(X=torch.squeeze(X,0), X_est_EKF=torch.squeeze(X_estimated_ekf,0), X_est_DANSE=torch.squeeze(X_estimated_filtered,0))
    #plot_state_trajectory(X=torch.squeeze(X,0), X_est_EKF=torch.squeeze(X_estimated_ekf,0), X_est_DANSE=torch.squeeze(X_estimated_filtered,0))
    
    #plt.show()
    return nmse_ekf, nmse_danse, nmse_ukf, nmse_ls, time_elapsed_ekf, time_elapsed_danse, time_elapsed_ukf, snr

if __name__ == "__main__":
    device = 'cpu'
    inverse_r2_dB_arr = np.array([0.0, 10.0, 20.0, 30.0])
    #inverse_r2_dB_arr = np.array([20.0])
    nmse_ls_arr = np.zeros((len(inverse_r2_dB_arr,)))
    nmse_ekf_arr = np.zeros((len(inverse_r2_dB_arr,)))
    nmse_ukf_arr = np.zeros((len(inverse_r2_dB_arr,)))
    nmse_danse_arr = np.zeros((len(inverse_r2_dB_arr,)))
    t_ekf_arr = np.zeros((len(inverse_r2_dB_arr,)))
    t_ukf_arr = np.zeros((len(inverse_r2_dB_arr,)))
    t_danse_arr = np.zeros((len(inverse_r2_dB_arr,)))
    snr_arr = np.zeros((len(inverse_r2_dB_arr,)))

    model_file_saved_dict = {
        "-10.0dB":"./models/LorenzSSM_danse_gru_m_3_n_3_T_1000_N_500_-10.0dB_-20.0dB/danse_gru_ckpt_epoch_671_best.pt",
        "0.0dB":"./models/LorenzSSM_danse_gru_m_3_n_3_T_1000_N_500_0.0dB_-20.0dB/danse_gru_ckpt_epoch_671_best.pt",
        "10.0dB":"./models/LorenzSSM_danse_gru_m_3_n_3_T_1000_N_500_10.0dB_-20.0dB/danse_gru_ckpt_epoch_671_best.pt",
        "20.0dB":"./models/LorenzSSM_danse_gru_m_3_n_3_T_1000_N_500_20.0dB_-20.0dB/danse_gru_ckpt_epoch_681_best.pt",
        "30.0dB":"./models/LorenzSSM_danse_gru_m_3_n_3_T_1000_N_500_30.0dB_-20.0dB/danse_gru_ckpt_epoch_684_best.pt"
    }

    for i, inverse_r2_dB in enumerate(inverse_r2_dB_arr):
        
        print("Data Load")
        print(dataFileName[i])
        [train_input_long,train_target_long, cv_input, cv_target, test_input, test_target] =  torch.load(DatafolderName + dataFileName[i],map_location='cpu')
        #model_file_saved_i = 'models/LorenzSSM_danse_gru_m_3_n_3_T_1000_N_500_{}dB_-20.0dB/danse_gru_ckpt_epoch_300_best.pt'.format(inverse_r2_dB)
        test_target = torch.transpose(test_target,1,2)
        test_input = torch.transpose(test_input,1,2)
        q_i = qopt[i].numpy()**2
        r_i = r[i].numpy()**2
        model_file_saved_i = model_file_saved_dict['{}dB'.format(inverse_r2_dB)]
        print(model_file_saved_i)
        nmse_ekf_i, nmse_danse_i, nmse_ukf_i, nmse_ls_i,  time_elapsed_ekf_i, time_elapsed_danse_i, time_elapsed_ukf_i, snr_i = test_lorenz(device=device, 
            model_file_saved=model_file_saved_i, X=test_target, Y=test_input, q_available=q_i, r_available=r_i)
        nmse_ls_arr[i] = nmse_ls_i.numpy().item()
        nmse_ekf_arr[i] = nmse_ekf_i.numpy().item()
        nmse_ukf_arr[i] = nmse_ukf_i.numpy().item()
        nmse_danse_arr[i] = nmse_danse_i.numpy().item()
        t_ekf_arr[i] = time_elapsed_ekf_i
        t_ukf_arr[i] = time_elapsed_ukf_i
        t_danse_arr[i] = time_elapsed_danse_i
        snr_arr[i] = 10*np.log10(snr_i.numpy().item())

    # Plotting the NMSE Curve
    plt.rcParams['font.family'] = 'serif'
    plt.figure()
    plt.plot(inverse_r2_dB_arr, nmse_ls_arr, 'gp--', linewidth=1.5, label="NMSE-LS")
    plt.plot(inverse_r2_dB_arr, nmse_ekf_arr, 'rd--', linewidth=1.5, label="NMSE-EKF")
    plt.plot(inverse_r2_dB_arr, nmse_danse_arr, 'bo-', linewidth=2.0, label="NMSE-DANSE")
    plt.plot(inverse_r2_dB_arr, nmse_ukf_arr, 'ks-', linewidth=2.0, label="NMSE-UKF")
    plt.xlabel('$\\frac{1}{r^2}$ (in dB)')
    plt.ylabel('NMSE (in dB)')
    plt.grid(True)
    plt.legend()
    #plt.subplot(212)
    #plt.savefig('./figs/NMSE_vs_inverse_r2dB_Lorenz.pdf')

    plt.figure()
    plt.plot(snr_arr, nmse_ls_arr, 'gp--', linewidth=1.5, label="NMSE-LS")
    plt.plot(snr_arr, nmse_ekf_arr, 'rd--', linewidth=1.5, label="NMSE-EKF")
    plt.plot(snr_arr, nmse_danse_arr, 'bo-', linewidth=2.0, label="NMSE-DANSE")
    plt.plot(snr_arr, nmse_ukf_arr, 'ks-', linewidth=2.0, label="NMSE-UKF")
    plt.xlabel('SNR (in dB)')
    plt.ylabel('NMSE (in dB)')
    plt.grid(True)
    plt.legend()
    #plt.savefig('./figs/NMSE_vs_SNR_Lorenz.pdf')

    # Plotting the Time-elapsed Curve
    plt.figure()
    #plt.subplot(211)
    plt.plot(inverse_r2_dB_arr, t_ekf_arr, 'rd--', linewidth=1.5, label="Inference time-EKF")
    plt.plot(inverse_r2_dB_arr, t_ukf_arr, 'ks--', linewidth=1.5, label="Inference time-UKF")
    plt.plot(inverse_r2_dB_arr, t_danse_arr, 'bo-', linewidth=2.0, label="Inference time-DANSE")
    plt.xlabel('$\\frac{1}{r^2}$ (in dB)')
    plt.ylabel('Time (in s)')
    plt.grid(True)
    plt.legend()
    #plt.savefig('./figs/InferTime_vs_inverse_r2dB_Lorenz.pdf')

    #plt.subplot(212)
    plt.figure()
    plt.plot(snr_arr, t_ekf_arr, 'rd--', linewidth=1.5, label="Inference time-EKF")
    plt.plot(snr_arr, t_ukf_arr, 'ks-', linewidth=2.0, label="Inference time-UKF")
    plt.plot(snr_arr, t_danse_arr, 'bo-', linewidth=2.0, label="Inference time-DANSE")
    plt.xlabel('SNR (in dB)')
    plt.ylabel('Time (in s)')
    plt.grid(True)
    plt.legend()
    #plt.savefig('./figs/InferTime_vs_SNR_Lorenz.pdf')

    plt.show()

