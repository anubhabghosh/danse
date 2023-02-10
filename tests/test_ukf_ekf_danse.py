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
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.plot_functions import *
from utils.utils import generate_normal, dB_to_lin, lin_to_dB, mse_loss, nmse_loss, \
    mse_loss_dB, load_saved_dataset, save_dataset, nmse_loss_std, mse_loss_dB_std, NDArrayEncoder
from parameters import get_parameters, A_fn, h_fn, f_lorenz, f_lorenz_danse, delta_t
from generate_data import LorenzAttractorModel, generate_SSM_data
from src.ekf import EKF
from src.ukf import UKF
from src.ukf_aliter import UKF_Aliter
from src.danse import DANSE, push_model

# traj_resultName = ['traj_lor_KNetFull_rq1030_T2000_NT100.pt']#,'partial_lor_r4.pt','partial_lor_r5.pt','partial_lor_r6.pt']
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

def test_lorenz(device='cpu', model_file_saved=None, test_data_file=None, test_logfile=None, X=None, Y=None, q_available=None, r_available=None):

    _, rnn_type, m, n, T, _, inverse_r2_dB, nu_dB = parse("{}_danse_{}_m_{:d}_n_{:d}_T_{:d}_N_{:d}_{:f}dB_{:f}dB", model_file_saved.split('/')[-2])
    d = m
    #d = 3
    T_test = 2_000
    #nu_dB = 0.0
    delta = delta_t # If decimate is True, then set this delta to 1e-5 and run it for long time
    delta_d = 0.02
    J = 5
    J_test = 5
    #inverse_r2_dB = 20
    decimate=False
    use_Taylor = False 

    orig_stdout = sys.stdout
    f_tmp = open(test_logfile, 'a')
    sys.stdout = f_tmp

    if not os.path.isfile(test_data_file):
        
        print('Dataset is not present, creating at {}'.format(test_data_file))
        # My own data generation scheme
        m, n, T_test, N_test, inverse_r2_dB_test, nu_dB_test = parse("test_trajectories_m_{:d}_n_{:d}_LorenzSSM_data_T_{:d}_N_{:d}_r2_{:f}dB_nu_{:f}dB.pkl", test_data_file.split('/')[-1])
        #N_test = 100 # No. of trajectories at test time / evaluation
        X = torch.zeros((N_test, T_test+1, m))
        Y = torch.zeros((N_test, T_test, n))

        lorenz_model = LorenzAttractorModel(d=d, J=J, delta=delta, delta_d=delta_d,
                                        A_fn=A_fn, h_fn=h_fn, decimate=decimate,
                                        mu_e=np.zeros((d,)), mu_w=np.zeros((d,)),
                                        use_Taylor=use_Taylor)

        print("Test data generated using r2: {} dB, nu: {} dB".format(inverse_r2_dB_test, nu_dB_test))
        for i in range(N_test):
            x_lorenz_i, y_lorenz_i  = get_test_sequence(lorenz_model=lorenz_model, T=T_test, nu_dB=nu_dB_test,
                                            inverse_r2_dB=inverse_r2_dB_test)
            X[i, :, :] = torch.from_numpy(x_lorenz_i).type(torch.FloatTensor)
            Y[i, :, :] = torch.from_numpy(y_lorenz_i).type(torch.FloatTensor)

        test_data_dict = {}
        test_data_dict["X"] = X
        test_data_dict["Y"] = Y
        test_data_dict["model"] = lorenz_model
        save_dataset(Z_XY=test_data_dict, filename=test_data_file)

    else:

        print("Dataset at {} already present!".format(test_data_file))
        m, n, T_test, N_test, inverse_r2_dB_test, nu_dB_test = parse("test_trajectories_m_{:d}_n_{:d}_LorenzSSM_data_T_{:d}_N_{:d}_r2_{:f}dB_nu_{:f}dB.pkl", test_data_file.split('/')[-1])
        test_data_dict = load_saved_dataset(filename=test_data_file)
        X = test_data_dict["X"]
        Y = test_data_dict["Y"]
        lorenz_model = test_data_dict["model"]

    print("*"*100)
    print("*"*100,file=orig_stdout)
    i_test = np.random.choice(N_test)
    print("1/r2: {}dB, nu: {}dB".format(inverse_r2_dB_test, nu_dB_test))
    print("1/r2: {}dB, nu: {}dB".format(inverse_r2_dB_test, nu_dB_test), file=orig_stdout)
    #print(i_test)
    #Y = Y[:5]
    #X = X[:5]
    
    N_test, Ty, dy = Y.shape
    N_test, Tx, dx = X.shape

    # Get the estimate using the baseline
    H_tensor = torch.from_numpy(jacobian(h_fn, torch.randn(lorenz_model.n_states,)).numpy()).type(torch.FloatTensor)
    H_tensor = torch.repeat_interleave(H_tensor.unsqueeze(0),N_test,dim=0)
    X_LS = torch.einsum('ijj,ikj->ikj',torch.pinverse(H_tensor),Y)

    # Initialize the extended Kalman filter model in PyTorch
    
    ekf_model = EKF(
        n_states=lorenz_model.n_states,
        n_obs=lorenz_model.n_obs,
        J=J_test,
        f=f_lorenz_danse,#lorenz_model.A_fn, f_lorenz for KalmanNet paper, f_lorenz_danse for our work
        h=lorenz_model.h_fn,
        Q=lorenz_model.Q, #For KalmanNet
        R=lorenz_model.R, # For KalmanNet
        inverse_r2_dB=inverse_r2_dB_test,
        nu_dB=nu_dB_test,
        device=device,
        use_Taylor=use_Taylor
    )

    # Get the estimates using an extended Kalman filter model
    
    X_estimated_ekf = None
    Pk_estimated_ekf = None

    start_time_ekf = timer()
    X_estimated_ekf, Pk_estimated_ekf, mse_arr_ekf = test_ekf_lorenz(X=X, Y=Y, ekf_model=ekf_model)
    time_elapsed_ekf = timer() - start_time_ekf
    
    # Initialize the Kalman filter model in PyTorch
    ukf_model = UKF_Aliter(
        n_states=lorenz_model.n_states,
        n_obs=lorenz_model.n_obs,
        f=None,
        h=lorenz_model.h_fn,
        Q=lorenz_model.Q, # For KalmanNet, else None
        R=lorenz_model.R, # For KalmanNet, else None,
        kappa=-1, # Usually kept 0
        alpha=0.1, # Usually small 1e-3
        delta_t=lorenz_model.delta,
        beta=2,
        n_sigma=2*lorenz_model.n_states+1,
        inverse_r2_dB=inverse_r2_dB_test,
        nu_dB=nu_dB_test,
        device=device
    )

    # Get the estimates using an extended Kalman filter model
    X_estimated_ukf = None
    Pk_estimated_ukf = None
    start_time_ukf = timer()
    X_estimated_ukf, Pk_estimated_ukf, mse_arr_ukf_lin, mse_arr_ukf = test_ukf_lorenz(X=X, Y=Y, ukf_model=ukf_model)
    time_elapsed_ukf = timer() - start_time_ukf
    # Initialize the DANSE model in PyTorch
    
    ssm_dict, est_dict = get_parameters(N=1, T=Ty, n_states=lorenz_model.n_states,
                                        n_obs=lorenz_model.n_obs, 
                                        inverse_r2_dB=inverse_r2_dB_test, 
                                        nu_dB=nu_dB_test)
    

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
    
    print("DANSE Model file: {}".format(model_file_saved))

    X_estimated_pred = None
    X_estimated_filtered = None
    Pk_estimated_filtered = None

    start_time_danse = timer()
    X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered = test_danse_lorenz(danse_model=danse_model, 
                                                                                                saved_model_file=model_file_saved,
                                                                                                Y=Y,
                                                                                                device=device)
    time_elapsed_danse = timer() - start_time_danse

    nmse_ls = nmse_loss(X[:,1:,:], X_LS[:,0:,:])
    nmse_ls_std = nmse_loss_std(X[:,1:,:], X_LS[:,0:,:])
    nmse_ekf = nmse_loss(X[:,1:,:], X_estimated_ekf[:,1:,:])
    nmse_ekf_std = nmse_loss_std(X[:,1:,:], X_estimated_ekf[:,1:,:])
    nmse_ukf = nmse_loss(X[:,1:,:], X_estimated_ukf[:,1:,:])
    nmse_ukf_std = nmse_loss_std(X[:,1:,:], X_estimated_ukf[:,1:,:])
    nmse_danse = nmse_loss(X[:,1:,:], X_estimated_filtered[:,0:,:])
    nmse_danse_std = nmse_loss_std(X[:,1:,:], X_estimated_filtered[:,0:,:])
    nmse_danse_pred = nmse_loss(X[:,1:,:], X_estimated_pred[:,0:,:])
    nmse_danse_pred_std = nmse_loss_std(X[:,1:,:], X_estimated_pred[:,0:,:])

    mse_dB_ls = mse_loss_dB(X[:,1:,:], X_LS[:,0:,:])
    mse_dB_ls_std = mse_loss_dB_std(X[:,1:,:], X_LS[:,0:,:])
    mse_dB_ekf = mse_loss_dB(X[:,1:,:], X_estimated_ekf[:,1:,:])
    mse_dB_ekf_std = mse_loss_dB_std(X[:,1:,:], X_estimated_ekf[:,1:,:])
    mse_dB_ukf = mse_loss_dB(X[:,1:,:], X_estimated_ukf[:,1:,:])
    mse_dB_ukf_std = mse_loss_dB_std(X[:,1:,:], X_estimated_ukf[:,1:,:])
    mse_dB_danse = mse_loss_dB(X[:,1:,:], X_estimated_filtered[:,0:,:])
    mse_dB_danse_std = mse_loss_dB_std(X[:,1:,:], X_estimated_filtered[:,0:,:])
    mse_dB_danse_pred = mse_loss_dB(X[:,1:,:], X_estimated_pred[:,0:,:])
    mse_dB_danse_pred_std = mse_loss_dB_std(X[:,1:,:], X_estimated_pred[:,0:,:])
    
    print("DANSE - MSE LOSS:",mse_dB_danse, "[dB]")
    print("DANSE - MSE STD:", mse_dB_danse_std, "[dB]")

    snr = mse_loss(X[:,1:,:], torch.zeros_like(X[:,1:,:])) * dB_to_lin(inverse_r2_dB_test)

    print("LS, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB]".format(N_test, nmse_ls, nmse_ls_std, mse_dB_ls, mse_dB_ls_std))
    print("ekf, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_ekf, nmse_ekf_std, mse_dB_ekf, mse_dB_ekf_std, time_elapsed_ekf))
    print("ukf, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_ukf, nmse_ukf_std, mse_dB_ukf, mse_dB_ukf_std, time_elapsed_ukf))
    print("danse (pred.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_danse_pred, nmse_danse_pred_std, mse_dB_danse_pred, mse_dB_danse_pred_std, time_elapsed_danse))
    print("danse (fil.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_danse, nmse_danse_std, mse_dB_danse, mse_dB_danse_std, time_elapsed_danse))

    # System console print
    print("LS, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB]".format(N_test, nmse_ls, nmse_ls_std, mse_dB_ls, mse_dB_ls_std), file=orig_stdout)
    print("ekf, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {} secs".format(N_test, nmse_ekf, nmse_ekf_std, mse_dB_ekf, mse_dB_ekf_std, time_elapsed_ekf), file=orig_stdout)
    print("ukf, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {} secs".format(N_test, nmse_ukf, nmse_ukf_std, mse_dB_ukf, mse_dB_ukf_std, time_elapsed_ukf), file=orig_stdout)
    print("danse (pred.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {} secs".format(N_test, nmse_danse_pred, nmse_danse_pred_std, mse_dB_danse_pred, mse_dB_danse_pred_std, time_elapsed_danse), file=orig_stdout)
    print("danse (fil.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {} secs".format(N_test, nmse_danse, nmse_danse_std, mse_dB_danse, mse_dB_danse_std, time_elapsed_danse), file=orig_stdout)
    
    # Plot the result
    plot_state_trajectory_axes(X=torch.squeeze(X[0,1:,:],0), 
                                X_est_EKF=torch.squeeze(X_estimated_ekf[0,1:,:],0), 
                                X_est_UKF=torch.squeeze(X_estimated_ukf[0,1:,:],0), 
                                X_est_DANSE=torch.squeeze(X_estimated_filtered[0],0), 
                                savefig=True,
                                savefig_name="./figs/LorenzModel/Partial/AxesWisePlot_r2_{}dB_nu_{}dB.pdf".format(inverse_r2_dB_test, nu_dB_test))
    plot_state_trajectory(X=torch.squeeze(X[0,1:,:],0), 
                        X_est_EKF=torch.squeeze(X_estimated_ekf[0,1:,:],0), 
                        X_est_UKF=torch.squeeze(X_estimated_ukf[0,1:,:],0), 
                        X_est_DANSE=torch.squeeze(X_estimated_filtered[0],0),
                        savefig=True,
                        savefig_name="./figs/LorenzModel/Partial/3dPlot_r2_{}dB_nu_{}dB.pdf".format(inverse_r2_dB_test, nu_dB_test))
    #plot_state_trajectory_axes(X=torch.squeeze(X,0), X_est_EKF=torch.squeeze(X_estimated_ekf,0), X_est_DANSE=torch.squeeze(X_estimated_filtered,0))
    #plot_state_trajectory(X=torch.squeeze(X,0), X_est_EKF=torch.squeeze(X_estimated_ekf,0), X_est_DANSE=torch.squeeze(X_estimated_filtered,0))
    
    #plt.show()
    sys.stdout = orig_stdout
    return nmse_ekf, nmse_ekf_std, nmse_danse, nmse_danse_std, nmse_ukf, nmse_ukf_std, nmse_ls, nmse_ls_std, time_elapsed_ekf, time_elapsed_danse, time_elapsed_ukf, snr

if __name__ == "__main__":

    device = 'cpu'
    inverse_r2_dB_arr = np.array([-10.0, 0.0, 10.0, 20.0, 30.0])
    #inverse_r2_dB_arr = np.array([20.0])
    nmse_ls_arr = np.zeros((len(inverse_r2_dB_arr,)))
    nmse_ekf_arr = np.zeros((len(inverse_r2_dB_arr,)))
    nmse_ukf_arr = np.zeros((len(inverse_r2_dB_arr,)))
    nmse_danse_arr = np.zeros((len(inverse_r2_dB_arr,)))
    nmse_ls_std_arr = np.zeros((len(inverse_r2_dB_arr,)))
    nmse_ekf_std_arr = np.zeros((len(inverse_r2_dB_arr,)))
    nmse_ukf_std_arr = np.zeros((len(inverse_r2_dB_arr,)))
    nmse_danse_std_arr = np.zeros((len(inverse_r2_dB_arr,)))
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

    test_data_file_dict = {
        "-10.0dB":"./data/synthetic_data/test_trajectories_m_3_n_3_LorenzSSM_data_T_2000_N_100_r2_-10.0dB_nu_-20.0dB.pkl",
        "0.0dB":"./data/synthetic_data/test_trajectories_m_3_n_3_LorenzSSM_data_T_2000_N_100_r2_0.0dB_nu_-20.0dB.pkl",
        "10.0dB":"./data/synthetic_data/test_trajectories_m_3_n_3_LorenzSSM_data_T_2000_N_100_r2_10.0dB_nu_-20.0dB.pkl",
        "20.0dB":"./data/synthetic_data/test_trajectories_m_3_n_3_LorenzSSM_data_T_2000_N_100_r2_20.0dB_nu_-20.0dB.pkl",
        "30.0dB":"./data/synthetic_data/test_trajectories_m_3_n_3_LorenzSSM_data_T_2000_N_100_r2_30.0dB_nu_-20.0dB.pkl"
    }

    test_logfile = "./log/Lorenz_test_Partial.log"
    test_jsonfile = "./log/Lorenz_test_Partial.json"

    for i, inverse_r2_dB in enumerate(inverse_r2_dB_arr):
        
        model_file_saved_i = model_file_saved_dict['{}dB'.format(inverse_r2_dB)]
        test_data_file_i = test_data_file_dict['{}dB'.format(inverse_r2_dB)]
        
        nmse_ekf_i, nmse_ekf_i_std, nmse_danse_i, nmse_danse_i_std, nmse_ukf_i, nmse_ukf_i_std, nmse_ls_i, nmse_ls_i_std, time_elapsed_ekf_i, time_elapsed_danse_i, time_elapsed_ukf_i, snr_i = test_lorenz(device=device, 
            model_file_saved=model_file_saved_i, test_data_file=test_data_file_i, test_logfile=test_logfile)
        nmse_ls_arr[i] = nmse_ls_i.numpy().item()
        nmse_ekf_arr[i] = nmse_ekf_i.numpy().item()
        nmse_ukf_arr[i] = nmse_ukf_i.numpy().item()
        nmse_danse_arr[i] = nmse_danse_i.numpy().item()
        nmse_ls_std_arr[i] = nmse_ls_i_std.numpy().item()
        nmse_ekf_std_arr[i] = nmse_ekf_i_std.numpy().item()
        nmse_ukf_std_arr[i] = nmse_ukf_i_std.numpy().item()
        nmse_danse_std_arr[i] = nmse_danse_i_std.numpy().item()
        t_ekf_arr[i] = time_elapsed_ekf_i
        t_ukf_arr[i] = time_elapsed_ukf_i
        t_danse_arr[i] = time_elapsed_danse_i
        snr_arr[i] = 10*np.log10(snr_i.numpy().item())
    
    test_stats = {}
    test_stats['UKF_mean_nmse'] = nmse_ukf_arr
    test_stats['EKF_mean_nmse'] = nmse_ekf_arr
    test_stats['DANSE_mean_nmse'] = nmse_danse_arr
    test_stats['UKF_std_nmse'] = nmse_ukf_std_arr
    test_stats['EKF_std_nmse'] = nmse_ekf_std_arr
    test_stats['DANSE_std_nmse'] = nmse_danse_std_arr
    test_stats['UKF_time'] = t_ukf_arr
    test_stats['EKF_time'] = t_ekf_arr
    test_stats['DANSE_time'] = t_danse_arr
    test_stats['LS_mean_nmse'] = nmse_ls_arr
    test_stats['LS_std_nmse'] = nmse_ls_std_arr
    test_stats['SNR'] = snr_arr

    with open(test_jsonfile, 'w') as f:
        f.write(json.dumps(test_stats, cls=NDArrayEncoder, indent=2))

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
    plt.savefig('./figs/LorenzModel/Partial/NMSE_vs_inverse_r2dB_Lorenz.pdf')

    plt.figure()
    plt.plot(snr_arr, nmse_ls_arr, 'gp--', linewidth=1.5, label="NMSE-LS")
    plt.plot(snr_arr, nmse_ekf_arr, 'rd--', linewidth=1.5, label="NMSE-EKF")
    plt.plot(snr_arr, nmse_danse_arr, 'bo-', linewidth=2.0, label="NMSE-DANSE")
    plt.plot(snr_arr, nmse_ukf_arr, 'ks-', linewidth=2.0, label="NMSE-UKF")
    plt.xlabel('SNR (in dB)')
    plt.ylabel('NMSE (in dB)')
    plt.grid(True)
    plt.legend()
    plt.savefig('./figs/LorenzModel/Partial/NMSE_vs_SNR_Lorenz.pdf')

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
    plt.savefig('./figs/LorenzModel/Partial/InferTime_vs_inverse_r2dB_Lorenz.pdf')

    #plt.subplot(212)
    plt.figure()
    plt.plot(snr_arr, t_ekf_arr, 'rd--', linewidth=1.5, label="Inference time-EKF")
    plt.plot(snr_arr, t_ukf_arr, 'ks-', linewidth=2.0, label="Inference time-UKF")
    plt.plot(snr_arr, t_danse_arr, 'bo-', linewidth=2.0, label="Inference time-DANSE")
    plt.xlabel('SNR (in dB)')
    plt.ylabel('Time (in s)')
    plt.grid(True)
    plt.legend()
    plt.savefig('./figs/LorenzModel/Partial/InferTime_vs_SNR_Lorenz.pdf')

    plt.show()
