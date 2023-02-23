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
from utils.utils import generate_normal, dB_to_lin, lin_to_dB, \
    mse_loss, nmse_loss, nmse_loss_std, mse_loss_dB, mse_loss_dB_std, \
        load_saved_dataset, save_dataset, NDArrayEncoder
from parameters import get_parameters
from generate_data import LinearSSM, generate_SSM_data
from src.kf import KF
from src.danse import DANSE, push_model
from parse import parse
from timeit import default_timer as timer
import json

def test_kf_linear(X, Y, kf_model):

    X_estimated_kf, Pk_estimated_kf, mse_arr_kf = kf_model.run_mb_filter(X, Y)
    return X_estimated_kf, Pk_estimated_kf, mse_arr_kf

'''
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
        #print("kf, sample: {}, mse_loss: {}".format(i+1, mse_arr_KF[i]))

    mse_kf_lin_avg = torch.mean(mse_arr_KF, dim=0) # Calculate the MSE by averaging over all examples in a batch
    mse_kf_dB_avg = 10*torch.log10(mse_kf_lin_avg)
    print("KF - MSE LOSS:", mse_kf_dB_avg, "[dB]")
    print("KF - MSE STD:", 10*torch.log10(torch.std(mse_arr_KF, dim=0).abs()), "[dB]")
    return X_estimated_KF, Pk_estimated_KF, mse_arr_KF
'''

def test_danse_linear(danse_model, saved_model_file, Y, device='cpu'):

    danse_model.load_state_dict(torch.load(saved_model_file, map_location=device))
    danse_model = push_model(nets=danse_model, device=device)
    danse_model.eval()

    with torch.no_grad():

        Y_test_batch = Variable(Y, requires_grad=False).type(torch.FloatTensor).to(device)
        X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered = danse_model.compute_predictions(Y_test_batch)
    
    return X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered

def test_linear(device='cpu', model_file_saved=None, test_data_file=None, test_logfile=None, evaluation_mode=None):

    _, rnn_type, m, n, T, _, inverse_r2_dB, nu_dB = parse("{}_danse_{}_m_{:d}_n_{:d}_T_{:d}_N_{:d}_{:f}dB_{:f}dB", model_file_saved.split('/')[-2])
    
    #m = 10
    #n = 10
    q2 = 1.0
    r2 = 1.0
    T_test = 500
    #T = 6_000
    #nu_dB = 0
    #inverse_r2_dB = 40
    orig_stdout = sys.stdout
    f_tmp = open(test_logfile, 'a')
    sys.stdout = f_tmp

    if not os.path.isfile(test_data_file):
        
        print('Dataset is not present, creating at {}'.format(test_data_file))
        # My own data generation scheme
        m, n, T_test, N_test, inverse_r2_dB_test, nu_dB_test = parse("test_trajectories_m_{:d}_n_{:d}_LinearSSM_data_T_{:d}_N_{:d}_r2_{:f}dB_nu_{:f}dB.pkl", test_data_file.split('/')[-1])
        #N_test = 100 # No. of trajectories at test time / evaluation
        X = torch.zeros((N_test, T_test+1, m))
        Y = torch.zeros((N_test, T_test, n))

        # Initialize a Linear SSM with the extracted parameters
        linear_ssm = LinearSSM(n_states=m, n_obs=n, F=None, G=np.zeros((m,1)), H=None, 
                            mu_e=np.zeros((m,)), mu_w=np.zeros((n,)), q2=q2, r2=r2, 
                            Q=None, R=None)

        print("Test data generated using r2: {} dB, nu: {} dB".format(inverse_r2_dB_test, nu_dB_test))
        for i in range(N_test):
            x_lin_i, y_lin_i = linear_ssm.generate_single_sequence(T=T_test, inverse_r2_dB=inverse_r2_dB, nu_dB=nu_dB)
            X[i, :, :] = torch.from_numpy(x_lin_i).type(torch.FloatTensor)
            Y[i, :, :] = torch.from_numpy(y_lin_i).type(torch.FloatTensor)

        test_data_dict = {}
        test_data_dict["X"] = X
        test_data_dict["Y"] = Y
        test_data_dict["model"] = linear_ssm
        save_dataset(Z_XY=test_data_dict, filename=test_data_file)

    else:

        print("Dataset at {} already present!".format(test_data_file))
        m, n, T_test, N_test, inverse_r2_dB_test, nu_dB_test = parse("test_trajectories_m_{:d}_n_{:d}_LinearSSM_data_T_{:d}_N_{:d}_r2_{:f}dB_nu_{:f}dB.pkl", test_data_file.split('/')[-1])
        test_data_dict = load_saved_dataset(filename=test_data_file)
        X = test_data_dict["X"]
        Y = test_data_dict["Y"]
        linear_ssm = test_data_dict["model"]

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
    H_tensor = torch.from_numpy(linear_ssm.construct_H()).type(torch.FloatTensor)
    H_tensor = torch.repeat_interleave(H_tensor.unsqueeze(0),N_test,dim=0)
    X_LS = torch.einsum('ijj,ikj->ikj',torch.pinverse(H_tensor),Y)#torch.einsum('ijj,ikj->ikj',H_tensor,Y)

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
    start_time_kf = timer()
    X_estimated_kf = None
    Pk_estimated_kf = None
    
    X_estimated_kf, Pk_estimated_kf, mse_arr_kf = test_kf_linear(X=X, Y=Y, kf_model=kf_model)
    time_elapsed_kf = timer() - start_time_kf

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
    Pk_estimated_filtered = None

    start_time_danse = timer()
    X_estimated_pred, Pk_estimated_pred, X_estimated_filtered, Pk_estimated_filtered = test_danse_linear(danse_model=danse_model, 
                                                                                                        saved_model_file=model_file_saved,
                                                                                                        Y=Y,
                                                                                                        device=device)
    
    time_elapsed_danse = timer() - start_time_danse

    nmse_ls = nmse_loss(X[:,1:,:], X_LS[:,0:,:])
    nmse_ls_std = nmse_loss_std(X[:,1:,:], X_LS[:,0:,:])
    nmse_kf = nmse_loss(X[:,1:,:], X_estimated_kf[:,1:,:])
    nmse_kf_std = nmse_loss_std(X[:,1:,:], X_estimated_kf[:,1:,:])
    nmse_danse = nmse_loss(X[:,1:,:], X_estimated_filtered[:,0:,:])
    nmse_danse_std = nmse_loss_std(X[:,1:,:], X_estimated_filtered[:,0:,:])
    nmse_danse_pred = nmse_loss(X[:,1:,:], X_estimated_pred[:,0:,:])
    nmse_danse_pred_std = nmse_loss_std(X[:,1:,:], X_estimated_pred[:,0:,:])

    mse_dB_ls = mse_loss_dB(X[:,1:,:], X_LS[:,0:,:])
    mse_dB_ls_std = mse_loss_dB_std(X[:,1:,:], X_LS[:,0:,:])
    mse_dB_kf = mse_loss_dB(X[:,1:,:], X_estimated_kf[:,1:,:])
    mse_dB_kf_std = mse_loss_dB_std(X[:,1:,:], X_estimated_kf[:,1:,:])
    mse_dB_danse = mse_loss_dB(X[:,1:,:], X_estimated_filtered[:,0:,:])
    mse_dB_danse_std = mse_loss_dB_std(X[:,1:,:], X_estimated_filtered[:,0:,:])
    mse_dB_danse_pred = mse_loss_dB(X[:,1:,:], X_estimated_pred[:,0:,:])
    mse_dB_danse_pred_std = mse_loss_dB_std(X[:,1:,:], X_estimated_pred[:,0:,:])
    
    print("DANSE - MSE LOSS:",mse_dB_danse, "[dB]")
    print("DANSE - MSE STD:", mse_dB_danse_std, "[dB]")

    snr = mse_loss(X[:,1:,:], torch.zeros_like(X[:,1:,:])).mean() * dB_to_lin(inverse_r2_dB_test)

    print("LS, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB]".format(N_test, nmse_ls, nmse_ls_std, mse_dB_ls, mse_dB_ls_std))
    print("kf, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_kf, nmse_kf_std, mse_dB_kf, mse_dB_kf_std, time_elapsed_kf))
    #print("danse, batch size: {}, nmse: {} ± {}[dB], mse: {} ± {}[dB], time: {} secs".format(N_test, nmse_danse, nmse_danse_std, mse_dB_danse, mse_dB_danse_std, time_elapsed_danse))
    print("danse (pred.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_danse_pred, nmse_danse_pred_std, mse_dB_danse_pred, mse_dB_danse_pred_std, time_elapsed_danse))
    print("danse (fil.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_danse, nmse_danse_std, mse_dB_danse, mse_dB_danse_std, time_elapsed_danse))

    print("LS, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB]".format(N_test, nmse_ls, nmse_ls_std, mse_dB_ls, mse_dB_ls_std), file=orig_stdout)
    print("kf, batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_kf, nmse_kf_std, mse_dB_kf, mse_dB_kf_std, time_elapsed_kf), file=orig_stdout)
    #print("danse, batch size: {}, nmse: {} ± {}[dB], mse: {} ± {}[dB], time: {} secs".format(N_test, nmse_danse, nmse_danse_std, mse_dB_danse, mse_dB_danse_std, time_elapsed_danse))
    print("danse (pred.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_danse_pred, nmse_danse_pred_std, mse_dB_danse_pred, mse_dB_danse_pred_std, time_elapsed_danse), file=orig_stdout)
    print("danse (fil.), batch size: {}, nmse: {:.4f} ± {:.4f}[dB], mse: {:.4f} ± {:.4f}[dB], time: {:.4f} secs".format(N_test, nmse_danse, nmse_danse_std, mse_dB_danse, mse_dB_danse_std, time_elapsed_danse), file=orig_stdout)

    # Plot the result
    plot_state_trajectory_axes(X=torch.squeeze(X[0,1:,:],0), 
                        X_est_KF=torch.squeeze(X_estimated_kf[0,1:,:], 0), 
                        X_est_DANSE=torch.squeeze(X_estimated_filtered[0], 0),
                        savefig=True,
                        savefig_name="./figs/LinearModel/{}/AxesWisePlot_r2_{}dB_nu_{}dB.pdf".format(evaluation_mode, inverse_r2_dB_test, nu_dB_test))
    
    plot_state_trajectory(X=torch.squeeze(X[0,1:,:],0), 
                        X_est_KF=torch.squeeze(X_estimated_kf[0,1:,:], 0), 
                        X_est_DANSE=torch.squeeze(X_estimated_filtered[0], 0),
                        savefig=True,
                        savefig_name="./figs/LinearModel/{}/3dPlot_r2_{}dB_nu_{}dB.pdf".format(evaluation_mode, inverse_r2_dB_test, nu_dB_test))
    
    sys.stdout = orig_stdout
    #plt.show()
    return nmse_kf, nmse_kf_std, nmse_danse, nmse_danse_std, nmse_ls, nmse_ls_std, \
        mse_dB_kf, mse_dB_kf_std, mse_dB_danse, mse_dB_danse_std, mse_dB_ls, mse_dB_ls_std, \
        time_elapsed_kf, time_elapsed_danse, snr

if __name__ == "__main__":
    device = 'cpu'
    evaluation_mode = 'Full'
    inverse_r2_dB_arr = np.array([-10.0, 0.0, 10.0, 20.0, 30.0])
    #inverse_r2_dB_arr = np.array([20.0])
    nmse_ls_arr = np.zeros((len(inverse_r2_dB_arr,)))
    nmse_kf_arr = np.zeros((len(inverse_r2_dB_arr,)))
    nmse_danse_arr = np.zeros((len(inverse_r2_dB_arr,)))
    nmse_ls_std_arr = np.zeros((len(inverse_r2_dB_arr,)))
    nmse_kf_std_arr = np.zeros((len(inverse_r2_dB_arr,)))
    nmse_danse_std_arr = np.zeros((len(inverse_r2_dB_arr,)))
    mse_ls_dB_arr = np.zeros((len(inverse_r2_dB_arr,)))
    mse_kf_dB_arr = np.zeros((len(inverse_r2_dB_arr,)))
    mse_danse_dB_arr = np.zeros((len(inverse_r2_dB_arr,)))
    mse_ls_dB_std_arr = np.zeros((len(inverse_r2_dB_arr,)))
    mse_kf_dB_std_arr = np.zeros((len(inverse_r2_dB_arr,)))
    mse_danse_dB_std_arr = np.zeros((len(inverse_r2_dB_arr,)))
    t_kf_arr = np.zeros((len(inverse_r2_dB_arr,)))
    t_danse_arr = np.zeros((len(inverse_r2_dB_arr,)))
    snr_arr = np.zeros((len(inverse_r2_dB_arr,)))

    model_file_saved_dict = {
        "-10.0dB":'./models/LinearSSM_danse_gru_m_5_n_5_T_500_N_500_-10.0dB_0.0dB/danse_gru_ckpt_epoch_693_best.pt',
        "0.0dB":'./models/LinearSSM_danse_gru_m_5_n_5_T_500_N_500_0.0dB_0.0dB/danse_gru_ckpt_epoch_720_best.pt',
        "10.0dB":'./models/LinearSSM_danse_gru_m_5_n_5_T_500_N_500_10.0dB_0.0dB/danse_gru_ckpt_epoch_1438_best.pt',
        "20.0dB":'./models/LinearSSM_danse_gru_m_5_n_5_T_500_N_500_20.0dB_0.0dB/danse_gru_ckpt_epoch_1008_best.pt',
        "30.0dB":'./models/LinearSSM_danse_gru_m_5_n_5_T_500_N_500_30.0dB_0.0dB/danse_gru_ckpt_epoch_371_best.pt'
    }
    
    T_test = 1000
    N_test = 100
    test_data_file_dict = {
        "-10.0dB":"./data/synthetic_data/test_trajectories_m_5_n_5_LinearSSM_data_T_{}_N_{}_r2_-10.0dB_nu_0.0dB.pkl".format(T_test, N_test),
        "0.0dB":"./data/synthetic_data/test_trajectories_m_5_n_5_LinearSSM_data_T_{}_N_{}_r2_0.0dB_nu_0.0dB.pkl".format(T_test, N_test),
        "10.0dB":"./data/synthetic_data/test_trajectories_m_5_n_5_LinearSSM_data_T_{}_N_{}_r2_10.0dB_nu_0.0dB.pkl".format(T_test, N_test),
        "20.0dB":"./data/synthetic_data/test_trajectories_m_5_n_5_LinearSSM_data_T_{}_N_{}_r2_20.0dB_nu_0.0dB.pkl".format(T_test, N_test),
        "30.0dB":"./data/synthetic_data/test_trajectories_m_5_n_5_LinearSSM_data_T_{}_N_{}_r2_30.0dB_nu_0.0dB.pkl".format(T_test, N_test)
    }

    test_logfile = "./log/Linear_test_{}_T_{}_N_{}.log".format(evaluation_mode, T_test, N_test)
    test_jsonfile = "./log/Linear_test_{}_T_{}_N_{}.json".format(evaluation_mode, T_test, N_test)

    for i, inverse_r2_dB in enumerate(inverse_r2_dB_arr):
        
        model_file_saved_i = model_file_saved_dict['{}dB'.format(inverse_r2_dB)]
        test_data_file_i = test_data_file_dict['{}dB'.format(inverse_r2_dB)]
        
        nmse_kf_i, nmse_kf_std_i, nmse_danse_i, nmse_danse_std_i, nmse_ls_i, nmse_ls_std_i, \
            mse_dB_kf_i, mse_dB_kf_std_i, mse_dB_danse_i, mse_dB_danse_std_i, mse_dB_ls_i, mse_dB_ls_std_i, \
            time_elapsed_kf_i, time_elapsed_danse_i, snr_i = test_linear(device=device, 
            model_file_saved=model_file_saved_i, test_data_file=test_data_file_i, test_logfile=test_logfile, evaluation_mode=evaluation_mode)
        
        # Store the NMSE values and std devs of the NMSE values
        nmse_ls_arr[i] = nmse_ls_i.numpy().item()
        nmse_kf_arr[i] = nmse_kf_i.numpy().item()
        nmse_danse_arr[i] = nmse_danse_i.numpy().item()
        nmse_ls_std_arr[i] = nmse_ls_std_i.numpy().item()
        nmse_kf_std_arr[i] = nmse_kf_std_i.numpy().item()
        nmse_danse_std_arr[i] = nmse_danse_std_i.numpy().item()

        # Store the MSE values and std devs of the MSE values (in dB)
        mse_ls_dB_arr[i] = mse_dB_ls_i.numpy().item()
        mse_kf_dB_arr[i] = mse_dB_kf_i.numpy().item()
        mse_danse_dB_arr[i] = mse_dB_danse_i.numpy().item()
        mse_ls_dB_std_arr[i] = mse_dB_ls_std_i.numpy().item()
        mse_kf_dB_std_arr[i] = mse_dB_kf_std_i.numpy().item()
        mse_danse_dB_std_arr[i] = mse_dB_danse_std_i.numpy().item()

        # Store the inference times 
        t_kf_arr[i] = time_elapsed_kf_i
        t_danse_arr[i] = time_elapsed_danse_i
        snr_arr[i] = 10*np.log10(snr_i.numpy().item())

    test_stats = {}
    test_stats['KF_mean_nmse'] = nmse_kf_arr
    test_stats['DANSE_mean_nmse'] = nmse_danse_arr
    test_stats['KF_std_nmse'] = nmse_kf_std_arr
    test_stats['DANSE_std_nmse'] = nmse_danse_std_arr
    test_stats['KF_mean_mse'] = mse_kf_dB_arr
    test_stats['DANSE_mean_mse'] = mse_danse_dB_arr
    test_stats['KF_std_mse'] = mse_kf_dB_std_arr
    test_stats['DANSE_std_mse'] = mse_danse_dB_std_arr
    test_stats['LS_mean_mse'] = mse_ls_dB_arr
    test_stats['LS_std_mse'] = mse_ls_dB_std_arr
    test_stats['LS_mean_nmse'] = nmse_ls_arr
    test_stats['LS_std_nmse'] = nmse_ls_std_arr
    test_stats['KF_time'] = t_kf_arr
    test_stats['DANSE_time'] = t_danse_arr
    test_stats['SNR'] = snr_arr

    with open(test_jsonfile, 'w') as f:
        f.write(json.dumps(test_stats, cls=NDArrayEncoder, indent=2))

    # Plotting the NMSE Curve
    plt.rcParams['font.family'] = 'serif'
    plt.figure()
    plt.plot(inverse_r2_dB_arr, nmse_ls_arr, 'gp--', linewidth=1.5, label="NMSE-LS")
    plt.plot(inverse_r2_dB_arr, nmse_kf_arr, 'rd--', linewidth=1.5, label="NMSE-KF")
    plt.plot(inverse_r2_dB_arr, nmse_danse_arr, 'bo-', linewidth=2.0, label="NMSE-DANSE")
    plt.xlabel('$\\frac{1}{r^2}$ (in dB)')
    plt.ylabel('NMSE (in dB)')
    plt.grid(True)
    plt.legend()
    #plt.subplot(212)
    plt.savefig('./figs/LinearModel/{}/NMSE_vs_inverse_r2dB_Linear.pdf'.format(evaluation_mode))

    plt.figure()
    plt.plot(snr_arr, nmse_ls_arr, 'gp--', linewidth=1.5, label="NMSE-LS")
    plt.plot(snr_arr, nmse_kf_arr, 'rd--', linewidth=1.5, label="NMSE-KF")
    plt.plot(snr_arr, nmse_danse_arr, 'bo-', linewidth=2.0, label="NMSE-DANSE")
    plt.xlabel('SNR (in dB)')
    plt.ylabel('NMSE (in dB)')
    plt.grid(True)
    plt.legend()
    plt.savefig('./figs/LinearModel/{}/NMSE_vs_SNR_Linear.pdf'.format(evaluation_mode))
    
    # Plotting the MSE Curve
    plt.rcParams['font.family'] = 'serif'
    plt.figure()
    plt.plot(inverse_r2_dB_arr, mse_ls_dB_arr, 'gp--', linewidth=1.5, label="MSE-LS")
    plt.plot(inverse_r2_dB_arr, mse_kf_dB_arr, 'rd--', linewidth=1.5, label="MSE-KF")
    plt.plot(inverse_r2_dB_arr, mse_danse_dB_arr, 'bo-', linewidth=2.0, label="MSE-DANSE")
    plt.xlabel('$\\frac{1}{r^2}$ (in dB)')
    plt.ylabel('MSE (in dB)')
    plt.grid(True)
    plt.legend()
    #plt.subplot(212)
    plt.savefig('./figs/LinearModel/{}/MSE_vs_inverse_r2dB_Linear.pdf'.format(evaluation_mode))

    plt.figure()
    plt.plot(snr_arr, mse_ls_dB_arr, 'gp--', linewidth=1.5, label="MSE-LS")
    plt.plot(snr_arr, mse_kf_dB_arr, 'rd--', linewidth=1.5, label="MSE-KF")
    plt.plot(snr_arr, mse_danse_dB_arr, 'bo-', linewidth=2.0, label="MSE-DANSE")
    plt.xlabel('SNR (in dB)')
    plt.ylabel('MSE (in dB)')
    plt.grid(True)
    plt.legend()
    plt.savefig('./figs/LinearModel/{}/MSE_vs_SNR_Linear.pdf'.format(evaluation_mode))

    # Plotting the Time-elapsed Curve
    plt.figure()
    #plt.subplot(211)
    plt.plot(inverse_r2_dB_arr, t_kf_arr, 'rd--', linewidth=1.5, label="Inference time-KF")
    plt.plot(inverse_r2_dB_arr, t_danse_arr, 'bo-', linewidth=2.0, label="Inference time-DANSE")
    plt.xlabel('$\\frac{1}{r^2}$ (in dB)')
    plt.ylabel('Time (in s)')
    plt.grid(True)
    plt.legend()
    plt.savefig('./figs/LinearModel/{}/InferTime_vs_inverse_r2dB_Linear.pdf'.format(evaluation_mode))

    #plt.subplot(212)
    plt.figure()
    plt.plot(snr_arr, t_kf_arr, 'rd--', linewidth=1.5, label="Inference time-KF")
    plt.plot(snr_arr, t_danse_arr, 'bo-', linewidth=2.0, label="Inference time-DANSE")
    plt.xlabel('SNR (in dB)')
    plt.ylabel('Time (in s)')
    plt.grid(True)
    plt.legend()
    plt.savefig('./figs/LinearModel/{}/InferTime_vs_SNR_Linear.pdf'.format(evaluation_mode))

    plt.show()
