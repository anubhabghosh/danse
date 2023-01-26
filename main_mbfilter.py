# Import necessary libraries
import os
import json
import argparse
from parse import parse
import numpy as np
import scipy
import matplotlib.pyplot as plt
import torch
import pickle as pkl
from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils.utils import load_saved_dataset, Series_Dataset, obtain_tr_val_test_idx, create_splits_file_name, \
    create_file_paths, check_if_dir_or_file_exists, load_splits_file, get_dataloaders, NDArrayEncoder
# Import the parameters
from parameters import get_parameters
from utils.plot_functions import plot_measurement_data, plot_measurement_data_axes, plot_state_trajectory, plot_state_trajectory_axes

# Import estimator model and functions
from src.kf import KF
from src.ekf import EKF
from src.ukf import UKF

def main():

    usage = "Train DANSE using trajectories of SSMs \n"\
        "python3.8 main_mbfilter.py --filter_type [kf/ekf/ukf] --dataset_mode [LinearSSM/LorenzSSM] \n"\
        "--datafile [fullpath to datafile] --splits [fullpath to splits file]"
    
    parser = argparse.ArgumentParser(description="Input a string indicating the mode of the script \n"\
        "train - training and testing is done, test-only evlaution is carried out")
    parser.add_argument("--filter_type", help="Enter the filter type (kf / ekf / ukf)", type=str)
    parser.add_argument("--dataset_type", help="Enter the type of dataset (LinearSSM / LorenzSSM)", type=str)
    parser.add_argument("--model_file_saved", help="In case of testing mode, Enter the desired model checkpoint with full path (gru/lstm/rnn)", type=str, default=None)
    parser.add_argument("--datafile", help="Enter the full path to the dataset", type=str)
    parser.add_argument("--splits", help="Enter full path to splits file", type=str)
    
    args = parser.parse_args() 
    filter_type = args.filter_type
    datafile = args.datafile
    dataset_type = args.dataset_type
    datafolder = "".join(datafile.split("/")[i]+"/" for i in range(len(datafile.split("/")) - 1))
    model_file_saved = args.model_file_saved
    splits_file = args.splits

    print(datafile.split('/')[-1])
    # Dataset parameters obtained from the 'datafile' variable
    _, n_states, n_obs, _, T, N_samples, inverse_r2_dB = parse("{}_m_{:d}_n_{:d}_{}_data_T_{:d}_N_{:d}_r2_{:f}dB.pkl", datafile.split('/')[-1])
    
    ssm_parameters_dict, est_parameters_dict = get_parameters(
                                            N=N_samples,
                                            T=T,
                                            n_states=n_states,
                                            n_obs=n_obs,
                                            inverse_r2_dB=inverse_r2_dB
                                        )

    batch_size = ssm_parameters_dict[dataset_type]["batch_size"] # Set the batch size
    estimator_options = est_parameters_dict["danse"] # Get the options for the estimator

    if not os.path.isfile(datafile):
        
        print("Dataset is not present, run 'generate_data.py / run_generate_data.sh' to create the dataset")
        #plot_trajectories(Z_pM, ncols=1, nrows=10)
    else:

        print("Dataset already present!")
        Z_XY = load_saved_dataset(filename=datafile)
    
    Z_XY_dataset = Series_Dataset(Z_XY_dict=Z_XY)

    if not os.path.isfile(splits_file):
        tr_indices, val_indices, test_indices = obtain_tr_val_test_idx(dataset=Z_XY_dataset,
                                                                    tr_to_test_split=0.9,
                                                                    tr_to_val_split=0.833)
        print(len(tr_indices), len(val_indices), len(test_indices))
        splits = {}
        splits["train"] = tr_indices
        splits["val"] = val_indices
        splits["test"] = test_indices
        splits_file_name = create_splits_file_name(dataset_filename=datafile,
                                                splits_filename=splits_file
                                                )
        
        print("Creating split file at:{}".format(splits_file_name))
        with open(splits_file_name, 'wb') as handle:
            pkl.dump(splits, handle, protocol=pkl.HIGHEST_PROTOCOL)
    else:
        print("Loading the splits file from {}".format(splits_file))
        splits = load_splits_file(splits_filename=splits_file)
        tr_indices, val_indices, test_indices = splits["train"], splits["val"], splits["test"]

    train_loader, val_loader, test_loader = get_dataloaders(Z_XY_dataset, batch_size, tr_indices, val_indices, test_indices)

    print("No. of training, validation and testing batches: {}, {}, {}".format(len(train_loader), 
                                                                                len(val_loader), 
                                                                                len(test_loader)))

    ngpu = 1 # Comment this out if you want to run on cpu and the next line just set device to "cpu"
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu>0) else "cpu")
    print("Device Used:{}".format(device))
    
    logfile_path = "./log/".format(dataset_type)
    modelfile_path = "./models/".format(dataset_type)

    #NOTE: Currently this is hardcoded into the system
    main_exp_name = "{}_{}".format(dataset_type, filter_type)

    #print(params)
    tr_log_file_name = "training_{}_m_{}_n_{}_T_{}_N_{}_{}dB.log".format(
                                                            filter_type,
                                                            n_states,
                                                            n_obs,
                                                            T,
                                                            N_samples,
                                                            inverse_r2_dB
                                                            )
    
    te_log_file_name = "testing_{}_m_{}_n_{}_T_{}_N_{}_{}dB.log".format(
                                                            filter_type,
                                                            n_states,
                                                            n_obs,
                                                            T,
                                                            N_samples,
                                                            inverse_r2_dB
                                                            )
    
    flag_log_dir, flag_log_file = check_if_dir_or_file_exists(os.path.join(logfile_path, main_exp_name),
                                                            file_name=tr_log_file_name)
    
    print("Is log-directory present:? - {}".format(flag_log_dir))
    print("Is log-file present:? - {}".format(flag_log_file))
    
    flag_models_dir, _ = check_if_dir_or_file_exists(os.path.join(modelfile_path, main_exp_name),
                                                    file_name=None)
    
    print("Is model-directory present:? - {}".format(flag_models_dir))
    #print("Is file present:? - {}".format(flag_file))
    
    tr_logfile_name_with_path = os.path.join(os.path.join(logfile_path, main_exp_name), tr_log_file_name)
    te_logfile_name_with_path = os.path.join(os.path.join(logfile_path, main_exp_name), te_log_file_name)

    if flag_log_dir == False:
        print("Creating {}".format(os.path.join(logfile_path, main_exp_name)))
        os.makedirs(os.path.join(logfile_path, main_exp_name), exist_ok=True)
    
    if flag_models_dir == False:
        print("Creating {}".format(os.path.join(modelfile_path, main_exp_name)))
        os.makedirs(os.path.join(modelfile_path, main_exp_name), exist_ok=True)
    
    if filter_type.lower() == "kf":
        ssm_data_model = Z_XY['ssm_model']
        kf_model = KF(n_states=ssm_data_model.n_states,
                    n_obs=ssm_data_model.n_obs,
                    F=ssm_data_model.F,
                    G=ssm_data_model.G,
                    H=ssm_data_model.H,
                    Q=ssm_data_model.Q,
                    R=ssm_data_model.R,
                    inverse_r2_dB=ssm_parameters_dict[dataset_type]["inverse_r2_dB"],
                    nu_dB=ssm_parameters_dict[dataset_type]["nu_dB"],
                    device=device)

        X_estimated, Pk_X_estimated, avg_test_mse = kf_model.run_mb_filter(testloader=test_loader,
                                                                    te_logfile=te_logfile_name_with_path)
    
    return None

if __name__ == "__main__":
    main()