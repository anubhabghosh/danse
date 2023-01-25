# Importing the necessary libraries
import numpy as np
import scipy
import torch
from torch import distributions
import matplotlib.pyplot as plt
from scipy.linalg import expm
from utils.utils import dB_to_lin, generate_normal, save_dataset
from parameters import get_parameters
import argparse
from parse import parse
import os

class LinearSSM(object):

    def __init__(self, n_states, n_obs, F, G, H, mu_e, mu_w, q, r, Q, R) -> None:
        
        # Initialize system model parameters
        self.n_states = n_states
        self.n_obs = n_obs

        if F is None and H is None:
            self.F = self.construct_F()
            self.H = self.construct_H()
        else:
            self.F = F
            self.H = H

        self.G = G

        # Initialize noise variances
        self.mu_e = mu_e
        self.mu_w = mu_w
        self.q = q 
        self.r = r
        self.Q = Q
        self.R = R

        if self.Q is None and self.R is None:
            self.init_noise_covs()

    def construct_F(self):
        m = self.n_states
        F_sys = np.eye(m) + np.concatenate((np.zeros((m,1)), 
                                np.concatenate((np.ones((1,m-1)), 
                                                np.zeros((m-1,m-1))), 
                                               axis=0)), 
                               axis=1)
        return F_sys

    def construct_H(self):
        H_sys = np.rot90(np.eye(self.n_states, self.n_states)) + np.concatenate((np.concatenate((np.ones((1, self.n_states-1)), 
                                                              np.zeros((self.n_states-1, self.n_states-1))), 
                                                             axis=0), 
                                              np.zeros((self.n_states,1))), 
                                             axis=1)
        return H_sys[:self.n_obs, :self.n_states]

    def init_noise_covs(self):

        self.Q = self.q**2 * np.eye(self.n_states)
        self.R = self.r**2 * np.eye(self.n_obs)
        return None

    def generate_driving_noise(k, a=1.2, add_noise=False):
    
        #u_k = np.cos(a*k) # Previous idea (considering start at k=0)
        if add_noise == False:
            u_k = np.cos(a*(k+1)) # Current modification (considering start at k=1)
        elif add_noise == True:
            u_k = np.cos(a*(k+1) + np.random.normal(loc=0, scale=np.pi, size=(1,1))) # Adding noise to the sample
        return u_k


    def generate_single_sequence(self, T, inverse_r2_dB=0, nu_dB=0, drive_noise=False, add_noise_flag=False):
    
        x_arr = np.zeros((T+1, self.n_states))
        y_arr = np.zeros((T, self.n_obs))
        
        r2 = 1.0 / dB_to_lin(inverse_r2_dB)
        q2 = dB_to_lin(nu_dB - inverse_r2_dB)
        
        self.r = r2
        self.q = q2
        
        self.init_noise_covs()
        
        #NOTE: Since theta_5 and theta_6 are modeling variances in this code, 
        # for direct comparison with the MATLAB code, the std param input should be
        # a square root version
        e_k_arr = generate_normal(N=T, mean=np.zeros((self.n_states,)), Sigma=self.Q)
        w_k_arr = generate_normal(N=T, mean=np.zeros((self.n_obs,)), Sigma=self.R)
        
        # Generate the sequence iteratively
        for k in range(T):
            
            # Generate driving noise (which is time varying)
            # Driving noise should be carefully selected as per value of k (start from k=0 or =1)
            if drive_noise == True: 
                u_k = self.generate_driving_noise(k, a=1.2, add_noise=add_noise_flag)
            else:
                u_k = np.array([0.0]).reshape((-1,1))
            
            # For each instant k, sample e_k, w_k
            e_k = e_k_arr[k]
            w_k = w_k_arr[k]
            
            # Equation for updating the hidden state
            x_arr[k+1] = (self.F @ x_arr[k].reshape((-1,1)) + self.G @ u_k + e_k.reshape((-1,1))).reshape((-1,))
            
            # Equation for calculating the output state
            y_arr[k] = self.H @ (x_arr[k]) + w_k
        
        return x_arr, y_arr

class LorenzAttractorModel(object):

    def __init__(self, d, J, delta, delta_d, A_fn, h_fn, decimate=False) -> None:
        
        self.d = d
        self.J = J
        self.delta = delta
        self.delta_d = delta_d
        self.A_fn = A_fn
        self.h_fn = h_fn
        self.decimate = decimate

    def h_fn(self, x):
        return x

    def f_linearize(self, x):

        self.F = np.eye(self.d)
        for j in range(1, self.J+1):
            self.F += np.linalg.matrix_power(self.A_fn(x)*self.delta, j) / np.math.factorial(j)

        return self.F @ x

    def generate_single_sequence(self, T, inverse_r2_dB, nu_dB):
    
        x = np.zeros((T+1, self.d))
        y = np.zeros((T, self.d))
        
        r2 = 1.0 / dB_to_lin(inverse_r2_dB)
        q2 = dB_to_lin(nu_dB - inverse_r2_dB)
        
        #print("Measurement variance: {}, Process variance: {}".format(r2, q2))
        
        e = np.random.multivariate_normal(np.zeros(self.d,), q2*np.eye(self.d),size=(T+1,))
        v = np.random.multivariate_normal(np.zeros(self.d,), r2*np.eye(self.d),size=(T,))
        
        for t in range(0,T):
            
            x[t+1] = self.f_linearize(x[t]) + e[t]
            y[t] = self.h_fn(x[t]) + v[t]
        
        if self.decimate == True:
            K = self.delta_d // self.delta
            x_lorenz_d = x[0:T:K,:]
            y_lorenz_d = self.h_fn(x_lorenz_d) + np.random.multivariate_normal(np.zeros(self.d,), r2*np.eye(self.d),size=(len(x_lorenz_d),))
        else:
            x_lorenz_d = None
            y_lorenz_d = None

        return x, y, x_lorenz_d, y_lorenz_d

def generate_SSM_data(type_, T, parameters):

    if type_ == "LinearSSM":

        model = LinearSSM(n_states=parameters["n_states"],
                        n_obs=parameters["n_obs"],
                        F=parameters["F"],
                        G=parameters["G"],
                        H=parameters["H"],
                        mu_e=parameters["mu_e"],
                        mu_w=parameters["mu_w"],
                        q=parameters["q"],
                        r=parameters["r"],
                        Q=parameters["Q"],
                        R=parameters["R"])

        X_arr = np.zeros((T, model.n_states))
        Y_arr = np.zeros((T, model.n_obs))

        X_arr, Y_arr = model.generate_single_sequence(
                T=T,
                inverse_r2_dB=parameters["inverse_r2_dB"],
                nu_dB=parameters["nu_dB"],
                drive_noise=False,
                add_noise_flag=False
            )

    elif type_ == "LorenzSSM":

        model = LorenzAttractorModel(
            d=3,
            J=parameters["J"],
            delta=parameters["delta"],
            A_fn=parameters["A_fn"],
            h_fn=parameters["h_fn"],
            delta_d=parameters["delta_d"],
            decimate=parameters["decimate"]
                    )
        
        X_arr = np.zeros((T, model.d))
        Y_arr = np.zeros((T, model.d))

        X_arr, Y_arr, X_arr_d, Y_arr_d = model.generate_single_sequence(
                                        T=T,
                                        inverse_r2_dB=parameters["inverse_r2_dB"],
                                        nu_dB=parameters["nu_dB"]
                                    )
        
    return model, X_arr, Y_arr


def generate_state_observation_pairs(type_, parameters, T=200, N_samples=1000):

    # Define the parameters of the model
    #N = 1000

    # Plot the trajectory versus sample points
    #num_trajs = 5

    Z_XY = {}
    Z_XY["num_samples"] = N_samples
    Z_XY_data_lengths = []

    count = 0
    Z_XY_data = []

    for i in range(N_samples):
        
        model, Xi, Yi = generate_SSM_data(type_, T, parameters)
        Z_XY_data_lengths.append(T)
        Z_XY_data.append([Xi, Yi])
        
    Z_XY["data"] = np.row_stack(Z_XY_data).astype(object)
    #Z_pM["data"] = Z_pM_data
    Z_XY["trajectory_lengths"] = np.vstack(Z_XY_data_lengths)

    return Z_XY

def create_filename(T=100, N_samples=200, m=5, n=5, dataset_basepath="./data/", type_="LinearSSM", inverse_r2_dB=40):
    # Create the dataset based on the dataset parameters
    
    datafile = "trajectories_m_{}_n_{}_{}_data_T_{}_N_{}_r2_{}dB.pkl".format(m, n, type_, int(T), int(N_samples), inverse_r2_dB)
    dataset_fullpath = os.path.join(dataset_basepath, datafile)
    return dataset_fullpath

def create_and_save_dataset(T, N_samples, filename, parameters, type_="LinearSSM"):

    #NOTE: Generates for pfixed theta estimation experiment
    # Currently this uses the 'modified' function
    #Z_pM = generate_trajectory_modified_param_pairs(N=N, 
    #                                                M=num_trajs, 
    #                                                P=num_realizations, 
    #                                                usenorm_flag=usenorm_flag)
    #np.random.seed(10) # This can be kept at a fixed step for being consistent
    Z_XY = generate_state_observation_pairs(type_=type_, parameters=parameters, T=T, N_samples=N_samples)
    save_dataset(Z_XY, filename=filename)

if __name__ == "__main__":
    
    usage = "Create datasets by simulating state space models \n"\
            "python generate_data.py --sequence_length T --num_samples N --dataset_type [LinearSSM/LorenzSSM] --output_path [output path name]\n"\
            "Creates the dataset at the location output_path"\
        
    parser = argparse.ArgumentParser(description="Input arguments related to creating a dataset for training RNNs")
    

    parser.add_argument("--n_states", help="denotes the number of states in the latent model", type=int, default=5)
    parser.add_argument("--n_obs", help="denotes the number of observations", type=int, default=5)
    parser.add_argument("--num_samples", help="denotes the number of trajectories to be simulated for each realization", type=int, default=500)
    parser.add_argument("--sequence_length", help="denotes the length of each trajectory", type=int, default=200)
    parser.add_argument("--inverse_r2_dB", help="denotes the inverse of measurement noise power", type=float, default=40.0)
    parser.add_argument("--dataset_type", help="specify mode=pfixed (all theta, except theta_3, theta_4) / vars (variances) / all (full theta vector)", type=str, default=None)
    parser.add_argument("--output_path", help="Enter full path to store the data file", type=str, default=None)
    
    args = parser.parse_args() 

    n_states = args.n_states
    n_obs = args.n_obs
    T = args.sequence_length
    N_samples = args.num_samples
    type_ = args.dataset_type
    output_path = args.output_path
    inverse_r2_dB = args.inverse_r2_dB

    # Create the full path for the datafile
    datafilename = create_filename(T=T, N_samples=N_samples, m=n_states, n=n_obs, dataset_basepath=output_path, type_=type_, inverse_r2_dB=inverse_r2_dB)
    ssm_parameters, _ = get_parameters(N=N_samples, T=T, n_states=n_states, n_obs=n_obs, inverse_r2_dB=inverse_r2_dB)

    # If the dataset hasn't been already created, create the dataset
    if not os.path.isfile(datafilename):
        print("Creating the data file: {}".format(datafilename))
        create_and_save_dataset(T=T, N_samples=N_samples, filename=datafilename, type_=type_, parameters=ssm_parameters[type_])
    
    else:
        print("Dataset {} is already present!".format(datafilename))
    
    print("Done...")