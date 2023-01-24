# Importing the necessary libraries
import numpy as np
import scipy
import torch
from torch import distributions
import matplotlib.pyplot as plt
from scipy.linalg import expm
from utils.utils import dB_to_lin, generate_normal

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
        m = self.n_obs
        H_sys = np.rot90(np.eye(m)) + np.concatenate((np.concatenate((np.ones((1, m-1)), 
                                                              np.zeros((m-1, m-1))), 
                                                             axis=0), 
                                              np.zeros((m,1))), 
                                             axis=1)
        return H_sys

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
        
        print("Measurement variance: {}, Process variance: {}".format(r2, q2))
        
        e = np.random.multivariate_normal(np.zeros(self.d,), q2*np.eye(self.d),size=(T+1,))
        v = np.random.multivariate_normal(np.zeros(self.d,), r2*np.eye(self.d),size=(T,))
        
        for t in range(0,T):
            
            x[t+1] = self.f_linearize(x[t]) + e[t]
            y[t] = self.h_fn(x[t]) + v[t]
        
        if self.decimate == True:
            K = int(self.delta_d / self.delta)
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
        
        X_arr = np.zeros((T, model.n_states))
        Y_arr = np.zeros((T, model.n_obs))

        X_arr, Y_arr = model.generate_single_sequence(
                T=T,
                inverse_r2_dB=parameters["inverse_r2_dB"],
                nu_dB=parameters["nu_dB"],
                drive_noise=False,
                add_noise_flag=False
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