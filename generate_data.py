# Importing the necessary libraries
import numpy as np
import scipy
import torch
from torch import distributions
import matplotlib.pyplot as plt
from scipy.linalg import expm

def dB_to_lin(x):
    return 10**(x/10)

def lin_to_dB(x):
    assert x != 0, "X is zero"
    return 10*np.log10(x)

def generate_normal(N, mean, Sigma):

    # n = N(mean, std**2)
    n = np.random.multivariate_normal(mean=mean, cov=Sigma, size=(N,))
    return n

class LinearSSM(object):

    def __init__(self, n_states, n_obs, F, G, H, mu_e, mu_w, q, r, Q, R) -> None:
        
        # Initialize system model parameters
        self.n_states = n_states
        self.n_obs = n_obs

        if self.F is None and self.H is None:
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


    def generate_single_sequence(self, N, drive_noise=False, add_noise_flag=False):
    
        x_arr = np.zeros((N+1, self.n_states))
        y_arr = np.zeros((N, self.n_obs))
        
        #NOTE: Since theta_5 and theta_6 are modeling variances in this code, 
        # for direct comparison with the MATLAB code, the std param input should be
        # a square root version
        e_k_arr = generate_normal(N=N, mean=np.zeros((self.n_states,)), Sigma=self.Q)
        w_k_arr = generate_normal(N=N, mean=np.zeros((self.n_obs,)), Sigma=self.R)
        
        # Generate the sequence iteratively
        for k in range(N):
            
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

    def __init__(self, d, J, delta, A_fn, h_fn) -> None:
        
        self.d = d
        self.J = J
        self.delta = delta
        self.A_fn = A_fn
        self.h_fn = h_fn

    def h_fn(self, x):
        return x

    def f_linearize(self, x):

        self.F = np.eye(self.d)
        for j in range(1, self.J+1):
            self.F += np.linalg.matrix_power(self.A_fn(x)*self.delta, j) / np.math.factorial(j)

        return self.F @ x

    def simulate_Lorenz_attractor(self, T, inverse_r2_dB, nu_dB, delta_d=None, decimate=False):
    
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
        
        if decimate == True:
            K = int(delta_d / self.delta)
            x_lorenz_d = x[0:T:K,:]
            y_lorenz_d = self.h_fn(x_lorenz_d) + np.random.multivariate_normal(
                np.zeros(self.d,), r2*np.eye(self.d),size=(len(x_lorenz_d),))
        else:
            x_lorenz_d = None
            y_lorenz_d = None

        return x, y, x_lorenz_d, y_lorenz_d
