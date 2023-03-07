#####################################################
# Creator: Anubhab Ghosh 
# Feb 2023
#####################################################
import numpy as np
import math
from utils.utils import dB_to_lin, generate_normal

class LinearSSM(object):

    def __init__(self, n_states, n_obs, F=None, G=None, H=None, mu_e=0.0, mu_w=0.0, q2=1.0, r2=1.0, Q=None, R=None) -> None:
        
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
        self.q2 = q2 
        self.r2 = r2
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

        self.Q = self.q2 * np.eye(self.n_states)
        self.R = self.r2 * np.eye(self.n_obs)
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
        
        self.r2 = r2
        self.q2 = q2
        
        self.init_noise_covs()
        
        #NOTE: Since theta_5 and theta_6 are modeling variances in this code, 
        # for direct comparison with the MATLAB code, the std param input should be
        # a square root version
        e_k_arr = generate_normal(N=T, mean=self.mu_e, Sigma2=self.Q)
        w_k_arr = generate_normal(N=T, mean=self.mu_w, Sigma2=self.R)
        
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

    def __init__(self, d, J, delta, delta_d, A_fn, h_fn, decimate=False, mu_e=None, mu_w=None, use_Taylor=True) -> None:
        
        self.n_states = d
        self.J = J
        self.delta = delta
        self.delta_d = delta_d
        self.A_fn = A_fn
        self.h_fn = h_fn
        self.n_obs = self.h_fn(np.random.randn(d,1)).shape[0]
        self.decimate = decimate
        self.mu_e = mu_e
        self.mu_w = mu_w
        self.use_Taylor = use_Taylor

    def h_fn(self, x):
        return x

    def f_linearize(self, x):

        self.F = np.eye(self.n_states)
        for j in range(1, self.J+1):
            #self.F += np.linalg.matrix_power(self.A_fn(x)*self.delta, j) / np.math.factorial(j)
            self.F += np.linalg.matrix_power(self.A_fn(x[0])*self.delta, j) / np.math.factorial(j)

        return self.F @ x

    def init_noise_covs(self):

        self.Q = self.q2 * np.eye(self.n_states)
        self.R = self.r2 * np.eye(self.n_obs)
        return None

    def generate_single_sequence(self, T, inverse_r2_dB, nu_dB):
    
        x = np.zeros((T+1, self.n_states))
        y = np.zeros((T, self.n_obs))
        
        r2 = 1.0 / dB_to_lin(inverse_r2_dB)
        q2 = dB_to_lin(nu_dB - inverse_r2_dB)
        
        self.r2 = r2
        self.q2 = q2
        
        self.init_noise_covs()
        #print("Measurement variance: {}, Process variance: {}".format(r2, q2))
        
        e = np.random.multivariate_normal(self.mu_e, self.Q, size=(T+1,))
        v = np.random.multivariate_normal(self.mu_w, self.R, size=(T,))
        
        for t in range(0,T):
            
            x[t+1] = self.f_linearize(x[t]) + e[t]
            y[t] = self.h_fn(x[t]) + v[t]
        
        if self.decimate == True:
            K = self.delta_d // self.delta
            x_lorenz_d = x[0:T:K,:]
            y_lorenz_d = self.h_fn(x_lorenz_d) + np.random.multivariate_normal(self.mu_e, self.R, size=(len(x_lorenz_d),))
        else:
            x_lorenz_d = x
            y_lorenz_d = y

        return x_lorenz_d, y_lorenz_d

class SinusoidalSSM(object):

    def __init__(self, n_states, alpha=0.9, beta=1.1, phi=0.1*math.pi, delta=0.01, a=1.0, b=1.0, c=0.0, decimate=False, mu_e=None, mu_w=None, use_Taylor=False):
        
        self.n_states = n_states
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.phi = phi
        self.a = a
        self.b = b
        self.c = c
        self.n_obs = self.h_fn(np.random.randn(self.n_states,1)).shape[0]
        self.decimate = decimate
        self.mu_e = mu_e
        self.mu_w = mu_w
        self.use_Taylor = use_Taylor

    def init_noise_covs(self):

        self.Q = self.q * np.eye(self.n_states)
        self.R = self.r * np.eye(self.n_obs)
        return None

    def h_fn(self, x):
        return self.a * (self.b * x + self.c)

    def f_fn(self, x):
        return self.alpha * np.sin(self.beta * x + self.phi) + self.delta

    def generate_single_sequence(self, T, inverse_r2_dB, nu_dB):
    
        x = np.zeros((T+1, self.n_states))
        y = np.zeros((T, self.n_obs))
        
        r2 = 1.0 / dB_to_lin(inverse_r2_dB)
        q2 = dB_to_lin(nu_dB - inverse_r2_dB)
        
        self.r = r2
        self.q = q2
        
        self.init_noise_covs()

        #print("Measurement variance: {}, Process variance: {}".format(r2, q2))
        
        e = np.random.multivariate_normal(np.zeros(self.n_states,), q2*np.eye(self.n_states),size=(T+1,))
        v = np.random.multivariate_normal(np.zeros(self.n_obs,), r2*np.eye(self.n_obs),size=(T,))
        
        for t in range(0,T):
            x[t+1] = self.f_fn(x[t]) + e[t]
            y[t] = self.h_fn(x[t]) + v[t]
        
        if self.decimate == True:
            K = self.delta_d // self.delta
            x_d = x[0:T:K,:]
            y_d = self.h_fn(x_d) + np.random.multivariate_normal(self.mu_e, self.R, size=(len(x_d),))
        else:
            x_d = x
            y_d = y

        return x_d, y_d
