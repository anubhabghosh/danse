from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints, JulierSigmaPoints 
import torch
from torch import nn
from timeit import default_timer as timer
from utils.utils import dB_to_lin, mse_loss
import numpy as np

class UKF_Aliter(nn.Module):
    """ This class implements an unscented Kalman filter in PyTorch
    """
    def __init__(self, n_states, n_obs, f=None, h=None, Q=None, R=None, kappa=-1, 
                alpha=0.1, beta=2, n_sigma=None, inverse_r2_dB=None, 
                nu_dB=None, device='cpu', init_cond=None):
        super(UKF_Aliter, self).__init__()

        # Initialize the device
        self.device = device

        # Initializing the system model
        self.n_states = n_states # Setting the number of states of the Kalman filter
        self.n_obs = n_obs
        self.f_k = f # State transition function (relates x_k, u_k to x_{k+1})
        self.h_k = h # Output function (relates state x_k to output y_k)
         
        if (not inverse_r2_dB is None) and (not nu_dB is None):
            r2 = 1.0 / dB_to_lin(inverse_r2_dB)
            q2 = dB_to_lin(nu_dB - inverse_r2_dB)
            Q = q2 * np.eye(self.n_states)
            R = r2 * np.eye(self.n_obs)

        self.Q_k = self.push_to_device(Q) # Covariance matrix of the process noise, we assume process noise w_k ~ N(0, Q)
        self.R_k = self.push_to_device(R) # Covariance matrix of the measurement noise, we assume mesaurement noise v_k ~ N(0, R)
        
        # Defining the parameters for the unscented transformation
        self.kappa = kappa # Hyperparameter in the UT
        self.alpha = alpha # Hyperparameter in the UT
        self.beta = beta # Hyperparameter in the UT

        self.get_sigma_points()
        #self.set_mean_weights()
        #self.set_cov_weights()

        #self.init_cond = init_cond

        self.ukf = UnscentedKalmanFilter(dim_x=self.n_states, dim_z=self.n_obs, dt=self.delta_t, fx=self.f_k, hx=self.h_k, points=self.sigma_points)
        self.ukf.R = self.R_k.numpy()
        self.ukf.Q = self.Q_k.numpy()
        self.ukf.x = torch.zeros((self.n_states,)).numpy()
        self.ukf.P = (torch.eye(self.n_states)*1e-5).numpy()
        return None

    def get_sigma_points(self):
        self.sigma_points = MerweScaledSigmaPoints(self.n_states, alpha=self.alpha, beta=self.beta, kappa=self.kappa)

    def run_mb_filter(self, X, Y, U=None):

        _, Ty, dy = Y.shape
        _, Tx, dx = X.shape

        if len(Y.shape) == 3:
            N, T, d = Y.shape
        elif len(Y.shape) == 2:
            T, d = Y.shape
            N = 1
            Y = Y.reshape((N, Ty, d))
 
        traj_estimated = torch.zeros((N, Tx, self.n_states), device=self.device).type(torch.FloatTensor)
        Pk_estimated = torch.zeros((N, Tx, self.n_states, self.n_states), device=self.device).type(torch.FloatTensor)
        
        # MSE [Linear]
        MSE_UKF_linear_arr = torch.zeros((N,)).type(torch.FloatTensor)
        # points = JulierSigmaPoints(n=SysModel.m)
        
        start = timer()
        for i in range(0, N):
            
            #if self.init_cond is not None:
            #    self.ukf.x = torch.unsqueeze(self.init_cond[i, :], 1).numpy()
            
            for k in range(0, Ty):
                
                self.ukf.predict()
                self.ukf.update(Y[i,k,:].numpy())       
                traj_estimated[i,k,:] = torch.from_numpy(self.ukf.x)
                Pk_estimated[i,k,:,:] = torch.from_numpy(self.ukf.P)

            MSE_UKF_linear_arr[i] = mse_loss(traj_estimated[i], X[i]).item()
        
        end = timer()
        t = end - start

        MSE_UKF_linear_avg = torch.mean(MSE_UKF_linear_arr)
        MSE_UKF_dB_avg = 10 * torch.log10(MSE_UKF_linear_avg)
        # Standard deviation
        MSE_UKF_dB_std = torch.std(MSE_UKF_linear_arr, unbiased=True)
        MSE_UKF_dB_std = 10 * torch.log10(MSE_UKF_dB_std)

        print("UKF - MSE LOSS:", MSE_UKF_dB_avg, "[dB]")
        print("UKF - MSE STD:", MSE_UKF_dB_std, "[dB]")
        # Print Run Time
        print("Inference Time:", t)

        return traj_estimated, Pk_estimated, MSE_UKF_linear_avg, MSE_UKF_dB_avg