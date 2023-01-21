######################################################################## 
# Implementing a Extended Kalman filter - SQ
#########################################################################
import numpy as np
import torch
from torch import autograd, nn

class UKF(nn.Module):
    """ This class implements an unscented Kalman filter in PyTorch
    """
    def __init__(self, n_states, f, h, Q, R, kappa, alpha, n_sigma):
        super(UKF, self).__init__()

        # Initializing the system model
        self.n_states = n_states # Setting the number of states of the Kalman filter
        self.f_k = f # State transition function (relates x_k, u_k to x_{k+1})
        self.h_k = h # Output function (relates state x_k to output y_k)
        self.Q_k = Q # Covariance matrix of the process noise, we assume process noise w_k ~ N(0, Q)
        self.R_k = R # Covariance matrix of the measurement noise, we assume mesaurement noise v_k ~ N(0, R)
        
        # Defining the required state to be estimate 
        self.x_hat_pos_k = torch.zeros((self.n_states, 1)) # Assuming initial value of the state is zero, i.e. \hat{p}_0^{+} = 0
        self.Pk_pos = torch.eye(self.n_states) # Assuming initial value of the state covariance for the filtered estimate is identity, i.e. P_{0 \vert 0} = I
        self.Sk_pos_sqrt = torch.linalg.cholesky(self.Pk_pos) # Assuming initial value of the state covariance for the filtered estimate is identity, i.e. P_{0 \vert 0} = I
        self.x_hat_neg_k = None # Prediction state \hat{x}_{k \vert k-1}
        self.Sk_neg_sqrt = None # Cholesky factor of the state covariance matrix S_{k \vert k-1} = \sqrt{P_{k \vert k-1}}
        self.Pk_neg = None # State covariance matrix P_{k \vert k-1}
        self.Kk = torch.zeros(self.n_states) # The Kalman gain K_k (filtered version)

        # Defining the parameters for the unscented transformation
        self.kappa = kappa # Hyperparameter in the UT
        self.alpha = alpha # Hyperparameter in the UT

        if n_sigma % 2 == 0:
            self.n_sigma = n_sigma # No. of sigma points in the unscented transformation
        else:
            self.n_sigma = n_sigma + 1 # To make number of sigma points as even

        self.x_hat_neg_k_sigma = torch.empty(self.n_states, self.n_sigma)
        self.x_hat_pos_k_sigma = torch.empty(self.n_states, self.n_sigma)

        return None

    def unscented_transform(self, xk, Pk):
        
        Sigma_tilde_points = torch.empty_like(self.x_hat_neg_k_sigma)
        for i in range(self.n_sigma):
            if i < self.n_sigma // 2:
                Sigma_tilde_points[:, i] = torch.transpose(torch.linalg.cholesky((self.n_sigma // 2)*Pk))[:, i]
            else:
                Sigma_tilde_points[:, i] = torch.transpose(-torch.linalg.cholesky((self.n_sigma // 2)*Pk))[:, i]
        
        return Sigma_tilde_points + torch.repeat_interleave(xk.reshape((-1,1)), self.n_sigma, dim=1)

    def transform_sigma_points_prediction(self, u_k=0):
        return self.f_k(self.x_hat_neg_k_sigma, u_k)

    def transform_sigma_points_filtering(self):
        return self.h_k(self.x_hat_pos_k_sigma)    

    def predict_estimate(self, Q_k_prev, u_k=0.0):
        """ This function helps implement the prediction step / time-update step of the Kalman filter, i.e. using 
        available observations, and previous state estimates, what is the next state estimate?

        Args:
            F_k_prev: State transition matrix F_k at time instant k
            Pk_pos_prev: Filtered state covariance matrix P_{k \vert k}
            G_k_prev: Input matrix G_k at time instant k
            Q_k_prev: Process noise covariance Q_k at instant k

        Returns:
            _type_: _description_
        """
        self.x_hat_pos_k_sigma = self.unscented_transform(self.x_hat_pos_k, self.Pk_pos)
        self.x_hat_pos_k_sigma_transformed = self.transform_sigma_points_prediction(u_k)
        self.x_hat_neg_k = torch.mean(
            self.x_hat_pos_k_sigma_transformed,
            dim=1
        ).reshape((self.n_states,1))
        self.Pk_neg = (1.0/self.n_sigma) * (self.x_hat_pos_k_sigma_transformed - self.x_hat_neg_k) @ (self.x_hat_pos_k_sigma_transformed - self.x_hat_neg_k).T
        self.Pk_neg += Q_k_prev
        return self.x_hat_neg_k, self.Pk_neg

    def filtered_estimate(self, y_k):
        """ This function implements the filtering step of the Kalman filter
        Args:
            y_k: The measurement at time instant k 
        Returns:
            _type_: _description_
        """
        
        self.x_hat_neg_k_sigma = self.unscented_transform(self.x_hat_neg_k, self.Pk_neg)
        self.x_hat_neg_k_sigma_transformed = self.transform_sigma_points_prediction()
        y_hat_k = torch.mean(
            self.x_hat_neg_k_sigma_transformed,
            dim=1
        ).reshape((-1,1))
        Py = (1.0/self.n_sigma) * (self.x_hat_neg_k_sigma_transformed - y_hat_k) @ (self.x_hat_neg_k_sigma_transformed - y_hat_k).T
        Py += self.R_k
        Pxy = (1.0/self.n_sigma) * (self.x_hat_neg_k_sigma - self.x_hat_neg_k) @ (self.x_hat_neg_k_sigma_transformed - y_hat_k).T
        self.K_k = Pxy @ torch.linalg.inv(Py)
        self.x_hat_pos_k = self.x_hat_neg_k + self.K_k @ (y_k - y_hat_k)
        self.Pk_pos = self.Pk_neg - self.K_k @ Py @ self.K_k.T
        return self.x_hat_pos_k, self.Pk_pos