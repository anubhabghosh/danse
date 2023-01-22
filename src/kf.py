######################################################################## 
# Implementing a Kalman filter - SQ
#########################################################################
import numpy as np
import torch
from torch import nn, linalg

class KF(nn.Module):
    """ This class implements a Kalman Filter in PyTorch
    """
    def __init__(self, n_states, F, G, H, Q, R) -> None:
        super(KF, self).__init__()
        
        # Initializing the system model
        self.n_states = n_states # Setting the number of states of the Kalman filter
        self.F_k = F # State transition matrix (relates x_k to x_{k+1})
        self.G_k = G # Input matrix (relates input u_k to x_k)
        self.H_k = H # Output matrix (relates state x_k to output y_k)
        self.Q_k = Q # Covariance matrix of the process noise, we assume process noise w_k ~ N(0, Q)
        self.R_k = R # Covariance matrix of the measurement noise, we assume mesaurement noise v_k ~ N(0, R)
        
        # Defining the required state to be estimate 
        self.x_hat_pos_k = torch.zeros((n_states, 1)) # Assuming initial value of the state is zero, i.e. \hat{p}_0^{+} = 0
        self.Pk_pos = torch.eye(n_states) # Assuming initial value of the state covariance for the filtered estimate is identity, i.e. P_{0 \vert 0} = I
        self.Sk_pos_sqrt = torch.linalg.cholesky(self.Pk_pos) # Assuming initial value of the state covariance for the filtered estimate is identity, i.e. P_{0 \vert 0} = I
        self.x_hat_neg_k = None # Prediction state \hat{x}_{k \vert k-1}
        self.Sk_neg_sqrt = None # Cholesky factor of the state covariance matrix S_{k \vert k-1} = \sqrt{P_{k \vert k-1}}
        self.Pk_neg = None # State covariance matrix P_{k \vert k-1}
        self.Kk = torch.zeros(self.n_states) # The Kalman gain K_k (filtered version)

        return None
    
    def predict_estimate(self, F_k_prev, Pk_pos_prev, G_k_prev, Q_k_prev):
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
        #self.Pk_neg = F_k_prev @ (Pk_pos_prev @ F_k_prev.T) + G_k_prev @ (Q_k_prev @ G_k_prev.T) # Calculating the predicted state covariance P_{k \vert k -1} \equiv P^{-}_{k}
        
        # Implemeting a Square-Root Filtering version for numerical stability
        # Trying QR decomposition to get Pk_neg from Pk_pos
        self.Sk_pos_sqrt = torch.linalg.cholesky(Pk_pos_prev)
        Qk_prev_sqrt = torch.linalg.cholesky(Q_k_prev)
        A_state = torch.cat((F_k_prev @ self.Sk_pos_sqrt, G_k_prev @ Qk_prev_sqrt), dim=1)
        assert A_state.shape[1] == 2*self.n_states, "A_state has a dimension problem"
        _, Ra = torch.linalg.qr(A_state.T)
        assert torch.allclose(Ra, torch.triu(Ra)), "Ra is not upper triangular" # check if upper triangular
        self.Sk_neg_sqrt = Ra.T[:,:self.n_states]

        # Time update equation
        self.x_hat_neg_k = F_k_prev @ self.x_hat_pos_k # Calculating the predicted state estimate using the previous filtered state estimate \hat{x}_{k-1 \vert k-1}^{+}
        self.Pk_neg = self.Sk_neg_sqrt @ self.Sk_neg_sqrt.T # Calculating the predicted state estimate covariance 

        return self.x_hat_neg_k, self.Pk_neg

    def filtered_estimate(self, y_k):
        """ This function implements the filtering step of the Kalman filter
        Args:
            y_k: The measurement at time instant k 
        Returns:
            _type_: _description_
        """
        
        assert self.H_k.shape[1] == self.n_states, "Dimension of H_k is not consistent"

        # Trying to Square root algorithm since K_k is ill conditioned at the moment.
        # So, we use QR decomposition in the measurement equation
        A_measurement = torch.block_diag([
            [torch.sqrt(self.R_k) * torch.eye(self.H_k.shape[0]), self.H_k @ self.Sk_neg_sqrt],
            [torch.zeros((self.n_states, self.H_k.shape[0])), self.Sk_neg_sqrt]
            ])
        
        Qa_measurement, Ra_measurement = torch.linalg.qr(A_measurement.T)
        Re_k_sqrt = Ra_measurement.T[:self.H_k.shape[0],:self.H_k.shape[0]]
        K_k_Re_k_sqrt = Ra_measurement.T[self.H_k.shape[0]:,:self.H_k.shape[0]]
        self.K_k = K_k_Re_k_sqrt @ torch.linalg.inv(Re_k_sqrt)
        assert self.K_k.shape[0] == self.n_states, "Kalman gain has a dimension issue"

        self.Sk_pos_sqrt = Ra_measurement.T[self.H_k.shape[0]:,self.H_k.shape[0]:]
        
        # Measurement update equation
        self.x_hat_pos_k = self.x_hat_neg_k + self.K_k @ (y_k - self.H_k @ self.x_hat_neg_k)
        self.Pk_pos = self.Sk_pos_sqrt @ self.Sk_pos_sqrt.T
        assert self.Pk_pos.shape[0] == self.n_states and self.Pk_pos.shape[1] == self.n_states, "Dimension problem for P_k"

        return self.x_hat_pos_k, self.Pk_pos

    def compute_K(self):
        """ Altenrative function for Kalman Gain computation for the linearized model

        Returns:
            _type_: _description_
        """
        Re_k = self.H_k @ (self.Pk_neg @ self.H_k.T) + self.R_k
        if len(torch.flatten(Re_k)) > 1:
            K_k = self.Pk_neg @ (self.H_k.T @ torch.linalg.inv(Re_k))
        elif len(torch.flatten(Re_k)) == 1:
            K_k = self.Pk_neg @ (self.H_k.T @ (1.0 / Re_k))
        return K_k

    def run_mb_filter(self, X, Y):

        if len(Y.shape) == 3:
            N, T, d = Y.shape
        elif len(Y.shape) == 2:
            T, d = Y.shape
            N = 1
            Y = Y.reshape((N, T, d))

        traj_estimated = torch.zeros((N, T, self.n_states))
        Pk_estimated = torch.zeros((N, T, self.n_states, self.n_states))
        mse_arr = torch.zeros((N,1))

        for i in range(0, N):

            for k in range(0, T):

                x_rec_hat_neg_k, Pk_neg = self.predict_estimate(F_k_prev=self.F_k, Pk_pos_prev=self.Pk_pos,
                                        G_k_prev=self.G_k, Q_k_prev=self.Q_k)
                
                x_rec_hat_pos_k, Pk_pos = self.filtered_estimate(y_k=Y[i])
            
                # Save filtered state estimates
                traj_estimated[i,k,:] = x_rec_hat_pos_k
                #Also save covariances
                Pk_estimated[i,k,:,:] = Pk_pos

                print("i: {}, k: {}, norm of kalman gain: {}".format(k, np.linalg.norm(self.K_k)))

            mse_arr[i] = torch.linalg.norm(traj_estimated[i] - X[i])  # Calculate the squared error across the length of a single sequence

        mse = torch.mean(mse_arr, dim=0) # Calculate the MSE by averaging over all examples in a batch
        
        return traj_estimated, Pk_estimated, mse
