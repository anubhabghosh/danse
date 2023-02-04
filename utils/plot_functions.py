from cProfile import label
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_state_trajectory(X, X_est_KF=None, X_est_EKF=None, X_est_UKF=None, X_est_DANSE=None, savefig=False, savefig_name=None):
    
    # Creating 3d plot of the data
    print(X.shape)
    
    if X.shape[-1] == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(X[:,0], X[:,1],'--',label='$\\mathbf{x}^{true}$')
        if not X_est_KF is None:
            ax.plot(X_est_KF[:,0], X_est_KF[:,1],':',label='$\\hat{\mathbf{x}}_{KF}$')
        if not X_est_EKF is None:
            ax.plot(X_est_EKF[:,0], X_est_EKF[:,1],':',label='$\\hat{\mathbf{x}}_{EKF}$')
        if not X_est_UKF is None:
            ax.plot(X_est_UKF[:,0], X_est_UKF[:,1],'-.',label='$\\hat{\mathbf{x}}_{UKF}$')
        if not X_est_DANSE is None:
            ax.plot(X_est_DANSE[:,0], X_est_DANSE[:,1],'--',label='$\\hat{\mathbf{x}}_{DANSE}$')
        ax.set_xlabel('$X_1$')
        ax.set_ylabel('$X_2$')
        plt.legend()
        
    elif X.shape[-1] > 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(X[:,0], X[:,1], X[:,2], '--',label='$\\mathbf{x}^{true}$')
        if not X_est_KF is None:
            ax.plot(X_est_KF[:,0], X_est_KF[:,1], X_est_KF[:,2], ':',label='$\\hat{\mathbf{x}}_{KF}$')
        if not X_est_EKF is None:
            ax.plot(X_est_EKF[:,0], X_est_EKF[:,1], X_est_EKF[:,2], ':',label='$\\hat{\mathbf{x}}_{EKF}$')
        if not X_est_UKF is None:
            ax.plot(X_est_UKF[:,0], X_est_UKF[:,1], X_est_UKF[:,2], '-.',label='$\\hat{\mathbf{x}}_{UKF}$')
        if not X_est_DANSE is None:
            ax.plot(X_est_DANSE[:,0], X_est_DANSE[:,1], X_est_DANSE[:,2], '--',label='$\\hat{\mathbf{x}}_{DANSE}$')
        ax.set_xlabel('$X_1$')
        ax.set_ylabel('$X_2$')
        ax.set_zlabel('$X_3$')
        plt.legend()
    
    if savefig:
        plt.savefig(savefig_name)
    #plt.show()
    return None

def plot_measurement_data(Y, savefig=False, savefig_name=None):
    
    # Plot the measurement data
    fig = plt.figure()

    if Y.shape[-1] == 2:

        ax = fig.add_subplot(111)
        ax.plot(Y[:,0], Y[:,1], '--', label='$\\mathbf{y}^{measured}$')
        ax.set_xlabel('$Y_1$')
        ax.set_ylabel('$Y_2$')
        plt.legend()
        if savefig:
            plt.savefig(savefig_name)

    elif Y.shape[-1] > 2:

        ax = fig.add_subplot(111, projection='3d')
        ax.plot(Y[:,0], Y[:,1], Y[:, 2], '--', label='$\\mathbf{y}^{measured}$')
        ax.set_xlabel('$Y_1$')
        ax.set_ylabel('$Y_2$')
        ax.set_zlabel('$Y_3$')
        plt.legend()
        if savefig:
            plt.savefig(savefig_name)

    #plt.show()
    return None

def plot_state_trajectory_axes(X, X_est_KF=None, X_est_EKF=None, X_est_UKF=None, X_est_DANSE=None, savefig=False, savefig_name=None):
    
    # Creating 3d plot of the data
    print(X.shape)
    
    if X.shape[-1] == 2:
        fig = plt.figure(figsize=(20,10))
        plt.subplot(311)
        plt.plot(X[:,0],'--',label='$\\mathbf{x}^{true} (x-component) $')
        if not X_est_KF is None:
            plt.plot(X_est_KF[:,0], ':',label='$\\hat{\mathbf{x}}_{KF} (x-component) $')
        if not X_est_DANSE is None:
            plt.plot(X_est_DANSE[:,0], '--',label='$\\hat{\mathbf{x}}_{DANSE} (x-component) $')
        if not X_est_EKF is None:
            plt.plot(X_est_EKF[:,0], ':',label='$\\hat{\mathbf{x}}_{EKF} (x-component) $')
        if not X_est_UKF is None:
            plt.plot(X_est_UKF[:,0], ':',label='$\\hat{\mathbf{x}}_{UKF} (x-component) $')
        plt.ylabel('$X_1$')
        plt.xlabel('$n$')
        plt.legend()

        plt.subplot(312)
        plt.plot(X[:,1], '--',label='$\\mathbf{x}^{true} (y-component)$')
        if not X_est_DANSE is None:
            plt.plot(X_est_DANSE[:,1], '--',label='$\\hat{\mathbf{x}}_{DANSE} (y-component) $')
        if not X_est_KF is None:
            plt.plot(X_est_KF[:,1], ':',label='$\\hat{\mathbf{x}}_{KF} (y-component) $')
        if not X_est_EKF is None:
            plt.plot(X_est_EKF[:,1], ':',label='$\\hat{\mathbf{x}}_{EKF} (y-component) $')
        if not X_est_UKF is None:
            plt.plot(X_est_UKF[:,1], ':',label='$\\hat{\mathbf{x}}_{UKF} (y-component) $')
        plt.ylabel('$X_2$')
        plt.xlabel('$n$')
        plt.legend()
        
    elif X.shape[-1] > 2:
        
        fig = plt.figure(figsize=(20,10))
        plt.subplot(311)
        plt.plot(X[:,0],'--',label='$\\mathbf{x}^{true} (x-component) $')
        if not X_est_DANSE is None:
            plt.plot(X_est_DANSE[:,0], '--',label='$\\hat{\mathbf{x}}_{DANSE} (x-component) $')
        if not X_est_KF is None:
            plt.plot(X_est_KF[:,0], ':',label='$\\hat{\mathbf{x}}_{KF} (x-component) $')
        if not X_est_EKF is None:
            plt.plot(X_est_EKF[:,0], ':',label='$\\hat{\mathbf{x}}_{EKF} (x-component) $')
        if not X_est_UKF is None:
            plt.plot(X_est_UKF[:,0], ':',label='$\\hat{\mathbf{x}}_{UKF} (x-component) $')
        plt.ylabel('$X_1$')
        plt.xlabel('$n$')
        plt.legend()

        plt.subplot(312)
        plt.plot(X[:,1], '--',label='$\\mathbf{x}^{true} (y-component)$')
        if not X_est_DANSE is None:
            plt.plot(X_est_DANSE[:,1], '--',label='$\\hat{\mathbf{x}}_{DANSE} (y-component) $')
        if not X_est_KF is None:
            plt.plot(X_est_KF[:,1], ':',label='$\\hat{\mathbf{x}}_{KF} (y-component) $')
        if not X_est_EKF is None:
            plt.plot(X_est_EKF[:,1], ':',label='$\\hat{\mathbf{x}}_{EKF} (y-component) $')
        if not X_est_UKF is None:
            plt.plot(X_est_UKF[:,1], ':',label='$\\hat{\mathbf{x}}_{UKF} (y-component) $')
        plt.ylabel('$X_2$')
        plt.xlabel('$n$')
        plt.legend()
    
        plt.subplot(313)
        plt.plot(X[:,2],'--',label='$\\mathbf{x}^{true} (z-component)$')
        if not X_est_DANSE is None:
            plt.plot(X_est_DANSE[:,2], '--',label='$\\hat{\mathbf{x}}_{DANSE} (z-component) $')
        if not X_est_KF is None:
            plt.plot(X_est_KF[:,2], ':',label='$\\hat{\mathbf{x}}_{KF} (z-component) $')
        if not X_est_EKF is None:
            plt.plot(X_est_EKF[:,2], ':',label='$\\hat{\mathbf{x}}_{EKF} (z-component) $')
        if not X_est_UKF is None:
            plt.plot(X_est_UKF[:,2], ':',label='$\\hat{\mathbf{x}}_{UKF} (z-component) $')
        plt.ylabel('$X_3$')
        plt.xlabel('$n$')
        plt.legend()
    
    if savefig:
        plt.savefig(savefig_name)
    #plt.show()
    return None

def plot_measurement_data_axes(Y, Y_est=None, savefig=False, savefig_name=None):
    
    # Creating 3d plot of the data
    fig = plt.figure(figsize=(20,10))

    if Y.shape[-1] == 2:
        plt.subplot(311)
        plt.plot(Y[:,0],'--',label='$\\mathbf{Y}^{true} (x-component) $')
        if not Y_est is None:
            plt.plot(Y_est[:,0], '--',label='$\\hat{\mathbf{Y}} (x-component) $')
        plt.ylabel('$Y_1$')
        plt.xlabel('$n$')
        plt.legend()
        
        plt.subplot(312)
        plt.plot(Y[:,1], '--',label='$\\mathbf{Y}^{true} (y-component)$')
        if not Y_est is None:
            plt.plot(Y_est[:,1], '--',label='$\\hat{\mathbf{Y}} (y-component)$')
        plt.ylabel('$Y_2$')
        plt.xlabel('$n$')
        plt.legend()

    elif Y.shape[-1] > 2:
        plt.subplot(311)
        plt.plot(Y[:,0],'--',label='$\\mathbf{Y}^{true} (x-component) $')
        if not Y_est is None:
            plt.plot(Y_est[:,0], '--',label='$\\hat{\mathbf{Y}} (x-component) $')
        plt.ylabel('$Y_1$')
        plt.xlabel('$n$')
        plt.legend()
        
        plt.subplot(312)
        plt.plot(Y[:,1], '--',label='$\\mathbf{Y}^{true} (y-component)$')
        if not Y_est is None:
            plt.plot(Y_est[:,1], '--',label='$\\hat{\mathbf{Y}} (y-component)$')
        plt.ylabel('$Y_2$')
        plt.xlabel('$n$')
        plt.legend()

        plt.subplot(313)
        plt.plot(Y[:,2],'--',label='$\\mathbf{Y}^{true} (z-component)$')
        if not Y_est is None:
            plt.plot(Y_est[:,2],'--',label='$\\hat{\mathbf{Y}} (z-component)$')
        plt.ylabel('$Y_3$')
        plt.xlabel('$n$')
        plt.legend()
    
    if savefig:
        plt.savefig(savefig_name)
    #plt.show()
    return None
