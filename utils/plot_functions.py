from cProfile import label
from turtle import color
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_state_trajectory(X, X_est=None, savefig=False, savefig_name=None):
    
    # Creating 3d plot of the data
    print(X.shape)
    
    if X.shape[-1] == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(X[:,0], X[:,1],'--',label='$\\mathbf{x}^{true}$')
        if not X_est is None:
            ax.plot(X_est[:,0], X_est[:,1],'--',label='$\\hat{\mathbf{x}}$')
        ax.set_xlabel('$X_1$')
        ax.set_ylabel('$X_2$')
        ax.set_ylabel('$X_3$')
        plt.legend()
        
    elif X.shape[-1] > 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(X[:,0], X[:,1], X[:,2], '--',label='$\\mathbf{x}^{true}$')
        if not X_est is None:
            ax.plot(X_est[:,0], X_est[:,1], X_est[:,2], '--',label='$\\hat{\mathbf{x}}$')
        ax.set_xlabel('$X_1$')
        ax.set_ylabel('$X_2$')
        ax.set_zlabel('$X_3$')
        plt.legend()
    
    if savefig:
        plt.savefig(savefig_name)
    plt.show()
    return None

def plot_measurement_data(Y, savefig=False, savefig_name=None):
    
    # Plot the measurement data
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(Y[:,0], Y[:,1], '--',label='$\\mathbf{y}^{measured}$')
    ax.set_xlabel('$Y_1$')
    ax.set_ylabel('$Y_2$')
    plt.legend()
    if savefig:
        plt.savefig(savefig_name)
    plt.show()
    return None

def plot_state_trajectory_axes(X, X_est=None, savefig=False, savefig_name=None):
    
    # Creating 3d plot of the data
    print(X.shape)
    
    if X.shape[-1] == 2:
        fig = plt.figure(figsize=(20,10))
        plt.subplot(311)
        plt.plot(X[:,0],'--',label='$\\mathbf{x}^{true} (x-component) $')
        if not X_est is None:
            plt.plot(X_est[:,0], '--',label='$\\hat{\mathbf{x}} (x-component) $')
        plt.ylabel('$X_1$')
        plt.xlabel('$n$')
        plt.legend()

        plt.subplot(312)
        plt.plot(X[:,1], '--',label='$\\mathbf{x}^{true} (y-component)$')
        if not X_est is None:
            plt.plot(X_est[:,1], '--',label='$\\hat{\mathbf{x}} (y-component)$')
        plt.ylabel('$X_2$')
        plt.xlabel('$n$')
        plt.legend()
        
    elif X.shape[-1] > 2:
        
        fig = plt.figure(figsize=(20,10))
        plt.subplot(311)
        plt.plot(X[:,0],'--',label='$\\mathbf{x}^{true} (x-component) $')
        if not X_est is None:
            plt.plot(X_est[:,0], '--',label='$\\hat{\mathbf{x}} (x-component) $')
        plt.ylabel('$X_1$')
        plt.xlabel('$n$')
        plt.legend()

        plt.subplot(312)
        plt.plot(X[:,1], '--',label='$\\mathbf{x}^{true} (y-component)$')
        if not X_est is None:
            plt.plot(X_est[:,1], '--',label='$\\hat{\mathbf{x}} (y-component)$')
        plt.ylabel('$X_2$')
        plt.xlabel('$n$')
        plt.legend()
    
        plt.subplot(313)
        plt.plot(X[:,2],'--',label='$\\mathbf{x}^{true} (z-component)$')
        if not X_est is None:
            plt.plot(X_est[:,2],'--',label='$\\hat{\mathbf{x}} (z-component)$')
        plt.ylabel('$X_3$')
        plt.xlabel('$n$')
        plt.legend()
    
    if savefig:
        plt.savefig(savefig_name)
    plt.show()
    return None

def plot_measurement_data_axes(Y, Y_est=None, savefig=False, savefig_name=None):
    
    # Creating 3d plot of the data
    fig = plt.figure(figsize=(20,10))
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
    
    #plt.subplot(313)
    #plt.plot(X[:,2],'--',label='$\\mathbf{x}^{true} (z-component)$')
    #if not X_est is None:
    #    plt.plot(X_est[:,2],'--',label='$\\hat{\mathbf{x}} (z-component)$')
    #plt.ylabel('$X_3$')
    #plt.xlabel('$n$')
    #plt.legend()
    
    if savefig:
        plt.savefig(savefig_name)
    plt.show()
    return None
