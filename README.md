# DANSE - Data-driven Nonlinear State Estimation of a Model-free process with Linear measurements
This is the repository for implementing a nonlinear state estimation of a model-free process with Linear measurements

## Reference models
### Extended Kalman filter (EKF)

We assume that the model is

$\mathbf{x}_{k} = f_{k-1}\left(\mathbf{x}_{k-1}, \mathbf{u}_{k-1}, \mathbf{w}_{k-1}\right) \\
\mathbf{y}_{k} = h_{k}\left(\mathbf{x}_{k}, \mathbf{v}_{k}\right)
$

where we assume that the equations $f()$ and $h()$ represent nonlinear functions. The core idea of the extended Kalman filter is to linearize the functions $f()$ and $h()$ around the posterior and prior estimates of the state respectively, using a first-order, Taylor series approximation. Also, the assumption is that the process and measurement noises are white, and distributed as $\mathbf{w}_k \sim \left(\pmb{0}, \mathbf{Q}_k\right), \mathbf{v}_k \sim \left(\pmb{0}, \mathbf{R}_k\right)$. 

The algorithm steps are:

1. Initialization (k=1): 

    $\hat{\mathbf{x}}_{k-1 \vert k-1} = \hat{\mathbf{x}}_{0 \vert 0} := \mathbb{E}\left[\mathbf{x}_0\right]$,

    $\mathbf{P}_{k-1 \vert k-1} = \mathbf{P}_{0 \vert 0} := \mathbb{E}\left[(\mathbf{x}_0 - \hat{\mathbf{x}}_{0 \vert 0}) (\mathbf{x}_0 - \hat{\mathbf{x}}_{0 \vert 0})^{\top}
    \right]$,

2. For k = 2, 3, ...

    a. Linearize the state equation by computing $\mathbf{F}_{k-1} = \frac{\partial f_{k-1}}{\partial x_{k-1}} \Bigg \vert_{\hat{\mathbf{x}}_{k-1 \vert k-1}}$ and $\mathbf{G}_{k-1} = \frac{\partial f_{k-1}}{\partial w_{k-1}} \Bigg \vert_{\hat{\mathbf{x}}_{k-1 \vert k-1}}$. The Resulting state equation becomes: $\mathbf{x}_{k} = \mathbf{F}_{k-1}\mathbf{x}_{k-1} +  \tilde{\mathbf{u}}_{k-1} + \mathbf{G}_{k-1}\mathbf{w}_{k-1}$

    b. Prediction step:

    $\hat{\mathbf{x}}_{k \vert k-1} = f_{k-1}\left(\hat{\mathbf{x}}_{k-1 \vert k-1}, \mathbf{u}_{k-1}, 0\right)$

    $\mathbf{P}_{k \vert k-1} = \mathbf{F}_{k-1}\mathbf{P}_{k-1 \vert k-1}\mathbf{F}_{k-1}^{\top} + \mathbf{G}_{k-1}\mathbf{Q}_{k-1 \vert k-1}\mathbf{G}_{k-1}^{\top}$

    c. Filtering step:

    Linearize the measurement equation by computing $\mathbf{H}_{k} = \frac{\partial h_{k}}{\partial x_{k}} \Bigg \vert_{\hat{\mathbf{x}}_{k \vert k-1}}$ and $\mathbf{M}_{k} = \frac{\partial h_{k}}{\partial v_{k}} \Bigg \vert_{\hat{\mathbf{x}}_{k \vert k-1}}$ and use them to compute

    $\mathbf{K}_k = \mathbf{P}_{k \vert k-1} \mathbf{H}_k^{\top} \left(\mathbf{H}_k \mathbf{P}_{k \vert k-1}\mathbf{H}_k^{\top} + \mathbf{M}_k \mathbf{R}_{k}\mathbf{M}_k^{\top}\right)^{-1}$

    $\hat{\mathbf{x}}_{k \vert k} = \hat{\mathbf{x}}_{k \vert k-1} + \mathbf{K}_k \left[y_k - h_{k}\left(\hat{\mathbf{x}}_{k \vert k-1}, 0\right)\right]$

    $\mathbf{P}_{k \vert k} = \left(\mathbf{I} - \mathbf{K}_k \mathbf{H}_k\right) \mathbf{P}_{k \vert k-1}$

#### Extended Kalman filter (with square root filtering) (EKF-SQ)

The model description remains the same, to ensure that we have a stable numerical formulation of the state covariance matrices, we perform Cholesky decomposition of the state covariance matrices to get $\mathbf{P}_{k-1 \vert k-1}^{1/2} = \texttt{cholesky}(\mathbf{P}_{k-1 \vert k-1})$, then we 


1. Replace Step 2.b.2. by the following matrix equation to get $\mathbf{P}_{k \vert k-1}$

    $
    \begin{bmatrix}
    \mathbf{F}_{k-1}\mathbf{P}_{k-1 \vert k-1}^{1/2} & \mathbf{G}_{k-1}\mathbf{Q}_{k-1}^{1/2}
    \end{bmatrix} \Theta = \begin{bmatrix}
    \mathbf{X} & \pmb{0}
    \end{bmatrix}
    $

    Then $\mathbf{P}_{k \vert k-1}^{1/2} = \mathbf{X}$.

2. Replace Step 2.c.3. by the following matrix equation to get $\mathbf{P}_{k \vert k}$

    $
    \begin{bmatrix}
    \mathbf{R}_{k} & \mathbf{H}_k\mathbf{P}_{k\vert k-1}^{1/2} \\
    \pmb{0} & \mathbf{P}_{k \vert k-1}^{1/2}
    \end{bmatrix} \Theta = \begin{bmatrix}
    \mathbf{X} & \pmb{0} \\
    \mathbf{Y} & \mathbf{Z} \\
    \end{bmatrix}
    $

    Then $\mathbf{P}_{k \vert k}^{1/2} = \mathbf{Z}, \mathbf{R}_{e_k}^{1/2} = \mathbf{X}, \mathbf{K}_k\mathbf{R}_{e_k}^{1/2}  = \mathbf{Y}$.

The rest of the steps are exactly same as that of the standard-EKF algorithm. 