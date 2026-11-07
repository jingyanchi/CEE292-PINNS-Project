# Burgers’ Equation with PINNs: Soft vs. Hard Constraints and NTK Analysis

This project investigates how soft and hard enforcement of initial and boundary conditions affects the training of physics-informed neural networks (PINNs). It uses the one-dimensional viscous Burgers’ equation and explores training behavior through the empirical neural tangent kernel (NTK).

The project builds on the Burgers’ equation example of Raissi, Perdikaris, and Karniadakis (2019), with NTK analysis motivated by Wang, Yu, and Perdikaris (2022).

## Research Questions

- How does hard constraint enforcement affect solution accuracy compared with soft loss penalties?
- How do the empirical NTK spectra and kernel evolution differ between the two formulations?
- Can NTK-based loss weighting improve the soft-constrained model?

## Problem Definition

We consider the viscous Burgers’ equation:

$$
u_t + u u_x - \nu u_{xx} = 0,
\qquad \nu = \frac{0.01}{\pi},
$$

on the domain:

$$
x \in [-1,1], \qquad t \in [0,1],
$$

with initial condition:

$$
u(x,0) = -\sin(\pi x),
$$

and boundary conditions:

$$
u(-1,t) = u(1,t) = 0.
$$

The neural network approximates the solution using the coordinates $(x,t)$ as inputs.

## Comparison Design

### 1. Soft Constraints

The network directly predicts the solution:

$$
u_\theta(x,t) = N_\theta(x,t).
$$

Initial and boundary conditions are enforced through loss penalties:

$$
L_{\text{soft}} = L_c + L_r,
$$

where $L_c$ is the mean squared initial/boundary error and $L_r$ is the mean squared PDE residual.

### 2. Hard Constraints

The network output is transformed:

$$
u_\theta(x,t)
=
-\sin(\pi x)
+
t(1-x^2)N_\theta(x,t).
$$

This construction satisfies the prescribed initial and boundary conditions analytically, up to floating-point roundoff.

Training therefore minimizes only the PDE residual:

$$
L_{\text{hard}} = L_r.
$$

The residual is calculated by differentiating the complete transformed output.

### 3. NTK-Weighted Soft Constraints

The soft loss is extended to:

$$
L_{\text{soft,NTK}}
=
\lambda_c L_c + \lambda_r L_r.
$$

The weights are adapted using empirical NTK traces to investigate whether balancing the loss components improves training.

## Neural Tangent Kernel Analysis

For a vector of model outputs or residuals $q_\theta$, the empirical NTK is:

$$
J_q = \frac{\partial q_\theta}{\partial \theta},
\qquad
K_q = J_q J_q^\top.
$$

The analysis considers constraint, PDE-residual, and interior-output kernels on fixed probe points.

For mean-square losses, Jacobian rows are normalized by the square root of the corresponding probe count. Kernel diagnostics include:

- Eigenvalue spectra.
- Kernel traces.
- Relative changes from initialization.
- Differences between soft and hard residual kernels.

The hard model has a zero constraint Jacobian because its constraints are satisfied by construction. NTK loss balancing therefore applies to the soft model, while kernel analysis applies to both formulations.

## Dataset

The reference dataset is `burgers_shock.mat` from the original PINNs repository:

https://github.com/maziarraissi/PINNs/blob/master/appendix/Data/burgers_shock.mat

It contains spatial coordinates, time coordinates, and reference solution values.

The reference solution is used to evaluate prediction accuracy. Training uses initial/boundary observations and PDE collocation points rather than interior reference labels.

The dataset belongs to its original authors. Preserve their attribution and applicable upstream license terms when redistributing it.
