# Burgers’ Equation with PINNs: Soft vs. Hard Constraints and NTK Analysis

This project uses PyTorch to investigate how soft and hard enforcement of initial and boundary conditions affects physics-informed neural networks (PINNs). It also uses the empirical neural tangent kernel (NTK) to study their training behavior.

The project builds on the Burgers’ equation example from Raissi, Perdikaris, and Karniadakis (2019), with NTK analysis motivated by Wang, Yu, and Perdikaris (2022).

## Problem

We solve the one-dimensional viscous Burgers’ equation:

$$
u_t + u u_x - \frac{0.01}{\pi}u_{xx} = 0.
$$

The spatial domain is `[-1, 1]`, and the time interval is `[0, 1]`. The initial condition is `u(x, 0) = -sin(πx)`, with zero values at both spatial boundaries.

## Methods

The study compares three approaches:

| Approach | Description |
|---|---|
| Soft constraints | Initial and boundary conditions are enforced through loss penalties. |
| NTK-weighted soft constraints | Loss weights are adjusted using empirical NTK traces. |
| Hard constraints | The network output is constructed to satisfy the initial and boundary conditions automatically. |

For the hard-constrained model, the solution is constructed as:

$$
u_\theta(x,t) = -\sin(\pi x) + t(1-x^2)N_\theta(x,t).
$$

This construction removes the need for initial and boundary loss penalties. Training minimizes the PDE residual.

## NTK Analysis

The empirical NTK measures how changes in network parameters affect predictions and PDE residuals.

The analysis examines:

- Kernel eigenvalue spectra.
- Relative strengths of constraint and residual kernels.
- Kernel changes during training.
- Differences between soft and hard formulations.

NTK-based loss balancing applies to the soft model. The hard model has no constraint penalties to rebalance, but its residual kernel can still be analyzed.

## Dataset

The reference dataset, `burgers_shock.mat`, comes from the original PINNs repository:

https://github.com/maziarraissi/PINNs/blob/master/appendix/Data/burgers_shock.mat

It contains spatial coordinates, time coordinates, and reference solution values.

Training uses initial/boundary observations and PDE collocation points. Interior reference solution values are used for evaluation.

The dataset is credited to its original authors. Applicable upstream license terms should be preserved when redistributing it.
