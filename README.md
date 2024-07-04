# Slow Invariant Manifolds of Fast-Slow Systems of ODEs with Physics-Informed Neural Networks

*If you use or modify for research purposes this software, please cite our paper as below:*

**Patsatzis, D. G., Russo, L., & Siettos, C. (2024). A physics-informed neural network method for the approximation of slow invariant manifolds for the general class of stiff systems of ODEs. arXiv preprint arXiv:2403.11591.**

*Under review in SIAM Journal of Applied Dynamical Systems*

Last updated by Patsatzis D. G., 4 Jul 2024

We present a ``Physics-Informed Machine-Learning`` method for the approximation of ``Slow Invariant Manifolds`` for the general class of ``Fast-Slow dynamical systems`` of ODEs.
The provided functionals are in an 
Our approach, simultaneously decomposes the vector field into fast and slow components and provides a functional of the ``Slow Invariant Manifold`` (SIM) in an explicit form that facilitates the construction and numerical integration of ``Reduced Order Models`` (ROMs).
We use ``Feedforward Neural Networks`` (FNNs) with symbolic differentiation to solve the PDE corresponding to the ``Invariance Equation`` (IE) within the ``Geometric Singular Perturbation Theory`` (GSPT) framework.

The performance of the scheme is compared with analytic/numeric SIM approximations provided by the well-established GSPT methods of ``Computational Singular Perturbation``, ``Partial Equilibirum Approximation`` and ``Quasi-Steady State Approximation``.

For illustration, we provide three benchmark problems: the Michaelis-Menten (MM), the Target Mediated Drug Disposition (TMDD) reaction mechanism, and a fully Competitive Substrate Inhibition (fCSI) mechanism.

Keywords: Physics-Informed Neural Networks, Slow invariant manifolds, Fast-Slow dynamical systems

DISCLAIMER:
This software is provided "as is" without warranty of any kind.

# Software outline

The three benchmark problems (MM, TMDD and fCSI systems) are include in folders MM_src, TMDD_src and fCSI_src, respenctively.
Each folder contains 2 main routines (for the fCSI system, one has to select the original state variables or the transformed variables case):
1) createData.m that creates the training/validation and testing data sets; random initial conditions, integration, restriction to the desired domain and the CSP-criterion for the dimension of the SIM are implemented therein, and
2) PI_SLFNN_LinearTrans_<case_name>.m that that solves the PINN optimization problem using single layer FNNs and symoblic differentiation; see Algorithm 1 in the manuscript

<case_name>: MM1, MM2, MM3 for the 3 parameter sets considered for the MM mechanism, P2 and P4 for the 2 SIMs of the TMDD mechanism and empty for the fCSI systems for both original and transformed variables

Dependancies:

The createData.m main routine depends on get"Problem_name"solGRID.m for producing trajectories in the desired domain.

The PI_SLFNN_LinearTrans_<case_name>.m routines require (i) the training and test data sets (constructed from createData.m), (ii) the RHS and Jacobian of the given problem (provided as external functions) and (iii) the analytic/numeric SIM approximations provided by GSPT in <sys_name_>knownSIMS.m

**Reproducing our results**

For each general fast-slow systems of ODEs 
(i)   run createData.m to get the training and test data sets 
(ii)  run PI_SLFNN_LinearTrans_<case_name>.m to train the PINN scheme, obtain metrics on training and evaluate its accuracy by comparing with the GSPT SIM approximations provided in the Appendix
(iii) run PlotScript.m to reproduce the plots in the manuscript (select figure ID)

# Finding SIM approximations with the PINN scheme for your own fast/slow dynamical system

The following is an outline of the steps you would need to follow
1) Given your fast-slow system, set up the RHS and Jacobian external functions 
2) Define the domain $\Omega \times I$ where you want to derive a SIM approximation (in routine createData.m)
3) Create the training/validation and test data sets using a set of appropriate initial conditions.
4) Set hyperparameters for the solution of the PINN scheme 
5) Run many runs in PI_SLFNN_LinearTrans_<case_name>.m to select best learned parameters (according to validation set)
6) For comparison with GSPT-based SIM approximation, you need to derive the analytic/numeric approximations based on your problem
7) Produce your own plots for visualization