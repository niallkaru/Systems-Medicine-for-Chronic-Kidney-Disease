"""Main file 22/12/2024
This script models the dynamics of fibrosis and senescence in a biological system. It includes:
1. Fixed-point analysis to identify steady states.
2. Vector field visualization in 3D.
3. Simulation of system trajectories under different conditions.
4. Data fitting to experimental results.
"""

import senescence_model as sm
import fibrosis_model as fm
import fibrosis_senescence_model as fsm
import numpy as np
import time
import scipy.integrate 
import matplotlib.pyplot as plt
import plotting as ping
import state_parameter_maker as spm
import pandas as pd
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from sys import exit
import matplotlib.cm as cm
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 150

def main():
    """Main function to execute the fibrosis-senescence model analysis."""

    t1 = time.time()  # Start runtime measurement
    fit = False
    # Constants for converting between time units
    min2day = 24 * 60  # Minutes to days
    day2min = 1 / min2day  # Days to minutes

    # Initialize the model with initial states and parameters
    initial_state_fsm = spm.state_fsm(0, 10, 10, 10, 100)  # Initial cell populations, M, F, P, C, S
    params_fsm = spm.params_fsm(lam1=0.9, lam2=0.8, mu1=0.3, mu2=0.3, K=1e6, k1=1e9,k2=1e9,
                                beta1=470 * min2day,beta2=70 * min2day,beta3=240 * min2day,
                                alpha1=940 *min2day,alpha2=510 *min2day, gamma=2,
                                  n=0.003,h=50, r=24, q=1e4)  # Parameters for the model

    # Create the fibrosis-senescence model instance
    cell_fsm = fsm.fibrosis_senescence_model(params_fsm, initial_state_fsm)
    cell_fsm.nullclines_3D(-2,7,50)
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection="3d")
    # … draw your 3D nullclines & mesh …

    # now draw the separatrix, auto-finding the saddle:
    print("3D Separatrix")
    cell_fsm.plot_separatrix_surface_fd(
    M_range=(0.01,1e7),
    F_range=(0.01,1e7),
    S_range=(0.01,1e7),
    resolution=80,
    t_end=200,
    threshold=1.0
    )
    cell_fsm.plot_separatrix_finite_difference_parallel(
        M_range=(1e2, 1e6),
        F_range=(1e2, 1e6),
        S_range=(1e2, 1e6),
        resolution=20,
        t_end=100
    )

    sep3d = cell_fsm.plot_separatrix_surface_3D()

    # finalize axes
    ax.set_xlabel("log₁₀(Macrophages)")
    ax.set_ylabel("log₁₀(Myofibroblasts)")
    ax.set_zlabel("log₁₀(Senescent Cells)")
    ax.legend()
    plt.show()
    """
    FIXED POINTS
    """

    fixed_S = 2000
    fixed_pts = cell_fsm.fixed_pt_sweep([-2,7],[-2,7],fixed_S)
    print(fixed_pts)
    # Collect results
    pty, ptx = [], []
    labels = []
    colors = []

    for M, F in fixed_pts:
        S = fixed_S
        info = cell_fsm.classify_slice(F, M, S, fixed={'S': S})
        verdict = info['verdict']

        # Choose colour and label based on verdict
        if "stable" in verdict.lower() and "saddle" not in verdict.lower():
            color = 'red'
            label = 'Stable'
        elif "saddle" in verdict.lower():
            color = 'purple'
            label = 'Saddle'
        else:
            color = 'blue'
            label = 'Unstable'

        ptx.append(F)
        pty.append(M)
        colors.append(color)
        labels.append(label)
    print("Fixed points Calculated")
    print(ptx,pty)
################
#%%
    """
    3D Quiver
    """
    # Define ranges for variables (adjust according to your system)
    M_vals = np.logspace(-2, 7, 15)  # Macrophages range
    F_vals = np.logspace(-2, 7, 15)  # Myofibroblasts range
    S_vals = np.logspace(-2, 7, 15)  # Senescent cells range

    # Create a 3D grid for (M, F, S)
    Mg, Fg, Sg = np.meshgrid(M_vals, F_vals, S_vals, indexing='ij')

    # Initialize derivative arrays for the vector field
    U = np.zeros(Mg.shape)  # dM/dt
    V = np.zeros(Mg.shape)  # dF/dt
    W = np.zeros(Mg.shape)  # dS/dt

    print("Initialisation")
    # Compute the vector field at each grid point
    for i in range(Mg.shape[0]):
        for j in range(Mg.shape[1]):
            for k in range(Mg.shape[2]):
                state = [Mg[i, j, k], Fg[i, j, k], Sg[i, j, k]]  # Current state
                dstate = cell_fsm.residual_fixed_point(state)  # Compute derivatives
                U[i, j, k] = dstate[0]  # dM/dt
                V[i, j, k] = dstate[1]  # dF/dt
                W[i, j, k] = dstate[2]  # dS/dt
    print("Vector Field done")
    # Normalize the vector lengths for clarity in visualization
    norm = np.sqrt(U**2 + V**2 + W**2)  # Compute vector magnitudes
    norm[norm == 0] = 1.  # Avoid division by zero
    U_norm = U / norm  # Normalise dM/dt
    V_norm = V / norm  # Normalise dF/dt
    W_norm = W / norm  # Normalise dS/dt
    print("field lengths normalised")
    # Flatten the grid and vector field for plotting
    X = np.log10(Mg.flatten())  # Log-transformed macrophages
    Y = np.log10(Fg.flatten())  # Log-transformed fibroblasts
    Z = np.log10(Sg.flatten())  # Log-transformed senescent cells
    U_flat = U_norm.flatten()  # Flattened dM/dt
    V_flat = V_norm.flatten()  # Flattened dF/dt
    W_flat = W_norm.flatten()  # Flattened dS/dt
    mag = norm.flatten()  # Flattened magnitudes
    print("flattened")
# Normalise magnitudes to [0, 1] for colormap
    mag_norm = (mag - mag.min()) / (mag.max() - mag.min())
    print("normalised")
# Create the 3D quiver plot for the vector field
    cmap = cm.plasma  # Choose colormap
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for xi, yi, zi, ui, vi, wi, m_val in zip(X, Y, Z, U_flat, V_flat, W_flat, mag_norm):
        col = cmap(m_val)  # Map magnitude to color
        ax.quiver(xi, yi, zi, ui, vi, wi, color=col, length=0.4, linewidth=0.5, alpha=0.5)
    print("QPlot created")
    ax.set_xlabel("Macrophages (log₁₀ M)")
    ax.set_ylabel("Myofibroblasts (log₁₀ F)")
    ax.set_zlabel("Senescent Cells (log₁₀ S)")
    plt.title("3D Quiver Plot of the Vector Field with Fixed Points")
    ax.legend()  # Add legend for fixed points
    plt.show()  # Display the plot
    #%%
#####################################
    """
    2D Quiver
    """
    # Define your grid using np.logspace.

   # plt.xscale('log')
    #plt.yscale('log')

    # Call the method on your model instance.

    sep_pos_1,sep_neg_1 = cell_fsm.separatrix(np.linspace(0,100,50),(ptx[1],pty[1],2000),epsilon=10)
    sep_pos_2,sep_neg_2 = cell_fsm.separatrix(np.linspace(0,100,50),(pty[1],ptx[1],2000),epsilon=100)

    plt.plot(sep_neg_2[1],sep_neg_2[0],'g')
    plt.plot(sep_pos_2[1],sep_pos_2[0],'g')
    #print("Fixed point in 3D (M, F, S):", fp1)
    plt.scatter(ptx[0],pty[0],color='g',label="Stable Healing")
    plt.scatter(ptx[1],pty[1])
    plt.scatter(ptx[2],pty[2],color='r',label='Stable Hot Fibrosis Point')

    # mFnull1, mFnull2, mFnull3 are intervals that do not contain any poles
    FM_space = np.logspace(-2, 7, 10**3)
    Fnull1 = np.logspace(-2, 5.7, 10**3)
    Fnull2 = np.logspace(5.85, 5.95, 10**3)
    Fnull3 = np.logspace(6.05, 7, 10**3)
    xsmooth1 = [10**5.7, 10**5.85]
    ysmooth1 = [cell_fsm.nullclines_F(pt)[1] for pt in xsmooth1]

    F_vals, M_vals = np.vectorize(cell_fsm.nullclines_M, otypes=[float, float])(FM_space)
    plt.plot(F_vals, M_vals, 'r', label='Macrophage nullcline')
    # print(F_vals)

    plt.plot(xsmooth1, ysmooth1, 'b')
    plt.plot(cell_fsm.nullclines_F(Fnull1)[0], cell_fsm.nullclines_F(Fnull1)[1], 'b', label = 'Myofibroblasts nullcline')
    plt.plot(cell_fsm.nullclines_F(Fnull2)[0], cell_fsm.nullclines_F(Fnull2)[1], 'b')
    plt.plot(cell_fsm.nullclines_F(Fnull3)[0], cell_fsm.nullclines_F(Fnull3)[1], 'b')

    plt.legend(fontsize=5)
    plt.xlabel("Fibroblasts")
    plt.ylabel("Macrophages")
    plt.xscale('log')
    plt.yscale('log')
    M_vals = np.logspace(-2,7,40) # Myofibroblast concentrations (x-axis)
    F_vals = np.logspace(-2,7,40)  # Macrophage concentrations (y-axis)
   # M_vals = np.linspace(0,20,40)
   # F_vals = np.linspace(0,20,40)

    fixed_S = 2000  # Set a biologically relevant fixed value for Senescent cells

    cell_fsm.plot_2D_quiver_field_fixed_S(M_vals, F_vals, fixed_S,x_label="Myofibroblast Conc.",y_label="Macrophage Conc.")

####### TRAJECTORIES #######
    # pulses_M = [(0,0,0)]
    # pulses_S = [(0,0,0)]
    # t = np.linspace(0, 50, 50)
    # #intitial conditions, start with small amount of F or only M population changes
    # X0 = np.array([0, 50, 1e5])

    # params_fsm_2 = spm.params_fsm(lam1=0.9,lam2=0.8,mu1=0.3,mu2=0.3,K=1e6,k1=1e9,\
    #                             k2=1e9,beta1=470*min2day,beta2=70*min2day,\
    #                             beta3=240*min2day,alpha1=940*min2day,alpha2=510*min2day,\
    #                             gamma=2,n=0.0003,h=50,r=100,q=110)

    # cell_fsm_2 = fsm.fibrosis_senescence_model(params_fsm_2,initial_state_fsm)
    # X1 = cell_fsm_2.solve_constant_injury(t,X0,pulses_M,pulses_S)
    # plt.plot(X1.t,X1.y[1],label = 'Fibroblasts 2') #Int good idea?
    # plt.plot(X1.t,X1.y[0],label = "Macrophages 2")
    # plt.plot(X1.t,X1.y[2],label='Senescent cells 2')
    # plt.legend()
    # plt.xlabel("t")
    # plt.ylabel("Cell Count")
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()
    # plt.plot(X1.y[1],X1.y[0])
    # plt.ylabel("Macrophages")
    # plt.xlabel("Fibroblasts")
    # plt.show()
    # # fig = plt.figure()
    # # ax = fig.add_subplot(projection='3d')
    # # # Plot the trajectory with M on x, F on y, and S on z axes.
    # # ax.plot(X1.y[0], X1.y[2], X1.y[1], label="Trajectory")
    # # ax.set_xlabel("Macrophages")
    # # ax.set_ylabel("SnC")
    # # ax.set_zlabel("Fibroblasts")
    # # ax.legend()
    # # plt.show()

 
    """
    FITTING: To simplify the system, we have non-dimensionalised the system to reduce the number of parameters
    Sensitivity analysis first to find out what the more important factors are
    """
    if fit==True:
        print("Fitting")
        initial_state_fit = spm.state_fsm(10, 10, 10, 10, 10)  # Initial cell populations, M, F, P, C, S
        params_fit = spm.params_fsm(lam1=0.9, lam2=0.8, mu1=0.3, mu2=0.3, K=1e6, k1=1e9,k2=1e9,
                                    beta1=470 * min2day,beta2=70 * min2day,beta3=240 * min2day,
                                    alpha1=940 *min2day,alpha2=510 *min2day, gamma=2,
                                    n=0.0003,h=50, r=24, q=1e5)  # Parameters for the model

        fitting_fsm = fsm.fibrosis_senescence_model(params_fit, initial_state_fit)
        fitting_fsm.define_dimensionless_parameters(M0=22,S0=80) # Create dimensionless parameters, use max for M0 and s)
        data_to_fit = fitting_fsm.preprocess_data(r"C:\Users\niall\Documents\MPhys\Systems-Medicine-for-Chronic-Kidney-Disease\cell_numbers_for_model.xlsx")
        param_names = ['zeta1', 'zeta2', 'phi2', 'rho1', 'rho2', 'eta', 'chi', 'theta']
        t_span = (0, 50)        # dimensionless time
        t_eval = np.linspace(0, 50, 500)
        x0 = [0.1,1e-5,0.1]
        fitting_fsm.sensitivity_barplot_all_outputs(param_names=param_names,y0=x0,t_span=t_span,t_eval=t_eval,delta=0.05,central_diff=True,time_index=-1,show_plot=True)

    # print(data_to_fit)
        params_to_fit=['zeta1','phi2','eta','chi','theta']
        initial_guesses = [(fitting_fsm.lam1/fitting_fsm.mu1),(fitting_fsm.mu2/fitting_fsm.mu1),((fitting_fsm.n*fitting_fsm.S0)/(fitting_fsm.M0*fitting_fsm.mu1)),(fitting_fsm.h/(fitting_fsm.S0*fitting_fsm.mu1)),((fitting_fsm.r*fitting_fsm.M0)/fitting_fsm.mu1)]
        bounds_for_fit = [(0.1,20),(0.1,10),(0.10,10),(0.10,10),(0.01,200)]
        fitting_fsm.fit_and_plot(params_to_fit,initial_guesses,bounds_for_fit,data_to_fit,x0)

    t2=time.time()
    print(f"DONE: Total runtime: {t2-t1}s")

if __name__ == '__main__':
    main()