"""Main file 22/12/2024"""

import senescence_model as sm
import fibrosis_model as fm
import numpy as np
import time
import scipy.integrate 
import matplotlib.pyplot as plt
import plotting as ping
import state_parameter_maker as spm
import matplotlib as mpl
from brokenaxes import brokenaxes
#mpl.rcParams['figure.figsize'] = [5, 3]
mpl.rcParams['figure.dpi'] = 150
mpl.rcParams['savefig.dpi'] = 150
def main():
    t1 = time.time()
    ## For senescence ##
    params_sn = [0.15/365, 0.27, 1.1,0.14]
    initial_state = 100
    t = np.linspace(0, 100, 10000)
    cell_sm = sm.senescence_model(params_sn,initial_state)
    sol = cell_sm.solve(t)
    sol_stoch = cell_sm.euler_maruyama() 
    ## For fibrosis
    initial_state_fm = [10,10,10,10]
    # Constants for converting between times
    min2day = 24*60 
    day2min = 1/min2day

    # Order for params below :lam1,lam2,mu1,mu2,K,k1,k2,beta1,beta2,beta3,alpha1,alpha2,gamma
    params_fm = [0.9,0.8,0.3,0.3,1E6,1E9,1e9,470*min2day,70*min2day,240*min2day,940*min2day,510*min2day,2]
    cell_fm = fm.fibrosis_model(params_fm,initial_state_fm)




    # straight lines to replace/ignore the sharp increase near the poles
    xsmooth1 = [10**5.7, 10**5.85]
    ysmooth1 = [cell_fm.nullclines_F(pt)[1] for pt in xsmooth1]

    fp1 = cell_fm.fixed_points(initial_guess=np.array([1e4, 1e4]))
    fp1 = cell_fm.perturb_fixed_point(fp1)

    fp2 = cell_fm.fixed_points(initial_guess=np.array([5e5, 5e5]))
    fp2 = cell_fm.perturb_fixed_point(fp2)

    fp3 = cell_fm.fixed_points(initial_guess=np.array([0, 1e4]))
    fp3 = cell_fm.perturb_fixed_point(fp3)

    fp4 = cell_fm.fixed_points(initial_guess=np.array([0, 2e5]))
    fp4 = cell_fm.perturb_fixed_point(fp4)

    fp5 = cell_fm.fixed_points(initial_guess=np.array([0, 0]))
    fp5 = cell_fm.perturb_fixed_point(fp5)
    # Create broken axis: y-limits are split into two ranges
    plt.scatter(fp1[1],fp1[0],color = 'b',alpha=0.5,label='Unstable Fixed Points')
    plt.scatter(fp2[1],fp2[0],color = 'r', alpha=0.5,label='Hot Fibrosis (Stable Fixed Point)')
    plt.scatter(fp3[1],fp3[0],color = 'b',alpha=0.5)
    plt.scatter(fp4[1],fp4[0],color = 'purple',alpha=0.5,label='Cold Fibrosis (Semi-Stable Fixed Point)')
    plt.scatter(fp5[1],fp5[0],color = 'g',s=60,alpha=0.5,label='Healing (Stable Fixed Point)')
    # mFnull1, mFnull2, mFnull3 are intervals that do not contain any poles
    FM_space = np.logspace(-2, 7, 10**4)
    Fnull1 = np.logspace(-2, 5.7, 10**3)
    Fnull2 = np.logspace(5.85, 5.95, 10**3)
    Fnull3 = np.logspace(6.05, 7, 10**3)
    xsmooth1 = [10**5.7, 10**5.85]
    ysmooth1 = [cell_fm.nullclines_F(pt)[1] for pt in xsmooth1]
    plt.plot(xsmooth1, ysmooth1, 'b')
    plt.plot(cell_fm.nullclines_M(FM_space)[0], cell_fm.nullclines_M(FM_space)[1], 'r', label = 'Macrophage nullcline')
    plt.plot(cell_fm.nullclines_F(Fnull1)[0], cell_fm.nullclines_F(Fnull1)[1], 'b', label = 'Myofibroblast nullcline')
    plt.plot(cell_fm.nullclines_F(Fnull2)[0], cell_fm.nullclines_F(Fnull2)[1], 'b')
    plt.plot(cell_fm.nullclines_F(Fnull3)[0], cell_fm.nullclines_F(Fnull3)[1], 'b')
    plt.legend(fontsize=7.5, loc='upper left')  # Adjust as needed    plt.show()
    plt.xlim((0.01, max(Fnull3)))
    plt.ylim((0.01, max(FM_space)))
    plt.xlabel("Fibroblast Concentration")
    plt.ylabel("Macrophage Concentration")
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


    nullclines_M = cell_fm.nullclines_M(FM_space)
    nullclines_F1 = cell_fm.nullclines_F(Fnull1)
    nullclines_F2 = cell_fm.nullclines_F(Fnull2)
    nullclines_F3 = cell_fm.nullclines_F(Fnull3)
    nullclines_plot = [nullclines_M,nullclines_F1,nullclines_F2,nullclines_F3,[xsmooth1,ysmooth1]]
    colours = ['orange','b','b','b','b']

    plot = ping.Plotter(cell_fm)
    #plot.plot_scatter([fixed_x,fixed_y],log=[True,True],show=False,lims=[[1,1e7],[1,1e7]])#Order matters
    #plot.plot_series(nullclines_plot,log=[True,True],lims=[[1,1e7],[1,1e7]],colours=colours)

    ### Plot trajectories ###
    x_traj = np.array([1e6,1e6])
    t_traj = np.arange(0,50,0.1)
    vals = [0.00000001,0.0001,0.001,0.01,0.1,1,10]#Length determines no. trajectories
    vcols = plt.cm.autumn_r(np.linspace(0.3, 1., len(vals)))  # Colours for each trajectory

    # Plot trajectories
    for v, col in zip(vals, vcols):
        x_traj_new = [x_traj[0], v * x_traj[1]]  # Starting point
        #print(x_traj_new)
        X = scipy.integrate.solve_ivp(cell_fm.change_in_m_f_to_int,(t[0],t[-1]), x_traj_new,t_eval=t)  # Integrate the system
    
    # define a grid and compute direction at each point
    x = np.logspace(-2, 7, 35) #np.linspace(0, xmax, nb_points)
    y = np.logspace(-2, 7, 35) #np.linspace(0, ymax, nb_points)

    X1 , Y1  = np.meshgrid(x, y)                       # create a grid

    DY1, DX1 = np.vectorize(lambda M, F: cell_fm.change_in_m_f(M, F), otypes=[float, float])(Y1, X1)    
    DX1 = DX1/X1
    DY1 = DY1/Y1
    M = (np.hypot(DX1, DY1))                           # Norm of the growth rate 
    M[ M == 0] = 1.                                 # Avoid zero division errors 
    DX1 /= M                                        # Normalize each arrows
    DY1 /= M
    log_M = np.log10(M+1e-9)
    Q = plt.quiver(X1, Y1, DX1, DY1, log_M, pivot='mid', cmap=plt.cm.jet)
    cbar = plt.colorbar()
    cbar.set_label('Rate of Change (Normalised)', fontsize=12)  # Add a label to the colorbar

#### Perturbation ####
    pulses = [(0,5,10**4.8),(2,5,-1e4)]
    #pulses =[(10,15,10**5.4)] #10**5.4 for just past, below for healin
    #pulses = [(0,5,10**4),(10,15,10**4),(20,25,10**4),(30,35,10**4),(40,45,10**4)]#,(70,130,-10**5)]
   # pulses=[(0,5,10**4.8),(12,17,-3e3)]#Treatment earlier leads to worse outcome 1 v 2 start
    t = np.linspace(0, 100,  1000)
    t_sep = np.linspace(0,150,1000)
    run1_state = [0,0,0,0]
    #intitial conditions, start with small amount of F or only M population changes
    run1 = fm.fibrosis_model(params_fm,run1_state)
    X0 = np.array([0, 50])
    X = scipy.integrate.solve_ivp(run1.constant_injury,(t[0],t[-1]),X0,t_eval=t,args = (pulses,))
    sep_traj = cell_fm.separatrix_traj_neg(t_sep,fp1,epsilon=0.1)
    #print(sep_traj[-20:-1])
    sep_traj_2 = cell_fm.separatrix_traj_neg(t_sep,fp1,epsilon=-0.1)
    plt.xlabel('Myofibroblast Conc. Cells/ml')
    plt.ylabel('Macrophage Conc. Cells/ml')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(0.01, max(x))
    plt.ylim(0.01, max(y))
    plt.plot(X.y[1],X.y[0],color='maroon',label='Trajectory')
    print(X.y)
    print("Time to settle: ",cell_fm.time_to_state(X.y,X.t))
    plt.scatter(fp1[1],fp1[0],color = 'b',alpha=0.5,label='Unstable Fixed Point')
    plt.scatter(fp2[1],fp2[0],color = 'r', alpha=0.5,label='Hot Fibrosis (Stable Fixed Point)')
    plt.scatter(fp3[1],fp3[0],color = 'b',alpha=0.5)
    plt.scatter(fp4[1],fp4[0],color = 'purple',alpha=0.5,label='Cold Fibrosis (Sem-Stable Fixed Point)')
    plt.scatter(fp5[1],fp5[0],color = 'g',alpha=0.5,label='Healing (Stable Fixed Point)')
    plt.plot(sep_traj[1],sep_traj[0],'k',label="Separatrix",alpha=0.6)
    plt.plot(sep_traj_2[1],sep_traj_2[0],'k',alpha=0.6)
    plt.legend(fontsize=5, loc='upper left',markerscale=0.8,framealpha=0.9)  # Adjust as needed    plt.show()
    plt.show()

    ### Trajectories###
    t = np.linspace(0, 100,  1000)
    run1_state = [0,0,0,0]
    pulses_1 = [(0,5,10**4.8),(2,6,-1e4)]#,(70,130,-10**5)]
    #pulses_2 = [(0,5,10**4),(10,15,10**4),(20,25,10**4),(30,35,10**4),(40,45,10**4)]
    pulses_2 = [(0,5,10**4.8),(1,4,-1e4)]
    #intitial conditions, start with small amount of F or only M population changes
    run1 = fm.fibrosis_model(params_fm,run1_state)
    X0 = np.array([0, 50])
    X1 = run1.solve_constant_injury(t,X0,pulses_1)
    plt.plot(X1.t,X1.y[1],label='Myofibroblast, Later Treatment')
    plt.plot(X1.t,X1.y[0],label='Macrophage, Later Treatment')
    plt.xlabel("Time (days)")
    plt.ylabel("Concentration (cells/ml)")
    X2= run1.solve_constant_injury(t,X0,pulses_2)
    plt.plot(X2.t,X2.y[1],label='Myofibroblast, Early Treatment')
    plt.plot(X2.t,X2.y[0],label='Macrophage, Early Treatment')
    plt.legend()
    plt.show()
    t2=time.time()
    print(f"DONE: Total runtime: {t2-t1}s")

if __name__ == '__main__':
    main()