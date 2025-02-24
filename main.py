"""Main file 22/12/2024"""

import senescence_model as sm
import fibrosis_model as fm
import numpy as np
import scipy.integrate 
import matplotlib.pyplot as plt
import plotting as ping
def main():
    min2day = 24*60 # Constants for converting between times
    day2min = 1/min2day
    ## For senescence ##
    params = [0.15/365, 0.27, 1.1,0.14]
    initial_state = 100
    t = np.linspace(0, 10, 1000)
    cell_sm = sm.senescence_model(params,initial_state)
    sol = cell_sm.solve(t)
    sol_stoch = cell_sm.euler_maruyama() 
    #ping.plot_tseries(sol_stoch[0],sol_stoch[1])
    initial_state_fm = [10,10,10,10]
    #           lam1,lam2,mu1,mu2,K,k1,k2,beta1,beta2,beta3,alpha1,alpha2,gamma
    params_fm = [0.9,0.8,0.3,0.3,1E6,1E9,1e9,470*min2day,70*min2day,240*min2day,940*min2day,510*min2day,2]
    cell_fm = fm.fibrosis_model(params_fm,initial_state_fm)
    sol_fm = cell_fm.solve(t)
    #ping.plot_tseries(sol_fm.t,sol_fm.y[0])
    M_test = 100000
    F_test = 100000
    #print(cell_fm.lam2,cell_fm.mu2)
    nullcline_M = cell_fm.nullclines_M(M_test)
    nullcline_F = cell_fm.nullclines_F(F_test)
    #print(f"nullclines_M({M_test}) = {nullcline_M}")
    #print(f"nullclines_F({F_test}) = {nullcline_F}")


    mFM_space = np.logspace(0, 7, 10**4)

    # mFnull1, mFnull2, mFnull3 are intervals that do not contain any poles
    # smoothmF1 and smoothmF2 are intervals that contain poles
    mFnull1 = np.logspace(0, 5.7, 10**3)
    #smoothmF1 = np.logspace(5.7, 5.85, 10**3)
    mFnull2 = np.logspace(5.85, 5.95, 10**3)
    #smoothmF2 = np.logspace(5.95, 6.05, 10**3)
    mFnull3 = np.logspace(6.05, 7, 10**3)

    # straight lines to replace/ignore the sharp increase near the poles
    xsmooth1 = [10**5.7, 10**5.85]
    ysmooth1 = [cell_fm.nullclines_F(pt)[1] for pt in xsmooth1]
    fixed_x = cell_fm.fixed_points(np.array([1e3,1e3]))
    fixed_y = cell_fm.fixed_points(np.array([5e5,3e5]))
    ### CHECK SPACE FOR FIXED POINTS ###
    # x_space = np.logspace(0,10,num=100)
    # y_space = np.logspace(0,10,num=100)
    # fixed_points = []
    # for x in x_space:
    #     for y in y_space:
    #         fixed_point = cell_fm.fixed_points(np.array([x,y]))
    #         if np.all(fixed_point>=0):
    #             fixed_points.append(np.round(fixed_point,decimals=2))
    # fixed_points = np.unique(np.array(fixed_points),axis=0)
    # print(f'Unique fixed points {fixed_points}, length {fixed_points.shape}')
    plt.figure()
    plt.plot(cell_fm.nullclines_M(mFM_space)[0], cell_fm.nullclines_M(mFM_space)[1], 'r', label = 'Macrophage nullcline')
    #plt.plot(nullcline_mF(mFM_space)[0], nullcline_mF(mFM_space)[1], 'g')
    #We have poles around 6.64*10^5 and 10^6
    cold_fixed = cell_fm.fixed_point_cold()
    # for array in fixed_points:
    #     plt.scatter(array[1],array[0])
    plt.scatter(cold_fixed[1],1, color = 'r',label='Fixed Point: End of separatrix')
    plt.scatter(cold_fixed[0],1,color = 'purple',label='Fixed Point: Cold Fibrosis')
    plt.scatter(fixed_x[1],fixed_x[0],color = 'r' ,label='Fixed Point')
    plt.scatter(fixed_y[1],fixed_y[0],color = 'b', label = 'Fixed Point: Hot Fibrosis')
    plt.scatter(1,1,color ='b', label = 'Fixed Point: Healing')
    plt.plot(cell_fm.nullclines_F(mFnull1)[0], cell_fm.nullclines_F(mFnull1)[1], 'b', label = 'Myofibroblasts nullcline')
    plt.plot(cell_fm.nullclines_F(mFnull2)[0], cell_fm.nullclines_F(mFnull2)[1], 'b')
    plt.plot(cell_fm.nullclines_F(mFnull3)[0], cell_fm.nullclines_F(mFnull3)[1], 'b')
    plt.plot(xsmooth1, ysmooth1, 'b')

    plt.xlabel("myofibroblasts")
    plt.ylabel("macrophages")
    plt.xlim((1, 10**7))
    plt.ylim((1, 10**7))
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.show()

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
        #plt.plot(X[:, 1], X[:, 0], color=col, label='X=(%.f, %.f)' % (x_traj[1], x_traj[0]))
        # define a grid and compute direction at each point

    x = np.logspace(0, 10, 35) #np.linspace(0, xmax, nb_points)
    y = np.logspace(0, 10, 35) #np.linspace(0, ymax, nb_points)

    X1 , Y1  = np.meshgrid(x, y)                       # create a grid

    DY1, DX1 = np.vectorize(lambda M, F: cell_fm.change_in_m_f(M, F), otypes=[float, float])(Y1, X1)    
    DX1 = DX1/X1
    DY1 = DY1/Y1
    M = (np.hypot(DX1, DY1))                           # Norm of the growth rate 
    M[ M == 0] = 1.                                 # Avoid zero division errors 
    DX1 /= M                                        # Normalize each arrows
    DY1 /= M



    #-------------------------------------------------------


#### Perturbation ####
    #print(f'fixed x {fixed_x}\nfixed y {fixed_y}')
    pulses = [(0,3,5e5),(6,9,1e4)]
    t = np.linspace(0, 50,  500)
    t_sep = np.linspace(0,75,1000)
    t_sep_neg = np.linspace(0, 75, 1000)
    #intitial conditions, start with small amount of F or only M population changes
    X0 = np.array([0, 50])
    X = scipy.integrate.solve_ivp(cell_fm.constant_injury,(t[0],t[-1]),X0,t_eval=t,args = (pulses,))

    #print(f'X {X}')
    # plt.plot(X.t,X.y[1],label = 'Fibroblasts')
    plt.plot(X.t,X.y[0],label = "Macrophages")
    plt.legend()
    plt.xlabel("M")
    plt.ylabel("t")
    plt.show()
    #print(cell_fm.separatrix_eigen(fixed_y))
    print("fixed x",fixed_x)
    sep_traj = cell_fm.separatrix_traj_neg(t_sep,fixed_x,epsilon=1)
    sep_traj_2 = cell_fm.separatrix_traj_neg(t_sep_neg,fixed_x,epsilon=-1)
    print("fixed_x",fixed_x)
    plt.plot(X.y[1],X.y[0],label = 'Run1')
    plt.plot(sep_traj[1],sep_traj[0],'g',label="Separatrix")
    plt.plot(sep_traj_2[1],sep_traj_2[0],'g')
    plt.scatter(fixed_x[1],fixed_x[0])
    plt.scatter(fixed_y[1],fixed_y[0])
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc = 'best')
    plt.xlabel("F")
    plt.ylabel("M")
    plt.show()
    #print(X)

    # Drow direction fields, using matplotlib 's quiver function
    # I choose to plot normalized arrows and to use colors to give information on
    # the growth speed
    plt.title('Trajectories and direction fields')
    Q = plt.quiver(X1, Y1, DX1, DY1, M, pivot='mid', cmap=plt.cm.jet)
    plt.xlabel('F')
    plt.ylabel('M')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlim(1, max(x))
    plt.ylim(1, max(y))
    plt.plot(X.y[1],X.y[0],label = 'Run1')

    plt.scatter(fixed_x[1],fixed_x[0],alpha=0.5)
    plt.scatter(fixed_y[1],fixed_y[0],alpha=0.5)
    plt.plot(sep_traj[1],sep_traj[0],'g',label="Separatrix",alpha=0.6)
    plt.plot(sep_traj_2[1],sep_traj_2[0],'g',alpha=0.6)
    plt.show()

    print("DONE")

if __name__ == '__main__':
    main()