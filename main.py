"""Main file 22/12/2024"""

import senescence_model as sm
import fibrosis_model as fm
import numpy as np
import matplotlib.pyplot as plt
import plotting
def main():
    min2day = 24*60 # Constants for converting between times
    day2min = 1/min2day
    ## For senescence ##
    params = [0.15/365, 0.27, 1.1,0.14]
    initial_state = 1000
    t = np.linspace(0, 10000, 10000)
    cell_sm = sm.senescence_model(params,initial_state)
    sol = cell_sm.solve(t)
    sol_stoch = cell_sm.euler_maruyama() 
    plt.figure()
    plt.plot(sol_stoch[0],sol_stoch[1],label='Euler-Maruyama')
    plt.ylabel("Number SC")
    plt.xlabel("Time (Days)")
    plt.title("Stoch Sim, Euler-Maruyama")
    plt.plot(sol.t,sol.y[0],label = 'without stoch')
    plt.legend()
    plt.show()
    initial_state_fm = [10,10,10,10]
    #params_fm = np.random.randint(0,10,13)
    #lam1,lam2,mu1,mu2,K,k1,k2,beta1,beta2,beta3,alpha1,alpha2,gamma
    params_fm = [0.9,0.8,0.3,0.3,470*min2day,70*min2day,240*min2day,940*min2day,510*min2day,2,1E9,1E9,1E6]
    cell_fm = fm.fibrosis_model(params_fm,initial_state_fm)
    """     sol_fm = cell_fm.solve(t)
    print(sol_fm)
    plt.plot(sol_fm.t,sol_fm.y[0])
    plt.xlabel("Time")
    plt.ylabel("Number/Fibrotic Cells")
    plt.show() """
    M_test = 100000
    F_test = 100000
    #print(cell_fm.lam2,cell_fm.mu2)
    nullcline_M = cell_fm.nullclines_M(M_test)
    nullcline_F = cell_fm.nullclines_F(F_test)
    print(f"nullclines_M({M_test}) = {nullcline_M}")
    print(f"nullclines_F({F_test}) = {nullcline_F}")
    X_test = [M_test,F_test]
    fixed_points = cell_fm.subtract_nulls(X_test)
    print(f"find_fixed_points({X_test}) = {fixed_points}")

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


    plt.figure()
    plt.plot(cell_fm.nullclines_M(mFM_space)[0], cell_fm.nullclines_M(mFM_space)[1], 'r', label = 'Macrophage nullcline')
    #plt.plot(nullcline_mF(mFM_space)[0], nullcline_mF(mFM_space)[1], 'g')
    #We have poles around 6.64*10^5 and 10^6
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

    print("DONE")
if __name__ == '__main__':
    main()