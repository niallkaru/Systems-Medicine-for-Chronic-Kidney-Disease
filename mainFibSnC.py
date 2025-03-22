"""Main file 22/12/2024"""

import senescence_model as sm
import fibrosis_model as fm
import fibrosis_senescence_model as fsm
import numpy as np
import time
import scipy.integrate 
import matplotlib.pyplot as plt
import plotting as ping
import state_parameter_maker as spm

def main():
    #print("This is running")
    t1 = time.time()
    # Constants for converting between times
    min2day = 24*60 
    day2min = 1/min2day
    ## For fibrosis
    initial_state_fsm = spm.state_fsm(10,10,10,10,10)
    params_fsm = spm.params_fsm(lam1=0.9,lam2=0.8,mu1=0.3,mu2=0.3,K=1e6,k1=1e9,\
                                k2=1e9,beta1=470*min2day,beta2=70*min2day,\
                                beta3=240*min2day,alpha1=940*min2day,alpha2=510*min2day,\
                                gamma=2,n=0.0003,h=50,r=1e2,q=1.1)

    cell_fsm = fsm.fibrosis_senescence_model(params_fsm,initial_state_fsm)


    FM_space = np.logspace(0, 7, 10**4)

    # mFnull1, mFnull2, mFnull3 are intervals that do not contain any poles
    Fnull1 = np.logspace(0, 5.7, 10**3)
    Fnull2 = np.logspace(5.85, 5.95, 10**3)
    Fnull3 = np.logspace(6.05, 7, 10**3)
    plt.plot(cell_fsm.nullclines_M(FM_space)[0], cell_fsm.nullclines_M(FM_space)[1], 'r', label = 'Macrophage nullcline')
    #plt.plot(nullcline_mF(mFM_space)[0], nullcline_mF(mFM_space)[1], 'g')
    #We have poles around 6.64*10^5 and 10^6

    plt.plot(cell_fsm.nullclines_F(Fnull1)[0], cell_fsm.nullclines_F(Fnull1)[1], 'b', label = 'Myofibroblasts nullcline')
    plt.plot(cell_fsm.nullclines_F(Fnull2)[0], cell_fsm.nullclines_F(Fnull2)[1], 'b')
    plt.plot(cell_fsm.nullclines_F(Fnull3)[0], cell_fsm.nullclines_F(Fnull3)[1], 'b')
    plt.legend()
    plt.xlabel("Fibroblasts")
    plt.ylabel("Macrophages")
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    #### Perturbation ####
#     #pulses_M =[(0,1,5e4),(6,9,1e4)]
#     pulses_M=[(0,0,0)]
#     pulses_S =[(0,2,100)]#,(10,14,-1e3)]
#    # pulses_S=[(0,0,0)]
#     t = np.linspace(0, 50, 50)
#     t_sep = np.linspace(0,75,1000)
#     t_sep_neg = np.linspace(0, 75, 1000)
#     #intitial conditions, start with small amount of F or only M population changes
#     X0 = np.array([0, 50, 1e5])
#     #print("Up to integral grand")
#     X = cell_fsm.constant_injury(t,X0,pulses_M,pulses_S)
#     #print("Here OK")
#     plt.plot(X.t,X.y[1],label = 'Fibroblasts')
#     plt.plot(X.t,X.y[0],label = "Macrophages")
#     plt.plot(X.t,X.y[2],label='Senescent cells')
#     plt.legend()
#     plt.xlabel("t")
#     plt.ylabel("Cell Count")
#     plt.show()
#     plt.plot(X.y[1],X.y[0],label = 'Trajectory')
#     plt.xscale('log')
#     plt.yscale('log')
#   plt.show()


    params_fsm2 = spm.params_fsm(lam1=0.9,lam2=0.8,mu1=0.3,mu2=0.3,K=1e6,k1=1e9,\
                                k2=1e9,beta1=470*min2day,beta2=70*min2day,\
                                beta3=240*min2day,alpha1=940*min2day,alpha2=510*min2day,\
                                gamma=2,n=1e2,h=50,r=0.0003,q=1.1)
    cell_fsm = fsm.fibrosis_senescence_model(params_fsm2,initial_state_fsm)

    pulses_M=[(0,0,0)]
    pulses_S =[(0,2,0)]#,(10,14,-1e3)]
    t = np.linspace(0, 50, 50)
    #intitial conditions, start with small amount of F or only M population changes
    X0 = np.array([0, 50, 1e5])

    X = cell_fsm.constant_injury(t,X0,pulses_M,pulses_S)
    plt.plot(X.t,X.y[1],label = 'Fibroblasts 1')
    plt.plot(X.t,X.y[0],label = "Macrophages 1")
    plt.plot(X.t,X.y[2],label='Senescent cells 1')
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("Cell Count")
    plt.xscale('log')
    plt.yscale('log')

    params_fsm_1 = spm.params_fsm(lam1=0.9,lam2=0.8,mu1=0.3,mu2=0.3,K=1e6,k1=1e9,\
                                k2=1e9,beta1=470*min2day,beta2=70*min2day,\
                                beta3=240*min2day,alpha1=940*min2day,alpha2=510*min2day,\
                                gamma=2,n=0.0003,h=50,r=1e2,q=1.1)

    cell_fsm2 = fsm.fibrosis_senescence_model(params_fsm_1,initial_state_fsm)
    X1 = cell_fsm2.constant_injury(t,X0,pulses_M,pulses_S)
    #print("Here OK")
    plt.plot(X1.t,X1.y[1],label = 'Fibroblasts 2')
    plt.plot(X1.t,X1.y[0],label = "Macrophages 2")
    plt.plot(X1.t,X1.y[2],label='Senescent cells 2')
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("Cell Count")
    #plt.plot(X.y[1],X.y[0],label = 'Trajectory')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    t = np.linspace(0, 25, 100)
    ### Heat map ###
    #cell_fsm.snc_param_heatmap(t,X0)
    t2=time.time()
    print(f"DONE: Total runtime: {t2-t1}s")

if __name__ == '__main__':
    main()