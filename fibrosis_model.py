import numpy as np
from scipy.integrate import solve_ivp
"""Niall Karunaratne 22/12/2024
Class for fibrosis model for cells (within kidney)
Adapted from Principles of Cell Circuits for Tissue Repair and Fibrosis Adler et al. 2020"""

class fibrosis_model:
    def __init__(self,params,initial_state):
        """"Extract several million parameters (is there a better way?)"""
        self.stateM = initial_state[0]
        self.stateF = initial_state[1]
        self.stateP = initial_state[2]
        self.stateC = initial_state[3]

        self.lam1 = params[0]
        self.mu1 = params[1]
        self.lam2 = params[2]
        self.mu2 = params[3]
        self.beta1 = params[4]
        self.beta2 = params[5]
        self.beta3 = params[6]
        self.alpha1 = params[7]
        self.alpha2 = params[8]
        self.gamma = params[9]
        self.k1 = params[10]
        self.k2 = params[11]
        self.K = params[12]

    def equations(self,t,y,lam1,lam2,mu1,mu2,beta1,beta2,beta3,alpha1,alpha2,gamma,k1,k2,K):
        M, F, C, P = y
        dFdt = F*(lam1*(P/(k1+P))*(1-(F/K))-mu1)
        dMdt = M*(lam2*(C/(k2+C))-mu2)
        dCdt = beta1*F - alpha1*M*(C/(k2+C))-gamma*C
        dPdt = beta2*M+beta3*F-alpha2*F*(P/(k1+P))-gamma*P
        return np.array([dFdt,dMdt,dCdt,dPdt])
    
    def solve(self,t):
        initial_conditions = np.array([self.stateM, self.stateF, self.stateP, self.stateC])

        sol = solve_ivp(self.equations, [t[0], t[-1]], initial_conditions, t_eval=t, args=(self.lam1, self.lam2, self.mu1, self.mu2, self.beta1, self.beta2, self.beta3, self.alpha1, self.alpha2, self.gamma, self.k1, self.k2, self.K))
        return sol
