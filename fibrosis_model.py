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

    def equations(self, t, y):
        M, F, C, P = y
        dFdt = F * (self.lam1 * (P / (self.k1 + P)) * (1 - (F / self.K)) - self.mu1)
        dMdt = M * (self.lam2 * (C / (self.k2 + C)) - self.mu2)
        dCdt = self.beta1 * F - self.alpha1 * M * (C / (self.k2 + C)) - self.gamma * C
        dPdt = self.beta2 * M + self.beta3 * F - self.alpha2 * F * (P / (self.k1 + P)) - self.gamma * P
        return np.array([dFdt,dMdt,dCdt,dPdt])
    def threshold_event(self,t,y):
        return y[0] #Stop when mF reaches zero
    def solve(self,t):
        def threshold(t,y):
            return self.threshold_event(t,y)
        threshold.terminal = True
        threshold.direction = 0
        initial_conditions = np.array([self.stateM, self.stateF, self.stateP, self.stateC])

        sol = solve_ivp(self.equations, [t[0], t[-1]], initial_conditions, t_eval=t)#,events = (threshold))
        return sol
