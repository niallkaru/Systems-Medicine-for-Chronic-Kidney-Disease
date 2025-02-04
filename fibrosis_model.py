import numpy as np
from scipy.integrate import solve_ivp
import scipy.optimize as opt
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
        self.lam2 = params[1]
        self.mu2 = params[2]
        self.mu1 = params[3]
        self.K = params[4]
        self.k1 = params[5]
        self.k2 = params[6]
        self.beta1 = params[7]
        self.beta2 = params[8]
        self.beta3 = params[9]
        self.alpha1 = params[10]
        self.alpha2 = params[11]
        self.gamma = params[12]

    def equations(self, t, y):
        M, F, C, P = y
        dFdt = F * (self.lam1 * (P / (self.k1 + P)) * (1 - (F / self.K)) - self.mu1)
        dMdt = M * (self.lam2 * (C / (self.k2 + C)) - self.mu2)
        dCdt = self.beta1 * F - self.alpha1 * M * (C / (self.k2 + C)) - self.gamma * C
        dPdt = self.beta2 * M + self.beta3 * F - self.alpha2 * F * (P / (self.k1 + P)) - self.gamma * P
        return np.array([dFdt,dMdt,dCdt,dPdt])
    def nullcline_equations(self,y):
        M, F, C, P = y
        M_1 = 0
        M_2 = (self.mu2*self.k2)/(self.lam2-self.mu2)
        F_1 = 0
        F_2 = 1- (self.K*self.mu1)/(self.lam1*(P/(self.k1+P)))
        nullclines = [M_1,M_2,F_1,F_2]
        return np.array(nullclines)
    def nullclines_CP(self,M,F):
        nullclines = []
        delta_C = (self.alpha1 * M - self.k2 * self.gamma - self.beta1 * F)**2 - 4 * self.gamma * self.beta1 * F * self.k2
        delta_P = (self.beta2 * M + self.beta3 * F - self.alpha2 * F + self.gamma * self.k1)**2 - 4 * self.gamma * (self.k1 * self.beta2 * M + self.k1 * self.beta3 * F)
        C_1 = (-1 * (self.alpha1 * M - self.k2 * self.gamma - self.beta1 * F) + np.sqrt(delta_C)) / (2 * self.gamma)
        C_2 = (-1 * (self.alpha1 * M - self.k2 * self.gamma - self.beta1 * F) - np.sqrt(delta_C)) / (2 * self.gamma)
        P_1 = (-1 * (self.beta2 * M + self.beta3 * F - self.alpha2 * F + self.gamma * self.k1) + np.sqrt(delta_P)) / (2 * self.gamma)
        P_2 = (-1 * (self.beta2 * M + self.beta3 * F - self.alpha2 * F + self.gamma * self.k1) - np.sqrt(delta_P)) / (2 * self.gamma)

        if np.isreal(C_1) and C_1 >= 0:
            nullclines.append(C_1)
        else: print("C_1 not positive/real")
        if np.isreal(C_2) and C_2 >= 0:
            nullclines.append(C_2)
        else: print("C_2 not positive/real")
        if np.isreal(P_1) and P_1 >= 0:
            nullclines.append(P_1)
        else: print("P_1 not positive/real")
        if np.isreal(P_2) and P_2 >= 0:
            nullclines.append(P_2)
        else: print("P_2 not positive/real")
        return np.array(nullclines)
    def nullclines_M(self,M):
        C = (self.mu2*self.k2)/(self.lam2-self.mu2)
        F = (1/self.beta1)*((self.alpha1*M*C)/(self.k2+C))+((self.gamma*C)/self.beta1)
        return [F,M]
    def nullclines_F(self,F):
        P = (self.mu1*self.k1*self.K)/(self.lam1*self.K-F*self.lam1-self.mu1*self.K)
        M = -1*(self.beta3*F-self.alpha2*F*P/(self.k1+P)-self.gamma*P)/self.beta2
        return [F,M]
    def subtract_nulls(self,X0):
        """ Returns the one nullcline subtracted from the other accurately"""
        M0, F0 = X0
        return [np.subtract(self.nullclines_M(M0)[0],self.nullclines_F(F0)[0]), np.subtract(self.nullclines_M(M0)[1],self.nullclines_F(F0)[1])]
    def fixed_points(self):
        x = opt.fsolve(self.subtract_nulls(), np.logspace(0, 7, 7))
    def change_in_m_f_to_int(self,y, t = 0):
        """ Return the growth rate of M and F assuming steady state of c1 and c2"""
    
        M = y[0]
        F = y[1]
        CF_steady = self.nullclines_CP(M,F)
        print(f'CF_steady {CF_steady}')
        C = CF_steady(M, F)[0]
        P = CF_steady(M, F)[1]
        dMdt = M * (self.lam2 * (C / (self.k2 + C)) - self.mu2)
        dFdt = F * (self.lam1 * (P / (self.k1 + P)) * (1 - (F / self.K)) - self.mu1)    
        return np.array([dMdt, dFdt])
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

        
