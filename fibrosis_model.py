import numpy as np
from scipy.integrate import solve_ivp
import scipy.optimize as opt
"""Niall Karunaratne 22/12/2024
Class for fibrosis model for cells (within kidney)
Adapted from Principles of Cell Circuits for Tissue Repair and Fibrosis Adler et al. 2020"""

class fibrosis_model:
    def __init__(self,params,initial_state):
        """"Extract several million parameters
        Inputs:
        self: Instance of class
        params: List of parameters for simulation
        initial_state: Initial values for simulation

        Outputs
        stateM: Starting macrophage population
        stateF: Starting myofibroblast population
        stateP: Starting PDGF population
        stateC: Starting CSF population
        lam1: Lambda_1, Max. proliferation of myofibroblasts
        lam2: Lambda_2, Max. proliferation of macrophages
        mu2: removal rate of macrophages
        mu1: removal rate of myofibroblasts
        K: myofibroblast carrying capacity
        k1: binding affinity
        k2: binding affinity
        beta1: Max. CSF secretion by myofibroblasts
        beta2: Max. PDGF secretion by macrophages
        beta3: Max. PDGF secretion by myofibroblasts
        alpha1: Max. endocytosis of CSF by macrophages
        alpha2: Max. endocytosis of PDGF by myofibroblats
        gamma: Growth factor degredation rate

        """
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
        """
        Equations (without any quasi-steady state assumptions yet) for system
        Inputs:
        self: instance of class
        t: array of times
        y: array of populations of M,F,C and P

        Outputs:
        dFdt,dMdt,dCdt,dPdt: Array of differential equations for system
        """
        M, F, C, P = y
        dFdt = F * (self.lam1 * (P / (self.k1 + P)) * (1 - (F / self.K)) - self.mu1)
        dMdt = M * (self.lam2 * (C / (self.k2 + C)) - self.mu2)
        dCdt = self.beta1 * F - self.alpha1 * M * (C / (self.k2 + C)) - self.gamma * C
        dPdt = self.beta2 * M + self.beta3 * F - self.alpha2 * F * (P / (self.k1 + P)) - self.gamma * P
        return np.array([dFdt,dMdt,dCdt,dPdt])

    def steady_state_CP(self,M,F):
        """
        Important one. The quasi-steady state values of C and P. Found by setting dCdt and dPdt to zero and 
        substituting into dMdt and dFdt. Yields quadratic equation, solved below
        Input:
        M: Macrophage levels
        F: Myofibroblast levels
        Outputs:
        steady_state_C: Value of C at steady state for each M,F combination
        steady_state_P: Value of C at steady state for each M,F combination
        """
        steady_state_C = []
        steady_state_P = []
        ## If the values for the populations drop below zero, we will get non-physical solutions
        ## to avoid this, set to zero, i.e. no cells. Adapted from Georg Eicher's code
        if M < 0:
            M = 0
        if F < 0:
            F = 0

        # a,b,c for solving quadratic formula for P
        ap = self.gamma
        bp = self.gamma*self.k1 +self.alpha2*F -self.beta2*M - self.beta3*F
        cp = -1*self.k1*(self.beta2*M+self.beta3*F)
        # Likewise, a,b,c, for C
        ac = self.gamma
        bc = self.alpha1*M+self.gamma*self.k2-self.beta1*F
        cc = -1*(self.beta1*F*self.k2)
        delta_c = bc**2 - 4*ac*cc
        delta_p = bp**2 -4*ap*cp
        C_1 = (-bc+np.sqrt(delta_c))/(2*ac)
        C_2 = (-bc-np.sqrt(delta_c))/(2*ac)
        P_1 = (-bp+np.sqrt(delta_p))/(2*ap)
        P_2 = (-bp-np.sqrt(delta_p))/(2*ap)

        # Check if they are real and positive
        if np.isreal(C_1) and C_1 >= 0:
            steady_state_C.append(C_1)
        else: print("C_1 not positive/real. C_1 = ",C_1)
        if np.isreal(C_2) and C_2 >= 0:
            steady_state_C.append(C_2)
        #else: print("C_2 not positive/real. C_2 = ",C_2)
        if np.isreal(P_1) and P_1 >= 0:
            steady_state_P.append(P_1)
        else: print("P_1 not positive/real. P_1 = ",P_1)
        if np.isreal(P_2) and P_2 >= 0:
            steady_state_P.append(P_2)
        #else: print("P_2 not positive/real. P_2 = ", P_2)
        # if len(nullclines) <= 1:
        #     nullclines.append(0.0)
        return steady_state_C,steady_state_P
    def nullclines_M(self,M):
        """
        Find nullclines for macrophages. Start with dMdt = 0, rearrange for C, sub into
        dCdt and rearrange for F
        Input:
        M: Macrophage levels
        Return:
        F,M: F,M values for a given value of M
        """
        C = (self.mu2*self.k2)/(self.lam2-self.mu2)
        F = (1/self.beta1)*((self.alpha1*M*C)/(self.k2+C))+((self.gamma*C)/self.beta1)
        return [F,M]
    def nullclines_F(self,F):
        """
        Find nullclines for myofibroblasts. Start with dFdt = 0, rearrange for P, sub into dPdt, 
        rearrange for M
        Input:
        F: Macrophage levels
        Return:
        F,M: F,M values for a given value of M
        """
        P = (self.mu1*self.k1*self.K)/(self.lam1*self.K-F*self.lam1-self.mu1*self.K)
        M = -1*(self.beta3*F-self.alpha2*F*P/(self.k1+P)-self.gamma*P)/self.beta2
        return [F,M]
    def subtract_nulls(self,X0):
        """ Returns the one nullcline subtracted from the other accurately"""
        M0, F0 = X0
        return [np.subtract(self.nullclines_M(M0)[0],self.nullclines_F(F0)[0]), np.subtract(self.nullclines_M(M0)[1],self.nullclines_F(F0)[1])]
    
    def fixed_points(self,initial_guess = np.array([1e4,1e4])):
        ## Optimise to get fixed points, not very accurate as it stands
        x = opt.fsolve(self.subtract_nulls, initial_guess)
        return np.array(x)
    def change_in_m_f_to_int(self,t,y):
        """ Return the growth rate of M and F assuming steady state of P and C. This is the same as
        the other change_in_m_f but returns arrays, better for using with a numerical integrator"""
    
        M = y[0]
        F = y[1]
        CF_steady = self.steady_state_CP(M,F)
        #print(f'CF_steady {CF_steady}')
        C = CF_steady[0][0]
        P = CF_steady[1][0]
        #print(C,P)
        dMdt = M * (self.lam2 * (C / (self.k2 + C)) - self.mu2)
        dFdt = F * (self.lam1 * (P / (self.k1 + P)) * (1 - (F / self.K)) - self.mu1)    
        return np.array([dMdt, dFdt])
    
    def change_in_m_f_to_int_neg(self,t,y):
        """ Return the growth rate of M and F assuming steady state of P and C. This is the same as
        the other change_in_m_f but returns arrays, better for using with a numerical integrator"""
    
        M = y[0]
        F = y[1]
        CF_steady = self.steady_state_CP(M,F)
        #print(f'CF_steady {CF_steady}')
        C = CF_steady[0][0]
        P = CF_steady[1][0]
        #print(C,P)
        dMdt = M * (self.lam2 * (C / (self.k2 + C)) - self.mu2)
        dFdt = F * (self.lam1 * (P / (self.k1 + P)) * (1 - (F / self.K)) - self.mu1)    
        return np.array([-1*dMdt, -1*dFdt])
    
    def change_in_m_f(self,M,F, t = 0):
        """ Return the growth rate of M and F assuming steady state of P and C"""
    
        CF_steady = self.steady_state_CP(M,F)
        C = CF_steady[0][0]
        P = CF_steady[1][0]
        dMdt = M * (self.lam2 * (C / (self.k2 + C)) - self.mu2)
        dFdt = F * (self.lam1 * (P / (self.k1 + P)) * (1 - (F / self.K)) - self.mu1)    
        return dMdt, dFdt
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
    
    def heavyside_pulses(self,pulses,t):
        """ Heavyside function returns amp between start and stop, otherwise 0 when single value is given
        This takes in pulses which allows multiple injuries
        Inputs:
        pulses: start, stop, amp. Start time, stopping time and amplitude (in macrophages) of injury
        t: time

        Return:
        total: total injury (in macrophages) present
        """
        total = 0
        for start,stop,amp in pulses: # Go through every injury individually
            if start < t < stop:
                total += amp
        return total

    def constant_injury(self,t, X, pulses):
        """ Simulate the equations but with an injury added to the macrophage term and quasi-steady state
        approximation for dCdt and dPdt.
        
        Input:
        t: time
        X: starting values for M and P
        pulses: injuries to be added

        Return:
        Mdot: Number of macrophages present at a time
        Fdot: Number of myofibroblasts present at a time
        """
        #print("t:", t, "y:", X, "start:", start, "stop:", stop, "amp:", amp)

        M = X[0]
        F = X[1]

        C1_C2_steady = self.steady_state_CP(M,F)
        #print(f'C1_C2_steady {C1_C2_steady}')
        C = C1_C2_steady[0][0]#Funny way of grabbing them due to how I organised C1_C2_steady, just roll with it
        P = C1_C2_steady[1][0]
        #print(M,F)
        M_dot = M*(self.lam2*(C/(self.k2+C)) - self.mu2) + self.heavyside_pulses(pulses, t)
        F_dot = F*(self.lam1*(P/(self.k1+P))*(1 - F/self.K) - self.mu1)
        
        #print(M_dot, F_dot)
        return np.array([M_dot, F_dot])
    
    def separatrix_eigen(self,X):

        M = X[0]
        F = X[1]

        C1_C2_steady = self.steady_state_CP(M,F)
        #print(f'C1_C2_steady {C1_C2_steady}')
        C = C1_C2_steady[0][0]#Funny way of grabbing them due to how I organised C1_C2_steady, just roll with it
        P = C1_C2_steady[1][0]
        dMdM = (self.lam2*C)/(self.k2+C) - self.mu2
        dMdF = 0
        dFdM = 0
        dFdF = (self.lam1*P)/(self.k1+P)-(2*F*self.lam1*P)/(self.K*(self.k1+P))-self.mu1
        jacobian = np.zeros((2,2))
        jacobian[0,0] = dMdM
        jacobian[0,1] = dMdF
        jacobian[1,1] = dFdF
        jacobian[1,0] = dFdM
        eigenvals,eigenvecs = np.linalg.eig(jacobian)
        #print(eigenvals,eigenvecs)
        unstable_index = np.argmax(eigenvals.real)
        #Index of largest (real) eigenvalue, a positive eigenval
        #corresponds to an unstable fixed point (along separatrix)
        unstable_vector = eigenvecs[:,unstable_index]

        return eigenvals[unstable_index],unstable_vector/np.linalg.norm(unstable_vector) #Normalise it      

    def separatrix_traj(self,t,X,epsilon=1):
        """
        Plot Separatrix, a bit sketchy at the moment
        """
        eigenval,unstable_vector = self.separatrix_eigen(X)

        initial = X+epsilon*unstable_vector #Perturb a little
        print(initial)
        sep_traj = solve_ivp(self.change_in_m_f_to_int, (t[0], t[-1]), initial, t_eval=t)
        return [sep_traj.y[0],sep_traj.y[1]]
        
