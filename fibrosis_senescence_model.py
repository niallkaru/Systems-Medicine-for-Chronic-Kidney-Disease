import numpy as np
from scipy.integrate import solve_ivp
import scipy.optimize as opt
import state_parameter_maker as spm
import matplotlib.pyplot as plt
"""Niall Karunaratne 19/03/2025
Class for fibrosis model with senescence for cells (within kidney)
Adapted from Principles of Cell Circuits for Tissue Repair and Fibrosis Adler et al. 2020"""

class fibrosis_senescence_model:
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
        n: Proliferation of macrophages from senescence cells
        h: Proliferation of senescent cells due to aging
        r: Removal rate of senescent cells by macrophages
        q: Saturation/Carrying capacity for senescent cells

        """
        self.stateM = initial_state[0]
        self.stateF = initial_state[1]
        self.stateP = initial_state[2]
        self.stateC = initial_state[3]
        self.stateS = initial_state[4]
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
        self.n = params[13]
        self.h = params[14]
        self.r = params[15]
        self.q = params[16]

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
        M, F, C, P, S = y
        dFdt = F * (self.lam1 * (P / (self.k1 + P)) * (1 - (F / self.K)) - self.mu1)
        dMdt = self.n*S + M * (self.lam2 * (C / (self.k2 + C)) - self.mu2)
        dCdt = self.beta1 * F - self.alpha1 * M * (C / (self.k2 + C)) - self.gamma * C
        dPdt = self.beta2 * M + self.beta3 * F - self.alpha2 * F * (P / (self.k1 + P)) - self.gamma * P
        dSdt = self.h - (self.r*S*M)/(S+self.q)
        return np.array([dFdt,dMdt,dCdt,dPdt,dSdt])
    

    def steady_state_CP(self,M,F):
        """
        Important one. The quasi-steady state values of C and P. Found by setting dCdt and dPdt to zero and 
        substituting into dMdt and dFdt. Yields quadratic equation, solved below.

        C and P are independent of S so these should remain the same
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
        return steady_state_C,steady_state_P
    def nullclines_M(self,M):
        """
        Find nullclines for macrophages. Start with dMdt = 0, rearrange for C, sub into
        dCdt and rearrange for F, return both.
        Input:
        M: Macrophage levels
        Return:
        F,M: F,M values for a given value of M
        """

        S = (self.h*self.q)/(self.r*M-self.h)
        C = (self.mu2*self.k2-self.n*S*self.k2/M)/(self.lam2+self.n*S/M-self.mu2)
        F = (1/self.beta1)*((self.alpha1*M*C)/(self.k2+C))+((self.gamma*C)/self.beta1)
        return [F,M]
    def nullclines_F(self,F):
        """
        Find nullclines for myofibroblasts. Start with dFdt = 0, rearrange for P, sub into dPdt, 
        rearrange for M, again, return both.
        Input:
        F: Fibroblast levels
        Return:
        F,M: F,M values for a given value of M
        """
        P = (self.mu1*self.k1*self.K)/(self.lam1*self.K-F*self.lam1-self.mu1*self.K)
        M = -1*(self.beta3*F-self.alpha2*F*P/(self.k1+P)-self.gamma*P)/self.beta2
        return [F,M]
    def nullclines_S(self,M):
        """
        Find nullclines for senescent cells. Start with dSdt = 0, rearrange for S, 
        sub in M, return both.
        Input:
        M: Macrophage levels
        Return:
        M,S: S,M values for a given value of M
        
        """
        S = (self.h*self.q)/(self.r*M-self.h)
        return [M,S]
    
    def subtract_nulls(self,X0):
        """ Returns the one nullcline subtracted from the other accurately, this is
        used for finding fixed points"""
        M0, F0 = X0
        return [np.subtract(self.nullclines_M(M0)[0],self.nullclines_F(F0)[0]), np.subtract(self.nullclines_M(M0)[1],self.nullclines_F(F0)[1])]
    
    def fixed_points(self,initial_guess = np.array([1e4,1e4])):
        """
        We want fixed points, where the nullclines cross ie. Fdot = Mdot
        So using scipy.optimize (sic, American spelling)
        and a function to find the difference between them
        """
        x = opt.fsolve(self.subtract_nulls, initial_guess)
        return np.array(x)
    def fixed_point_cold(self,initial_guess=np.array([1e5,0])):
            P_coeff = np.array([-self.gamma,
                           (self.K / self.lam1) * (self.lam1 - self.mu1) * (self.beta3 - self.alpha2) - self.gamma * self.k1,
                           (self.K * self.k1 / self.lam1) * (self.beta3 * self.lam1 - 2 * self.mu1 * self.beta3 + self.mu1 *self.alpha2),
                           -self.k1**2 * self.mu1 * self.K * self.beta3 / self.lam1])
                # rearranged from eqns in transparent methods
            coldP= np.roots(P_coeff)
            coldF = []
            
            for coldroot in coldP:
                if np.isreal(coldroot) and coldroot >= 0:
                    coldF.append(self.K * ((self.lam1 - self.mu1) / (self.lam1) - (self.mu1 * self.k1) / (self.lam1 * np.real(coldroot)))) # finds mF value given PDGF value
            
            return coldF
    def change_in_m_f_to_int(self,t,y):
        """ Return the growth rate of M and F assuming steady state of P and C. This is the same as
        the other change_in_m_f but returns arrays, better for using with a numerical integrator"""
    
        M = y[0]
        F = y[1]
        CF_steady = self.steady_state_CP(M,F)
        C = CF_steady[0][0]
        P = CF_steady[1][0]
        dMdt = M * (self.lam2 * (C / (self.k2 + C)) - self.mu2)
        dFdt = F * (self.lam1 * (P / (self.k1 + P)) * (1 - (F / self.K)) - self.mu1)    
        return np.array([dMdt, dFdt])
    
    def change_in_m_f_to_int_neg(self,t,y):
        """ Return the growth rate of M and F assuming steady state of P and C. This is the same as
        the other change_in_m_f but returns arrays, better for using with a numerical integrator"""
    
        M = y[0]
        F = y[1]
        CF_steady = self.steady_state_CP(M,F)
        C = CF_steady[0][0]
        P = CF_steady[1][0]
        dMdt = M * (self.lam2 * (C / (self.k2 + C)) - self.mu2)
        dFdt = F * (self.lam1 * (P / (self.k1 + P)) * (1 - (F / self.K)) - self.mu1)    
        return np.array([-1*dMdt, -1*dFdt])
    
    def change_in_m_f(self,M,F, t = 0):
        """ Return the growth rate of M and F assuming steady state of P and C, essentially
        same equations as the equations function, but with steady C and P"""
    
        CF_steady = self.steady_state_CP(M,F)
        C = CF_steady[0][0]
        P = CF_steady[1][0]
        dMdt = M * (self.lam2 * (C / (self.k2 + C)) - self.mu2)
        dFdt = F * (self.lam1 * (P / (self.k1 + P)) * (1 - (F / self.K)) - self.mu1)    
        return dMdt, dFdt
    def threshold_event(self,t,y):
        """Function to stop integrator when it hits zero"""
        return min(y[0]-1,y[1]-1) #Stop when mF reaches zero
    def solve(self,t):
        """Solve functions (no steady state) using solve_ivp"""
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
        total: total injury (in cells) present
        """
        total = 0
        for start,stop,amp in pulses: # Go through every injury individually
            if start < t < stop:
                total += amp
        return total

    def constant_injury(self,t, X, pulses_M,pulses_S):
        """ Simulate the equations but with an injury added to the macrophage term and quasi-steady state
        approximation for dCdt and dPdt. We use solve_ivp to do this and assume a
        A steady state for C and P
        
        Input:
        t: time
        X: starting values for M, P and S
        pulses_M: injuries to be added (via macrophage levels)
        pulses_S: injuries to be added (via SnC levels)

        Return:
        Mdot: Number of macrophages present at a time
        Fdot: Number of myofibroblasts present at a time
        """
        #print("t:", t, "y:", X, "start:", start, "stop:", stop, "amp:", amp)

        M = X[0]
        F = X[1]
        S = X[2]

        C1_C2_steady = self.steady_state_CP(M,F)
        C = C1_C2_steady[0][0]#Funny way of grabbing them due to how I organised C1_C2_steady, just roll with it
        P = C1_C2_steady[1][0]
        #print(f'C1_C2_steady {C1_C2_steady}')

        #print(M,F)
        def equations_constant_injury(t,y,pulses_M,pulses_S):    
            M, F, S = y  
            M_dot = self.n*S+M*(self.lam2*(C/(self.k2+C)) - self.mu2) + self.heavyside_pulses(pulses_M, t)
            F_dot = F*(self.lam1*(P/(self.k1+P))*(1 - F/self.K) - self.mu1)
            S_dot = self.heavyside_pulses(pulses_S, t) + self.h-(self.r*S*M)/(S+self.q)
            return np.array([M_dot,F_dot,S_dot])

        X0 = [M,F,S]

        sol = solve_ivp(equations_constant_injury,(t[0],t[-1]),X0,t_eval=t,args = (pulses_M,pulses_S,),method='Radau')

        #print(M_dot, F_dot)
        return sol
    

    def snc_param_heatmap(self,t,X):
        """
        Calculate the levels of SnCs for different values of h and q, 
        plot the results as a heatmap.

        :Inputs:
        self: class instance
        t: array-like, time points for simulation
        X: array-like, initial values for M, F and S

        :Returns:
        None
        Heatmap
        
        Notes:
        We use the Radau method within solve_ivp as the ODEs are stiff, i.e.
        do not solve quickly
        """
        M = X[0]
        F = X[1]
        S = X[2]

        C1_C2_steady = self.steady_state_CP(M,F)
        C = C1_C2_steady[0][0]#Funny way of grabbing them due to how I organised C1_C2_steady, just roll with it
        P = C1_C2_steady[1][0]
        l = 20
        #print(f'C1_C2_steady {C1_C2_steady}')
        r_range = np.logspace(0,7,l)
        n_range = np.logspace(0,7,l)
        #print(M,F)
        results = np.empty((l,l))


        for i,r in enumerate(r_range):
            for j,n in enumerate(n_range):
                print(f'Running for r: {r} , n: {n}')
                def equation(t,y):    
                    M, F, S = y
                    M_dot = n*S+M*(self.lam2*(C/(self.k2+C)) - self.mu2)
                    F_dot = F*(self.lam1*(P/(self.k1+P))*(1 - F/self.K) - self.mu1)
                    S_dot = self.h-(r*S*M)/(S+self.q)
                    return np.array([M_dot,F_dot,S_dot])
                X0 = [M,F,S]
                sol = solve_ivp(equation,(t[0],t[-1]),X0,t_eval=t,method='Radau')
                #Use Radau as it seems kinda stiff
                results[i,j] = sol.y[0,-1]
        plt.imshow(results, cmap='plasma', extent=[n_range[0], n_range[-1], r_range[0], r_range[-1]],\
           origin='lower')
        plt.colorbar(label='Final Fibroblast Cell Population (F)')
        plt.xlabel('n (Production rate of M due to S)')
        plt.xscale('log')
        plt.ylabel('r (Senescent Cell Removal Rate)')
        plt.yscale('log')
        plt.title('Senescent Cell Population Heatmap')
        plt.show()
        plt.savefig("figure.png")
    #intitial conditions, start with small amount of F or only M population changes

        #print(M_dot, F_dot)
            
