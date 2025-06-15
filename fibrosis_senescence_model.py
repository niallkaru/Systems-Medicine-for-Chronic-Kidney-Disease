import numpy as np
import pandas as pd
import scipy as sp
from scipy.integrate import solve_ivp
import scipy.optimize as opt
import state_parameter_maker as spm
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import itertools
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plotly.graph_objects as go
from scipy.optimize import differential_evolution
from scipy.optimize import basinhopping

"""Niall Karunaratne 19/03/2025
Class for fibrosis model with senescence for cells
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
        stateS: Starting SnC population

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
        These are not normally used as they do not include pulses of macrophages/SnCs
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
        Mmin = self.h / self.r
        if M <= Mmin:               # outside admissible domain, need +
            return np.nan, np.nan   # plotting libs ignore NaN
        S = (self.h*self.q)/(self.r*M-self.h)
        C = (self.mu2*self.k2-self.n*S*self.k2/M)/(self.lam2+self.n*S/M-self.mu2)
        F = (1/self.beta1)*((self.alpha1*M*C)/(self.k2+C))+((self.gamma*C)/self.beta1)
        return F,M
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
    def nullclines_S(self,S):
        """
        Find nullclines for senescent cells. Start with dSdt = 0, rearrange for S, 
        sub in M, return both.
        Input:
        S: SnC levels
        Return:
        M,S: S,M values for a given value of S
        
        """
        M = self.h * (self.q + S) / (self.r * S)
        return [S, M]
    def nullclines_M_fixed_S(self, M_vals, S_fixed):
        """
        Compute the macrophage nullcline (dM/dt = 0) for fixed S.
        Return F values as a function of M.
        Input:
        F: Fibroblast levels
        Return:
        M,F: M,F values for a given value of S
        """
        F_vals = []
        for M in M_vals:
            try:
                if M <= 0 or self.lam2 == 0:
                    raise ValueError("Invalid M or λ₂")

                term = self.mu2 - (self.n * S_fixed) / M
                term = self.mu2 - (self.n * S_fixed) / M
                if term <= 1e-6:
                    term = 1e-6  # avoid poles

                if term <= 0 or term >= self.lam2:
                    raise ValueError("Invalid C expression (would be negative or undefined)")

                # Compute C from rearranged dM/dt = 0
                C = (self.k2 * term) / (self.lam2 - term)

                # Now use dC/dt = 0 to compute F
                frac = C / (self.k2 + C)
                F = (self.alpha1 * M * frac + self.gamma * C) / self.beta1
                F_vals.append(F)

            except:
                F_vals.append(np.nan)

        return np.array(M_vals), np.array(F_vals)


    def change_in_m_f_to_int(self,t,y):
        """ Return the growth rate of M and F assuming steady state of P and C. This is the same as
        the other change_in_m_f but negative and returns arrays, better for using with a numerical integrator.
        It is needed for finding the separatrix"""
    
        M = y[0]
        F = y[1]
        S = y[2]
        CF_steady = self.steady_state_CP(M,F)
        C = CF_steady[0][0]
        P = CF_steady[1][0]
        dMdt = self.n*S+M * (self.lam2 * (C / (self.k2 + C)) - self.mu2)
        dFdt = F * (self.lam1 * (P / (self.k1 + P)) * (1 - (F / self.K)) - self.mu1)  
        dSdt = self.h -(self.r*S*M)/(S+self.q)    
  
        return np.array([dMdt, dFdt,dSdt])
    
    def change_in_m_f_to_int_neg(self,t,y):
        """ Return the growth rate of M and F assuming steady state of P and C. This is the same as
        the other change_in_m_f but negative and returns arrays, better for using with a numerical integrator.
        It is needed for finding the separatrix"""
    
        M = y[0]
        F = y[1]
        S = y[2]
        CF_steady = self.steady_state_CP(M,F)
        C = CF_steady[0][0]
        P = CF_steady[1][0]
        dMdt = self.n*S+M * (self.lam2 * (C / (self.k2 + C)) - self.mu2)
        dFdt = F * (self.lam1 * (P / (self.k1 + P)) * (1 - (F / self.K)) - self.mu1)  
        dSdt = self.h -(self.r*S*M)/(S+self.q)    
  
        return np.array([-1*dMdt, -1*dFdt,-1*dSdt])
    def change_in_m_f_to_int_neg_2D(self, t, y):
        """
        2D “negative” RHS for (M,F) at fixed S – returns dM, dF, dS=0 so that S is frozen.
        """
        M, F, S = y
        # compute your quasi‑steady C,P exactly as before
        C, P = self.steady_state_CP(M, F)
        C = C[0];  P = P[0]

        # the original 2D growth rates
        dMdt = self.n*S + M*(self.lam2*(C/(self.k2+C)) - self.mu2)
        dFdt = F*(self.lam1*(P/(self.k1+P))*(1 - F/self.K) - self.mu1)

        # negating them gives the backward integration along the unstable mode,
        # and we force dSdt=0 so that S never moves off its slice
        return np.array([-dMdt, -dFdt, 0.0])
    def change_in_m_f_to_int_2D(self, t, y):
        """
        2D “negative” RHS for (M,F) at fixed S – returns dM, dF, dS=0 so that S is frozen.
        Similar to above, but not negative.
        """
        M, F, S = y
        C, P = self.steady_state_CP(M, F)
        C = C[0];  P = P[0]
        dMdt = self.n*S + M*(self.lam2*(C/(self.k2+C)) - self.mu2)
        dFdt = F*(self.lam1*(P/(self.k1+P))*(1 - F/self.K) - self.mu1)
        return np.array([ dMdt,  dFdt,  0.0])

    
    def change_in_m_f(self,M,F,S,t=0):
        """ Return the growth rate of M and F assuming steady state of P and C, essentially
        same equations as the equations function, but with steady C and P. This version is not
        used in the numerical integrators"""
        # M = y[0]
        # F = y[1]
        # S = y[2]
        CF_steady = self.steady_state_CP(M,F)
        C = CF_steady[0][0]
        P = CF_steady[1][0]
        dMdt = self.n*S + M * (self.lam2 * (C / (self.k2 + C)) - self.mu2)
        dFdt = F * (self.lam1 * (P / (self.k1 + P)) * (1 - (F / self.K)) - self.mu1)
        dSdt = self.h -(self.r*S*M)/(S+self.q)    
        return dMdt, dFdt, dSdt
    
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

        M = max(0, X[0])
        F = max(0, X[1])
        S = max(0,X[2])
        C1_C2_steady = self.steady_state_CP(M,F)
        C = C1_C2_steady[0][0]#Funny way of grabbing them due to how I organised C1_C2_steady, just roll with it
        P = C1_C2_steady[1][0]


        M_dot = self.n*S+M*(self.lam2*(C/(self.k2+C)) - self.mu2) + self.heavyside_pulses(pulses_M, t)
        F_dot = F*(self.lam1*(P/(self.k1+P))*(1 - F/self.K) - self.mu1)
        S_dot = self.heavyside_pulses(pulses_S, t) + self.h-(self.r*S*M)/(S+self.q)
        return np.array([M_dot,F_dot,S_dot])

    def solve_constant_injury(self,t,X,pulses_M,pulses_S):
        """
        Using solve_ivp (from scipy), solve the above equation with optional pulses for M and S
        """
        sol = solve_ivp(self.constant_injury,(t[0],t[-1]),X,t_eval=t,args = (pulses_M,pulses_S,),method='Radau')
        def enforce_non_negative(y):
            # Enforce non-negative values for the solver.
            return np.maximum(y, 0)
        sol.y = np.apply_along_axis(enforce_non_negative, 0, sol.y)

        #print(M_dot, F_dot)
        return sol
    

    def snc_param_heatmap(self,t,X,r_range,n_range):
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
        C = C1_C2_steady[0][0]#Funny way of grabbing them due to how I organised C1_C2_steady
        P = C1_C2_steady[1][0]
        length_r = len(r_range)
        length_n = len(n_range)
       # if length_r != length_n:
            #raise ValueError("Must have equal numbers of r and n.")
        #print(f'C1_C2_steady {C1_C2_steady}')

        #print(M,F)
        results_M = np.empty((length_r, length_n))
        results_F = np.empty_like(results_M)
        results_S = np.empty_like(results_M)
        for i, r in enumerate(tqdm(r_range, desc="Heatmap Progress")):
            for j, n in enumerate(n_range):
                def equation(t, y):
                    M, F, S = y
                    M_dot = n*S + M*(self.lam2*(C/(self.k2+C)) - self.mu2)
                    F_dot = F*(self.lam1*(P/(self.k1+P))*(1 - F/self.K) - self.mu1)
                    S_dot = self.h - (r*S*M)/(S + self.q)
                    return [M_dot, F_dot, S_dot]

                sol = solve_ivp(equation, (t[0], t[-1]), [M, F, S], t_eval=t, method='Radau')
                results_F[i, j] = sol.y[1, -1]
                results_M[i, j] = sol.y[0, -1]
                results_S[i, j] = sol.y[2, -1]

        plt.pcolormesh(n_range, r_range, results_F, shading='auto', cmap='plasma')
        #plt.xscale('log')
        #plt.yscale('log')
        plt.colorbar(label='Final Myofibroblast Cell Population (F)')
        plt.xlabel('n (Production rate of M due to S)')
        plt.ylabel('r (Senescent Cell Removal Rate)')
        plt.title('Myofibroblast Population Heatmap')
        plt.show(block=True)
        plt.pcolormesh(n_range, r_range, results_M, shading='auto', cmap='plasma')
        #plt.xscale('log')
        #plt.yscale('log')
        plt.colorbar(label='Final Macrophage Cell Population (M)')
        plt.xlabel('n (Production rate of M due to S)')
        plt.ylabel('r (Senescent Cell Removal Rate)')
        plt.title('Macrophage Population Heatmap')
        plt.show(block=True)
        plt.pcolormesh(n_range, r_range, results_S, shading='auto', cmap='plasma')
        #plt.xscale('log')
        #plt.yscale('log')
        plt.colorbar(label='Final SnC Population (F)')
        plt.xlabel('n (Production rate of M due to S)')
        plt.ylabel('r (Senescent Cell Removal Rate)')
        plt.title('SnC Population Heatmap')
        plt.show(block=True)

    def residual_fixed_point(self, X):
        """
        Compute the residuals for the full 3D steady-state conditions.
        X = [M, F, S]
        Uses the quasi-steady state for C and P.
        At a fixed point, we require:
        dM/dt = 0,  dF/dt = 0,  dS/dt = 0
        """
        M, F, S = X
        # Obtain C and P via your steady_state_CP function.
        CF_steady = self.steady_state_CP(M, F)
        C = CF_steady[0][0]
        P = CF_steady[1][0]
        
        # Compute each derivative
        dMdt = self.n * S + M * (self.lam2 * (C / (self.k2 + C)) - self.mu2)
        dFdt = F * (self.lam1 * (P / (self.k1 + P)) * (1 - (F / self.K)) - self.mu1)
        dSdt = self.h - (self.r * S * M) / (S + self.q)
        
        return np.array([dMdt, dFdt, dSdt])  
    def residual_fixed_point_slice(self, M, F, fixed_S):
        """
        Find fixed points for a slice of the 3D system, that is with a fixed value of S
        This is done using the change_in_m_f function
        """
        dM, dF, _ = self.change_in_m_f(M, F, fixed_S)
        return np.array([dM, dF])

    def perturb_fixed_point(self,fp, epsilon=0.01,tol=1e-4):
        """
        Replace any zero entries in the fixed point fp with epsilon.
        """
        fp = np.where(fp <= tol, epsilon, fp)
        return fp
    def fixed_points_3D(self, initial_guess=np.array([1e4, 1e4, 1e4])):
        """
        Find the fixed point in 3D by solving residual_fixed_point(X) = [0, 0, 0].
        """
        res = opt.root(self.residual_fixed_point, initial_guess,method='hybr')
        if not res.success:
            #print("Fixed Point failure: ",initial_guess)
            return np.array([np.nan, np.nan, np.nan])

        fixed_pt = res.x
        # fixed_pt = self.perturb_fixed_point(res.x,epsilon=1e-2,tol=1e-1)
        return fixed_pt
    def fixed_pt_sweep(self, xrange, yrange, z_fixed, mode='slice', method='eigen', perturb=True, classify=True):
        """
        Sweep over a 2D grid (M, F) at fixed S=z_fixed and find fixed points.

        Parameters:
            xrange, yrange (list): Log10 bounds for M and F.
            z_fixed (float): Fixed value of S.
            mode (str): 'slice' or 'full' — which classification strategy to use.
            method (str): 'eigen', 'dynamics', or 'both' — how to classify each point.
            perturb (bool): Whether to apply perturbation to clean up points.
            classify (bool): Whether to return classifications.

        Returns:
            List of fixed points, and optionally their classification metadata.
        """
        xfull = np.logspace(xrange[0], xrange[-1], 35)
        yfull = np.logspace(yrange[0], yrange[-1], 35)
        xvals, yvals = xfull[::3], yfull[::3]

        grid_x, grid_y = np.meshgrid(xvals, yvals, indexing='ij')

        vect_fun = np.vectorize(lambda M, F: (self.fixed_points_3D([M, F, z_fixed])[0], self.fixed_points_3D([M, F, z_fixed])[1]))
        sol1, sol2 = vect_fun(grid_x, grid_y)
        combined_sols = np.stack((sol1, sol2), axis=-1).reshape(-1, 2)
        combined_sols = combined_sols[~np.isnan(combined_sols).any(axis=1)]
        unique_sols = np.unique(np.round(combined_sols, 2), axis=0)
        filtered_sols = unique_sols[np.all(unique_sols >= 0, axis=1)]

        sols = filtered_sols
        meta = {}

        for i, sol in enumerate(filtered_sols):
            M, F = sol
            S = z_fixed
            print(f"Initial guess: M={M:.2e}, F={F:.2e}, S={S:.2e}")
            fp = self.fixed_points_3D([M, F, S])
            print("Returned:", fp)
            print("Residual norm:", np.linalg.norm(self.dynamics_3D(*fp)))
            if mode == 'slice':
                res = self.classify_slice(M, F, S, fixed={'S': S}, method=method)
            elif mode == 'full':
                res = self.classify_full(M, F, S, method=method)
            else:
                raise ValueError("mode must be 'slice' or 'full'")

            meta[i] = res
           # print(f"{mode.upper()} @ S={S}: {res['verdict'] if 'verdict' in res else 'no verdict'}")

        if perturb:
            sols = [self.perturb_fixed_point(sol, epsilon=1e-2, tol=1e-1) for sol in filtered_sols]

        if classify:
            return sols, meta
        else:
            return sols


    def sweep_fixed_S(self, xrange, yrange, fixed_S=2000,
                    classify=True, method="eigen", perturb=True):
        """
        Find fixed points for a fixed S value over a log-log grid in M-F space.

        Parameters:
            xrange   : tuple of log10(M) range (e.g. [-2, 7])
            yrange   : tuple of log10(F) range
            fixed_S  : fixed value of senescent cell population (S)
            classify : whether to classify each fixed point
            method   : classification method ('eigen', 'dynamics', 'both')
            perturb  : whether to perturb near-zero values

        Returns:
            fixed_pts : list of unique valid fixed points (M, F, S)
            metadata  : dict of classification results
        """
        xvals = np.logspace(xrange[0], xrange[1], 35)[::2]
        yvals = np.logspace(yrange[0], yrange[1], 35)[::2]
        grid_x, grid_y = np.meshgrid(xvals, yvals, indexing='ij')
        grid_pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)

        raw_pts = []
        metadata = {}

        for i, (M0, F0) in enumerate(grid_pts):
            res = opt.root(
                lambda XY: self.residual_fixed_point_slice(XY[0], XY[1], fixed_S),
                [M0, F0],
                method='hybr'
            )

            if not res.success or np.any(np.isnan(res.x)) or np.any(res.x < 0):
                continue

            M, F = res.x
            raw_pts.append([M, F, fixed_S])  # include S explicitly

        raw_pts = np.array(raw_pts)
        rounded = np.round(raw_pts, 2)
        unique = np.unique(rounded, axis=0)
        fixed_pts = unique[np.all(unique >= 0, axis=1)]

        for idx, pt in enumerate(fixed_pts):
            if classify:
                res = self.classify_slice(*pt, fixed={"S": fixed_S}, method=method)
                metadata[idx] = res
            if perturb:
                pt = self.perturb_fixed_point(pt)
                fixed_pts[idx] = pt


        return fixed_pts.tolist(), metadata

    def sweep_full_3D(self, xrange, yrange, zrange,
                    classify=True, method="eigen", perturb=True):
        """
        Find and classify fixed points in full 3D (M,F,S) space.

        Parameters:
            xrange, yrange, zrange: log10 ranges for M, F, S
            classify: whether to classify each fixed point
            method: 'eigen', 'dynamics', or 'both'
            perturb: whether to apply perturbation cleanup

        Returns:
            fixed_pts: list of [M, F, S]
            metadata: dict with verdicts, eigenvalues, etc.
        """

        xvals = np.logspace(xrange[0], xrange[1], 20)
        yvals = np.logspace(yrange[0], yrange[1], 20)
        zvals = np.logspace(zrange[0], zrange[1], 20)

        raw_pts = []
        metadata = {}

        for M0 in xvals:
            for F0 in yvals:
                for S0 in zvals:
                    guess = [M0, F0, S0]
                    fp = self.fixed_points_3D(guess)
                    if np.any(np.isnan(fp)) or np.any(fp < 0):
                        continue
                    raw_pts.append(fp)

        # Round and filter unique positive fixed points
        raw_pts = np.array(raw_pts)
        rounded = np.round(raw_pts, 2)
        unique = np.unique(rounded, axis=0)
        fixed_pts = unique[np.all(unique >= 0, axis=1)]

        for idx, pt in enumerate(fixed_pts):
            if perturb:
                pt = self.perturb_fixed_point(pt)
                fixed_pts[idx] = pt
            if classify:
                res = self.classify_full(*pt, method=method)
                metadata[idx] = res

        return fixed_pts.tolist(), metadata


   
    def jacobian_full(self, M, F, S):
        """
        Return the full 3×3 Jacobian matrix d(dM,dF,dS)/d(M,F,S)
        using your analytic formulas for the partial derivatives.
        """
        # quasi‐steady C,P
        C = self.steady_state_CP(M, F)[0][0]
        P = self.steady_state_CP(M, F)[1][0]

        # fill entries
        dMdM = (self.lam2*C)/(self.k2 + C) - self.mu2
        dMdF = 0
        dMdS = self.n

        dFdM = 0
        dFdF = (self.lam1*P)/(self.k1+P) \
            - (2*F*self.lam1*P)/(self.K*(self.k1+P)) \
            - self.mu1
        dFdS = 0

        dSdM = -(self.r*S)/(S + self.q)
        dSdF = 0
        dSdS = (self.r*M)/(S + self.q) * (S/(S + self.q) - 1)

        return np.array([
            [dMdM,  dMdF,  dMdS],
            [dFdM,  dFdF,  dFdS],
            [dSdM,  dSdF,  dSdS],
        ])

    def eigen(self,M,F,S):
        """
        We find the eigenvalues/vectors of the fixed points to determine
        if they are stable/unstable (negative or positive). Use later.

        Input: X, coordinates of steady state in M-F-S space
        Return: Eigenvalues and normalised eigenvector of steady state
        """



        C1_C2_steady = self.steady_state_CP(M,F)
        #print(f'C1_C2_steady {C1_C2_steady}')
        C = C1_C2_steady[0][0]#Funny way of grabbing them due to how I organised C1_C2_steady, just roll with it
        P = C1_C2_steady[1][0]
        #rows z/dM,z/dF,z/dS
        #cols dM/z,dF/z,dS/z
        dMdM = (self.lam2*C)/(self.k2+C) - self.mu2
        dMdF = 0
        dMdS = self.n
        dFdM = 0
        dFdF = (self.lam1*P)/(self.k1+P)-(2*F*self.lam1*P)/(self.K*(self.k1+P))-self.mu1
        dFdS = 0
        dSdM = (-1*self.r*S)/(S+self.q)
        dSdF = 0
        dSdS = ((self.r*M)/(S+self.q))*(S/(S+self.q)-1)
        jacobian = np.zeros((3,3))
        jacobian[0,0] = dMdM
        jacobian[0,1] = dMdF
        jacobian[0,2] = dMdS
        jacobian[1,1] = dFdF
        jacobian[1,0] = dFdM
        jacobian[1,2] = dFdS
        jacobian[2,0] = dSdM
        jacobian[2,1] = dSdF
        jacobian[2,2] = dSdS
        eigenvals,eigenvecs = np.linalg.eig(jacobian)
        return eigenvals,eigenvecs
        #print(eigenvals,eigenvecs)

    def separatrix_eigen(self,M,F,S):
        eigenvals,eigenvecs = self.eigen(M,F,S)
        print(f'eigenvals {eigenvals}\neigenvecs {eigenvecs}')
        unstable_index = np.argmax(eigenvals.real)
        print(f'unstable index {unstable_index}')
        #Index of largest (real) eigenvalue, a positive eigenval
        #corresponds to an unstable fixed point (along separatrix)
        unstable_vector = eigenvecs[:,unstable_index]
        print(f'unstable vector {unstable_vector}')
        return eigenvals[unstable_index],unstable_vector/np.linalg.norm(unstable_vector) #Normalise it   

    
    def separatrix(self,t,X,epsilon=10):
            """
            Plot Separatrix, we need to use negative changes to go *against* the normal direction
            (see quiver plot to understand better).
            The idea is to take in the eigenvector of the unstable point, multiply it by
            a small value and add it to the fixed point as a perturbation, then plot to get
            the separatrix
            Input: t, range of times to integrate over
            X, fixed point location
            epsilon, value to perturb by

            Return: list of trajectory of separatrix
            """
            M = X[0]
            F = X[1]
            S = X[2]
            # eigenval,unstable_vector = self.separatrix_eigen(M,F,S)
            # # Try perturbing in just M
            (vals2, vecs2), _ = self.eigen_slice(M, F, S, fixed={'S':S})
            # find the index of the positive (most unstable) eigenvalue
            idx = np.argmin(np.real(vals2))  
            # extract the 2D eigenvector
            v2d = vecs2[:, idx]
            # normalize
            v2d = v2d / np.linalg.norm(v2d)
            unstable_vector = np.array([ v2d[0],  # M‑component
                 v2d[1],  # F‑component
                 0.0 ])   # S‑component fixed
            #unstable_vector = np.array((1,0,0))
            print(unstable_vector,vals2)
            initial_pos = X+epsilon*np.array([v2d[0], v2d[1], 0.0]) #Perturb a little
            initial_neg = X-epsilon**np.array([v2d[0], v2d[1], 0.0]) #Perturb a little, other direction

            sep_traj_pos = solve_ivp(self.change_in_m_f_to_int_neg_2D, (t[0], t[-1]), initial_pos, t_eval=t,method='Radau',rtol=1e-9, atol=1e-12)
            sep_traj_neg = solve_ivp(self.change_in_m_f_to_int_neg_2D, (t[0], t[-1]), initial_neg, t_eval=t,method='Radau',rtol=1e-9, atol=1e-12)
            pos_M,pos_F,pos_S = sep_traj_pos.y[0],sep_traj_pos.y[1],sep_traj_pos.y[2]
            neg_M,neg_F,neg_S = sep_traj_neg.y[0],sep_traj_neg.y[1],sep_traj_neg.y[2]
            # Chop if it loops back
            pos_M = pos_M[:np.argmin(pos_M)] if np.any(np.diff(pos_M) < 0) else pos_M
            pos_F = pos_F[:len(pos_M)]

            pos_traj =[pos_M,pos_F,pos_S]
            neg_traj= [neg_M,neg_F,neg_S]
            #print(f'pos_traj {pos_traj}\nneg_traj {neg_traj}')
            return pos_traj, neg_traj

    def planar_rhs(self,t, y):
        M, F = y
        dM, dF, _ = self.change_in_m_f(M, F, S=2000)
        return [dM, dF]


    def plot_2D_quiver_field_fixed_S(self, x_vals, y_vals, fixed_S,
                                    x_label="Myofibroblast Conc.",
                                    y_label="Macrophage Conc."):
        """
        Creates a 2D quiver plot of the M-F vector field for the new fibrosis-senescence
        model by fixing the Senescent (S) value. It uses a log-spaced grid for the x- and y-axes,
        computes the derivatives at each (M, F) point for the fixed S, then normalizes the vectors
        and colors them by their (relative) magnitude.
        
        Parameters:
        x_vals: 1D array of Myofibroblast concentrations (e.g. generated via np.logspace)
        y_vals: 1D array of Macrophage concentrations (e.g. via np.logspace)
        fixed_S: Fixed value for the Senescent cell concentration
        x_label: Label for the x-axis (default "Myofibroblast Conc.")
        y_label: Label for the y-axis (default "Macrophage Conc.")
        
        Assumes that the model has a method `change_in_m_f(M, F, S)` that returns a tuple:
        (dM/dt, dF/dt)
        """
        # Create a grid using the provided x and y values
        X, Y = np.meshgrid(x_vals, y_vals)
        
        # Use np.vectorize to compute the directional derivatives for each point.
        # Note: We pass Y as the first argument and X as the second so that the lambda receives:
        #       M = value from Y (macrophages) and F = value from X (myofibroblasts)
        # Adjust this order if your model expects a different convention.
        vector_func = np.vectorize(
            lambda M, F: self.change_in_m_f(M, F, fixed_S)[0:2],
            otypes=[float, float]
        )
        # Here, the convention is: first returned value is dM/dt, second is dF/dt.
        DY, DX = vector_func(Y, X)
        
        # Scale the derivatives by the state to get relative (or percentage) rates.
        # This step mimics your original normalization where you divide by the coordinate values.
        DX = DX / X
        DY = DY / Y
        
        # Compute the norm of the (DX, DY) arrows.
        norm = np.hypot(DX, DY)
        norm[norm == 0] = 1.0  # avoid division by zero
        
        # Normalize the derivatives so that all arrows have the same length;
        # the original magnitude (norm) is kept for color-coding.
        DX_norm = DX / norm
        DY_norm = DY / norm
        log_norm = np.log10(norm + 1e-9)
        # Create the quiver plot.
        Q = plt.quiver(X, Y, DX_norm, DY_norm, log_norm, pivot='mid', cmap=plt.cm.rainbow,scale=40,headwidth=5)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        #plt.title(f"2D Quiver Plot with S fixed = {fixed_S:g}")
        

        
        # Add a colorbar for the vector magnitude.
        cbar = plt.colorbar(Q)
        cbar.set_label("Relative Growth Rate (Normalised)")
        

    ####### Adimensionalised ##########
    def define_dimensionless_parameters(self, M0=1.0, S0=1.0):
        """
        Define dimensionless parameters from the dimensional ones.
        Optionally takes scaling constants for M and S (else set to 1.0).
        """
        # Store scaling constants
        self.M0 = M0
        self.S0 = S0

        # Time and population scalings
        mu1 = self.mu1
        K = self.K

        # Dimensionless parameters
        self.zeta1 = self.lam1 / mu1
        self.zeta2 = self.lam2 / mu1
        self.phi2 = self.mu2 / mu1
        self.phi_g = self.gamma / mu1
        self.rho1 = (self.alpha1 * M0) / mu1
        self.rho2 = (self.alpha2 * K) / mu1
        self.eta = (self.n * S0) / (M0 * mu1)
        self.chi = self.h / (S0 * mu1)
        self.theta = (self.r * M0) / mu1
        self.q = self.q / S0
        self.xi = self.beta2 / (self.beta2 + self.beta3) if (self.beta2 + self.beta3) > 0 else 0.5

    def steady_state_cp_dimless(self, m, f):
        """
        Compute steady-state values of dimensionless CSF (c) and PDGF (p)
        using corrected unit-consistent scaling:
            - c = C / k2
            - p = P / k1
        Parameters:
            m: dimensionless macrophage level (M / M0)
            f: dimensionless myofibroblast level (F / K)
        Returns:
            c, p: dimensionless concentrations of CSF and PDGF
        """
        m = max(m, 0)
        f = max(f, 0)

        # -- CSF (c = C / k2) --
        ac = self.phi_g
        bc = self.rho1 * m + self.phi_g - (self.beta1 / self.mu1) * f
        cc = - (self.beta1 / self.mu1) * f
        delta_c = bc**2 - 4 * ac * cc
        c1 = (-bc + np.sqrt(delta_c)) / (2 * ac)
        c2 = (-bc - np.sqrt(delta_c)) / (2 * ac)
        c = c1 if (np.isreal(c1) and c1 >= 0) else c2

        # -- PDGF (p = P / k1) --
        ap = self.phi_g
        xi = self.xi
        bp = self.phi_g + self.rho2 * f - xi * m - (1 - xi) * f
        cp = - (xi * m + (1 - xi) * f)
        delta_p = bp**2 - 4 * ap * cp
        p1 = (-bp + np.sqrt(delta_p)) / (2 * ap)
        p2 = (-bp - np.sqrt(delta_p)) / (2 * ap)
        p = p1 if (np.isreal(p1) and p1 >= 0) else p2

        return c, p


    
    def reduced_rhs(self, tau, y):
        m, f, s = y
        c, p = self.steady_state_cp_dimless(m, f)
        #c, _ = self.steady_state_cp_dimless(m, f)
        #p=1.0
        dm = self.eta * s + m * (self.zeta2 * c / (1 + c) - self.phi2)
        df = f * (self.zeta1 * p / (1 + p) * (1 - f) - 1)

        ds = self.chi - (self.theta * s * m) / (s + self.q)

        return np.array([dm, df, ds])


    
    def simulate_model(self, y0, t_span=(0, 100), t_eval=None):
        from scipy.integrate import solve_ivp
        import numpy as np

        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 500)

        result = solve_ivp(
            fun=self.reduced_rhs,
            t_span=t_span,
            y0=y0,
            t_eval=t_eval,
            method='Radau',
            vectorized=False
        )
        #print("Requested t_eval length:", len(t_eval))
        #print("Returned solution length:", len(result.t))
        if len(result.t) != len(t_eval):
            print("Solver returned fewer timepoints than requested.")
            raise RuntimeError("ODE solver failed to complete all timepoints.")

        return result
    def parameter_sensitivity(self, param_name, delta=0.01, y0=None, t_span=(0, 100), t_eval=None, output_var='f', time_index=-1, central_diff=True):

        if y0 is None:
            y0 = [0.01, 0.01, 0.01]
        if t_eval is None:
            t_eval = np.linspace(t_span[0], t_span[1], 500)

        var_index = {'m': 0, 'f': 1, 's': 2}[output_var]
        original_value = getattr(self, param_name)

        setattr(self, param_name, original_value)
        result_base = self.simulate_model(y0, t_span=t_span, t_eval=t_eval)
        output_base = result_base.y[var_index, time_index]

        if central_diff:
            setattr(self, param_name, original_value * (1 + delta))
            output_plus = self.simulate_model(y0, t_span=t_span, t_eval=t_eval).y[var_index, time_index]

            setattr(self, param_name, original_value * (1 - delta))
            output_minus = self.simulate_model(y0, t_span=t_span, t_eval=t_eval).y[var_index, time_index]

            dfdtheta = (output_plus - output_minus) / (2 * original_value * delta)
        else:
            setattr(self, param_name, original_value * (1 + delta))
            output_plus = self.simulate_model(y0, t_span=t_span, t_eval=t_eval).y[var_index, time_index]
            dfdtheta = (output_plus - output_base) / (original_value * delta)

        setattr(self, param_name, original_value)
        sensitivity = dfdtheta * (original_value / output_base)
        return sensitivity
    def sensitivity_barplot_all_outputs(self, param_names, delta=0.01, y0=None, t_span=(0, 100), t_eval=None, time_index=-1, central_diff=True, show_plot=True):
        import matplotlib.pyplot as plt
        import numpy as np

        output_vars = ['m', 'f', 's']
        sensitivities = {param: {} for param in param_names}

        for param in param_names:
            for var in output_vars:
                try:
                    s = self.parameter_sensitivity(
                        param_name=param,
                        delta=delta,
                        output_var=var,
                        y0=y0,
                        t_span=t_span,
                        t_eval=t_eval,
                        time_index=time_index,
                        central_diff=central_diff
                    )
                    sensitivities[param][var] = s
                except Exception as e:
                    print(f"Error for {param} ({var}): {e}")
                    sensitivities[param][var] = np.nan

        if show_plot:
            x = np.arange(len(param_names))
            width = 0.25

            m_vals = [sensitivities[p]['m'] for p in param_names]
            f_vals = [sensitivities[p]['f'] for p in param_names]
            s_vals = [sensitivities[p]['s'] for p in param_names]

            plt.figure(figsize=(12, 6))
            plt.bar(x - width, m_vals, width, label='Macrophages (m)', color='skyblue')
            plt.bar(x,         f_vals, width, label='Myofibroblasts (f)', color='salmon')
            plt.bar(x + width, s_vals, width, label='Senescent (s)', color='lightgreen')

            plt.xticks(x, param_names, rotation=45, ha='right')
            plt.ylabel("Normalized Sensitivity")
            plt.title(f"Parameter Sensitivities (\u0394 = {delta*100:.1f}%, Central Diff = {central_diff})")
            plt.legend()
            plt.grid(True, axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()

        return sensitivities

    def least_squares_loss(self, theta, param_names, data_dict, y0):

        for name, val in zip(param_names, theta):
            setattr(self, name, val)

            if not np.isfinite(val) or val < 0 or val > 1e3:
                print(f"Invalid value for {name}: {val}, returning inf.")
                return np.inf



        tau_data = np.array(data_dict['t']) * self.mu1

        try:
            result = self.simulate_model(y0, t_span=(tau_data[0], tau_data[-1]), t_eval=tau_data)
            m_sim, f_sim, s_sim = result.y
        except Exception as e:
            print(f"Simulation failed for theta={theta}: {e}")
            return np.inf

        m_obs = np.array(data_dict.get('m', []))
        f_obs = np.array(data_dict.get('f', []))
        s_obs = np.array(data_dict.get('s', []))

        # --- Added weights to balance importance of each variable ---
        # Since f is often much smaller due to being scaled by K, we up-weight its loss term
        w_m, w_f, w_s = 1.0, 1.0, 1.0
        #print(f"theta = {theta}")
       # print(f"m_sim shape: {m_sim.shape}, m_obs: {len(m_obs)}")

        sse = 0.0
        if len(f_obs):
            sse += w_f * np.sum((f_sim - f_obs)**2)
        if len(m_obs):
            sse += w_m * np.sum((m_sim - m_obs)**2)
        if len(s_obs):
            sse += w_s * np.sum((s_sim - s_obs)**2)

        return sse

    def preprocess_data(self,filepath):

        df = pd.read_excel(filepath, skiprows=0)
        t_data = df.iloc[:, 0].to_numpy() * 7  # from weeks to days
        M_data = df.iloc[:, 1].to_numpy()
        F_data = df.iloc[:, 3].to_numpy()
        S_data = df.iloc[:, 5].to_numpy()
       # print(F_data)
        tau_data = t_data * self.mu1
        m_data = M_data / self.M0 if hasattr(self, 'M0') else M_data
        f_data = F_data / self.K  if hasattr(self, 'K')  else F_data
        s_data = S_data / self.S0 if hasattr(self, 'S0') else S_data

        data_dict = {
            't': t_data,  # still in days
            'm': m_data,
            'f': f_data,
            's': s_data
        }
        print("Length of f_data:", len(data_dict['f']))

        return data_dict
    def get_dimensional_from_fitted(self):
        return {
            'mu2': self.phi2 * self.mu1,
            'n':   self.eta * self.M0 * self.mu1 / self.S0,
            'h':   self.chi * self.S0 * self.mu1,
            'r':   self.theta * self.mu1 / self.M0,
        }
    def plot_residuals(self, tau_data, sim_outputs, data_dict):
        """
        Plot residuals (model - data) for all three dimensionless variables: m, f, s.
        Myofibroblast residuals (f) are shown on a separate secondary y-axis.

        Parameters:
            tau_data: array-like, dimensionless time points
            sim_outputs: tuple of (m_sim, f_sim, s_sim)
            data_dict: dict with keys 'm', 'f', 's' containing data arrays
        """
        m_sim, f_sim, s_sim = sim_outputs
        m_data = np.array(data_dict['m'])
        f_data = np.array(data_dict['f'])
        s_data = np.array(data_dict['s'])

        m_resid = m_sim - m_data
        f_resid = f_sim - f_data
        s_resid = s_sim - s_data

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Primary axis for macrophages and senescent cells
        ax1.plot(tau_data, m_resid, marker='s', linestyle='-', label='Macrophage residuals', color='blue')
        ax1.plot(tau_data, s_resid, marker='s', linestyle='-', label='Senescent residuals', color='green')
        ax1.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax1.set_xlabel('Time (dimensionless τ)')
        ax1.set_ylabel('Residuals: Macrophages / Senescent')
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Secondary axis for myofibroblast residuals
        ax2 = ax1.twinx()
        ax2.plot(tau_data, f_resid, marker='s', linestyle='-', label='Myofibroblast residuals', color='red')
        ax2.set_ylabel('Residuals: Myofibroblasts')

        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.title('Residuals for Model Fit')
        plt.tight_layout()
        plt.show()

    def compute_p_from_sim(self, m_array, f_array):
        return [self.steady_state_cp_dimless(m, f)[1] for m, f in zip(m_array, f_array)]
    def compute_P_from_sim_dim(self, m_array, f_array):
        return [self.k1 * self.steady_state_cp_dimless(m, f)[1] for m, f in zip(m_array, f_array)]

    def fit_and_plot(self, param_names, initial_guess, bounds, data_dict, y0):
        """
        Fits the model to data and plots M, F, S from both simulation and data.

        Args:
            param_names: list of parameter names to fit
            initial_guess: list of initial guesses for those parameters
            bounds: list of (min, max) tuples for each parameter
            data_dict: dictionary with keys 't', 'm', 'f', 's' (in dimensionless form)
            y0: initial condition [m0, f0, s0] (dimensionless)
        """
        print("Beginning fit...\n")

        minimizer_kwargs = {
            'method': 'SLSQP',
            'bounds': bounds,
            'args': (param_names, data_dict, y0)
        }

        result = opt.basinhopping(
            func=self.least_squares_loss,
            x0=initial_guess,
            minimizer_kwargs=minimizer_kwargs,
            niter=100
        )
        # from scipy.optimize import differential_evolution

        # result = differential_evolution(
        #     func=self.least_squares_loss,
        #     bounds=bounds,
        #     args=(param_names, data_dict, y0),
        #     maxiter=500,       # number of generations to run
        #     popsize=15,        # population size multiplier
        #     tol=1e-6,          # convergence tolerance
        #     polish=True        # refine best member with L-BFGS-B at the end
        # )

        best_params = result.x
        print("Best fit parameters:")
        for name, val in zip(param_names, best_params):
            print(f"{name}: {val:.4g}")

        # Update model parameters
        for name, val in zip(param_names, best_params):
            setattr(self, name, val)

        tau_data = np.array(data_dict['t']) * self.mu1
        print("tau_data for simulation:", tau_data)

        sol = self.simulate_model(y0, t_span=(tau_data[0], tau_data[-1]), t_eval=tau_data)
        tau_data = sol.t
        m_sim, f_sim, s_sim = sol.y
        p_series = self.compute_p_from_sim(m_sim, f_sim)
        #p_series = self.compute_P_from_sim_dim(m_sim, f_sim)

        plt.plot(tau_data, p_series, label='PDGF (dimensionless p)', color='orange')
        plt.xlabel('Time (τ)')
        plt.ylabel('PDGF (p)')
        plt.title('PDGF over Time')
        plt.grid(False)
        plt.tight_layout()
        plt.show()

        print("f_sim shape:", f_sim.shape)
        print("sol.t:", sol.t)
        print("Expected timepoints:", tau_data)
        print(f't model: {tau_data}\nM model: {m_sim}\nS Model: {s_sim}\nF Model: {f_sim}\n Data: {data_dict}')
        print("Shape of f_sim:", f_sim.shape)

        # ---- Convert back to dimensional parameters
        print("\nMapped back to dimensional parameters:")
        mapped = {}
        if 'phi2' in param_names:
            mapped['mu2'] = self.phi2 * self.mu1
        if 'eta' in param_names:
            mapped['n'] = self.eta * self.M0 * self.mu1 / self.S0
        if 'chi' in param_names:
            mapped['h'] = self.chi * self.S0 * self.mu1
        if 'theta' in param_names:
            mapped['r'] = self.theta * self.mu1 / self.M0
        for k, v in mapped.items():
            print(f"  {k} = {v:.4g}")


            # ---- Plot with second y-axis for F (dimensionless)
        fig, ax1 = plt.subplots()

        ax1.plot(tau_data, data_dict['m'], 'o', linestyle='-', label='Macrophages (data)', color='skyblue')
        ax1.plot(tau_data, data_dict['s'], 'o', linestyle='-', label='Senescent (data)', color='lightgreen')
        ax1.plot(tau_data, m_sim, 's', linestyle='--', label='Macrophages (model)', color='blue')
        ax1.plot(tau_data, s_sim, 's', linestyle='--', label='Senescent (model)', color='green')

        ax1.set_xlabel('Time (dimensionless τ)')
        ax1.set_ylabel('Macrophages / Senescent (dimensionless)')
        ax1.set_yscale('log')

        # ---- Second axis for F (myofibroblasts, dimensionless)
        ax2 = ax1.twinx()
        ax2.plot(tau_data, np.array(data_dict['f']), 'o', linestyle='-', label='Myofibroblasts (data)', color='salmon')
        ax2.plot(tau_data, f_sim, 's', linestyle='--', label='Myofibroblasts (model)', color='red')
        ax2.set_ylabel('Myofibroblasts (dimensionless)')
        ax2.set_yscale('log')

        # ---- Combined legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2)

        plt.title('Model Fit to Data')
        #plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        # Plot residuals
        self.plot_residuals(tau_data, (m_sim, f_sim, s_sim), data_dict)

        return result


    def fit_dimensional(self, param_names, initial_guess, bounds, data_dict, y0):
        """
        Fit model parameters in dimensional form using Basinhopping

        Parameters
        ----------
        param_names : list of str
            Names of parameters to fit (must be attributes on self).
        initial_guess : list of float
            Starting values for each parameter.
        bounds : list of (min, max)
            Bounds for each parameter.
        data_dict : dict
            Contains 't', 'm', 'f', 's' arrays in dimensional units.
        y0 : array-like [M0, F0, S0]
            Initial conditions for macrophages, fibroblasts, senescent cells.
        """
        # Prepare minimizer arguments (SLSQP with bounds)
        minimizer_kwargs = {
            'method': 'SLSQP',
            'bounds': bounds
        }

        # 1. Run Basinhopping
        result = basinhopping(
            func=lambda theta: self.least_squares_loss(theta, param_names, data_dict, y0),
            x0=initial_guess,
            minimizer_kwargs=minimizer_kwargs,
            niter=100,
            stepsize=0.1,
            T=1.0
        )

        # 2. Update only the fitted parameters
        best_theta = result.x
        for name, val in zip(param_names, best_theta):
            setattr(self, name, val)

        print("Optimization success:", result.lowest_optimization_result.success)
        print("Best-fit parameters:")
        for name, val in zip(param_names, best_theta):
            print(f"  {name} = {val:.4g}")

        # 3. Simulate the dimensional 3D ODE (M,F,S) using QSSA for C & P
        t = np.array(data_dict['t'])
        def dyn(t, y):
            M, F, S = y
            dM, dF, dS = self.change_in_m_f(M, F, S)
            return [dM, dF, dS]

        sol = solve_ivp(
            fun=dyn,
            t_span=(t[0], t[-1]),
            y0=y0,
            t_eval=t,
            method='Radau'
        )
        M_sim, F_sim, S_sim = sol.y

        # 4. Plot data vs. model
        plt.figure(figsize=(10, 6))
        plt.plot(t, data_dict['m'], 'o', label='M data')
        plt.plot(t, M_sim, '-', label='M model')
        plt.plot(t, data_dict['f'], 's', label='F data')
        plt.plot(t, F_sim, '--', label='F model')
        plt.plot(t, data_dict['s'], 'd', label='S data')
        plt.plot(t, S_sim, '-.', label='S model')
        plt.xlabel('Time')
        plt.ylabel('Cell count')
        plt.legend()
        plt.title('Dimensional Fit (Basinhopping)')
        plt.show()

        return result




###################
    def classify_slice(self, M, F, S, fixed={'S': 2000}, method='eigen', tol=1e-7, eps=1e-2):
        """
        Classify the stability of a fixed point in a 2D slice (e.g. M-F) while holding one variable fixed.

        Parameters:
            M, F, S (float): Coordinates of the fixed point.
            fixed (dict): Variable(s) to hold fixed. E.g. {'S': 2000}
            method (str): 'eigen', 'dynamics', or 'both'
            tol (float): Eigenvalue threshold for stability.
            eps (float): Step size for dynamics probing.

        Returns:
            dict: Contains verdict, and optionally eigenvalues and direction scores.
        """
        var_names = ['M', 'F', 'S']
        J_full = self.jacobian_full(M, F, S)

        # Identify indices
        fixed_idx = [var_names.index(k) for k in fixed.keys()]
        free_idx = [i for i in range(3) if i not in fixed_idx]

        if method in ['eigen', 'both']:
            # Get submatrices
            J_ff = J_full[np.ix_(free_idx, free_idx)]
            J_perp = J_full[np.ix_(fixed_idx, fixed_idx)]

            lam_free = np.linalg.eigvals(J_ff)
            lam_perp = np.linalg.eigvals(J_perp) if J_perp.size > 0 else np.array([])

            stable_in_slice = np.max(np.real(lam_free)) < tol
            stable_trans = np.max(np.real(lam_perp)) < tol if lam_perp.size > 0 else True
            #print(f'')
            if stable_in_slice and stable_trans:
                verdict_eigen = "stable in slice and transverse"
            elif stable_in_slice and not stable_trans:
                verdict_eigen = "saddle (stable in slice, unstable transverse)"
            else:
                verdict_eigen = "saddle / unstable in slice"

        if method in ['dynamics', 'both']:
            # Probe 2D slice directions: ±x, ±y, diagonals
            directions = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == dy == 0:
                        continue
                    directions.append(np.array([dx, dy]))

            scores = []
            for d in directions:
                d_scaled = eps * d / np.linalg.norm(d)
                m_, f_ = M + d_scaled[0], F + d_scaled[1]

                # Keep fixed variable (e.g., S) constant
                s_ = S

                v = self.dynamics_3D(m_, f_, s_)
                v_free = np.array([v[i] for i in free_idx])
                score = -np.dot(v_free, d_scaled)  # stability if vector points inward
                scores.append(score)

            num_pos = sum(s > 0 for s in scores)
            num_neg = sum(s < 0 for s in scores)

            if num_pos == len(scores):
                verdict_dyn = "stable in slice and transverse"
            elif num_neg == len(scores):
                verdict_dyn = "unstable in slice"
            else:
                verdict_dyn = "saddle / unstable in slice"

        # Return
        if method == 'eigen':
            return {
                "lam_free": lam_free,
                "lam_perp": lam_perp,
                "verdict": verdict_eigen
            }
        elif method == 'dynamics':
            return {
                "scores": scores,
                "verdict": verdict_dyn
            }
        elif method == 'both':
            return {
                "lam_free": lam_free,
                "lam_perp": lam_perp,
                "verdict_eigen": verdict_eigen,
                "scores": scores,
                "verdict_dynamics": verdict_dyn,
                "agree": verdict_eigen == verdict_dyn
            }
        else:
            raise ValueError("Method must be one of: 'eigen', 'dynamics', 'both'")

    def dynamics_3D(self, M, F, S):
        C, P = self.steady_state_CP(M, F)[0][0], self.steady_state_CP(M, F)[1][0]
        dMdt = M * ((self.lam2 * C) / (self.k2 + C) - self.mu2) + self.n * S
        dFdt = F * ((self.lam1 * P) / (self.k1 + P) * (1 - F / self.K) - self.mu1)
        dSdt = self.h - (self.r * S * M) / (S + self.q)
        return np.array([dMdt, dFdt, dSdt])

    def classify_full(self, M, F, S, method='eigen', tol=1e-6, eps=1e-2):
        """
        Classify a fixed point in the full 3D system using Jacobian eigenvalues,
        dynamics-based direction probing, or both.

        Parameters:
            M, F, S (float): Coordinates of the fixed point.
            method (str): 'eigen', 'dynamics', or 'both'.
            tol (float): Tolerance for eigenvalue-based stability.
            eps (float): Radius for perturbations (dynamics method).

        Returns:
            dict: verdict, eigenvalues and/or direction scores.
        """
        if method in ['eigen', 'both']:
            J = self.jacobian_full(M, F, S)
            eigvals = np.linalg.eigvals(J)
            real_parts = np.real(eigvals)

            if np.all(real_parts < -tol):
                verdict_eigen = "stable"
            elif np.all(real_parts > tol):
                verdict_eigen = "unstable"
            else:
                verdict_eigen = "saddle"

        if method in ['dynamics', 'both']:
            # Probe 26 directions around the fixed point
            directions = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        if dx == dy == dz == 0:
                            continue
                        directions.append(np.array([dx, dy, dz]))

            scores = []
            for d in directions:
                d_scaled = eps * d / np.linalg.norm(d)
                Mp, Fp, Sp = M + d_scaled[0], F + d_scaled[1], S + d_scaled[2]
                v = self.dynamics_3D(Mp, Fp, Sp)
                score = -np.dot(v, d_scaled)  # positive = stable in that direction
                scores.append(score)

            num_positive = sum(s > 0 for s in scores)
            num_negative = sum(s < 0 for s in scores)

            if num_positive == len(directions):
                verdict_dynamics = "stable"
            elif num_negative == len(directions):
                verdict_dynamics = "unstable"
            else:
                verdict_dynamics = "saddle"

        # Final output
        if method == 'eigen':
            return {"eigvals": eigvals, "verdict": verdict_eigen}
        elif method == 'dynamics':
            return {"scores": scores, "verdict": verdict_dynamics}
        elif method == 'both':
            return {
                "eigvals": eigvals,
                "verdict_eigen": verdict_eigen,
                "scores": scores,
                "verdict_dynamics": verdict_dynamics,
                "agree": verdict_eigen == verdict_dynamics
            }
        else:
            raise ValueError("method must be one of: 'eigen', 'dynamics', 'both'")

    def eigen_slice(self, M, F, S, fixed={'S':2000}, tol=1e-6):
            var_names = ['M','F','S']
            J_full    = self.jacobian_full(M, F, S)

            # which indices are fixed?
            fixed_idx = [var_names.index(k) for k in fixed.keys()]
            free_idx  = [i for i in range(3) if i not in fixed_idx]

            # extract the in‐slice block J_ff and transverse block J_perp
            J_ff   = J_full[np.ix_(free_idx, free_idx)]
            J_perp = J_full[np.ix_(fixed_idx, fixed_idx)]

            eigplane = np.linalg.eig(J_ff)
            eigperp = np.linalg.eig(J_perp) if J_perp.size>0 else np.array([])

            return eigplane,eigperp
    
    def fixed_curve_3D_old(self, F_range=(1e-2, 10**8), num_points=400, fig=None, plot=True,show=False):
        """
        Compute and optionally plot the fixed curve (intersection of dM/dt=0 and dF/dt=0),
        skipping the pole region between 1e4 and 2.8e4 in F and avoiding visual bridging.
        If a Plotly figure is passed in via `fig`, add the trace to it.
        """

        # Create F in two segments (below and above the pole)
        F_segments = [
            np.logspace(np.log10(F_range[0]), np.log10(1.5e4), num_points // 2),
            np.logspace(np.log10(2.2e4), np.log10(F_range[1]), num_points // 2)
        ]

        if fig is None and plot:
            fig = go.Figure()
        first = True
        for F_vals in F_segments:
            M_vals, S_vals = [], []
            for F in F_vals:
                try:
                    M = self.nullclines_F(F)[1]
                    S = (self.h * self.q) / (self.r * M - self.h)
                    if np.isreal(S) and np.isreal(M) and M > 0 and S > 0:
                        M_vals.append(M)
                        S_vals.append(S)
                    else:
                        M_vals.append(np.nan)
                        S_vals.append(np.nan)
                except:
                    M_vals.append(np.nan)
                    S_vals.append(np.nan)

            # Remove NaNs from each segment
            M_vals = np.array(M_vals)
            S_vals = np.array(S_vals)
            F_valid = np.array(F_vals)
            valid = ~np.isnan(M_vals) & ~np.isnan(S_vals)
            M_vals = M_vals[valid]
            S_vals = S_vals[valid]
            F_valid = F_valid[valid]

            if plot:
                fig.add_trace(go.Scatter3d(
                    x=np.log10(M_vals), y=np.log10(F_valid), z=np.log10(S_vals),
                    mode='lines',
                    line=dict(color='black', width=4),
                    name='M-F Nullcline Intersection',
                    legendgroup='fixed_curve',    # group both traces
                    showlegend=first            # only show legend on first trace
                ))
                first = False

        if plot:
            fig.update_layout(
                legend=dict(
                    x=0.85,         # move from 1.02 to ~0.85 (85% of figure width)
                    y=0.90,         # move slightly down from 1.0 so it doesn’t collide with the top margin
                    xanchor="left", # interpret x=0.85 as the left edge of the legend
                    yanchor="top",
                ),
                margin=dict(r=40)   # you might need a small right margin so the legend text doesn’t get cropped
            )
            fig.update_layout(
                scene=dict(
                    xaxis=dict(
                        title=f"log(Macrophages)<br>log((cells/ml))",
                        title_font=dict(size=9), 
                        tickfont=dict(size=9), 
                        range=[-2,6.0],     # force the x‐axis to start at 0
                        zeroline=False,      # draw the black line at x=0
                        autorange=False,         # turn off auto‐rescaling
                        zerolinewidth=2,
                        zerolinecolor="black"
                        # showbackground=True,
                        # backgroundcolor="rgba(225, 225, 225, 0.5)"
                    ),
                    yaxis=dict(
                        title=f"log(Myofibroblasts)<br>log((cells/ml))",
                        title_font=dict(size=9),
                        tickfont=dict(size=9),
                        range=[-2, 6.0],     # force y=0
                        autorange=False,         # turn off auto‐rescaling
                        zeroline=False,
                        zerolinewidth=2,
                        zerolinecolor="black"
                        # showbackground=True,
                        # backgroundcolor="rgba(225, 225, 225, 0.5)"
                    ),
                    zaxis=dict(
                        title=f"log(Senescent cells)<br>log((cells/ml))",
                        title_font=dict(size=9),
                        tickfont=dict(size=9),
                        range=[-2, 6.0],     # force z=0
                        autorange=False,         # turn off auto‐rescaling
                        zeroline=False,
                        zerolinewidth=2,
                        zerolinecolor="black"
                        # showbackground=True,
                        # backgroundcolor="rgba(225, 225, 225, 0.5)"
                    ),
                    aspectmode="cube"      # keeps x, y, z≙1:1:1 so the box is not distorted
                )
            )
            if show:
                fig.show()

        # Return just the concatenated values
        if fig == None:
            return None  # You could optionally return both chunks here if needed
        else:
            return fig
    def fixed_curve_3D(self, F_range=(1e-2, 10**8), num_points=400, fig=None, plot=True):
        """
        Compute & plot the fixed curve, coloring each segment by stability:
        • orangered = stable
        • plum      = semi‐stable (saddle)
        • skyblue   = unstable
        """

        # … (same F_segments construction as before) …
        F_segments = [
            np.logspace(np.log10(F_range[0]), np.log10(1.5e4), num_points // 2),
            np.logspace(np.log10(2.2e4), np.log10(F_range[1]), num_points // 2)
        ]
        if fig is None and plot:
            fig = go.Figure()

        # Replace this helper with your new color choices:
        def verdict_to_color(v):
            if   "stable"   in v: return "orangered"
            elif "saddle"   in v: return "plum"
            elif "unstable" in v: return "skyblue"
            else:                 return "gray"   # fallback if something unexpected appears

        for block_idx, F_vals in enumerate(F_segments):
            M_list, S_list, F_list, V_list = [], [], [], []

            for F in F_vals:
                try:
                    M = self.nullclines_F(F)[1]
                    S = (self.h * self.q) / (self.r * M - self.h)
                except:
                    M, S = np.nan, np.nan

                if np.isreal(M) and np.isreal(S) and (M > 0) and (S > 0):
                    M_list.append(M)
                    S_list.append(S)
                    F_list.append(F)

                    # Classify with the full‐3D Jacobian (verdict is "stable","saddle", or "unstable")
                    cinfo = self.classify_full(M, F, S, method="dynamics")
                    V_list.append(cinfo["verdict"])
                else:
                    M_list.append(np.nan)
                    S_list.append(np.nan)
                    F_list.append(np.nan)
                    V_list.append("invalid")

            M_arr = np.array(M_list)
            S_arr = np.array(S_list)
            F_arr = np.array(F_list)
            V_arr = np.array(V_list)

            good_mask = (~np.isnan(M_arr)) & (~np.isnan(S_arr)) & (V_arr != "invalid")
            M_arr = M_arr[good_mask]
            S_arr = S_arr[good_mask]
            F_arr = F_arr[good_mask]
            V_arr = V_arr[good_mask]

            if not plot or (len(M_arr) == 0):
                continue

            # Break V_arr into runs of consecutive identical verdicts
            runs = []
            start_idx = 0
            for i in range(1, len(V_arr)):
                if V_arr[i] != V_arr[i - 1]:
                    runs.append((start_idx, i - 1, V_arr[i - 1]))
                    start_idx = i
            runs.append((start_idx, len(V_arr) - 1, V_arr[-1]))

            # Plot each run as a separate 3D line with its mapped color
            for run_ix, run_fx, verdict in runs:
                Mi = M_arr[run_ix : run_fx + 1]
                Fi = F_arr[run_ix : run_fx + 1]
                Si = S_arr[run_ix : run_fx + 1]
                color = verdict_to_color(verdict)

                fig.add_trace(
                    go.Scatter3d(
                        x=np.log10(Mi),
                        y=np.log10(Fi),
                        z=np.log10(Si),
                        mode='lines',
                        line=dict(color=color, width=4),
                        name=f"Fixed Curve ({verdict})",
                        legendgroup='fixed_curve',
                        showlegend=(block_idx == 0 and run_ix == 0)
                    )
                )

        if plot:
            fig.update_layout(
                legend=dict(
                    x=0.85, y=0.90,
                    xanchor="left", yanchor="top",
                ),
                margin=dict(r=40)
            )
            fig.update_layout(
                scene=dict(
                    xaxis=dict(
                        title="log(Macrophages)",
                        range=[7.0, 0],
                        zeroline=False,
                        zerolinewidth=2,
                        zerolinecolor="black",
                        autorange=False
                    ),
                    yaxis=dict(
                        title="log(Myofibroblasts)",
                        range=[0, 6.0],
                        zeroline=False,
                        zerolinewidth=2,
                        zerolinecolor="black",
                        autorange=False
                    ),
                    zaxis=dict(
                        title="log(Senescent cells)",
                        range=[0, 8.0],
                        zeroline=False,
                        zerolinewidth=2,
                        zerolinecolor="black",
                        autorange=False
                    ),
                    aspectmode="cube"
                )
            )
            fig.show()

        return fig


    def nullclines_3D_plotly(self, start, stop, steps,return_fig=False,show=False):
        """
        Plot the three nullclines in 3D using Plotly:
        - dM/dt = 0 (magenta)
        - dF/dt = 0 (cyan), avoiding known pole region
        - dS/dt = 0 (green)

        All surfaces are log10-transformed and clipped to valid domains.
        """

        # Log-spaced grid
        M_vals = np.logspace(start, stop, steps)
        S_vals = np.logspace(start, stop, steps)

        # Safe F values (avoid pole between 10^5.7 and 10^5.85)
        Fnull1 = np.logspace(start, 5.7, steps // 2)
        Fnull2 = np.logspace(5.85, stop, steps // 2)
        F_vals = np.concatenate([Fnull1])#, Fnull2])

        # Grids for each nullcline
        M_grid1, F_grid1 = np.meshgrid(M_vals, F_vals, indexing='ij')
        F_grid2, S_grid2 = np.meshgrid(F_vals, S_vals, indexing='ij')
        S_grid3, F_grid3 = np.meshgrid(S_vals, F_vals, indexing='ij')
        # dM/dt = 0 surface (S from analytic formula)
        with np.errstate(divide='ignore', invalid='ignore'):
            S_null = (self.h * self.q) / (self.r * M_grid1 - self.h)
            S_null = np.where(S_null <= 0, np.nan, S_null)

        # dF/dt = 0 surface (M from nullclines_F)
        M_null = np.full_like(F_grid2, np.nan)
        for i in range(F_grid2.shape[0]):
            last_valid = 1e-6  # safe fallback
            for j in range(F_grid2.shape[1]):
                Fval = F_grid2[i, j]
                try:
                    Mval = self.nullclines_F(Fval)[1]
                    if np.isreal(Mval) and np.isfinite(Mval) and Mval > 0:
                        M_null[i, j] = Mval
                        last_valid = Mval
                    else:
                        M_null[i, j] = last_valid
                except:
                    M_null[i, j] = last_valid

        # dS/dt = 0 surface (M from analytic formula)
        with np.errstate(divide='ignore', invalid='ignore'):
            M_snull = (self.h * (self.q + S_grid3)) / (self.r * S_grid3)
            M_snull = np.clip(M_snull, 1e-6, 1e7)
            M_snull[S_grid3 < 1e-3] = np.nan

        # Clip values for log10
        def safe_log(arr, min_val=1e-6):
            arr = np.where(arr <= 0, min_val, arr)
            return np.log10(arr)

        # Create surface traces
        surfaces = [
            go.Surface(
                x=safe_log(M_grid1), y=safe_log(F_grid1), z=safe_log(S_null),
                colorscale=[[0,"seagreen"], [1, 'seagreen']],
                name='dM/dt = 0', showscale=False, opacity=0.5,showlegend=True
            ),
            go.Surface(
                x=safe_log(M_null), y=safe_log(F_grid2), z=safe_log(S_grid2),
                colorscale=[[0, 'navy'], [1, "navy"]],
                name='dF/dt = 0', showscale=False, opacity=0.5,showlegend=True
            )
        ]

        layout = go.Layout(
            scene=dict(
                xaxis_title="log(Macrophages), log(cells/ml)",
                yaxis_title="log₁₀(Myofibroblasts), log(cells/ml)",
                zaxis_title="log₁₀(Senescent Cells), log(cells/ml)",
                xaxis=dict(range=[start, stop]),
                yaxis=dict(range=[start, stop]),
                zaxis=dict(range=[start, stop]),
            ),
            legend=dict(
                x=0.8, y=0.9,
                bgcolor='rgba(255,255,255,0.5)'
            )
        )

        fig = go.Figure(data=surfaces, layout=layout)
        fig.update_layout(
            legend=dict(
                title="Nullclines",
                itemsizing='trace'
            )
        )
        fig.update_layout(
        scene=dict(
            xaxis=dict(
                title="Macrophages (cells/ml)",
                title_font=dict(size=18),    # axis title font size
                tickfont=dict(size=12)      # tick label font size
            ),
            yaxis=dict(
                title="Myofibroblasts (cells/ml)",
                title_font=dict(size=18),
                tickfont=dict(size=12)
            ),
            zaxis=dict(
                title="Senescent cells (cells/ml)",
                title_font=dict(size=18),
                tickfont=dict(size=12)
            )
        )
        )

        if return_fig:
            return fig
        if show:
            fig.show()



    def plot_log_quiver_3D(self,
                       M_range=(1e0, 1e5),
                       F_range=(1e0, 1e5),
                       S_range=(1e0, 1e5),
                       n=4,
                       sizeref=0.1,
                       uniform_color="steelblue",
                       opacity=0.7):
        """
        3D cone plot on log–log–log axes with:
        • All cones the same length (sizeref).
        • A single uniform color (uniform_color).
        • Direction = unit‐vector of (dM,dF,dS).
        """

        # 1) Build log‐grid
        exp_min_M, exp_max_M = np.log10(M_range[0]), np.log10(M_range[1])
        exp_min_F, exp_max_F = np.log10(F_range[0]), np.log10(F_range[1])
        exp_min_S, exp_max_S = np.log10(S_range[0]), np.log10(S_range[1])

        logM_vals = np.linspace(exp_min_M, exp_max_M, n)
        logF_vals = np.linspace(exp_min_F, exp_max_F, n)
        logS_vals = np.linspace(exp_min_S, exp_max_S, n)

        M_vals = 10 ** logM_vals
        F_vals = 10 ** logF_vals
        S_vals = 10 ** logS_vals

        Mg, Fg, Sg = np.meshgrid(M_vals, F_vals, S_vals, indexing="ij")
        X = Mg.ravel()
        Y = Fg.ravel()
        Z = Sg.ravel()

        # 2) Compute raw (dM, dF, dS)
        dM_flat = np.empty_like(X)
        dF_flat = np.empty_like(X)
        dS_flat = np.empty_like(X)

        for idx, (m, f, s) in enumerate(zip(X, Y, Z)):
            dM_flat[idx], dF_flat[idx], dS_flat[idx] = self.change_in_m_f(m, f, s)

        # 3) Convert to (delta_x, delta_y, delta_z) in log‐space
        eps = 1e-12
        delta_x = (dM_flat / np.maximum(X, eps)) / np.log(10)
        delta_y = (dF_flat / np.maximum(Y, eps)) / np.log(10)
        delta_z = (dS_flat / np.maximum(Z, eps)) / np.log(10)

        # 4) Compute magnitudes
        magnitude = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

        # 5) Normalize to unit length where magnitude > 0
        u = np.zeros_like(delta_x)
        v = np.zeros_like(delta_y)
        w = np.zeros_like(delta_z)

        nonzero = magnitude > 0
        u[nonzero] = delta_x[nonzero] / magnitude[nonzero]
        v[nonzero] = delta_y[nonzero] / magnitude[nonzero]
        w[nonzero] = delta_z[nonzero] / magnitude[nonzero]

        # 6) Build the Cone trace with a single uniform color
        X_tail = np.log10(X)
        Y_tail = np.log10(Y)
        Z_tail = np.log10(Z)

        cone_trace = go.Cone(
            x=X_tail,
            y=Y_tail,
            z=Z_tail,
            u=u,
            v=v,
            w=w,
            sizemode="absolute",
            sizeref=sizeref,
            anchor="tail",
            colorscale=[[0, uniform_color], [1, uniform_color]],
            showscale=False,opacity=opacity
        )

        fig = go.Figure(data=cone_trace)

        # 7) Configure scene for log‐axes, equal aspect ratio
        fig.update_layout(
            scene=dict(
                xaxis=dict(
                    title="log₁₀(Macrophages)",
                    range=[exp_min_M, exp_max_M],
                    showgrid=True
                ),
                yaxis=dict(
                    title="log₁₀(Myofibroblasts)",
                    range=[exp_min_F, exp_max_F],
                    showgrid=True
                ),
                zaxis=dict(
                    title="log₁₀(Senescent Cells)",
                    range=[exp_min_S, exp_max_S],
                    showgrid=True
                ),
                aspectmode="cube",
                camera=dict(eye=dict(x=1.4, y=1.4, z=1.2))
            ),
            margin=dict(l=0, r=0, t=30, b=0)
        )

        return fig






    def nullclines_3D(self, start, stop, steps):
        """
        Plot the three nullcline surfaces in 3D:
          - dM/dt = 0 over the (M,F) plane
          - dF/dt = 0 over the (F,S) plane
          - dS/dt = 0 over the (S,F) plane

        Parameters:
            start, stop : exponents for logspace
            steps       : number of points per axis
        """

        from matplotlib.patches import Patch

        # prepare log-spaced grids
        M_vals = np.logspace(start, stop, steps)
        F_vals = np.logspace(start, stop, steps)
        S_vals = np.logspace(start, stop, steps)

        fig = plt.figure()
        ax  = fig.add_subplot(projection='3d')
        ax.set_xlabel("log₁₀(Macrophages)")
        ax.set_ylabel("log₁₀(Myofibroblasts)")
        ax.set_zlabel("log₁₀(Senescent Cells)")

        # --- dM/dt = 0 surface (M × F) ---
        M_grid, F_grid = np.meshgrid(M_vals, F_vals, indexing='ij')
        # analytical S-nullcline formula for dM/dt=0
        S_M = (self.h * self.q) / (self.r * M_grid - self.h)
        surf_M = ax.plot_surface(
            np.log10(M_grid),
            np.log10(F_grid),
            np.log10(S_M),
            color='magenta', alpha=0.4
        )

        # --- dF/dt = 0 surface (F × S) ---
        F_grid, S_grid = np.meshgrid(F_vals, S_vals, indexing='ij')
        # for each F, nullclines_F returns [F, M]
        M_F = np.vectorize(lambda f: self.nullclines_F(f)[1])(F_grid)
        surf_F = ax.plot_surface(
            np.log10(M_F),
            np.log10(F_grid),
            np.log10(S_grid),
            color='cyan', alpha=0.4
        )

        # --- S-nullcline (dS/dt=0) as a full ribbon in F and S ---
        # build a full (S,F) grid
        S_grid, F_grid = np.meshgrid(S_vals, F_vals, indexing='ij')

        # compute M from the analytic S-nullcline M = h (q + S)/(r S)
        M_S_grid = (self.h * (self.q + S_grid)) / (self.r * S_grid)

        # now plot it as a proper surface
        ax.plot_surface(
            np.log10(M_S_grid),
            np.log10(F_grid),
            np.log10(S_grid),
            color='lime', alpha=0.3
        )

        # build a legend by proxy
        legend_handles = [
            Patch(color='magenta', alpha=0.4, label=r'$dM/dt=0$'),
            Patch(color='cyan',    alpha=0.4, label=r'$dF/dt=0$'),
            Patch(color='lime',    alpha=0.4, label=r'$dS/dt=0$'),
        ]
        ax.legend(handles=legend_handles, loc='upper left')

        plt.show()
    def plot_separatrix_surface_3D(self,
                                initial_guess=(5e3, 5e3, 1e4),
                                t_end=100,
                                n_dirs=20,
                                epsilon=500,
                                method="Radau",
                                color="black",
                                alpha=0.6):
        import numpy as np
        from scipy.integrate import solve_ivp
        import matplotlib.pyplot as plt

        print("🔍 Step 1: Locating fixed point from guess", initial_guess)
        fixed_pt = self.fixed_points_3D(np.array(initial_guess))
        if np.any(np.isnan(fixed_pt)):
            raise RuntimeError(f"  ✗ Could not find fixed point from {initial_guess}")
        M0, F0, S0 = fixed_pt
        res = self.residual_fixed_point(fixed_pt)
        print(f"  ✔ Found fixed point at M={M0:.3g}, F={F0:.3g}, S={S0:.3g}")
        print(f"  Residual at fixed point: {res}")

        print("🧮 Step 2: Computing Jacobian and eigen-decomposition")
        J = self.jacobian_full(M0, F0, S0)
        vals, vecs = np.linalg.eig(J)
        reals = np.real(vals)
        print("  Eigenvalues:", ", ".join(f"{v:.3g}" for v in vals))

        order = np.argsort(reals)
        stable_idx = order[:2]
        print(f"  → Stable directions: indices {stable_idx}, eigenvalues {reals[stable_idx]}")

        v1 = np.real(vecs[:, stable_idx[0]])
        v2 = np.real(vecs[:, stable_idx[1]])
        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)

        print("🌀 Step 3: Creating grid of initial perturbations")
        us = np.linspace(-1, 1, n_dirs)
        vs = np.linspace(-1, 1, n_dirs)
        U, V = np.meshgrid(us, vs, indexing="ij")
        perturbations = epsilon * (U[..., None] * v1 + V[..., None] * v2)
        initials = fixed_pt + perturbations

        print("⏪ Step 4: Backwards integration")
        Mf, Ff, Sf = np.full((3, n_dirs, n_dirs), np.nan)
        for i in range(n_dirs):
            for j in range(n_dirs):
                ic = initials[i, j]
                if np.any(ic <= 0) or np.any(np.isnan(ic)):
                    print(f"    ⚠ Skipping [{i},{j}] due to invalid init: {ic}")
                    continue
                try:
                    sol = solve_ivp(self.change_in_m_f_to_int,
                                    (0, -t_end),
                                    ic,
                                    t_eval=[-t_end],
                                    method=method,
                                    rtol=1e-6, atol=1e-9)
                    if sol.success and sol.y.shape[1] > 0:
                        Mf[i, j] = sol.y[0, -1]
                        Ff[i, j] = sol.y[1, -1]
                        Sf[i, j] = sol.y[2, -1]
                    else:
                        print(f"    ❗ Integration failed at [{i},{j}]: {sol.message}")
                except Exception as e:
                    print(f"    ❌ Exception at [{i},{j}]: {e}")

        print("🧱 Step 5: Plotting surface in log10-space")
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(projection='3d')
        surf = ax.plot_surface(np.log10(Mf), np.log10(Ff), np.log10(Sf),
                            color=color, alpha=alpha,
                            rcount=n_dirs, ccount=n_dirs)

        ax.set_xlabel("log10(Macrophages)")
        ax.set_ylabel("log10(Myofibroblasts)")
        ax.set_zlabel("log10(Senescent Cells)")
        ax.set_title("Stable-Manifold Separatrix Surface")

        ax.scatter(np.log10(M0), np.log10(F0), np.log10(S0),
                color="red", s=50, marker="o", label="Saddle Point", zorder=10)
        ax.legend()
        plt.tight_layout()
        plt.show()

        return fig, ax

    def separatrix_surface_finite_difference(self,
                                            M_range=(1e2, 1e4),
                                            F_range=(1e2, 1e4),
                                            S_range=(1e2, 1e4),
                                            steps=20,
                                            t_end=200,
                                            threshold=1e-1):
        """
        Approximate the 3D separatrix surface using finite differences.
        This method detects where neighboring initial conditions diverge
        to different final outcomes, identifying separatrix boundaries.

        Parameters:
            M_range : tuple, min and max for macrophages
            F_range : tuple, min and max for myofibroblasts
            S_range : tuple, min and max for senescent cells
            steps   : resolution of the grid in each dimension
            t_end   : integration time to allow settling
            threshold : minimal difference considered meaningful

        Returns:
            A plot showing separatrix points in 3D
        """
        import matplotlib.pyplot as plt

        M_vals = np.logspace(np.log10(M_range[0]), np.log10(M_range[1]), steps)
        F_vals = np.logspace(np.log10(F_range[0]), np.log10(F_range[1]), steps)
        S_vals = np.logspace(np.log10(S_range[0]), np.log10(S_range[1]), steps)

        print(f"📦 Building {steps**3} grid points to evaluate...")

        sep_points = []

        for i, M in enumerate(tqdm(M_vals, desc="Scanning grid")):
            for j, F in enumerate(F_vals):
                for k, S in enumerate(S_vals):
                    # Create 8 slightly perturbed neighbors in a small cube
                    eps = 1e-1
                    neighbors = np.array([
                        [M + eps, F + eps, S + eps],
                        [M + eps, F + eps, S - eps],
                        [M + eps, F - eps, S + eps],
                        [M + eps, F - eps, S - eps],
                        [M - eps, F + eps, S + eps],
                        [M - eps, F + eps, S - eps],
                        [M - eps, F - eps, S + eps],
                        [M - eps, F - eps, S - eps],
                    ])
                    outcomes = []

                    for x in neighbors:
                        try:
                            sol = solve_ivp(self.change_in_m_f_to_int,
                                            (0, t_end), x,
                                            t_eval=[t_end],
                                            method='Radau',
                                            rtol=1e-6, atol=1e-9)
                            if sol.success:
                                final_state = sol.y[:, -1]
                                outcomes.append(final_state)
                        except Exception as e:
                            continue

                    if len(outcomes) < 2:
                        continue

                    # Compute pairwise differences
                    for a in range(len(outcomes)):
                        for b in range(a + 1, len(outcomes)):
                            diff = np.linalg.norm(outcomes[a] - outcomes[b])
                            if diff > threshold:
                                sep_points.append([M, F, S])
                                break
                        else:
                            continue
                        break  # break outer loop if any pair differs

        sep_points = np.array(sep_points)

        if len(sep_points) == 0:
            print("⚠ No separatrix points found.")
            return

        print(f"🧠 Found {len(sep_points)} separatrix points.")

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(np.log10(sep_points[:, 0]),
                np.log10(sep_points[:, 1]),
                np.log10(sep_points[:, 2]),
                s=4, color="black", alpha=0.5)

        ax.set_xlabel("log10(Macrophages)")
        ax.set_ylabel("log10(Myofibroblasts)")
        ax.set_zlabel("log10(Senescent Cells)")
        ax.set_title("Separatrix Surface (Finite Difference Estimate)")
        plt.tight_layout()
        plt.show()

        return sep_points
    def classify_final_state(self, y, threshold=1.0):
        """
        Classify the outcome of a trajectory by examining its final state.
        Assumes that a high value of F (myofibroblasts) corresponds to fibrosis.
        Returns 0 for 'healing' (F < threshold), 1 for 'fibrotic' (F >= threshold)
        """
        return int(y[1] >= threshold)


    def integrate_and_classify(self,args):
        fsm, M0, F0, S0, t_span, threshold = args
        try:
            sol = solve_ivp(
                fun=fsm.change_in_m_f_to_int,
                t_span=t_span,
                y0=[M0, F0, S0],
                method='Radau',
                t_eval=[t_span[1]],
                rtol=1e-6,
                atol=1e-9
            )
            if sol.success and sol.y.shape[1] > 0:
                return (M0, F0, S0, self.classify_final_state(sol.y[:, -1], threshold))
            else:
                return (M0, F0, S0, -1)
        except Exception:
            return (M0, F0, S0, -1)


    def plot_separatrix_finite_difference_parallel(self, M_range, F_range, S_range,
                                                resolution=20, t_end=100, threshold=1.0):
        """
        Drop-in parallelized finite difference separatrix computation and plotting.

        Parameters:
        - fsm: your fibrosis_senescence_model instance
        - M_range, F_range, S_range: tuples (min, max) for the grid in each dimension
        - resolution: number of points per dimension (cubic grid)
        - t_end: integration time
        - threshold: threshold on F to define fibrosis

        Returns:
        - None (displays 3D scatter plot)
        """
        M_vals = np.logspace(np.log10(M_range[0]), np.log10(M_range[1]), resolution)
        F_vals = np.logspace(np.log10(F_range[0]), np.log10(F_range[1]), resolution)
        S_vals = np.logspace(np.log10(S_range[0]), np.log10(S_range[1]), resolution)

        points = list(itertools.product(M_vals, F_vals, S_vals))
        args = [(self, M, F, S, (0, t_end), threshold) for M, F, S in points]

        print(f"Launching parallel classification of {len(points)} points...")
        results = []
        with ProcessPoolExecutor() as executor:
            results = list(executor.map(self.integrate_and_classify, args))

        print("Integration complete. Plotting results...")
        M_heal, F_heal, S_heal = [], [], []
        M_fib, F_fib, S_fib = [], [], []

        for M, F, S, label in results:
            if label == 0:
                M_heal.append(M)
                F_heal.append(F)
                S_heal.append(S)
            elif label == 1:
                M_fib.append(M)
                F_fib.append(F)
                S_fib.append(S)

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(np.log10(M_heal), np.log10(F_heal), np.log10(S_heal),
                color='green', label='Healing', alpha=0.5)
        ax.scatter(np.log10(M_fib), np.log10(F_fib), np.log10(S_fib),
                color='red', label='Fibrotic', alpha=0.5)

        ax.set_xlabel("log10(Macrophages)")
        ax.set_ylabel("log10(Myofibroblasts)")
        ax.set_zlabel("log10(Senescent Cells)")
        ax.set_title("Separatrix via Finite Differences (Parallelized)")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_separatrix_surface_fd(self,
        M_range, F_range, S_range,
        resolution=40,
        t_end=100,
        threshold=1.0,
        cmap='Greys',
        alpha=0.4):
        """
        Finite-difference separatrix surface in (M,F,S), parallelized.

        1) Build a log‐spaced 3D grid of initial conditions.
        2) Classify each one in parallel (healing vs fibrotic).
        3) Reshape into a 3D array of labels.
        4) Extract the 0.5 iso‐surface via marching cubes.
        5) Plot the resulting triangular mesh.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # 1) log‐space axes
        logM = np.linspace(np.log10(M_range[0]), np.log10(M_range[1]), resolution)
        logF = np.linspace(np.log10(F_range[0]), np.log10(F_range[1]), resolution)
        logS = np.linspace(np.log10(S_range[0]), np.log10(S_range[1]), resolution)

        # prepare all (m,f,s) points in log‐space
        points = list(itertools.product(logM, logF, logS))
        npts = len(points)

        # build the args for integrate_and_classify(self,args)
        args = [
            (self, 10**m, 10**f, 10**s, (0, t_end), threshold)
            for m, f, s in points
        ]

        # 2) classify in parallel, with a progress bar
        print(f"🔎 Classifying {npts} points in parallel...")
        with ProcessPoolExecutor() as exe:
            # wrap the executor.map with tqdm so you get a live bar
            results = list(tqdm(
                exe.map(self.integrate_and_classify, args),
                total=npts,
                desc="Classifying"
            ))

        # 3) pack into a 3D array of labels
        labels = np.full((resolution,)*3, np.nan, dtype=float)
        for idx, (_, _, _, lab) in enumerate(results):
            i = idx // (resolution*resolution)
            j = (idx // resolution) % resolution
            k = idx % resolution
            if lab >= 0:
                labels[i,j,k] = lab

        # 4) marching cubes to extract the interface at level=0.5
        dM = logM[1] - logM[0]
        dF = logF[1] - logF[0]
        dS = logS[1] - logS[0]
        verts, faces, _, _ = measure.marching_cubes(
            labels, level=0.5, spacing=(dM, dF, dS)
        )
        # shift back into absolute log‐space
        verts += np.array([logM[0], logF[0], logS[0]])

        # 5) plot the mesh
        fig = plt.figure(figsize=(8, 6))
        ax  = fig.add_subplot(projection='3d')
        mesh = Poly3DCollection(verts[faces], alpha=alpha)
        mesh.set_facecolor(plt.get_cmap(cmap)(0.5))
        ax.add_collection3d(mesh)

        ax.set_xlim(logM[0], logM[-1])
        ax.set_ylim(logF[0], logF[-1])
        ax.set_zlim(logS[0], logS[-1])
        ax.set_xlabel("log₁₀(Macrophages)")
        ax.set_ylabel("log₁₀(Myofibroblasts)")
        ax.set_zlabel("log₁₀(Senescent Cells)")
        ax.set_title("Separatrix Surface (FD / Iso‐surface)")
        plt.tight_layout()
        plt.show()

        return fig, ax
    def monte_carlo_classify(self, 
                                    n_samples=5000,
                                    M_bounds=(0.01, 10**5.85),
                                    F_bounds=(0.01, 10**4.5),
                                    S_bounds=(0.01, 10**5.85),
                                    t_end=200,
                                    threshold=1.0,
                                    save_file="monte_carlo_classification_new.npz"):
        """
        Monte Carlo classification over random (M,F,S) points.
        Saves the results for use with marching cubes or Plotly.
        """
        M_samples = np.random.uniform(np.log10(M_bounds[0]), np.log10(M_bounds[1]), n_samples)
        F_samples = np.random.uniform(np.log10(F_bounds[0]), np.log10(F_bounds[1]), n_samples)
        S_samples = np.random.uniform(np.log10(S_bounds[0]), np.log10(S_bounds[1]), n_samples)

        initial_conditions = 10 ** np.vstack([M_samples, F_samples, S_samples]).T

        from concurrent.futures import ProcessPoolExecutor
        import tqdm

        print(f"🌱 Sampling {n_samples} points in (M,F,S) space...")
        args = [(self, M, F, S, (0, t_end), threshold) for M, F, S in initial_conditions]

        with ProcessPoolExecutor() as pool:
            results = list(tqdm.tqdm(pool.map(self.integrate_and_classify, args), total=n_samples))

        points = []
        labels = []

        for M, F, S, outcome in results:
            if outcome in [0, 1]:
                points.append([M, F, S])
                labels.append(outcome)

        points = np.array(points)
        labels = np.array(labels)

        # Save
        np.savez(save_file, points=points, labels=labels)
        print(f"💾 Saved {len(points)} classified samples to {save_file}")

        return points, labels


    def extract_marching_cubes_surface(self, pts, labels, resolution=50, level=0.5,sigma=1, cmap='Greys'):
        """
        Interpolate labeled Monte Carlo points onto a 3D grid and extract the separatrix via marching cubes.

        Parameters:
            pts      : ndarray of shape (N, 3), sampled (M,F,S) points
            labels   : ndarray of shape (N,), 0 or 1 classification
            resolution : number of bins per axis for voxelization
            level    : iso-surface value (typically 0.5 for binary separation)
            cmap     : colormap name for plotly surface

        Returns:
            verts, faces: for further use or export
        """
        from scipy.interpolate import griddata
        from skimage import measure
        import plotly.graph_objects as go
        import numpy as np

        logM = np.log10(pts[:, 0])
        logF = np.log10(pts[:, 1])
        logS = np.log10(pts[:, 2])

        domain = [
            (logM.min(), logM.max()),
            (logF.min(), logF.max()),
            (logS.min(), logS.max())
        ]

        print("📊 Interpolating data to 3D grid...")
        xi = np.linspace(*domain[0], resolution)
        yi = np.linspace(*domain[1], resolution)
        zi = np.linspace(*domain[2], resolution)
        grid_x, grid_y, grid_z = np.meshgrid(xi, yi, zi, indexing='ij')

        grid_vals = griddata(
            points=(logM, logF, logS),
            values=labels,
            xi=(grid_x, grid_y, grid_z),
            method='linear',
            fill_value=0.5
        )
        from scipy.ndimage import gaussian_filter

        # Smooth the interpolated 3D data
        smoothed_vals = gaussian_filter(grid_vals, sigma=sigma)  # adjust sigma as needed

        print("📐 Extracting isosurface...")
        verts, faces, _, _ = measure.marching_cubes(
            smoothed_vals,
            level=level,
            spacing=(xi[1]-xi[0], yi[1]-yi[0], zi[1]-zi[0])
        )

        verts += np.array([xi[0], yi[0], zi[0]])

        print("🎨 Rendering Plotly surface...")
        i, j, k = faces.T

        fig = go.Figure(data=[
            go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=i, j=j, k=k,
                opacity=0.5,
                color='lightgrey',
                name='Separatrix Surface'
            )
        ])

        fig.update_layout(
            title="Separatrix Surface (Monte Carlo + Marching Cubes)",
            scene=dict(
                xaxis_title="log₁₀(Macrophages)",
                yaxis_title="log₁₀(Myofibroblasts)",
                zaxis_title="log₁₀(Senescent Cells)",
                xaxis=dict(range=[domain[0][0], domain[0][1]]),
                yaxis=dict(range=[domain[1][0], domain[1][1]]),
                zaxis=dict(range=[domain[2][0], domain[2][1]])
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        fig.show()

        return verts, faces
