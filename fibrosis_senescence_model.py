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

    def debug_nullcline_M_S(self, M_vals, S_fixed, verbose=True):
        """
        Diagnose breakdowns in dM/dt=0 nullcline for fixed S.
        Returns: arrays of good and bad M values.
        """
        bad_indices = []
        for i, M in enumerate(M_vals):
            try:
                term = self.mu2 - (self.n * S_fixed) / M
                if term <= 0 or term >= self.lam2:
                    bad_indices.append(i)
                    if verbose:
                        print(f"⚠️ Invalid term at M={M:.2e}: term = {term:.2e}")
            except Exception as e:
                bad_indices.append(i)
                if verbose:
                    print(f"❌ Exception at M={M:.2e}: {e}")

        bad_Ms = M_vals[bad_indices]
        good_Ms = np.delete(M_vals, bad_indices)

        return good_Ms, bad_Ms



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
        M, F, S = y
        C, P = self.steady_state_CP(M, F)
        C = C[0];  P = P[0]
        dMdt = self.n*S + M*(self.lam2*(C/(self.k2+C)) - self.mu2)
        dFdt = F*(self.lam1*(P/(self.k1+P))*(1 - F/self.K) - self.mu1)
        return np.array([ dMdt,  dFdt,  0.0])

    
    def change_in_m_f(self,M,F,S,t=0):
        """ Return the growth rate of M and F assuming steady state of P and C, essentially
        same equations as the equations function, but with steady C and P"""
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
            print(f"{mode.upper()} @ S={S}: {res['verdict'] if 'verdict' in res else 'no verdict'}")

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
            if perturb:
                pt = self.perturb_fixed_point(pt)
                fixed_pts[idx] = pt
            if classify:
                res = self.classify_slice(*pt, fixed={"S": fixed_S}, method=method)
                metadata[idx] = res

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
        import numpy as np

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
        Return the full 3×3 Jacobian matrix ∂(dM,dF,dS)/∂(M,F,S)
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
    def threshold_event(self,t,y):
        """Function to stop integrator when it hits zero"""
        return min(y[0]-1,y[1]-1) #Stop when mF reaches zero
    
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

            sep_traj_pos = solve_ivp(self.change_in_m_f_to_int_neg, (t[0], t[-1]), initial_pos, t_eval=t,method='Radau',rtol=1e-9, atol=1e-12)
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
        Q = plt.quiver(X, Y, DX_norm, DY_norm, log_norm, pivot='mid', cmap=plt.cm.jet)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(f"2D Quiver Plot with S fixed = {fixed_S:g}")
        

        
        # Add a colorbar for the vector magnitude.
        cbar = plt.colorbar(Q)
        cbar.set_label("Relative Growth Rate (Normalised)")
        

    # def plot_2D_quiver_field_fixed_S(self, x_vals, y_vals, fixed_S,
    #                                 x_label="Myofibroblast Conc.",
    #                                 y_label="Macrophage Conc."):
    #     X, Y = np.meshgrid(x_vals, y_vals)
        
    #     vector_func = np.vectorize(
    #         lambda M, F: self.change_in_m_f(M, F, fixed_S)[0:2],
    #         otypes=[float, float]
    #     )
    #     dM_raw, dF_raw = vector_func(Y, X)

    #     # Raw magnitude
    #     norm = np.hypot(dF_raw, dM_raw)
    #     norm[norm == 0] = 1.0

    #     # Normalize for consistent arrow length
    #     dF_norm = dF_raw / norm
    #     dM_norm = dM_raw / norm

    #     # Optional: use np.log10(norm + eps) for better dynamic range visualization
    #     magnitude_color = norm  # Or np.log10(norm + 1e-9)

    #     # Plot (without axis scaling or show)
    #     Q = plt.quiver(X, Y, dF_norm, dM_norm, magnitude_color,
    #                 pivot='mid', cmap='jet', scale=40)

    #     # Axis labels
    #     plt.xlabel(x_label)
    #     plt.ylabel(y_label)
    #     plt.title(f"2D Quiver Plot with S fixed = {fixed_S:g}")

    #     # Add colorbar separately
    #     cbar = plt.colorbar(Q)
    #     cbar.set_label("Vector Magnitude")






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
        w_m, w_f, w_s = 1.0, 1000.0, 1.0
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
        plt.grid(True)
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
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.title('Model Fit to Data')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

        # Plot residuals
        self.plot_residuals(tau_data, (m_sim, f_sim, s_sim), data_dict)

        return result
###################
    def classify_slice(self, M, F, S, fixed={'S': 2000}, method='eigen', tol=1e-6, eps=1e-2):
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

        plt.tight_layout()
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
