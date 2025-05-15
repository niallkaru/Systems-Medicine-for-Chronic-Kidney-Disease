import numpy as np
from scipy.integrate import solve_ivp
import scipy.optimize as opt
"""Niall Karunaratne 22/12/2024
Class for fibrosis model for cells (within kidney)
Adapted from Principles of Cell Circuits for Tissue Repair and Fibrosis Adler et al. 2020"""

class fibrosis_model:
    def __init__(self,params,initial_state):
        """"Extract parameters
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
        C = (self.mu2*self.k2)/(self.lam2-self.mu2)
        F = (1/self.beta1)*((self.alpha1*M*C)/(self.k2+C))+((self.gamma*C)/self.beta1)
        return [F,M]
    def nullclines_F(self,F):
        """
        Find nullclines for myofibroblasts. Start with dFdt = 0, rearrange for P, sub into dPdt, 
        rearrange for M, again, return both.
        Input:
        F: Macrophage levels
        Return:
        F,M: F,M values for a given value of M
        """
        P = (self.mu1*self.k1*self.K)/(self.lam1*self.K-F*self.lam1-self.mu1*self.K)
        M = -1*(self.beta3*F-self.alpha2*F*P/(self.k1+P)-self.gamma*P)/self.beta2
        return [F,M]
    def subtract_nulls(self,X0):
        """ Returns the one nullcline subtracted from the other accurately, this is
        used for finding fixed points"""
        M0, F0 = X0
        return [np.subtract(self.nullclines_M(M0)[0],self.nullclines_F(F0)[0]), np.subtract(self.nullclines_M(M0)[1],self.nullclines_F(F0)[1])]
    
    # def fixed_points(self,initial_guess = np.array([1e4,1e4])):
    #     """
    #     We want fixed points, where the nullclines cross ie. Fdot = Mdot
    #     So using scipy.optimize (sic, American spelling)
    #     and a function to find the difference between them
    #     """
    #     x = opt.fsolve(self.subtract_nulls, initial_guess)
    #     return np.array(x)
    
    def residual_fixed_point(self, X):
        """
        Compute the residuals for steady-state conditions.
        X = [M, F]
        Uses the quasi-steady state for C and P.
        At a fixed point, we require:
        dM/dt = 0,  dF/dt = 0
        """
        M, F = X
        # Obtain C and P via your steady_state_CP function.
        CF_steady = self.steady_state_CP(M, F)
        C = CF_steady[0][0]
        P = CF_steady[1][0]

        # Compute each derivative
        dMdt = M * (self.lam2 * (C / (self.k2 + C)) - self.mu2)
        dFdt = F * (self.lam1 * (P / (self.k1 + P)) * (1 - (F / self.K)) - self.mu1)
      
        
        return np.array([dMdt, dFdt])  

    def fixed_points(self, initial_guess=np.array([1e4, 1e4])):
        """
        Find the fixed point in 3D by solving residual_fixed_point(X) = [0, 0, 0].
        """
        fixed_pt = opt.fsolve(self.residual_fixed_point, initial_guess)
        fixed_pt[np.where(np.abs(fixed_pt)<1e-6)]=0
        eigenvals,eigenvecs = self.eigen(fixed_pt)
        print(f'Eigenvalues at fixed point {fixed_pt}: {eigenvals}, {"Stable" if np.all(eigenvals<0) else "Unstable"}')
        return fixed_pt
    # def fixed_pt_sweep(self,xrange,yrange):
    #             # Example parameters (fill in with your own limits)
    #     xvals = np.logspace(xrange[0], xrange[-1], 35)  # e.g., for Myofibroblast concentration (F)
    #     yvals = np.logspace(yrange[0], yrange[-1], 35)  # e.g., for Macrophage concentration (M)
    #     #xvals = np.linspace(xrange[0], xrange[-1], 100)  # e.g., for Myofibroblast concentration (F)
    #     #yvals = np.linspace(yrange[0], yrange[-1], 100)  # e.g., for Macrophage concentration (M)
    #     # C
    #     # Create the grid
    #     grid_x, grid_y = np.meshgrid(xvals, yvals, indexing='ij')
    #     #print(grid_x,"\n",grid_y)

    #     # Use np.vectorize to apply your fixed_points_3D function.
    #     # Note: The lambda returns a tuple (or slice of an array) containing the first two coordinates.
    #     vect_fun = np.vectorize(lambda M, F: (self.fixed_points([M, F])[0], self.fixed_points([M, F])[1]))
    #     #print(vect_fun) # Issue is here!
    #     # This returns two arrays of shape (35, 35)
    #     #print("Finished lambda func")
    #     sol1, sol2 = vect_fun(grid_x, grid_y)
    #     # Now, combine these two arrays into an array of shape (35*35, 2)
    #     combined_sols = np.stack((sol1, sol2), axis=-1).reshape(-1, 2)
    #     #print(f'combined sols {combined_sols}')
    #     # Use np.unique to filter duplicate coordinate pairs.
    #     combined_sols = combined_sols[~np.isnan(combined_sols).any(axis=1)]

    #     # Make sure to call np.unique along axis 0.
    #     filtered_sols = np.unique(combined_sols, axis=0)
    #     #print("filtered sols",filtered_sols)
    #     #sols = [self.perturb_fixed_point(sol, epsilon=1e-2, tol=1e-5) for sol in filtered_sols]
    #    # print(sols)
    #     sols = filtered_sols
    #     return sols
    def perturb_fixed_point(self,fp, epsilon=1e-2,tol=1e-5):
        """
        Replace any zero entries in the fixed point fp with epsilon.
        """
        # If using numpy, you can use np.where:
        fp = np.where(fp <= tol, epsilon, fp)
        return fp
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
        total: total injury (in macrophages) present
        """
        total = 0
        for start,stop,amp in pulses: # Go through every injury individually
            if start < t < stop:
                total += amp
        return total

    def constant_injury(self,t, X, pulses):
        """ Simulate the equations but with an injury added to the macrophage term and quasi-steady state
        approximation for dCdt and dPdt. We use solve_ivp to do this and assume a
        A steady state for C and P
        
        Input:
        t: time
        X: starting values for M and P
        pulses: injuries to be added

        Return:
        Mdot: Number of macrophages present at a time
        Fdot: Number of myofibroblasts present at a time
        """
        # Ensure non-negative values for M and F
        M = max(0, X[0])
        F = max(0, X[1])

        C1_C2_steady = self.steady_state_CP(M,F)
        #print(f'C1_C2_steady {C1_C2_steady}')
        C = C1_C2_steady[0][0]#Funny way of grabbing them due to how I organised C1_C2_steady, just roll with it
        P = C1_C2_steady[1][0]
        #print(M,F)
        M_dot = M*(self.lam2*(C/(self.k2+C)) - self.mu2) + self.heavyside_pulses(pulses, t)
        F_dot = F*(self.lam1*(P/(self.k1+P))*(1 - F/self.K) - self.mu1)
        
        #print(M_dot, F_dot)
        return np.array([M_dot, F_dot])
    
    def eigen(self,X):
        """
        We find the eigenvalues/vectors of the fixed points to determine
        if they are stable/unstable (negative or positive). Use later.

        Input: X, coordinates of steady state in M-F space
        Return: Eigenvalues and normalised eigenvector of steady state
        """

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
        return eigenvals,eigenvecs
    def separatrix_eigen(self,X):
        """
        We find the eigenvalues/vectors of the fixed points to determine
        if they are stable/unstable (negative or positive). Use later.

        Input: X, coordinates of steady state in M-F space
        Return: Eigenvalues and normalised eigenvector of steady state
        """
        #print(eigenvals,eigenvecs)
        eigenvals,eigenvecs = self.eigen(X)
        unstable_index = np.argmax(eigenvals.real)
        #Index of largest (real) eigenvalue, a positive eigenval
        #corresponds to an unstable fixed point (along separatrix)
        unstable_vector = eigenvecs[:,unstable_index]

        return eigenvals[unstable_index],unstable_vector/np.linalg.norm(unstable_vector) #Normalise it      

    
    def separatrix_traj_neg(self,t,X,epsilon=1):
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
            eigenval,unstable_vector = self.separatrix_eigen(X)
            def threshold(t,y):
                # Allow the trajectory to reach zero but not go below.
                return min(y[0], y[1]) - 1e-1  # Stop when either M or F reaches zero.
            threshold.terminal = True
            threshold.direction = -1
            initial = X+epsilon*unstable_vector #Perturb a little
            sep_traj = solve_ivp(self.change_in_m_f_to_int_neg, (t[0], t[-1]), initial, t_eval=t,method='Radau')#,events=threshold)
            return [sep_traj.y[0],sep_traj.y[1]]

    def adimensionalised_funcs(self,t,y,adim):
        """
        We want non-dimensionalised versions of the equations to ascertain
        what the most important parameters and variables are.

        Here I am keeping it quite general and not inputting any of the factors
        directly. That is to say, not using the factors defined above.
        
        This would lead to a lot of cancelling but I want more flexibility.
        """
        M, F, C, P = y
        psi,phi,sigma,pi,tau = adim
        m = M/psi
        f = F/phi
        c = C/sigma
        p = P/pi
        t = t/tau

        dfdt = tau*f*(((self.lam1*pi*p)/(self.k1+pi*p))*(1-(f*phi/self.K))-self.mu1)
        dmdt = tau*m*(self.lam2*(c*sigma)/(self.k2+c*sigma)-self.mu2)
        dcdt = tau*(((self.beta1*phi*F)/sigma)-((self.alpha1*psi*m*c)/(self.k2+sigma*c))-self.gamma*C)
        dpdt = tau*(((self.beta2*psi*m+self.beta3*phi*f)/pi)-((self.alpha2*phi*p)/(self.k1+pi*p))-self.gamma*p)
        return dfdt,dmdt,dcdt,dpdt

    def solve_constant_injury(self, t, X0, pulses):

        def enforce_non_negative(y):
            # Enforce non-negative values for the solver.
            return np.maximum(y, 0)

        sol = solve_ivp(self.constant_injury, [t[0], t[-1]], X0, t_eval=t, args=(pulses,))
        sol.y = np.apply_along_axis(enforce_non_negative, 0, sol.y)
        return sol
    
    def time_to_state(self,y_matrix,t_values,tol=1e-5,min_steps=3):
        """
        Returns the earliest time when *all* variables in y_matrix have settled,
        i.e. their stepwise differences stay below tolerance for min_steps in a row.

        Parameters:
            y_matrix: 2D array-like (shape: [num_vars, num_timepoints])
            t_values: 1D array of time values (same length as y_matrix[0])
            tolerance: threshold for stepwise difference to consider settled
            min_steps: how many consecutive steps must be below tolerance

        Returns:
            Time when all variables are settled, or None
        """

        y_matrix = np.asarray(y_matrix)
        t_values = np.asarray(t_values)
        num_vars, num_t = y_matrix.shape

        settle_times = []

        for var_idx in range(num_vars):
            y = y_matrix[var_idx]
            diffs = np.abs(np.diff(y))
            final_val = np.abs(y[-1]) + 1e-8  # avoid divide-by-zero
            tol = tol * final_val
            for i in range(num_t - min_steps - 1):
                window = diffs[i:i + min_steps]
                if np.all(window < tol):
                    settle_times.append(t_values[i])
                    break
            else:
                # This variable never settled
                return None

        return max(settle_times)  # Wait until all have settled

    def residual_fixed_point(self, X):
            """
            Compute the residuals for the full 3D steady-state conditions.
            X = [M, F]
            Uses the quasi-steady state for C and P.
            At a fixed point, we require:
            dM/dt = 0,  dF/dt = 0
            """
            M, F = X
            # Obtain C and P via your steady_state_CP function.
            CF_steady = self.steady_state_CP(M, F)
            C = CF_steady[0]#[0]
            P = CF_steady[1]#[0]
            
            # Compute each derivative
            dMdt = M * (self.lam2 * (C / (self.k2 + C)) - self.mu2)
            dFdt = F * (self.lam1 * (P / (self.k1 + P)) * (1 - (F / self.K)) - self.mu1)
            
            return np.array([dMdt, dFdt])  

    def fixed_points_2D(self, initial_guess=np.array([1e4, 1e4])):
            """
            Find the fixed point in 3D by solving residual_fixed_point(X) = [0, 0, 0].
            """
            res = opt.root(self.residual_fixed_point, initial_guess,method='broyden1')
            if not res.success:
                #print("Fixed Point failure: ",initial_guess)
                return np.array([np.nan, np.nan])

            fixed_pt = res.x
            # fixed_pt = self.perturb_fixed_point(res.x,epsilon=1e-2,tol=1e-1)
            return fixed_pt   
    def jacobian_full(self, M, F):
            """
            Return the full 2x2 Jacobian matrix ∂(dM,dF)/∂(M,F)
            using your analytic formulas for the partial derivatives.
            """
            # quasi‐steady C,P
            # C_array, P_array = self.steady_state_CP(M, F)
            # C = C_array[0]
            # P = P_array[0]
            CF_steady = self.steady_state_CP(M, F)
            C = CF_steady[0][0]
            P = CF_steady[1][0]
            # fill entries
            dMdM = (self.lam2*C)/(self.k2+C) - self.mu2
            dMdF = 0
            dFdM = 0
            dFdF = (self.lam1*P)/(self.k1+P)-(2*F*self.lam1*P)/(self.K*(self.k1+P))-self.mu1


            return np.array([
                [dMdM,  dMdF],
                [dFdM,  dFdF]]) 
    def classify_by_dynamics(self, M0, F0, eps=1e-2):
        directions = [
            np.array([eps, 0]),
            np.array([-eps, 0]),
            np.array([0, eps]),
            np.array([0, -eps]),
            np.array([eps, eps]),
            np.array([-eps, -eps]),
            np.array([eps, -eps]),
            np.array([-eps, eps])
        ]
        
        stability_scores = []
        for d in directions:
            M, F = M0 + d[0], F0 + d[1]
            v = self.dynamics_2D(M, F)
            dot = -np.dot(v, d)
            stability_scores.append(dot)
        
        num_positive = sum(score > 0 for score in stability_scores)
        num_negative = sum(score < 0 for score in stability_scores)

        if num_positive == len(directions):
            verdict = "stable (attractor)"
        elif num_negative == len(directions):
            verdict = "unstable (repellor)"
        else:
            verdict = "saddle (mixed stability)"

        return verdict

    def classify_2D(self, M, F, method="eigen", tol_zero=1e-7, rel_tol=1e-2, eps=1e-2):
        """
        Classify the stability of a fixed point using either the Jacobian eigenvalues
        or by probing dynamics in nearby directions.

        Parameters:
            M, F (float): Fixed point coordinates.
            method (str): "eigen" or "dynamics".
            tol_zero (float): Tolerance for near-zero eigenvalues (eigen method).
            rel_tol (float): Relative tolerance for comparing eigenvalues (eigen method).
            eps (float): Perturbation radius (dynamics method).

        Returns:
            dict: Contains 'eigenvalues' (if available) and 'verdict'.
        """
        if method == "eigen":
            J = self.jacobian_full(M, F)
            eigvals = np.linalg.eigvals(J)
            real_parts = np.sort(np.real(eigvals))
            eig1, eig2 = real_parts

            if eig2 > tol_zero:
                verdict = "unstable (repellor)"
            elif abs(eig2) > abs(eig1) * (1 / rel_tol) and abs(eig1) < tol_zero:
                verdict = "saddle (mixed stability)"
            elif eig1 < -tol_zero and eig2 < -tol_zero:
                verdict = "stable (attractor)"
            else:
                verdict = "saddle (mixed stability)"

            return {
                "eigenvalues": eigvals,
                "verdict": verdict
            }

        elif method == "dynamics":
            # Directions to probe (8-point star)
            directions = [
                np.array([eps, 0]),
                np.array([-eps, 0]),
                np.array([0, eps]),
                np.array([0, -eps]),
                np.array([eps, eps]),
                np.array([-eps, -eps]),
                np.array([eps, -eps]),
                np.array([-eps, eps])
            ]

            scores = []
            for d in directions:
                Mp, Fp = M + d[0], F + d[1]
                v = self.dynamics_2D(Mp, Fp)
                score = -np.dot(v, d)
                scores.append(score)

            num_positive = sum(s > 0 for s in scores)
            num_negative = sum(s < 0 for s in scores)

            if num_positive == len(directions):
                verdict = "stable (attractor)"
            elif num_negative == len(directions):
                verdict = "unstable (repellor)"
            else:
                verdict = "saddle (mixed stability)"

            return {
                "verdict": verdict
            }

        else:
            raise ValueError("Unknown classification method: choose 'eigen' or 'dynamics'")


    def fixed_pt_sweep(self,xrange,yrange,perturb=True,classify=False):
                # Example parameters (fill in with your own limits)
        xfull = np.logspace(xrange[0], xrange[-1], 35)  # e.g., for Myofibroblast concentration (F)
        yfull = np.logspace(yrange[0], yrange[-1], 35)  # e.g., for Macrophage concentration (M)
        #xvals = np.linspace(xrange[0], xrange[-1], 100)  # e.g., for Myofibroblast concentration (F)
        #yvals = np.linspace(yrange[0], yrange[-1], 100)  # e.g., for Macrophage concentration (M)
        # C
        # Create the grid
        xvals = xfull[::3]
        yvals = yfull[::3]
        grid_x, grid_y = np.meshgrid(xvals, yvals, indexing='ij')
        #print(grid_x,"\n",grid_y)

        # Use np.vectorize to apply your fixed_points_3D function.
        # Note: The lambda returns a tuple (or slice of an array) containing the first two coordinates.
        vect_fun = np.vectorize(lambda M, F: (self.fixed_points_2D([M, F])[0], self.fixed_points_2D([M, F])[1]))
        #print(vect_fun) # Issue is here!
        # This returns two arrays of shape (35, 35)
        print("Finished lambda func")
        sol1, sol2 = vect_fun(grid_x, grid_y)
        # Now, combine these two arrays into an array of shape (35*35, 2)
        combined_sols = np.stack((sol1, sol2), axis=-1).reshape(-1, 2)
        #print(f'combined sols {combined_sols}')
        # Use np.unique to filter duplicate coordinate pairs.
        combined_sols = combined_sols[~np.isnan(combined_sols).any(axis=1)]

        # Make sure to call np.unique along axis 0.
        unique_sols = np.unique(np.round(combined_sols,2), axis=0)
        filtered_sols = unique_sols[np.all(unique_sols>=0,axis=1)]
        #print("filtered sols",filtered_sols)
        #sols = [self.perturb_fixed_point(sol, epsilon=1e-2, tol=1e-5) for sol in filtered_sols]
        # print(sols)
        # print(len(combined_sols),len(filtered_sols))
        sols = filtered_sols


        if classify:
            meta = {}
            for i, sol in enumerate(filtered_sols):
                M = sol[0]
                F = sol[1]
                res = self.classify_2D(M, F,method='dynamics')
                meta[i] = res
                print(f"{res['verdict']}")

            if perturb:
                sols = [self.perturb_fixed_point(sol, epsilon=1e-2, tol=1e-1) for sol in filtered_sols]

            return sols, meta

        else:
            
            if perturb:
                sols = [self.perturb_fixed_point(sol, epsilon=1e-2, tol=1e-1) for sol in filtered_sols]
            return sols
    def dynamics_2D(self, M, F):
        C_array, P_array = self.steady_state_CP(M, F)
        C = C_array[0]
        P = P_array[0]

        dMdt = M * ((self.lam2 * C) / (self.k2 + C) - self.mu2)
        dFdt = F * ((self.lam1 * P) / (self.k1 + P) * (1 - F / self.K) - self.mu1)
        return np.array([dMdt, dFdt])