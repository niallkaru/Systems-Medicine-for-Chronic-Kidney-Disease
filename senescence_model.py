"""Class for senescence model for cells (within kidney), created by Niall
 Karunaratne 21/12/2024
 Adapted from Senescent cell turnover slows with age providing
an explanation for the Gompertz law Karin et al, Nature Communications, 2019"""
import numpy as np
from scipy.integrate import solve_ivp,odeint
class senescence_model:
    def __init__(self,params,initial_state):
        """
        Extract parameters for the senescence model, set state to none
        """
        self.eta = params[0]
        self.beta = params[1]
        self.kappa = params[2]
        self.state = initial_state

    def senescent_model(self,t,x):
        """Minimal model for cellular senescence:
        dx/dt = eta*x - beta*x/(x-kappa)
        
        Parameters: 
        :x: Number of senescent cells
        :dx/dt: Rate of change of number of cells
        :eta: Increase of senescent cell production with increased age
        :beta: Rate of senescent cell removal
        :kappa: half-way saturation point for removal (Michaelis-Menten)
        """
        dxdt = self.eta*x-(self.beta*x)/(x-self.kappa)
        return dxdt
    def solve(self,t):
        """Solve using SciPy (deterministic here, no stochastic element)"""
        #State is array-like as solve_ivp requires it to be 1-dimensional
        sol = solve_ivp(self.senescent_model, [t[0],t[-1]],[self.state],t_eval=t)
        return sol
