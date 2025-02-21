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
        self.epsilon = params[3]
        self.state = initial_state

    def senescent_model(self,t,x):
        """Minimal model for cellular senescence:
        dx/dt = eta*x - beta*x/(x+kappa)
        
        Parameters: 
        :x: Number of senescent cells
        :dx/dt: Rate of change of number of cells
        :eta: Increase of senescent cell production with increased age
        :beta: Rate of senescent cell removal
        :kappa: half-way saturation point for removal (Michaelis-Menten)
        """
        dxdt = self.eta*x-(self.beta*x)/(x+self.kappa)
        return dxdt
    def nullclines(self):
        x_1 = 0
        x_2 = (self.beta-self.eta*self.kappa)/self.eta
        return np.array(x_1,x_2)
    def threshold_event(self,t,x):
        return x
    def solve(self,t):
        """Solve using SciPy (deterministic here, no stochastic element)"""
        def threshold(t,x):
            return self.threshold_event(t,x)
        threshold.terminal = True
        threshold.direction = 0
        #State is array-like as solve_ivp requires it to be 1-dimensional
        sol = solve_ivp(self.senescent_model, [t[0],t[-1]],[self.state],t_eval=t,events=threshold)
        return sol
    def stochastic_model(self,t,x,zeta):
        dxdt = self.eta*x-(self.beta*x)/(x+self.kappa) + np.sqrt(2*self.epsilon)*zeta
        return dxdt
    def euler_maruyama(self):
        """ Method of solving SDE
        """
        x = self.state
        t_max = 1000
        delta_t = t_max/100
        t=0
        times = [t]
        states = [x]
        while t < t_max:
            x1 = self.eta*x-(self.beta*x)/(x+self.kappa)
            x2 = np.random.normal(loc=(self.kappa*self.eta)/(self.beta-self.eta),scale=(np.abs(self.kappa*self.beta*self.epsilon+self.epsilon)/(self.beta-self.eta)))*np.sqrt(2*self.epsilon)
            #x2 = np.random.normal()*np.sqrt(2*self.epsilon)

            x+=(x1+x2)
            t+=delta_t
            times.append(t)
            states.append(x)
        return times,states
        

