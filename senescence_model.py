import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp,odeint

def senescent_model(t,x,params):
    eta = params[0]
    beta = params[1]
    kappa = params[2]

    dxdt = eta*x-(beta*x)/(x-kappa)
    return dxdt
t = np.linspace(0,365,365)
sol = solve_ivp(senescent_model, [t[0],t[-1]],[5],args=([0.084,0.15,0.5],))
print(len(sol.t))
plt.plot(sol.t,sol.y[0])
plt.xlabel("Time (days)")
plt.ylabel("Cell number")
plt.show()