"""Main file 22/12/2024"""

import senescence_model as sm
import fibrosis_model as fm
import numpy as np
import matplotlib.pyplot as plt
def main():
    ## For senescence ##
    params = [0.084, 0.15, 0.5]
    initial_state = 5
    t = np.linspace(0, 10, 100)
    cell = sm.senescence_model(params,initial_state)
    sol = cell.solve(t)    
    plt.plot(sol.t,sol.y[0])
    plt.show()





if __name__ == '__main__':
    main()