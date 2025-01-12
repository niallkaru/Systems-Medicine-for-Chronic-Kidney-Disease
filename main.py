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
    cell_sm = sm.senescence_model(params,initial_state)
    sol = cell_sm.solve(t)    
    # plt.plot(sol.t,sol.y[0])
    # plt.xlabel("Time")
    # plt.ylabel("Number/Senescent Cells")
    # plt.show()
    initial_state_fm = [5,0,0,0]
    #params_fm = np.random.randint(0,10,13)
    params_fm = [0.9,0.8,0.3,0.3,470,70,240,940,510,2,1E9,1E9,1E6]
    cell_fm = fm.fibrosis_model(params_fm,initial_state_fm)
    sol_fm = cell_fm.solve(t)
    print(sol_fm)
    plt.plot(sol_fm.t,sol_fm.y[1])
    plt.xlabel("Time")
    plt.ylabel("Number/Fibrotic Cells")
    plt.show()
    print("DONE")
if __name__ == '__main__':
    main()