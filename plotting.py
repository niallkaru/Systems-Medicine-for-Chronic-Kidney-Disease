import numpy as np
import matplotlib.pyplot as plt
import fibrosis_model as fm
import senescence_model as sm

def plot_tseries(time,data,xlabel='Time',ylabel='Number',label=None):
    plt.figure()
    plt.plot(time,data,label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()