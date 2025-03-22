"""
19/03/2025
Niall Karunaratne
State and Parameter list creator
Made to allow me to input states and parameters more explicitly, and
in a different order by using keyword arguments
"""
import numpy as np
def state_fm(stateM,stateF,stateC,stateP):
        """
        Fibrosis Model Initial State

        :Inputs:

        stateM: int, initial macrophage population
        stateF: int, initial myofibroblast population
        stateC: float, initial CSF value
        stateP: flaot, initial PDGF
        :Return:  array, array of values

        """
        M = stateM
        assert isinstance(M,int),"Macrophage population must be an integer"
        F = stateF
        assert isinstance(F,int),"Myofibroblast population must be an integer"
        C= stateC
        P = stateP
        return np.array([M,F,C,P])
def state_fsm(stateM,stateF,stateC,stateP,stateS):
        """
        Fibrosis Model with Senescence Initial State

        :Inputs:

        stateM: int, initial macrophage population
        stateF: int, initial myofibroblast population
        stateC: float, initial CSF value
        stateP: float, initial PDGF
        stateS: int, initial senescent cell population
        :Return:  array, array of values

        """
        M = stateM
        assert isinstance(M,int),"Macrophage population must be an integer"
        F = stateF
        assert isinstance(F,int),"Myofibroblast population must be an integer"
        C= stateC
        P = stateP
        S = stateS
        assert isinstance(S,int),"Senescent cell population must be an integer"
        return np.array([M,F,C,P,S])

def params_fm(lam1, lam2, mu2, mu1, K, k1, k2, beta1, beta2, beta3, alpha1, alpha2, gamma):
    """
    Fibrosis Model Parameters

    :Inputs:

    lam1: float, Max. proliferation rate of myofibroblasts
    lam2: float, Max. proliferation rate of macrophages
    mu2: float, Removal rate of macrophages
    mu1: float, Removal rate of myofibroblasts
    K: float, Myofibroblast carrying capacity
    k1: float, Binding affinity for PDGF
    k2: float, Binding affinity for CSF
    beta1: float, Max. CSF secretion by myofibroblasts
    beta2: float, Max. PDGF secretion by macrophages
    beta3: float, Max. PDGF secretion by myofibroblasts
    alpha1: float, Max. endocytosis of CSF by macrophages
    alpha2: float, Max. endocytosis of PDGF by myofibroblasts
    gamma: float, Growth factor degradation rate

    :Returns:
    array of ordered parameters: Array of parameter values in the correct order
    """
    # Ensure all inputs are floats
    assert all(isinstance(param, (int, float)) for param in [lam1, lam2, mu2, mu1, K, k1, k2, beta1, beta2, beta3, alpha1, alpha2, gamma]), \
        "All parameters must be integers or floats"

    # Return the parameters as a NumPy array
    return np.array([lam1, lam2, mu2, mu1, K, k1, k2, beta1, beta2, beta3, alpha1, alpha2, gamma])

def params_fsm(lam1, lam2, mu1, mu2, K, k1, k2, beta1, beta2, beta3, alpha1, alpha2, gamma,n,h,r,q):
    """
    # Order for params above MUST be :lam1,lam2,mu1,mu2,K,k1,k2,beta1,beta2,beta3,alpha1,alpha2,gamma.n,h,r,q

    Fibrosis/Senescence Model Parameters

    :Inputs:
    
    lam1: float, Max. proliferation rate of myofibroblasts
    lam2: float, Max. proliferation rate of macrophages
    mu2: float, Removal rate of macrophages
    mu1: float, Removal rate of myofibroblasts
    K: float, Myofibroblast carrying capacity
    k1: float, Binding affinity for PDGF
    k2: float, Binding affinity for CSF
    beta1: float, Max. CSF secretion by myofibroblasts
    beta2: float, Max. PDGF secretion by macrophages
    beta3: float, Max. PDGF secretion by myofibroblasts
    alpha1: float, Max. endocytosis of CSF by macrophages
    alpha2: float, Max. endocytosis of PDGF by myofibroblasts
    gamma: float, Growth factor degradation rate
    n: Proliferation of macrophages from senescence cells
    h: Proliferation of senescent cells due to aging
    r: Removal rate of senescent cells by macrophages
    q: Saturation/Carrying capacity for senescent cells

    :Returns:
    np.array: Array of parameter values in the correct order
    """
    # Ensure all inputs are floats
    assert all(isinstance(param, (int, float)) for param in [lam1, lam2, mu2, mu1, K, k1, k2, beta1, beta2, beta3, alpha1, alpha2, gamma,n,h,r,q]), \
        "All parameters must be integers or floats"

    # Return the parameters as a NumPy array
    # Order for params below :lam1,lam2,mu1,mu2,K,k1,k2,beta1,beta2,beta3,alpha1,alpha2,gamma

    return np.array([lam1, lam2, mu1, mu2, K, k1, k2, beta1, beta2, beta3, alpha1, alpha2, gamma,n,h,r,q])