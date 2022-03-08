import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from logistic_growth_model import general_logistic

def get_L_limits(s, n1, n2):
    """
    get the upper & lower value for L & initial guess
    
    Parameters
    ----------
    s: pandas.Series, smoothed data series
    
    n1: int, minimum day for exponetial function
    
    n2: int, maximum day for exponetial function
    
    Return
    ----------
    tuple of value with lower bound & upper bound & initial guess for L
    """
    last_val = s.value[-1] # get the last value from the smoothed data
    last_pct = s.pct_change()[-1] # get the last percentage change from the smoothed data
    L_min = last_val * last_pct ** n1 # get the lower bound
    L_max = last_val * last_pct ** (n2 +1) # get the upper bound
    L0 = (L_max - L_min) / 2 + L_min # get the initial guess for L
    if np.isnan(L_min): 
        L_min, L_max, L0 = 0, 1, 0 # assign the value if no lower bound found
    return L_min, L_max, L0

def get_bounds_p0(s, n1 = 5, n2 = 50):
    """
    get the bounds & p0 for inputted minimum & maximum day
    
    Parameters
    ----------
    s: pandas.Series, smoothed data series
    
    n1: int, minimum day for exponetial function
    
    n2: int, maximum day for exponetial function
    
    Return
    -------
    two item tuple: bound & p0
    """
    L_min, L_max, L0 = get_L_limits(s,n1,n2) # get the bounds of upper asymptote
    x0_min, x0_max = -50, 50 # get the bounds of horizontal shift
    k_min, k_max = 0.01, 0.1 # get the bounds of growth rate
    v_min, v_max = 0.01, 2 # get the bounds for asymmetry control
    s_min, s_max = 0, s.iloc[-1] + 0.01 # get the bounds of vertical shift
    s0 = s_max / 2 # get the initial point of vertical shift
    lower = L_min, x0_min, k_min, v_min, s_min # form the tuple of lower bound
    upper = L_max, x0_max, k_max, v_max, s_max # form the tuple of upper bound
    bound = lower, upper # form the tuple of bounds
    p0 = L0, 0, 0.1, 0.1, s0 # form the tuple of initial point
    return bound, p0