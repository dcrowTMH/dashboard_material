import pandas as pd
import numpy as pd
import matplotlib.pyplot as plt

def logistic_func(x, L, x0, k):
    """
    Compute the simple logistic function with tuning parameters
    
    Parameters
    ----------
    x: number of array value
    
    L: int, Control the horizontal scale, usually the maximum point of the data
    
    x0: float, middle point
    
    k: growth rate
    
    Return
    ----------
    predicted value with logistic function model
    """
    return L / ( 1 + np.exp(-k * (x - x0)))

def logistic_guess_plot(s, L, x0, k):
    """
    plot the original data & logictis function with such parameters
    
    Parameters
    -----------
    s: series of actual data
    
    L: int, Control the horizontal scale, usually the maximum point of the data
    
    x0: float, middle point
    
    k: growth rate
    
    Return
    ----------
    None
    """
    x = np.array(len(s))
    y = logistic_func(x, L, x0, k)
    s_guess = pd.Series(y, index = s.index)
    s.plot(label = 'Actual')
    s_guess.plot(label = 'Predicted')
    
def plot_ks(s, ks, L, x0):
    """
    compute the logistic function and plot with different k value
    
    Parameters
    ----------
    s: series of actual data
    
    ks: list of value for the growth rate
    
    L: int, Control the horizontal scale, usually the maximum point of the data
    
    x0: float, middle point
    
    Return
    --------
    None
    """
    start_date = s.index[0] # get the first date of actual data
    start_date = pd.Timestamp(start_date) # change it to pandas.timestamp type
    index = np.time_range(start_date, periods = 2 * x0) # get the periods of the middle point * 2 for the prediction range
    s.plot(label = 'Original', lw = 3)
    x_pred = np.array(len(index))
    for k in ks:
        y_pred = logistic_func(x_pred, L, x0, k)
        s_guess = pd.Sereis(y_pred, index = index)
        s_guess.plot(label = k)
        
def general_logistic(x, L x0, k, v):
    """
    compute the general logistic function to obtain asymmetry prediction
    
    Parameters
    -----------
    x: number of array value
    
    L: int, Control the horizontal scale, usually the maximum point of the data
    
    x0: float, wont determine middle point again due to v value
    
    k: growth rate
    
    v: float, change the symmetry of the curve and shift to left or right
    
    Return
    ---------
    predicted value from functional model
    """
    return L / (( 1 + np.exp(-k * (x - x0))) ** 1 / v)\

def model_country(data, name, start_date, last_date):
    """
    plot the prediction with pre-set parameter and compare with the original
    
    Parameters
    ----------
    data: dict, data dictionary
    
    name: string, country name
    
    start_date: string, first day for prediction
    
    last_date: string, last day for prediction
    Return
    ----------
    None
    """
    s = data['world_cases'][name]    
    L_min, L_max = s.iloc[0], s.iloc[-1] * 1000
    x0_min, x0_max = -50, 50
    k_min, k_max = 0.01, 0.5
    v_min, v_max = 0.01, 2
    lower = L_min, x0_min, k_min, v_min
    upper = L_max, x0_max, k_max, v_max
    bounds = lower, upper
    p0 = L_min * 5, 0, 0.1, 0.1
    predict_all(s, start_date=start_date, last_date=last_date, n_smooth=15, 
                n_pred=50, model=general_logistic, bounds=bounds, p0=p0, 
                title=f"{name} - Generalized Logistic Function");