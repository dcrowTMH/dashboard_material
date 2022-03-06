import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def simple_exp(x,a,b):
    """
    simple exponential model
    
    Parameters
    -----------
    x : array of x-values, np.array(len(y)) usually date number for time series data
    
    a : initial value
    
    b : growth rate (b > 1, growth & b < 1, decline)
    
    Return
    -----------
    Array of calculated values
    """
    return a * b ** x

def simple_cont_exp(x, a, b):
    """
    simple continuous exponential growth model
    
    Parameters:
    -------------
    x: numpy array of x-values, np.array(len(y)) ususally for the time series data
    
    a: initial value
    
    b: growth rate
    """
    return a * np.exp(b * x)

def exp_decline(x, a, b, c):
    """
    simple exponential decline model
    
    Parameters
    ----------
    x: numpy array of x-values, np.array(len(y)) ususally for the time series data
    
    a: initial value
    
    b: growth rate
    
    c : vertical shift deal with negative a
    """
    return a * b ** x + c

def exp_decline_cont(x, a, b, c):
    """
    simple continuous exponential decline model
    
    Parameters
    ----------
    x: numpy array of x-values, np.array(len(y)) ususally for the time series data
    
    a: initial value
    
    b: growth rate
    
    c : vertical shift deal with negative a
    """
    return a * np.exp(b * x) + c

def optimize_func(params, x, y, model):
    """
    function passed to scipy.optimize.least_squares as the first argument
    aim to get the error between original value & predicted values from model
    
    Parameters
    -----------
    params: tuple of values that passed to the model, should be the optimized value for the model
    
    x: x-value from the data (day for the time series date)
    
    y: y-value from the date (cases/ death from the data sets)
    
    model: function model used for optimization
    
    Return
    -----------
    error between predicted value and original value
    """
    
    y_pred = model(x, *params) # the p0 argument was passed to model and break it
    error = y - y_pred
    return error

def train_model(s, last_date, model, bounds, p0, **kwargs):
    """
    use scipy.optimize.least_squares to train the model and find the fitted parameters for the model
    
    Parameters
    ------------
    s: original set data/array used for modeling
    
    last_date: string for last date used for slicing the period for training
    
    model: functional model that used for prediction
    
    bounds: two-item tuple of lower and upper bound for the parameters (limit the range to prevent infinity)
    
    p0: initial parameters used to start the training
    
    kwargs: extra keyword that passed to the scipy.optimize.least_squares method
    
    Return
    ----------
    numpy array with fitted paramters
    """
    
    y = s.loc[:last_date].values # slice the period of time with loc
    x = np.arange(len(y)) # get the number of day for functional model work
    # p0 was passed to optimize_func as first argument & arg was passed as the remaining parameter for function model
    res = least_squares(optimize_func, p0 = p0, arg=(x,y,model), bounds = bounds, **kwargs)
    return res.x # return the fitted parameters

def get_daily_pred(model, params, n_train, n_pred):
    """
    get the prediction from the functional model with parameters
    
    Parameters
    -------------
    model: functional model used for prediction
    
    params: parameters passed to the model for prediction
    
    n_train: number of observation (number of days for prediction)
    
    n_pred: number of prediction maked
    """
    # make the range from the day before n_train to the day after the days of prediction
    x_pred = np.arange(n_train - 1, n_train + n_pred) 
    # passed the period of time & the parameters to the model
    y_pred = model(x_pred, *params)
    # get the diff of prediction between each day
    y_pred_daily = np.diff(y_pred)
    return y_pred_daily

def get_cumulative_pred(last_actual_value, y_pred_daily, last_date):
    """
    get the cumulative predicted value after the last date
    
    Parameters
    -----------
    last_actual_value: int, last record value
    
    y_pred_daily: numpy array of prediction
    
    last_date: string, last date used in the model
    
    Return
    ------------
    Series of cumlative predicted values
    """
    # add the last record value with the predicted value
    y_cumulative_pred = y_pred_daily + last_actual_value
    # change the string to pandas timestamp type and add one day for the date after the last date
    last_date = pd.Timestamp(last_date) + pd.Timedelta('1D')
    # get the index for the number of date with the number of prediction
    index = pd.date_range(last_date, period = len(y_cumulative_pred))
    # create another pandas series with corrected value & the date after the last date
    return pd.Series(y_cumulative_pred, index= index)

def plot_prediction(s, s_pred, title=""):
    """
    plot both original & predicted values
    
    Parameters
    -----------
    s: series of original data
    
    s_pred: series of predicted data
    
    title: string, title for the plot
    
    """
    # get the last date for prediction
    last_pred_date = s_pred.index[-1]
    # plot the original date the last date of prediction
    ax = s.loc[:last_pred_date].plot(label = 'Original')
    # plot the whole prediction series
    ax = s_pred.plot(label = 'Predicted')
    # get the legends & title for the plot
    ax.legends()
    ax.set_title(title)
    
def predict_all(s, start_date, last_date, n_smooth, n_pred, model, 
                bounds, p0, title="", **kwargs):
    """
    combine all the functions before and get the data process
    
    Parameters
    -----------
    s: series of original data
    
    start_date: string, start_date for the prediction
    
    last_date: string, last_date for the prediction
    
    n_smooth: number of point used for the lowess function
    
    n_pred: number of prediction wanted to make
    
    model: functional model used for the prediction
    
    bounds: tuple of values for the lower and upper bounds of the parameters for the least squares function
    
    p0: initial parameters used for the least squares function
    
    title: string, title for the plot
    
    kwargs: other keywords that passed to the least squares function
    
    Return
    -----------
    fitted parameters & predicted values
    """
    # smooth the data till the last date with number of point for smooth
    s_smooth = smooth(s.loc[:last_date], n_smooth = n_smooth)
    # slice the date period from start date to last date
    s_smooth = s_smooth.loc[start_date:]
    # get the fitted parameters for the prediction
    params = train_model(s_smooth, last_date, model = model, bounds=bounds, p0=p0, **kwargs)
    # predict the value with functional model with fitted paramters
    n_train = len(s_smooth)
    y_pred_daily = get_daily_pred(model, params, n_train, n_pred)
    # correct the date with the value from the last date
    y_pred_cum = get_cumulative_pred(s_smooth[last_date], y_pred_daily, last_date)
    # plot the original date with predicted date for comparison
    plot_prediction(s, y_pred_cum, title=title)
    return params, y_pred_daily
