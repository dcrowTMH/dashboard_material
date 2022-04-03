import pandas as pd
import plotly.graph_objects as go

def area_bar_plot(df, group, area, kind, last_date, first_pred_date):
    """
    create a bar plot of actual and predicted value for given kind from one area
    
    Parameters
    -----------
    df: pandas.dataframe, all data
    
    group: str, group of 'world' or 'usa'
    
    area: str, country or state in US
    
    kind: kind of value to display for actual value and prediction. "Daily Cases", "Daily Deaths", "Cases", "Deaths"
    
    last_date: str, last date of the actual data
    
    first_pred_date: str, first date for the prediction
    
    Return
    ------------
    fig: graph objects with the area bar plot
    """
    # query the data with specific group & area, @ was used for prefix in query method
    s = df.query("group == @group and area == @area").set_index("date")[kind]
    # get the actual data to last date and the predicted data from the first prediction date
    s_actual = s.loc[:last_date]
    s_pred = s.loc[first_pred_date:]
    # get the time series for x axis & the value for y axis (count)
    x_actual = s_actual.index
    y_actual = s_actual.values
    x_pred = s_pred.index
    y_pred = s_pred.values
    # create the figure object
    fig = go.Figure()
    # add the bar for acutal data on the figure object
    fig.add_bar(x = x_actual, y = y_actual, name= 'actual')
    # add the bar for predicted data on the figure object
    fig.add_bar(x = x_pred, y = y_pred, name = 'prediction')
    # update the layout with plot size and the title
    fig.update_layout(height = 400, width = 800, title = f'{group} {area}')
    return fig