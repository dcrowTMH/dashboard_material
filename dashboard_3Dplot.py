# Library for interactive dashboard
# library for 3D plot
import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Output, Input, State
import plotly
import plotly.graph_objects as go


# read the trial data
df = pd.read_csv('data/house_price.csv')
df.drop('Id', axis = 1, inplace = True)
number_column = [i for i in df.columns if df[i].dtype != 'object']
df_final = df[number_column]

plot_target = df[['MSSubClass','LotFrontage','LotArea']]

# plot the default graph
fig = go.Figure(data=[go.Surface(z = plot_target)])

fig.update_layout(title='Trial', autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90))

fig.show()

app = Dash(__name__)
# title for the dashboard
title = html.H2('3D testing plot')

# graph element refer to the function
graph = dcc.Graph(figure = fig, id='graph-3d')

# dropdown element for value input for interation
dropdown_0 = dcc.Dropdown(df_final.columns, 'MSSubClass', id='first-dropdown')
dropdown_1 = dcc.Dropdown(df_final.columns, 'LotFrontage', id='second-dropdown')
dropdown_2 = dcc.Dropdown(df_final.columns, 'LotArea', id='third-dropdown')
# group the three dropdown together
dropdown = html.Div([dropdown_0, dropdown_1, dropdown_2])

# final layout
layout = html.Div([title, dropdown, graph])

app.layout = layout

# call back interation with dropdown & graph as output
@app.callback(
    Output('graph-3d', 'figure'),
    [
    Input('first-dropdown','value'),
    Input('second-dropdown','value'),
    Input('third-dropdown','value'),
    ]
)

# define the function as previous graph
def generate_3d_plot(x,y,z):
    fig = go.Figure(data = go.Surface(z=df_final[[x,y,z]]))
    fig.update_layout(title = f'Surface plot: {x} vs {y} vs {z}',
                     autosize = True, width = 1000, height = 1000,
                     margin = dict(l=65, r=50, b=65, t=90))
    return fig

if __name__ == '__main__':
    app.run_server(debug = True)