# Library for interactive dashboard
import pandas as pd
import dash
from dash.dependencies import Input, Output, State
from dash.table import DataTable
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import plotly.graph_object as go
from plotly.subplot import make_subplots
from plotly.colors import qualitive