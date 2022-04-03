from jupyer_dash import JupyterDash # application for Dashboard in jupyter notebook
import dash_html_components as html # all the html element for dash board
from dash_table import DataTable # table element instead of build the table with HTML element one by one
import dash_core_components as dcc # tab/ radio bar as dash components
from plotly.subplots import make_subplots
from plotly.colors import qualitative

def create_table(group):
    # Columns needed to be shown (adjustable for other data set)
    used_columns = [
        "area",
        "Deaths",
        "Cases",
        "Deaths per Million",
        "Cases per Million"
    ]
    df = SUMMARY.loc[group, used_columns] # get the data from the loaded dataset
    first_col = "Country" if group == 'world' else 'State' # get the global data or US states data
    df = df.rename(columns={"area":first col}) # display Country or State bases on the input
    
    # needed to adjust for other dataset
    # string columns
    columns = [{"name": first_col, "id": first_col}]
    # numeric columns
    for name in df.columns[1:]:
        # DataTable format for the columns attribute
        col_info ={
            "name": name,
            "id": name,
            "type": "numeric",
            "format": {'specifier': ','},
        }
        columns.append(col_info)
    # display the sorted values base on Deaths at first
    data = df.sort_values("Deaths", ascending = False).to_dict("records")
    return DataTable(
        id=f"{group}-table",
        columns=columns,
        data = data,
        active_cell={"row":0, "column":0}, # active cell at first
        fixed_rows ={'headers': True}, # header was fixed with scrolling
        sort_action="native", # show the native sorted value
        derived_virtual_data=data,
        # CSS code for each class/item
        style_table={
            # both min height and height needed to be set due to the bug of Dash
            'minHeight':"80vh",
            "height":"80vh",
            "overflowY":"scroll", # vertical scroll for overflow row
            "borderRadius": "0px 0px 10px 10px",
        },
        style_cell={
            'whiteSpace':'normal',
            'height':'auto',
            'fontFamily':'verdana',
        },
        style_header={
            # format for the header column
            'textAligh": "center",
            "fontSize": 14,
        },
        style_data={
            "fontSize": 12,
        },
        style_data_conditional=[
            # conditional style code for specific cell
            {
                "if":{"column_id":first_col}, # specify the string(Country/ State) name
                "width": "120px",
                "textAligh": "left",
                # show it as clickable
                "textDecoration":"underline",
                "cursor":"pointer",
            },
            {
                "if":{"row_index": "odd"},
                # show contrast between rows
                "backgroundColor":"#fafbfb"
            }
        
        ],
    )

def create_tab(content, label, value):
    return dcc.Tab(
        content,
        label = label, # shown label
        value = value, # actual value
        id = f'{value}-tab',
        className='singel-tab',
        selected_className="single-tab-selected",
    )


# graph and data part
COLORS = qualitative.T10[:2] # plotly color package
LAST_DATE = SUMMARY['date'].iloc[-1] # obtain the last date
FIRST_PRED_DATE = LAST_DATE + pd.Timedelta('1D')

def create_figures(title ,n=3):
    figs = [] # list for the figures created
    annot_props = {"x": 0.1, "xref": "paper", "yref":"paper", "xanchor":"left",
                  "showarrow":False, "font" : {"size":18},}
    for _ in range(n):
        fig = make_subplots(rows=2 , cols = 1, vertical_spacing=0.1)
        fig.update_layout(
            title={'text': title, "x":0.5, "y":0.97, "font": {"size":20}},
            annotations=[
                {"y": 0.95, "text":"<b>Deaths</b>"},
                {"y": 0.3, "text":"<b>Cases</b>"},
            ],
            margin = {"t": 40, "l": 50, "r":10, "b":0},
            legend = {"x":0.5, "y" :-0.05, "xanchor":"center", "orientation": "h",
                     "font": {"size":15}})
        fig.update_traces(showlegned= False, row=2, col=1) # Just shoe the legend once
        fig.update_traces(hovertemplate="%{x} - %{y:,}")
        fir.update_annotations(annot_props)
        figs.append(fig)
    return figs

def make_cumulative_graphs(fig, df_dict, kinds):
    for row, kind in enumerate(kinds, start = 1):
        for i, (name, df) in enumerate(df_dict.items()):
            fig.add_scatter(x=df.index, y = df[kind], mode='line+markers',
                            showlegend=row==1, line={"color" : COLORS[i]},
                            name = name, row = row, col=1)
def make_daily_graph(fig, df_dict, kinds):
    for row, kind in enumerate(kinds, start = 1):
        for i, (name, df) in enumerate(df_dict.items()):
            fig.add_bar(x = df.index, y = df[kind], marker ={"color":COLORS[i]},
                        showlegend = row ==1, name=name, row=row, col = 1)
def make_weekly_graph(fig, df_dict, kinds):
    offset = "W-" + LAST_DATE.strftime("%a").upper()
    df_dict = {name: df.resample(offset, kind="timestamp", closed="right")[kinds].sum()
              for name, df in df_dict.items()}
    
    for row, kind in enumerate(kinds, start = 1):
        for i, (name, df) in enumerate(df_dict.items()):
            fig.add_scatter(x = df.index, y = df[kind], mode="lines+markers",
                            showlegned = row==1, line={"color":COLORS[i]},
                            name = name, row = row, col = 1)
            
def create_graphs(group, area):
    df = ALL_DATA.loc[(group, area)]
    df_dict = {"actual": df.loc[:LAST_DATE], "prediction":df.loc[FIRST_PRED_DATE:]}
    kinds = ["Deaths", "Cases"]
    new_kinds = ["Daily Deaths", "Daily Cases"]
    figs = create_figures(area)
    make_cumulative_graphs(figs[0], df_dict, kinds)
    make_daily_graphs(figs[1], df_dict, new_kinds)
    make_weekly_graphs(figs[2], df_dict, new_kinds)
    return figs

# map part
def hover_text(x):
    name = x["area"]
    deaths = x["Deaths"]
    cases = x["Cases"]
    deathsm = x["Deaths per Million"]
    casesm = x["Cases per Millsion"]
    return (
        f"<b>{name}</b><br>"
        f"Deaths - {deaths:,.0f}<br>"
        f"Cases - {cases:,.0f}<br>"
        f"Deaths per Million - {deathsm:,.0f}<br>"
        f"Cases per Million - {casesm:,.0f}<br>"
    )

def create_map(group, radio_value):
    df = SUMMARY.loc[group].query("population > 0.5")
    lm = None if group == "world" else "USA-states"
    proj = "robinson" if group == "world" else "albers usa"
    
    fig = go.Figure()
    fig.add_choropleth(
        locations = df["code"],
        z=df[radio_value],
        zmin = 0,
        locationmode =lm,
        clolorscale= "orrd",
        marker_line_wight = 0.5,
        text = df.apply(hover_text, axis=1),
        hoverinfo="text",
        colorbar=dict(len=0.6, x= 1, y =0.5),
    )
    fig.update_layout(
        geo={
            "lataxis": {"range" : [-50, 68]},
            "lonaxis": {"range" : [-130, 150]},
            "projection" :{"type" : proj},
            "showframe": False,
        },
        margin = {"t" :0, "l":10, "r":10,"b":0},
    )
    return fig