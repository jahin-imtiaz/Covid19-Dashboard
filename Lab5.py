import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime as dt
import re
import dash_bootstrap_components as dbc
import json
import numpy as np
import dash_table

external_stylesheets = [dbc.themes.SLATE]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

df = pd.read_csv('Final_Covid_Data.csv', dtype={"iso_code": str})
country_dict_list = [{'label': row['location'], 'value': row['iso_code']} for index, row in
                     df.groupby(['iso_code', 'location']).sum().reset_index().iterrows()]
states = json.load(open('us-states-mod.json', 'r'))
df_states = pd.read_csv("us-states-case&death.csv", dtype={"fips": str})
df_states['date'] = pd.to_datetime(df_states['date'])
states_dict_list = [{'label': row['state'], 'value': row['fips']} for index, row in
                    df_states.groupby(['state', 'fips']).sum().reset_index().iterrows()]
df_counties = pd.read_csv("us-counties_case&death.csv", dtype={"fips": str})
df_raceEthnicity = pd.read_csv('us-states-race&ethnicity.csv')
countries = json.load(open('countries-geojson.json', 'r'))

dropdown_option_array = [
                        {'label': 'New Cases', 'value': 'new_cases'},
                        {'label': 'New Deaths', 'value': 'new_deaths'},
                        {'label': 'New Tests', 'value': 'new_tests'},
                        {'label': 'Positive Rate', 'value': 'positive_rate'},
                        {'label': 'Daily Covid Related Tweet', 'value': 'covid_tweet_count'},
                        {'label': 'Total Cases', 'value': 'total_cases'},
                        {'label': 'Total Deaths', 'value': 'total_deaths'},
                        {'label': 'Total Tests', 'value': 'total_tests'},
                        {'label': 'Total Cases Per Million', 'value': 'total_cases_per_million'},
                        {'label': 'Total Deaths Per Million', 'value': 'total_deaths_per_million'},
                    ]

app.layout = dbc.Container(
    [
        html.H4(children='Covid-19 Dashboard for CSE-332 at Stony Brook University', style={'position': 'absolute',
                                                                                           'top': '2%',
                                                                                           'left': '50%',
                                                                                           'transform': 'translate(-50%, -50%)',
                                                                                           '-ms-transform': 'translate(-50%, -50%)',
                                                                                           'color': '#ccff00',
                                                                                           'font-size': '20px',
                                                                                           'padding': '3px 9px',
                                                                                           'border': 'none',
                                                                                           'cursor': 'pointer',
                                                                                           'border-radius': '18px',
                                                                                           'text-align': 'center'}),
        dbc.Tabs(
            [
                dbc.Tab(label="World", tab_id="tab-1"),
                dbc.Tab(label="United States", tab_id="tab-2"),
            ],
            id="tabs",
            active_tab="tab-1",
        ),
        html.Div(id="content"),
    ], fluid=True
)


@app.callback(Output("content", "children"), [Input("tabs", "active_tab")])
def switch_tab(at):
    if at == "tab-1":
        return tab1_content
    elif at == "tab-2":
        return tab2_content
    return html.P("This shouldn't ever be displayed...")

tab1_content = html.Div(children=[
    dbc.Row([
        dbc.Col([
            dbc.Row([
                html.P('Select a date:', style={'color': '#ccff00'}),
                dcc.DatePickerSingle(
                id='world_Map_calender',
                min_date_allowed=dt(2020, 1, 21),
                max_date_allowed=dt(2020, 11, 19),
                initial_visible_month=dt(2020, 9, 10),
                date=dt(2020, 10, 19, 23, 59, 59))], className='p-1 bg- text-dark border border-dark rounded',no_gutters=True),
            dbc.Row([
                html.Br(),
                html.P('Select a metric:', style={'color': '#ccff00'}),
                dbc.Select(
                    id='bar_variable',
                    options=dropdown_option_array,
                    value='new_cases', )], className='p-1 bg- text-dark border border-dark rounded', no_gutters=True),
            dbc.Row([
                html.Br(),
                html.Br(),
                html.Br(),
                html.P('Select comparison metric:', style={'color': '#ccff00'}),
                dbc.Select(
                    id='scatter_y_variable',
                    options=dropdown_option_array,
                    value='new_tests')], className='p-1 bg- text-dark border border-dark rounded',no_gutters=True),
        ], width=1),
        dbc.Col([
            dcc.Graph(id='world_map'),
        ], width=7),
        dbc.Col([
            dcc.Graph(id='scatter_plot'),
        ], width=4),
    ], className="p-1 bg-dark text-dark border border-dark rounded", align="center"),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='bar_graph'),
        ]),
        dbc.Col([
            dcc.Graph(id='country_parallel_coordinate'),
        ]),
    ]),
]);


tab2_content = html.Div(children=[
    dbc.Row([
        dbc.Col([
            dbc.Row([
                html.P('Select a date:', style={'color': '#ccff00'}),
                dcc.DatePickerSingle(
                    id='USA_Map_calender',
                    min_date_allowed=dt(2020, 1, 21),
                    max_date_allowed=dt(2020, 11, 19),
                    initial_visible_month=dt(2020, 9, 10),
                    date=dt(2020, 10, 19, 23, 59, 59),)
            ], id='USA_Map_calender_div', no_gutters=True),
            dbc.Row([
                html.P('Select a date:', style={'color': '#ccff00'}),
                dcc.DatePickerSingle(
                    id='USA_Map_calender_2',
                    min_date_allowed=dt(2020, 1, 21),
                    max_date_allowed=dt(2020, 11, 19),
                    initial_visible_month=dt(2020, 9, 10),
                    date=dt(2020, 10, 19, 23, 59, 59),)
            ],id='USA_Map_calender_2_div', style={'display': 'none'}, no_gutters=True),
            dbc.Row([
                html.Br(),
                html.P('Select a metric:', style={'color': '#ccff00'}),
                dbc.Select(
                    id='USA_Map_dropdown',
                    options=[
                        {'label': 'New Cases', 'value': 'new_cases'},
                        {'label': 'New Deaths', 'value': 'new_deaths'},
                    ],
                    value='new_cases')
            ], id='USA_Map_dropdown_div', style={'display': 'none'}, no_gutters=True),
            dbc.Row([
                html.Br(),
                html.Br(),
                html.P('Select state to compare:', style={'color': '#ccff00'}),
                dbc.Select(
                    id='Compare_with_dropdown',
                    options=states_dict_list,
                    value='36')
            ], id='Compare_with_dropdown_div', style={'display': 'none'}, no_gutters=True),
            dbc.Row([
                html.Br(),
                html.P('Select a metric:', style={'color': '#ccff00'}),
                dbc.Select(
                    id='usa_bar_chart_dropdown',
                    options=[
                        {'label': 'New Cases', 'value': 'new_cases'},
                        {'label': 'New Deaths', 'value': 'new_deaths'},
                    ],
                    value='new_cases', )
            ], id='usa_bar_chart_dropdown_div', no_gutters=True),
            dbc.Row([
                html.Br(),
                html.Br(),
                html.P('Select comparison metric:', style={'color': '#ccff00'}),
                dbc.Select(
                    id='usa_scatter_y_dropdown',
                    options=dropdown_option_array,
                    value='new_tests')
            ], id='usa_scatter_y_dropdown_div', no_gutters=True),
        ], width=1),
        dbc.Col([
            html.Div(id='USA_Map_div', children=dcc.Graph(id='USA_Map')),
            html.Div(id='USA_Map_2_div', children=dcc.Graph(id='USA_Map_2'), style={'display': 'none'}),
            html.Div(id='close_button_div', children=dbc.Button('X', id='close_button', style={'position': 'absolute',
                                                                                               'top': '8%',
                                                                                               'left': '95%',
                                                                                               'transform': 'translate(-50%, -50%)',
                                                                                               '-ms-transform': 'translate(-50%, -50%)',
                                                                                               'color': '#ccff00',
                                                                                               'font-size': '16px',
                                                                                               'padding': '3px 9px',
                                                                                               'border': 'none',
                                                                                               'cursor': 'pointer',
                                                                                               'border-radius': '18px',
                                                                                               'text-align': 'center'}),
                     style={'display': 'none'}),
        ], width=7),
        dbc.Col([
            html.Div([
                dbc.Row([
                    dbc.Col(dash_table.DataTable(id='data_table_1',
                                                 style_table={'overflowY': 'auto', 'height': '450px', 'widths': '50%',
                                                              'overflowX': 'false'},
                                                 style_header={
                                                     'backgroundColor': 'rgb(30, 30, 30)',
                                                     'fontWeight': 'bold'
                                                 },
                                                 style_data_conditional=[{
                                                     'if': {'row_index': 'odd'},
                                                     'backgroundColor': '#707070'}],
                                                 style_cell={'textAlign': 'center', 'backgroundColor': '#383838',
                                                             'color': 'white', 'width': '50%'},
                                                 fixed_rows={'headers': True},
                                                 style_as_list_view=False, )),
                    dbc.Col(dash_table.DataTable(id='data_table_2',
                                                 style_table={'overflowY': 'auto', 'height': '450px', 'widths': '50%',
                                                              'overflowX': 'false'},
                                                 style_header={
                                                     'backgroundColor': 'rgb(30, 30, 30)',
                                                     'fontWeight': 'bold'
                                                 },
                                                 style_data_conditional=[{
                                                     'if': {'row_index': 'odd'},
                                                     'backgroundColor': '#707070'}],
                                                 style_cell={'textAlign': 'center', 'backgroundColor': '#383838',
                                                             'color': 'white', 'width': '50%'},
                                                 fixed_rows={'headers': True},
                                                 style_as_list_view=False, )),
                ], no_gutters=True),
            ], id='table_div'),
            html.Div([
                dbc.Col(dcc.Graph(id='pie_race_cases')),
            ], id='pie_charts_div', style={'display': 'none'})
        ], width=4),
    ], className="p-1 bg-dark text-dark border border-dark rounded", align="center"),
    dbc.Row([
        dbc.Col([
            html.Div(dcc.Graph(id='timeseries_national'), id='timeseries_national_div', style={'display': 'none'}),
            html.Div(dcc.Graph(id='usa_bar_chart'), id='usa_bar_chart_div'),
        ]),
        dbc.Col([
            html.Div(dcc.Graph(id='timeseries_states'), id='timeseries_states_div', style={'display': 'none'}),
            html.Div(dcc.Graph(id='usa_comparison_chart'), id='usa_comparison_chart_div'),
        ]),
    ]),
]);


@app.callback(
    Output('USA_Map', 'selectedData'),
    Input('USA_Map_div', 'n_clicks'), prevent_initial_call=True
)
def deselect_on_click_map_div(n_clicks):
    return {}

@app.callback(
    [Output('timeseries_national_div', 'style'),
     Output('timeseries_states_div', 'style'),
     Output('USA_Map_div', 'style'),
     Output('USA_Map_2_div', 'style'),
     Output('pie_charts_div', 'style'),
     Output('table_div', 'style'),
     Output('close_button_div', 'style'),
     Output('usa_bar_chart_div', 'style'),
     Output('usa_comparison_chart_div', 'style'),
     Output('USA_Map_dropdown_div', 'style'),
     Output('Compare_with_dropdown_div', 'style'),
     Output('usa_bar_chart_dropdown_div', 'style'),
     Output('usa_scatter_y_dropdown_div', 'style'),
     Output('USA_Map_calender_div', 'style'),
     Output('USA_Map_calender_2_div', 'style')
     ],
    [Input('close_button', 'n_clicks'),
     Input('USA_Map', 'selectedData')], prevent_initial_call=True
)
def on_close_button_or_state_click(n_clicks, selectedData):
    ctx = dash.callback_context
    id = ctx.triggered[0]['prop_id'].split('.')[0]

    #check if the selected Data is [] or not, if true return first if statement(null because it was unselected by div click event)
    if (id == 'close_button') or not selectedData:
        return {'display': 'none'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'},\
               {'display': 'block'}, {'display': 'none'}, {'display': 'block'}, {'display': 'block'}, {'display': 'none'},\
               {'display': 'none'}, {'display': 'block'}, {'display': 'block'},  {'display': 'block'},  {'display': 'none'},
    else:
        return {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, {'display': 'block'}, {
            'display': 'block'}, {'display': 'none'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'},\
               {'display': 'block'}, {'display': 'block'}, {'display': 'none'}, {'display': 'none'},  {'display': 'none'},\
               {'display': 'block'}


@app.callback(
    Output('usa_bar_chart', 'figure'),
    Input('usa_bar_chart_dropdown', 'value')
)
def update_USA_bar_chart(bar_variable):
    bar_Country_code ='USA'
    bar_Country_name = df.loc[df['iso_code'] == bar_Country_code]['location'].iloc[1]

    trace_bar = go.Bar(
        x=df.loc[df['iso_code'] == bar_Country_code]['date'],
        y=df.loc[df['iso_code'] == bar_Country_code][bar_variable],
        text=bar_Country_name,
        name=bar_variable,
        marker_color='#ccff00',
        showlegend=True,
        hoverinfo='text+x+y',
    )
    layout = go.Layout(title=dict(
        text=bar_variable + ' in ' + bar_Country_name,
        font=dict(color='#ccff00')),
        xaxis=dict(
            tickfont_size=14,
            showgrid=False,
            color='#ccff00',
            tickwidth=1,
            ticks="outside", ),
        yaxis=dict(
            title='Count',
            tickwidth=1,
            ticks="outside",
            titlefont_size=16,
            tickfont_size=14,
            color='#ccff00',
            showgrid=False,
            showline=True, linewidth=1, linecolor='#707070',
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)',
            font=dict(color='#ccff00'),
        ),
        bargap=0.5,
        plot_bgcolor='#383838',
        paper_bgcolor='#383838',
    )
    data = [trace_bar]
    fig = go.Figure(data=data, layout=layout)
    fig.update_xaxes(rangeslider_visible=True,
                     rangeselector=dict(
                         buttons=list([
                             dict(count=1, label="1m", step="month", stepmode="backward"),
                             dict(count=6, label="6m", step="month", stepmode="backward"),
                             dict(count=1, label="YTD", step="year", stepmode="todate"),
                             dict(count=1, label="1y", step="year", stepmode="backward"),
                             dict(step="all")
                         ]),
                         bgcolor='#aaff00'))
    return fig


@app.callback(
    Output('usa_comparison_chart', 'figure'),
    [Input('usa_bar_chart_dropdown', 'value'),
     Input('usa_scatter_y_dropdown', 'value')]
)
def update_USA_comparison_chart(x_variable, y_variable):
    country_variable = 'USA'
    df_scatter = df.loc[df['iso_code'] == country_variable]

    x_values = df_scatter[x_variable]
    y_values = df_scatter[y_variable]

    trace_scatter = go.Scatter(x=x_values, y=y_values,
                               text=country_variable,
                               mode='markers',
                               marker_line_width=2,
                               marker_size=10,
                               marker_color='#ccff00',
                               )
    layout = go.Layout(title=dict(
        text=y_variable + ' vs ' + x_variable,
        font=dict(color='#ccff00')),
        xaxis=dict(
            title=x_variable,
            tickfont_size=14,
            gridwidth=.05, gridcolor='#707070',
            showline=True, linewidth=.5, linecolor='#707070',
            zeroline=True, zerolinewidth=.5, zerolinecolor='#707070',
            color='#ccff00', ),
        yaxis=dict(
            title=y_variable,
            titlefont_size=16,
            tickfont_size=14,
            gridwidth=.05, gridcolor='#707070',
            showline=True, linewidth=.5, linecolor='#707070',
            zeroline=True, zerolinewidth=.5, zerolinecolor='#707070',
            color='#ccff00', ),
        plot_bgcolor='#383838',
        paper_bgcolor='#383838',
    )

    data = [trace_scatter]
    return go.Figure(data=data, layout=layout)


@app.callback(
    Output('pie_race_cases', 'figure'),
    Input('USA_Map', 'selectedData'), prevent_initial_call=True
)
def update_pie_charts(selectedData):
    # check if the selected Data is [] or not, if true return empty graph or dont access this value (null because it was unselected by div click event)
    variable = selectedData['points'][0]['location']
    row = df_raceEthnicity[df_raceEthnicity['fips'] == int(variable)]
    state_name = row.iloc[0]['state']

    white_cases = int(row['White % of Cases'])
    Black_cases = int(row['Black % of Cases'])
    Hispanic_cases = int(row['Hispanic % of Cases'])
    Asian_cases = int(row['Asian % of Cases'])
    American_Indian_or_Alaska_Native_cases = int(row['American Indian or Alaska Native % of Cases'])
    Native_Hawaiian_or_Other_Pacific_Islander_cases = int(row['Native Hawaiian or Other Pacific Islander % of Cases'])

    white_deaths = int(row['White % of Deaths'])
    Black_deaths = int(row['Black % of Deaths'])
    Hispanic_deaths = int(row['Hispanic % of Deaths'])
    Asian_deaths = int(row['Asian % of Deaths'])
    American_Indian_or_Alaska_Native_deaths = int(row['American Indian or Alaska Native % of Deaths'])
    Native_Hawaiian_or_Other_Pacific_Islander_deaths = int(row['Native Hawaiian or Other Pacific Islander % of Deaths'])
    Labels = ['white', 'Black', 'Hispanic', 'Asian', 'American Indian or Alaska Native',
              'Native Hawaiian or Pacific Islander']

    Values = [white_cases, Black_cases, Hispanic_cases, Asian_cases, American_Indian_or_Alaska_Native_cases,
              Native_Hawaiian_or_Other_Pacific_Islander_cases]
    colors = ['#039ba8', '#8f5203', '#697d7b', '#8f0303', '#de00cf', '#a8d408', '#858700']
    trace_pie1 = go.Pie(
        labels=Labels,
        values=Values,
        domain=dict(column=0),
        hole=.4,
        insidetextorientation='radial',
        hoverinfo='label+percent',
        textinfo='percent',
        textfont_size=16,
        textfont_color='#f7f7f7',
        textposition='inside',
        marker=dict(colors=colors, line=dict(color='#000000', width=1)))

    layout = go.Layout(title=dict(text='Total cases by race/ethnicity in ' + state_name,
                                  x=0.5, y=.98, font=dict(color='#ccff00')),
                       grid={"rows": 1, "columns": 2},
                       annotations=[
                           {
                               "font": {
                                   "size": 18,
                                   "color": '#f7f7f7'
                               },
                               "showarrow": False,
                               "text": "Cases",
                               "x": 0.19,
                               "y": 0.5
                           },
                           {
                               "font": {
                                   "size": 18,
                                   "color": '#f7f7f7'
                               },
                               "showarrow": False,
                               "text": "Deaths",
                               "x": 0.82,
                               "y": 0.5
                           }
                       ],
                       paper_bgcolor='#383838',
                       plot_bgcolor='#383838',
                       margin={"r": 10, "t": 0, "l": 10, "b": 10},
                       legend=dict(
                           x=0.07,
                           y=0,
                           bgcolor='rgba(255, 255, 255, 0)',
                           bordercolor='rgba(255, 255, 255, 0)',
                           font={'color': '#ccff00'},
                           orientation="h"
                       ))

    Values = [white_deaths, Black_deaths, Hispanic_deaths, Asian_deaths, American_Indian_or_Alaska_Native_deaths,
              Native_Hawaiian_or_Other_Pacific_Islander_deaths]
    trace_pie2 = go.Pie(
        labels=Labels,
        values=Values,
        domain=dict(column=1),
        hole=.4,
        insidetextorientation='radial',
        hoverinfo='label+percent',
        textinfo='percent',
        textfont_size=16,
        textfont_color='#f7f7f7',
        textposition='inside',
        marker=dict(colors=colors, line=dict(color='#000000', width=1)))

    data = [trace_pie1, trace_pie2]
    fig = go.Figure(data=data, layout=layout)
    return fig


@app.callback(
    Output('timeseries_national', 'figure'),
    [Input('USA_Map', 'selectedData'),
     Input('USA_Map_dropdown', 'value')], prevent_initial_call=True
)
def update_time_series_national(selectedData, variable2):
    #check if the selected Data is [] or not, if true return empty graph or dont access this value (null because it was unselected by div click event)
    variable1 = selectedData['points'][0]['location']
    df_states_tmp = df_states[df_states['fips'].astype(int) == int(variable1)]

    fig = px.line(df_states_tmp, x='date', y=[variable2, 'national_avg_' + variable2])

    fig.update_layout(title='Daily ' + variable2 + ' compared to National average ' + variable2 + ' on that day',
                      xaxis=dict(
                          title='Covid-19 Time Period',
                          color='#ccff00',
                          tickfont_size=14,
                          showgrid=False,
                          tickcolor='#ccff00',
                          tickfont_color='#ccff00',
                          tickwidth=1,
                          ticks="outside"),
                      yaxis=dict(
                          title='Counts',
                          color='#ccff00',
                          tickcolor='#ccff00',
                          tickfont_color='#ccff00',
                          titlefont_size=16,
                          tickfont_size=14,
                          showgrid=False,
                          showline=True, linewidth=2, linecolor='black',
                      ),
                      legend=dict(
                          x=0.7,
                          y=1.0,
                          bgcolor='rgba(255, 255, 255, 0)',
                          bordercolor='rgba(255, 255, 255, 0)',
                          font={'color': '#ccff00'},
                      ),
                      legend_title_font_color='#ccff00',
                      title_font_color='#ccff00',
                      bargap=0.15,
                      paper_bgcolor='#383838',
                      plot_bgcolor='#383838')

    fig.update_xaxes(rangeslider_visible=True,
                     rangeselector=dict(
                         buttons=list([
                             dict(count=1, label="1m", step="month", stepmode="backward"),
                             dict(count=6, label="6m", step="month", stepmode="backward"),
                             dict(count=1, label="YTD", step="year", stepmode="todate"),
                             dict(count=1, label="1y", step="year", stepmode="backward"),
                             dict(step="all")
                         ]),
                         bgcolor='#aaff00',
                     ))
    return fig


@app.callback(
    Output('timeseries_states', 'figure'),
    [Input('USA_Map', 'selectedData'),
     Input('USA_Map_dropdown', 'value'),
     Input('Compare_with_dropdown', 'value')], prevent_initial_call=True
)
def update_time_series_states(selectedData, variable3, variable2):
    # check if the selected Data is [] or not, if true return empty graph or dont access this value (null because it was unselected by div click event)
    variable1 = selectedData['points'][0]['location']

    df_states_tmp = df_states[
        (df_states['fips'].astype(int) == int(variable1)) | (df_states['fips'].astype(int) == int(variable2))]

    fig = px.line(df_states_tmp, x='date', y=variable3, title='Time Series with Rangeslider', color="state",
                  line_group="state", hover_name="state")

    fig.update_layout(title='Daily ' + variable3 + ' comparison by states',
                      xaxis=dict(
                          title='Covid-19 Time Period',
                          color='#ccff00',
                          tickfont_size=14,
                          showgrid=False,
                          tickcolor='#ccff00',
                          tickfont_color='#ccff00',
                          tickwidth=1,
                          ticks="outside"),
                      yaxis=dict(
                          title='Counts',
                          color='#ccff00',
                          titlefont_size=16,
                          tickfont_size=14,
                          tickcolor='#ccff00',
                          tickfont_color='#ccff00',
                          showgrid=False,
                          showline=True, linewidth=2, linecolor='black',
                      ),
                      legend=dict(
                          x=0,
                          y=1.0,
                          title='',
                          bgcolor='rgba(255, 255, 255, 0)',
                          bordercolor='rgba(255, 255, 255, 0)',
                          font={'color': '#ccff00'},
                      ),
                      title_font_color='#ccff00',
                      legend_title_font_color='#ccff00',
                      paper_bgcolor='#383838',
                      plot_bgcolor='#383838')

    fig.update_xaxes(rangeslider_visible=True,
                     rangeselector=dict(
                         buttons=list([
                             dict(count=1, label="1m", step="month", stepmode="backward"),
                             dict(count=6, label="6m", step="month", stepmode="backward"),
                             dict(count=1, label="YTD", step="year", stepmode="todate"),
                             dict(count=1, label="1y", step="year", stepmode="backward"),
                             dict(step="all")
                         ]),
                         bgcolor='#aaff00'))
    return fig


@app.callback(
    Output('USA_Map_2', 'figure'),
    [Input('USA_Map', 'selectedData'),
     Input('USA_Map_calender_2', 'date'),
     Input('USA_Map_dropdown', 'value')], prevent_initial_call=True
)
def update_state_map(selectedData, date, variable):
    # check if the selected Data is [] or not, if true return empty graph or dont access this value (null because it was unselected by div click event)
    counties = json.load(open('us-counties-mod.json', 'r'))
    date = dt.strptime(re.split('T| ', date)[0], '%Y-%m-%d')
    date = date.strftime('%#m/%#d/%Y')
    state_fip = selectedData['points'][0]['location']

    county_fips = []
    new_FeatureList = []
    for feature in counties['features']:
        if (int(feature['properties']['STATE']) == int(state_fip)):
            new_FeatureList.append(feature)
            county_fips.append(feature['id'])
    counties['features'] = new_FeatureList
    center = counties['features'][0]['center']

    df_counties_tmp = df_counties[(df_counties['date'] == date) & (df_counties['fips'].isin(county_fips))]
    fig = go.Figure(
        go.Choroplethmapbox(geojson=counties, locations=df_counties_tmp.fips, z=df_counties_tmp[variable] + 1,
                            colorscale="Tealgrn",
                            showscale=False,
                            marker_opacity=0.5, marker_line_width=0,
                            customdata=df_counties_tmp[['new_cases', 'new_deaths']],
                            text=df_counties_tmp.county,
                            hovertemplate=
                            "<b>%{text}</b><br><br>" +
                            "New Cases: %{customdata[0]}<br>" +
                            "New Deaths: %{customdata[1]}<br>" +
                            "<extra></extra>"
                            ))
    fig.update_layout(mapbox_style="carto-darkmatter",
                      mapbox_zoom=4.8, mapbox_center={"lat": center[1], "lon": center[0]})
    fig.update_layout(clickmode='event+select')
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig


@app.callback(
    Output('USA_Map', 'figure'),
    [Input('usa_bar_chart_dropdown', 'value'),
     Input('USA_Map_calender', 'date')]
)
def update_usa_map(variable, date):
    date = dt.strptime(re.split('T| ', date)[0], '%Y-%m-%d')
    date = date.strftime('%#m/%#d/%Y')
    variable = variable

    df_states_tmp = df_states[df_states['date'] == date]

    fig = go.Figure(go.Choroplethmapbox(geojson=states, locations=df_states_tmp.fips, z=df_states_tmp[variable] + 1,
                                        colorscale="Tealgrn",
                                        showscale=False,
                                        marker_opacity=0.5, marker_line_width=0,
                                        text=df_states_tmp.state,
                                        customdata=df_states_tmp[['new_cases', 'new_deaths']],
                                        hovertemplate=
                                        "<b>%{text}</b><br><br>" +
                                        "New Cases: %{customdata[0]}<br>" +
                                        "New Deaths: %{customdata[1]}<br>" +
                                        "<extra></extra>"
                                        ))
    fig.update_layout(mapbox_style="carto-darkmatter",
                      mapbox_zoom=3, mapbox_center={"lat": 37.0902, "lon": -95.7129})
    fig.update_layout(clickmode='event+select')
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig;


@app.callback(
    [Output('data_table_1', 'columns'),
     Output('data_table_1', 'data')],
    Input('USA_Map_calender', 'date')
)
def update_usa_data_table_1(date):
    date = dt.strptime(re.split('T| ', date)[0], '%Y-%m-%d')
    date = date.strftime('%#m/%#d/%Y')

    df_states_temp = df_states[df_states['date'] == date]
    df_states_temp = df_states_temp[['state', 'new_cases']]

    return [{"name": i, "id": i} for i in df_states_temp.columns], df_states_temp.to_dict('records')


@app.callback(
    [Output('data_table_2', 'columns'),
     Output('data_table_2', 'data')],
    Input('USA_Map_calender', 'date')
)
def update_usa_data_table_2(date):
    date = dt.strptime(re.split('T| ', date)[0], '%Y-%m-%d')
    date = date.strftime('%#m/%#d/%Y')

    df_states_temp = df_states[df_states['date'] == date]
    df_states_temp = df_states_temp[['state', 'new_deaths']]

    return [{"name": i, "id": i} for i in df_states_temp.columns], df_states_temp.to_dict('records')


@app.callback(
    Output('bar_graph', 'figure'),
    [Input('world_map', 'selectedData'),
     Input('bar_variable', 'value'),
     ]
)
def update_bar_chart(selectedData, bar_variable):
    country_codes = []
    if not selectedData:
        country_codes = ['USA']
    else:
        for point in selectedData['points']:
            country_codes.append(point['location'])

    variable2 = bar_variable
    df_tmp = df
    df_tmp['date'] = pd.to_datetime(df_tmp['date'])
    df_tmp = df_tmp[df_tmp['iso_code'].isin(country_codes)]


    fig = px.line(df_tmp, x='date', y=variable2, title='Time Series with Rangeslider', color="location",
                  line_group="location", hover_name="location")

    fig.update_layout(title='Daily '+variable2+' comparison of selected countries',
                      xaxis=dict(
                          title='Covid-19 Time Period',
                          color='#ccff00',
                          tickfont_size=14,
                          showgrid=False,
                          tickcolor='#ccff00',
                          tickfont_color='#ccff00',
                          tickwidth=1,
                          ticks="outside"),
                      yaxis=dict(
                          title='Counts',
                          color='#ccff00',
                          titlefont_size=16,
                          tickfont_size=14,
                          tickcolor='#ccff00',
                          tickfont_color='#ccff00',
                          showgrid=False,
                          showline=True, linewidth=2, linecolor='black',
                      ),
                      legend=dict(
                          x=0,
                          y=1.0,
                          title='',
                          bgcolor='rgba(255, 255, 255, 0)',
                          bordercolor='rgba(255, 255, 255, 0)',
                          font={'color': '#ccff00'},
                      ),
                      title_font_color='#ccff00',
                      legend_title_font_color='#ccff00',
                      paper_bgcolor='#383838',
                      plot_bgcolor='#383838')

    fig.update_xaxes(rangeslider_visible=True,
                     rangeselector=dict(
                         buttons=list([
                             dict(count=1, label="1m", step="month", stepmode="backward"),
                             dict(count=6, label="6m", step="month", stepmode="backward"),
                             dict(count=1, label="YTD", step="year", stepmode="todate"),
                             dict(count=1, label="1y", step="year", stepmode="backward"),
                             dict(step="all")
                         ]),
                         bgcolor='#aaff00'))
    return fig


@app.callback(
    Output('world_map', 'figure'),
    [Input('bar_variable', 'value'),
     Input('world_Map_calender', 'date')]
)
def update_WORLD_map(variable, date):
    date = dt.strptime(re.split('T| ', date)[0], '%Y-%m-%d')
    date = date.strftime('%Y-%m-%d')
    df_tmp = df[df['date'] == date]

    z = []
    for v in df_tmp[variable]:
        if v != 0:
            z.append(np.log10(v))
        else:
            z.append(0)

    fig = go.Figure(go.Choroplethmapbox(geojson=countries, locations=df_tmp.iso_code, z=z,
                                        colorscale="Tealgrn",
                                        showscale=False,
                                        marker_opacity=0.6, marker_line_width=1,
                                        text=df_tmp.location,
                                        customdata=df_tmp[['new_cases', 'new_deaths', 'total_cases', 'total_deaths']],
                                        hovertemplate=
                                        "<b>%{text}</b><br><br>" +
                                        "New Cases: %{customdata[0]}<br>" +
                                        "New Deaths: %{customdata[1]}<br>" +
                                        "Total Cases: %{customdata[2]}<br>" +
                                        "Total Deaths: %{customdata[3]}<br>" +
                                        "<extra></extra>",
                                        ))
    fig.update_layout(mapbox_style="carto-darkmatter",
                      mapbox_zoom=.9, mapbox_center={"lat": 10, "lon": 0})
    fig.update_layout(clickmode='event+select')
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig

@app.callback(
    Output('country_parallel_coordinate', 'figure'),
    Input('world_map', 'selectedData')
)
def update_parallel_coordinates(selectedData):
    country_codes = []
    if not selectedData:
        country_codes = ['USA']
    else:
        for point in selectedData['points']:
            country_codes.append(point['location'])

    df_new = df
    df_new['date'] = pd.to_datetime(df['date'])
    df_new = df_new[(df_new['iso_code'].isin(country_codes)) & (df_new['date'] > dt(2020, 4, 1)) & (
                df_new['date'] < dt(2020, 11, 1))]
    df_new = df_new[['total_cases', 'total_deaths', 'positive_rate', 'new_tests', 'covid_tweet_count',
                     'new_cases', 'stringency_index', 'new_deaths']]
    fig = px.parallel_coordinates(df_new, color='new_cases', dimensions=df_new.columns)
    fig.update_traces(labelfont=dict(color='#ccff00', size=14),
                      tickfont=dict(color='#ccff00', size=14))
    fig.update_layout(
        plot_bgcolor='#383838',
        paper_bgcolor='#383838',
    )
    fig.update_layout(margin={"r": 30, "t": 60, "l": 50, "b": 50})
    return fig



@app.callback(
    Output('scatter_plot', 'figure'),
    [Input('world_map', 'selectedData'),
     Input('bar_variable', 'value'),
     Input('scatter_y_variable', 'value')]
)
def update_scatter_plot(selectedData, x_variable, y_variable):
    country_codes = []
    if not selectedData:
        country_codes = ['USA']
    else:
        for point in selectedData['points']:
            country_codes.append(point['location'])

    df_scatter = df.loc[df['iso_code'].isin(country_codes)]

    x_values = df_scatter[x_variable]
    y_values = df_scatter[y_variable]

    trace_scatter = go.Scatter(x=x_values, y=y_values,
                               mode='markers',
                               marker_line_width=2,
                               marker_size=10,
                               marker_color='#ccff00',
                               )
    layout = go.Layout(title=dict(
        text=y_variable + ' vs ' + x_variable,
        font=dict(color='#ccff00')),
        xaxis=dict(
            title=x_variable,
            tickfont_size=14,
            gridwidth=.05, gridcolor='#707070',
            showline=True, linewidth=.5, linecolor='#707070',
            zeroline=True, zerolinewidth=.5, zerolinecolor='#707070',
            color='#ccff00', ),
        yaxis=dict(
            title=y_variable,
            titlefont_size=16,
            tickfont_size=14,
            gridwidth=.05, gridcolor='#707070',
            showline=True, linewidth=.5, linecolor='#707070',
            zeroline=True, zerolinewidth=.5, zerolinecolor='#707070',
            color='#ccff00', ),
        plot_bgcolor='#383838',
        paper_bgcolor='#383838',
    )

    data = [trace_scatter]
    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(margin={"r": 20, "t": 60, "l": 50, "b": 50})
    return fig


if __name__ == '__main__':
    app.run_server()
