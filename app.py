import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go
import calendar

# Load default data once when app starts
train_df = pd.read_csv("train_egg_sales.csv", sep=';')
train_df['Date'] = pd.to_datetime(train_df['Date'], dayfirst=True, errors='coerce')
train_df['Year'] = train_df['Date'].dt.year
train_df['Month'] = train_df['Date'].dt.month

test_df = pd.read_csv("test_egg_sales.csv")
test_df['Date'] = pd.to_datetime(test_df['Date'], dayfirst=True, errors='coerce')
test_df['Year'] = test_df['Date'].dt.year
test_df['Month'] = test_df['Date'].dt.month

# Fit Prophet model (exclude invalid dates)
prophet_df = train_df.dropna(subset=['Date']).rename(columns={'Date': 'ds', 'Egg Sales': 'y'})
model = Prophet()
model.fit(prophet_df)

future = test_df.dropna(subset=['Date']).rename(columns={'Date': 'ds'})
forecast = model.predict(future)
test_df.loc[future.index, 'Egg Sales Prediction'] = forecast['yhat'].values

app = dash.Dash(__name__)

BACKGROUND_COLOR = '#121212'  # Pure black background (used in CSS only)
TEXT_COLOR = '#000000'        # Black text as requested
ACCENT_COLOR = '#FF6F61'      # Coral accent color for buttons and graph

app.layout = html.Div([
    # Background blur filter div
    html.Div(id='app-background-filter'),

    # Main content wrapped in a white transparent box for readability
    html.Div([
        html.H1("Egg Sales Monthly Analysis & Prediction",
                style={'textAlign': 'center', 'color': TEXT_COLOR, 'fontWeight': 'bold', 'marginBottom': '30px'}),

        html.Div([
            html.Div([
                html.Label("Select Month:", style={'color': TEXT_COLOR, 'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='month-dropdown',
                    options=[{'label': calendar.month_name[m], 'value': m} for m in range(1, 13)],
                    value=train_df['Month'].min() or 1,
                    clearable=False,
                    style={'width': '180px'}
                ),
            ], style={'display': 'inline-block', 'marginRight': '40px'}),

            html.Div([
                html.Label("Select Year:", style={'color': TEXT_COLOR, 'fontWeight': 'bold'}),
                dcc.Dropdown(
                    id='year-dropdown',
                    options=[{'label': y, 'value': y} for y in sorted(train_df['Year'].dropna().unique())],
                    value=train_df['Year'].min(),
                    clearable=False,
                    style={'width': '120px'}
                ),
            ], style={'display': 'inline-block'}),
        ], style={'marginBottom': '40px'}),

        html.Button("Show Sales Analysis", id='analyze-button', n_clicks=0,
                    style={'backgroundColor': ACCENT_COLOR, 'color': 'white',
                           'fontWeight': 'bold', 'padding': '10px 20px', 'border': 'none',
                           'borderRadius': '5px', 'cursor': 'pointer'}),

        html.Div(id='graph-container',
                 style={'marginTop': '40px'}),

        html.Div([
            html.Label("Select Date for Sales Prediction:",
                       style={'fontWeight': 'bold', 'color': TEXT_COLOR, 'marginTop': '40px',
                              'display': 'block'}),

            dcc.DatePickerSingle(
                id='prediction-date-picker',
                min_date_allowed=test_df['Date'].min().date(),
                max_date_allowed=test_df['Date'].max().date(),
                date=test_df['Date'].min().date(),
                display_format='YYYY-MM-DD',
                style={'marginTop': '10px'}
            ),

            html.Button("Get Prediction", id='predict-button', n_clicks=0,
                        style={'backgroundColor': ACCENT_COLOR, 'color': 'white',
                               'fontWeight': 'bold', 'padding': '10px 20px', 'border': 'none',
                               'borderRadius': '5px', 'cursor': 'pointer', 'marginLeft': '20px'}),

            html.Div(id='prediction-output',
                     style={'fontSize': '22px', 'fontWeight': 'bold', 'color': ACCENT_COLOR,
                            'marginTop': '20px'})
        ], style={'marginTop': '40px'}),

    ], id='main-content'),  # Important for CSS overlay

], style={'height': '100vh', 'padding': '0', 'margin': '0', 'fontFamily': 'Arial, sans-serif'}
)


@app.callback(
    Output('graph-container', 'children'),
    Input('analyze-button', 'n_clicks'),
    State('month-dropdown', 'value'),
    State('year-dropdown', 'value'),
    prevent_initial_call=True
)
def update_graph(n_clicks, month, year):
    filtered = train_df[(train_df['Month'] == month) & (train_df['Year'] == year)]

    if filtered.empty:
        return html.Div("No sales data for selected month and year.",
                        style={'color': TEXT_COLOR, 'fontSize': '18px', 'textAlign': 'center'})

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered['Date'],
        y=filtered['Egg Sales'],
        mode='lines+markers',
        line=dict(color=ACCENT_COLOR, width=3),
        marker=dict(size=6),
        name='Egg Sales'
    ))
    fig.update_layout(
        margin=dict(l=40, r=30, t=40, b=30),
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(title='Date'),
        yaxis=dict(title='Sales')
    )

    return dcc.Graph(figure=fig, style={'height': '400px'})


@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('prediction-date-picker', 'date'),
    prevent_initial_call=True
)
def show_prediction(n_clicks, selected_date):
    if selected_date:
        selected_date = pd.to_datetime(selected_date)
        row = test_df[test_df['Date'] == selected_date]
        if not row.empty and pd.notna(row.iloc[0]['Egg Sales Prediction']):
            pred_value = int(round(row.iloc[0]['Egg Sales Prediction']))  # Rounded whole number
            pred_date = row.iloc[0]['Date'].date()
            return f"The Prediction of Egg Sales for {pred_date} is {pred_value}"
        else:
            return "Selected date is out of prediction range or prediction unavailable."
    return ""


if __name__ == '__main__':
    app.run(debug=True)
