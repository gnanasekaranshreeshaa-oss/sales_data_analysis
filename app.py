import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
from prophet import Prophet
import plotly.graph_objs as go
import plotly.express as px
import calendar
import numpy as np
from datetime import datetime, date

# Enhanced color scheme
COLORS = {
    'primary': '#2E3440',      # Dark blue-gray
    'secondary': '#3B4252',    # Lighter blue-gray
    'accent': '#5E81AC',       # Blue accent
    'success': '#A3BE8C',      # Green
    'warning': '#EBCB8B',      # Yellow
    'danger': '#BF616A',       # Red
    'light': '#ECEFF4',        # Light gray
    'white': '#FFFFFF',
    'text': '#2E3440',
    'muted': '#4C566A'
}

# Load default data once when app starts
try:
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
    
    data_loaded = True
except Exception as e:
    print(f"Error loading data: {e}")
    data_loaded = False
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

app = dash.Dash(__name__)

# Enhanced CSS styles
app.layout = html.Div([
    
    # Header Section
    html.Div([
        html.Div([
            html.H1("🥚 Egg Sales Analytics Dashboard", 
                   className="dashboard-title"),
            html.P("Advanced forecasting and analysis platform", 
                   className="dashboard-subtitle")
        ], className="header-content")
    ], className="header-section"),
    
    # Main Container
    html.Div([
        # Control Panel
        html.Div([
            html.Div([
                html.H3("📊 Sales Analysis", className="section-title"),
                
                # Controls Row
                html.Div([
                    html.Div([
                        html.Label("Select Month:", className="control-label"),
                        dcc.Dropdown(
                            id='month-dropdown',
                            options=[{'label': calendar.month_name[m], 'value': m} for m in range(1, 13)] if data_loaded else [],
                            value=train_df['Month'].min() if data_loaded and not train_df.empty else 1,
                            clearable=False,
                            className="dropdown-style"
                        ),
                    ], className="control-group"),
                    
                    html.Div([
                        html.Label("Select Year:", className="control-label"),
                        dcc.Dropdown(
                            id='year-dropdown',
                            options=[{'label': int(y), 'value': int(y)} for y in sorted(train_df['Year'].dropna().unique())] if data_loaded and not train_df.empty else [],
                            value=int(train_df['Year'].min()) if data_loaded and not train_df.empty else 2021,
                            clearable=False,
                            className="dropdown-style"
                        ),
                    ], className="control-group"),
                    
                    html.Div([
                        html.Button(
                            [html.Span("📈", className="button-icon"), " Analyze Sales"], 
                            id='analyze-button', 
                            n_clicks=0,
                            className="primary-button"
                        )
                    ], className="button-group")
                ], className="controls-row")
            ], className="control-panel")
        ], className="panel-wrapper"),
        
        # Graph Container
        html.Div(id='graph-container', className="graph-section"),
        
        # Prediction Panel
        html.Div([
            html.Div([
                html.H3("🔮 Sales Prediction", className="section-title"),
                
                html.Div([
                    html.Div([
                        html.Label("Select Date for Prediction:", className="control-label"),
                        dcc.DatePickerSingle(
                            id='prediction-date-picker',
                            min_date_allowed=test_df['Date'].min().date() if data_loaded and not test_df.empty else date(2022, 1, 1),
                            max_date_allowed=test_df['Date'].max().date() if data_loaded and not test_df.empty else date(2022, 12, 31),
                            date=test_df['Date'].min().date() if data_loaded and not test_df.empty else date(2022, 1, 1),
                            display_format='DD/MM/YYYY',
                            className="date-picker-style"
                        ),
                    ], className="control-group"),
                    
                    html.Div([
                        html.Button(
                            [html.Span("🎯", className="button-icon"), " Get Prediction"], 
                            id='predict-button', 
                            n_clicks=0,
                            className="secondary-button"
                        )
                    ], className="button-group")
                ], className="controls-row"),
                
                # Prediction Output
                html.Div(id='prediction-output', className="prediction-output")
            ], className="control-panel")
        ], className="panel-wrapper"),
        
        # Statistics Cards (if data available)
        html.Div(id='stats-cards', className="stats-section") if data_loaded else html.Div()
        
    ], className="main-container"),
    
    # Footer
    html.Div([
        html.P("Built with Dash & Prophet | Enhanced UI Design", className="footer-text")
    ], className="footer")
    
], className="app-container", style={
    'fontFamily': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif'
})

# Enhanced callback for graph
@app.callback(
    [Output('graph-container', 'children'),
     Output('stats-cards', 'children')],
    Input('analyze-button', 'n_clicks'),
    [State('month-dropdown', 'value'),
     State('year-dropdown', 'value')],
    prevent_initial_call=True
)
def update_graph(n_clicks, month, year):
    if not data_loaded or train_df.empty:
        return [
            html.Div([
                html.Div([
                    html.H4("⚠️ Data Not Available"),
                    html.P("Please ensure train_egg_sales.csv is available in the directory.")
                ], className="error-message")
            ], className="graph-wrapper"),
            html.Div()
        ]
    
    filtered = train_df[(train_df['Month'] == month) & (train_df['Year'] == year)]
    
    if filtered.empty:
        return [
            html.Div([
                html.Div([
                    html.H4("📭 No Data Found"),
                    html.P(f"No sales data available for {calendar.month_name[month]} {year}")
                ], className="no-data-message")
            ], className="graph-wrapper"),
            html.Div()
        ]
    
    # Create enhanced graph
    fig = go.Figure()
    
    # Add main line
    fig.add_trace(go.Scatter(
        x=filtered['Date'],
        y=filtered['Egg Sales'],
        mode='lines+markers',
        line=dict(color=COLORS['accent'], width=3),
        marker=dict(
            size=8,
            color=COLORS['accent'],
            line=dict(width=2, color=COLORS['white'])
        ),
        name='Egg Sales',
        hovertemplate='<b>Date:</b> %{x}<br><b>Sales:</b> %{y:,.0f}<extra></extra>'
    ))
    
    # Add trend line if enough data points
    if len(filtered) > 5:
        z = np.polyfit(range(len(filtered)), filtered['Egg Sales'], 1)
        p = np.poly1d(z)
        fig.add_trace(go.Scatter(
            x=filtered['Date'],
            y=p(range(len(filtered))),
            mode='lines',
            line=dict(color=COLORS['warning'], width=2, dash='dash'),
            name='Trend Line',
            hovertemplate='<b>Trend:</b> %{y:,.0f}<extra></extra>'
        ))
    
    # Enhanced layout
    fig.update_layout(
        title=dict(
            text=f'Egg Sales - {calendar.month_name[month]} {year}',
            font=dict(size=20, color=COLORS['text'], family='Inter'),
            x=0.5
        ),
        margin=dict(l=60, r=40, t=80, b=60),
        plot_bgcolor=COLORS['white'],
        paper_bgcolor=COLORS['white'],
        font=dict(color=COLORS['text'], family='Inter'),
        xaxis=dict(
            title='Date',
            gridcolor=COLORS['light'],
            showgrid=True,
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title='Sales Volume',
            gridcolor=COLORS['light'],
            showgrid=True,
            tickfont=dict(size=12)
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    # Create statistics cards
    stats_data = filtered['Egg Sales']
    stats_cards = html.Div([
        html.Div([
            html.Div([
                html.H4(f"{stats_data.sum():,.0f}", className="stat-number"),
                html.P("Total Sales", className="stat-label")
            ], className="stat-card stat-primary"),
            
            html.Div([
                html.H4(f"{stats_data.mean():.0f}", className="stat-number"),
                html.P("Average Daily", className="stat-label")
            ], className="stat-card stat-success"),
            
            html.Div([
                html.H4(f"{stats_data.max():,.0f}", className="stat-number"),
                html.P("Peak Sales", className="stat-label")
            ], className="stat-card stat-warning"),
            
            html.Div([
                html.H4(f"{stats_data.std():.0f}", className="stat-number"),
                html.P("Std Deviation", className="stat-label")
            ], className="stat-card stat-info")
        ], className="stats-grid")
    ], className="stats-container")
    
    return [
        html.Div([
            dcc.Graph(figure=fig, className="main-graph")
        ], className="graph-wrapper"),
        stats_cards
    ]

# Enhanced prediction callback
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('prediction-date-picker', 'date'),
    prevent_initial_call=True
)
def show_prediction(n_clicks, selected_date):
    if not data_loaded or test_df.empty:
        return html.Div([
            html.Div([
                html.H4("⚠️ Prediction Unavailable"),
                html.P("Test data not loaded properly.")
            ], className="error-message")
        ])
    
    if selected_date:
        selected_date = pd.to_datetime(selected_date)
        row = test_df[test_df['Date'] == selected_date]
        
        if not row.empty and pd.notna(row.iloc[0]['Egg Sales Prediction']):
            pred_value = int(round(row.iloc[0]['Egg Sales Prediction']))
            pred_date = row.iloc[0]['Date'].strftime('%d %B %Y')
            
            return html.Div([
                html.Div([
                    html.H4("🎯 Prediction Result", className="result-title"),
                    html.Div([
                        html.Span("Date: ", className="result-label"),
                        html.Span(pred_date, className="result-date")
                    ]),
                    html.Div([
                        html.Span("Predicted Sales: ", className="result-label"),
                        html.Span(f"{pred_value:,}", className="result-value")
                    ])
                ], className="prediction-result success-result")
            ])
        else:
            return html.Div([
                html.Div([
                    html.H4("📅 Date Out of Range"),
                    html.P("Selected date is outside the prediction range or data unavailable.")
                ], className="warning-message")
            ])
    
    return html.Div()

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            .app-container {
                min-height: 100vh;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                font-family: 'Inter', sans-serif;
            }
            
            .header-section {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
                padding: 2rem 0;
                box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            }
            
            .header-content {
                max-width: 1200px;
                margin: 0 auto;
                text-align: center;
                padding: 0 2rem;
            }
            
            .dashboard-title {
                font-size: 2.5rem;
                font-weight: 700;
                color: #2E3440;
                margin-bottom: 0.5rem;
            }
            
            .dashboard-subtitle {
                font-size: 1.1rem;
                color: #4C566A;
                font-weight: 400;
            }
            
            .main-container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 2rem;
                gap: 2rem;
                display: flex;
                flex-direction: column;
            }
            
            .panel-wrapper {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 16px;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .control-panel {
                padding: 2rem;
            }
            
            .section-title {
                font-size: 1.5rem;
                font-weight: 600;
                color: #2E3440;
                margin-bottom: 1.5rem;
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            .controls-row {
                display: flex;
                gap: 2rem;
                align-items: end;
                flex-wrap: wrap;
            }
            
            .control-group {
                display: flex;
                flex-direction: column;
                gap: 0.5rem;
                min-width: 180px;
            }
            
            .control-label {
                font-weight: 500;
                color: #2E3440;
                font-size: 0.9rem;
            }
            
            .dropdown-style .Select-control {
                border: 2px solid #E5E9F0 !important;
                border-radius: 8px !important;
                min-height: 44px !important;
                font-family: 'Inter', sans-serif !important;
            }
            
            .dropdown-style .Select-control:hover {
                border-color: #5E81AC !important;
            }
            
            .primary-button, .secondary-button {
                padding: 12px 24px;
                border: none;
                border-radius: 8px;
                font-weight: 600;
                font-size: 0.9rem;
                cursor: pointer;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                gap: 0.5rem;
                font-family: 'Inter', sans-serif;
                min-height: 44px;
            }
            
            .primary-button {
                background: linear-gradient(135deg, #5E81AC, #81A1C1);
                color: white;
            }
            
            .secondary-button {
                background: linear-gradient(135deg, #A3BE8C, #B48EAD);
                color: white;
            }
            
            .primary-button:hover, .secondary-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 20px rgba(0,0,0,0.2);
            }
            
            .button-icon {
                font-size: 1.1rem;
            }
            
            .graph-wrapper {
                padding: 2rem;
                background: rgba(255, 255, 255, 0.95);
                border-radius: 16px;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .main-graph {
                border-radius: 12px;
                overflow: hidden;
            }
            
            .stats-section {
                margin-top: 1rem;
            }
            
            .stats-container {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 16px;
                padding: 2rem;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
            }
            
            .stat-card {
                padding: 1.5rem;
                border-radius: 12px;
                text-align: center;
                background: linear-gradient(135deg, #f8f9fa, #ffffff);
                border: 1px solid #E5E9F0;
                transition: all 0.3s ease;
            }
            
            .stat-card:hover {
                transform: translateY(-4px);
                box-shadow: 0 12px 24px rgba(0,0,0,0.1);
            }
            
            .stat-primary { border-left: 4px solid #5E81AC; }
            .stat-success { border-left: 4px solid #A3BE8C; }
            .stat-warning { border-left: 4px solid #EBCB8B; }
            .stat-info { border-left: 4px solid #B48EAD; }
            
            .stat-number {
                font-size: 2rem;
                font-weight: 700;
                color: #2E3440;
                margin-bottom: 0.5rem;
            }
            
            .stat-label {
                color: #4C566A;
                font-size: 0.9rem;
                font-weight: 500;
            }
            
            .prediction-output {
                margin-top: 1.5rem;
            }
            
            .prediction-result {
                padding: 1.5rem;
                border-radius: 12px;
                background: linear-gradient(135deg, #A3BE8C, #D8DEE9);
                border-left: 4px solid #A3BE8C;
            }
            
            .result-title {
                color: #2E3440;
                font-weight: 600;
                margin-bottom: 1rem;
            }
            
            .result-label {
                color: #4C566A;
                font-weight: 500;
            }
            
            .result-date, .result-value {
                color: #2E3440;
                font-weight: 600;
                font-size: 1.1rem;
            }
            
            .error-message, .warning-message, .no-data-message {
                padding: 2rem;
                text-align: center;
                border-radius: 12px;
                margin: 1rem 0;
            }
            
            .error-message {
                background: linear-gradient(135deg, #BF616A, #D8DEE9);
                border-left: 4px solid #BF616A;
            }
            
            .warning-message {
                background: linear-gradient(135deg, #EBCB8B, #D8DEE9);
                border-left: 4px solid #EBCB8B;
            }
            
            .no-data-message {
                background: linear-gradient(135deg, #4C566A, #D8DEE9);
                border-left: 4px solid #4C566A;
            }
            
            .footer {
                background: rgba(46, 52, 64, 0.9);
                color: #D8DEE9;
                text-align: center;
                padding: 1rem;
                margin-top: 2rem;
            }
            
            .footer-text {
                font-size: 0.9rem;
                font-weight: 400;
            }
            
            .date-picker-style .DateInput_input {
                border: 2px solid #E5E9F0 !important;
                border-radius: 8px !important;
                padding: 12px !important;
                font-family: 'Inter', sans-serif !important;
                min-height: 20px !important;
            }
            
            @media (max-width: 768px) {
                .controls-row {
                    flex-direction: column;
                    align-items: stretch;
                }
                
                .control-group {
                    min-width: auto;
                }
                
                .stats-grid {
                    grid-template-columns: 1fr 1fr;
                }
                
                .dashboard-title {
                    font-size: 2rem;
                }
                
                .main-container {
                    padding: 1rem;
                }
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True, port=8050)