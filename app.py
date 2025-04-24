#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# Load datasets
red_wine_df = pd.read_csv("winequality-red.csv", delimiter=";")
white_wine_df = pd.read_csv("winequality-white.csv", delimiter=";")

# Add wine type labels
red_wine_df['wine_type'] = 'Red Wine'
white_wine_df['wine_type'] = 'White Wine'
wine_df = pd.concat([red_wine_df, white_wine_df])

# Compute mean values for each feature (Treemap)
mean_red = red_wine_df.select_dtypes(include=['number']).mean().reset_index()
mean_red.columns = ["Feature", "Red Wine"]
mean_white = white_wine_df.select_dtypes(include=['number']).mean().reset_index()
mean_white.columns = ["Feature", "White Wine"]
mean_df = mean_red.merge(mean_white, on="Feature").melt(id_vars=["Feature"], var_name="Wine Type", value_name="Value")

# Compute correlations with quality (Correlation Line Chart)
corr_red = red_wine_df.select_dtypes(include=['number']).corr()['quality'].drop('quality')
corr_white = white_wine_df.select_dtypes(include=['number']).corr()['quality'].drop('quality')
corr_df = pd.DataFrame({"Feature": corr_red.index, "Red Wine": corr_red.values, "White Wine": corr_white.values})
corr_df = corr_df.melt(id_vars=["Feature"], var_name="Wine Type", value_name="Correlation")

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Wine Quality Analysis", style={'textAlign': 'center'}),

    # Treemap (Mean Feature Values)
    dcc.Graph(id='treemap'),

    # Correlation Line Chart (Feature Correlations)
    dcc.Graph(id='correlation-chart'),

      # Updated 3D Scatter Plot (Placed at the End)
    dcc.Graph(id='scatter-plot'),

     # Dropdown Selector for 3D Scatter Plot
    html.Div([
        html.Label("Select Wine Type for 3D Scatter Plot:", style={'fontSize': '18px', 'marginBottom': '10px'}),
        dcc.Dropdown(
            id='wine-dropdown',
            options=[
                {'label': 'Red Wine', 'value': 'Red Wine'},
                {'label': 'White Wine', 'value': 'White Wine'},
                {'label': 'Both Wines', 'value': 'Both'}
            ],
            value='Both',  # Default selection
            clearable=False,
            style={'width': '50%', 'marginBottom': '20px'}
        )
    ], style={'marginTop': '20px', 'textAlign': 'center'})

])

@app.callback(
    Output('treemap', 'figure'),
    Output('correlation-chart', 'figure'),
    Output('scatter-plot', 'figure'),
    Input('wine-dropdown', 'value')
)
def update_graphs(selected_wine):
    # Treemap: Mean Feature Comparison
    treemap_fig = px.treemap(mean_df, path=["Wine Type", "Feature"], values="Value",
                             color="Wine Type",
                             color_discrete_map={"Red Wine": "darkred", "White Wine": "gold"},
                             title="Treemap: Mean Values of Independent Variables for Red vs. White Wine",
                             hover_data=["Value"])
    treemap_fig.update_traces(texttemplate="%{label}: %{value:.2f}", textposition="middle center")

    # Correlation Line Chart: Feature Correlation Strength
    correlation_fig = px.line(corr_df, x="Feature", y="Correlation", color="Wine Type",
                              color_discrete_map={"Red Wine": "darkred", "White Wine": "gold"},
                              markers=True, line_shape="spline",
                              title="Correlation Strength of Physiochemical Properties with Wine Quality")

 # Highlight reference lines at 0, 0.5, and -0.5 correlation
    correlation_fig.add_shape(
        type="line",
        x0=corr_df["Feature"].min(), x1=corr_df["Feature"].max(),
        y0=0, y1=0,  # 0-axis reference line
        line=dict(color="black", width=2, dash="dash")
    )

    # Filter dataset based on dropdown selection
    if selected_wine == "Red Wine":
        filtered_df = red_wine_df
    elif selected_wine == "White Wine":
        filtered_df = white_wine_df
    else:
        filtered_df = wine_df  # Both wines

    # Updated 3D Scatter Plot
    scatter_fig = px.scatter_3d(
        filtered_df, x='alcohol', y='volatile acidity', z='quality',
        color='wine_type',
        color_discrete_map={"Red Wine": "darkred", "White Wine": "gold"},
        opacity=0.7,
        title=f"Wine Quality vs. Alcohol & Volatile Acidity ({selected_wine})",
        labels={"alcohol": "Alcohol Content", "volatile acidity": "Volatile Acidity", "quality": "Quality Score"}
    )

    return treemap_fig, correlation_fig, scatter_fig

# Run Dash app
if __name__ == '__main__':
    app.run (debug=True, port=8070)


# In[ ]:




