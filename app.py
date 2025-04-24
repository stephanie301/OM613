import streamlit as st
import pandas as pd
import plotly.express as px

# Load datasets
red_wine_df = pd.read_csv("winequality-red.csv", delimiter=";")
white_wine_df = pd.read_csv("winequality-white.csv", delimiter=";")

# Add wine type labels
red_wine_df['wine_type'] = 'Red Wine'
white_wine_df['wine_type'] = 'White Wine'
wine_df = pd.concat([red_wine_df, white_wine_df])

# Compute mean values for each feature
mean_red = red_wine_df.select_dtypes(include=['number']).mean().reset_index()
mean_red.columns = ["Feature", "Red Wine"]
mean_white = white_wine_df.select_dtypes(include=['number']).mean().reset_index()
mean_white.columns = ["Feature", "White Wine"]
mean_df = mean_red.merge(mean_white, on="Feature").melt(id_vars=["Feature"], var_name="Wine Type", value_name="Value")

# Compute correlations with quality
corr_red = red_wine_df.select_dtypes(include=['number']).corr()['quality'].drop('quality')
corr_white = white_wine_df.select_dtypes(include=['number']).corr()['quality'].drop('quality')
corr_df = pd.DataFrame({"Feature": corr_red.index, "Red Wine": corr_red.values, "White Wine": corr_white.values})
corr_df = corr_df.melt(id_vars=["Feature"], var_name="Wine Type", value_name="Correlation")

st.title("Wine Quality Analysis")

# Generate Treemap Chart
treemap_fig = px.treemap(mean_df, path=["Wine Type", "Feature"], values="Value",
                         color="Wine Type",
                         color_discrete_map={"Red Wine": "darkred", "White Wine": "gold"},
                         title="Mean Values of Independent Variables for Red vs. White Wine")
st.plotly_chart(treemap_fig)

# Generate Correlation Line Chart
correlation_fig = px.line(corr_df, x="Feature", y="Correlation", color="Wine Type",
                          color_discrete_map={"Red Wine": "darkred", "White Wine": "gold"},
                          markers=True, line_shape="spline",
                          title="Correlation Strength of Physiochemical Properties with Wine Quality")
correlation_fig.add_shape(
    type="line",
    x0=corr_df["Feature"].min(), x1=corr_df["Feature"].max(),
    y0=0, y1=0,  # Horizontal line at y=0
    line=dict(color="black", width=2, dash="dash")
)
st.plotly_chart(correlation_fig)

# Streamlit Dashboard Layout
wine_selection = st.selectbox("Select Wine Type:", ["Red Wine", "White Wine", "Both Wines"])

# Generate 3D Scatter Plot
filtered_df = red_wine_df if wine_selection == "Red Wine" else white_wine_df if wine_selection == "White Wine" else wine_df
scatter_fig = px.scatter_3d(filtered_df, x='alcohol', y='volatile acidity', z='quality',
                            color='wine_type',
                            color_discrete_map={"Red Wine": "darkred", "White Wine": "gold"},
                            opacity=0.7,
                            title=f"Wine Quality vs. Alcohol & Volatile Acidity ({wine_selection})")
st.plotly_chart(scatter_fig)





