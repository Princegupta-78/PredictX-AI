import plotly.graph_objects as go
import plotly.express as px

def create_health_gauge(score):
    """Creates a Plotly gauge chart for the engine health score."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Engine Health %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#1f77b4"},
                "steps": [
                    {"range": [0, 30], "color": "#ef4444"},   # Red (Danger)
                    {"range": [30, 60], "color": "#f59e0b"},  # Warning (Orange)
                    {"range": [60, 100], "color": "#10b981"}  # Green (Healthy)
                ]
            }
        )
    )
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def create_shap_bar_chart(exp_df):
    """Creates a horizontal bar chart of SHAP values."""
    # Sort for better visualization (largest impact at the top)
    exp_df = exp_df.sort_values(by="SHAP Value", key=abs, ascending=True)
    
    fig = px.bar(
        exp_df, 
        x="SHAP Value", 
        y="Feature", 
        orientation="h",
        title="Top Feature Contributions (SHAP)",
        color="SHAP Value",
        color_continuous_scale=px.colors.diverging.RdBu
    )
    fig.update_layout(height=400)
    return fig