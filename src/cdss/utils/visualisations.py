"""CDSS data visualisation functions."""
import plotly.graph_objects as go
import numpy as np

def create_uncertainty_gauge(uncertainty):
    '''Create a gauge chart for uncertainty visualization'''
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=uncertainty * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Model Uncertainty (%)"},
        gauge={
            'axis': {'range': [None, 10]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 3], 'color': "lightgreen"},
                {'range': [3, 6], 'color': "yellow"},
                {'range': [6, 10], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 5
            }
        }
    ))
    fig.update_layout(height=250)
    return fig

def create_probability_gauge(probability):
    '''Create a gauge chart for CKD probability'''
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "CKD Probability (%)"},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkred"},
            'steps': [
                {'range': [0, 20], 'color': "lightgreen"},
                {'range': [20, 40], 'color': "lightyellow"},
                {'range': [40, 60], 'color': "yellow"},
                {'range': [60, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    fig.update_layout(height=250)
    return fig

def create_feature_importance_chart(shap_values, feature_names, feature_values):
    '''Create horizontal bar chart showing SHAP feature contributions'''
    # Get top 5 features by absolute SHAP value
    abs_shap = np.abs(shap_values)
    top_indices = np.argsort(abs_shap)[-5:][::-1]

    top_features = [feature_names[i] for i in top_indices]
    top_shap = [shap_values[i] for i in top_indices]
    top_values = [feature_values[i] for i in top_indices]

    colors = ['red' if s > 0 else 'blue' for s in top_shap]

    fig = go.Figure(go.Bar(
        y=top_features,
        x=top_shap,
        orientation='h',
        marker=dict(color=colors),
        text=[f"Value: {v:.2f}" for v in top_values],
        textposition='auto'
    ))

    fig.update_layout(
        title="Top Contributing Features (SHAP Values)",
        xaxis_title="Impact on Prediction (positive = increases CKD risk)",
        yaxis_title="Feature",
        height=300,
        showlegend=False
    )

    return fig
