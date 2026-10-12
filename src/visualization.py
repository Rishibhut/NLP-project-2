"""
Visualization module for Hinglish ABSA and Cross-Lingual Evaluation.
Builds interactive Plotly charts and publication-grade Matplotlib/Seaborn figures.
"""

from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


def plot_model_comparison(df_models: pd.DataFrame) -> go.Figure:
    """
    Creates a grouped bar chart comparing Macro-F1 and Accuracy across models.
    """
    fig = go.Figure()
    models = df_models["Model"].tolist()

    if "Macro-F1" in df_models.columns:
        fig.add_trace(go.Bar(
            name="Macro-F1",
            x=models,
            y=df_models["Macro-F1"],
            text=[f"{v:.3f}" if pd.notna(v) else "N/A" for v in df_models["Macro-F1"]],
            textposition="auto",
            marker_color="#2b5c8f",
        ))

    if "Accuracy" in df_models.columns:
        fig.add_trace(go.Bar(
            name="Accuracy",
            x=models,
            y=df_models["Accuracy"],
            text=[f"{v:.3f}" if pd.notna(v) else "N/A" for v in df_models["Accuracy"]],
            textposition="auto",
            marker_color="#3caea3",
        ))

    fig.update_layout(
        title="Model Performance Comparison (Held-out Evaluation)",
        xaxis_title="Model Architecture",
        yaxis_title="Score (0.0 - 1.0)",
        yaxis=dict(range=[0, 1.05]),
        barmode="group",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_same_vs_cross_f1(df_gap: pd.DataFrame) -> go.Figure:
    """
    Side-by-side comparison of Same-Language F1 vs Cross-Lingual F1.
    """
    fig = go.Figure()
    models = df_gap["Model"].tolist()

    fig.add_trace(go.Bar(
        name="Same-Language Pairs",
        x=models,
        y=df_gap["Same-Language F1"],
        text=[f"{v:.3f}" if pd.notna(v) else "N/A" for v in df_gap["Same-Language F1"]],
        textposition="auto",
        marker_color="#173f5f",
    ))

    fig.add_trace(go.Bar(
        name="Cross-Lingual Pairs",
        x=models,
        y=df_gap["Cross-Lingual F1"],
        text=[f"{v:.3f}" if pd.notna(v) else "N/A" for v in df_gap["Cross-Lingual F1"]],
        textposition="auto",
        marker_color="#ed553b",
    ))

    fig.update_layout(
        title="Same-Language vs. Cross-Lingual Aspect-Opinion F1 Score",
        xaxis_title="Model Architecture",
        yaxis_title="Macro-F1 Score",
        yaxis=dict(range=[0, 1.05]),
        barmode="group",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_alignment_gap(df_gap: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart showing the Alignment Gap in percentage points.
    Positive value indicates degraded performance on cross-lingual pairs.
    """
    gaps = df_gap["Alignment Gap (pp)"].tolist()
    models = df_gap["Model"].tolist()
    colors = ["#ed553b" if (pd.notna(g) and g > 0) else "#3caea3" for g in gaps]

    fig = go.Figure(go.Bar(
        x=gaps,
        y=models,
        orientation="h",
        text=[f"+{g:.2f} pp" if (pd.notna(g) and g > 0) else f"{g:.2f} pp" if pd.notna(g) else "N/A" for g in gaps],
        textposition="auto",
        marker_color=colors,
    ))

    fig.update_layout(
        title="Cross-Lingual Aspect–Opinion Alignment Gap (Δ F1 in % Points)",
        xaxis_title="Alignment Gap (F1_same - F1_cross) in Percentage Points",
        yaxis_title="Model",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    fig.add_vline(x=0, line_width=1.5, line_dash="dash", line_color="black")
    return fig


def plot_normalization_impact(df_norm: pd.DataFrame) -> go.Figure:
    """
    Bar chart comparing original vs. normalized text performance.
    """
    fig = go.Figure()
    models = df_norm["Model"].tolist()

    fig.add_trace(go.Bar(
        name="Original Text (Cond A)",
        x=models,
        y=df_norm["Original F1"],
        text=[f"{v:.3f}" if pd.notna(v) else "N/A" for v in df_norm["Original F1"]],
        textposition="auto",
        marker_color="#6c757d",
    ))

    fig.add_trace(go.Bar(
        name="Normalized Text (Cond B)",
        x=models,
        y=df_norm["Normalized F1"],
        text=[f"{v:.3f}" if pd.notna(v) else "N/A" for v in df_norm["Normalized F1"]],
        textposition="auto",
        marker_color="#28a745",
    ))

    fig.update_layout(
        title="Effect of Transliteration Normalization on Aspect Sentiment F1",
        xaxis_title="Model Architecture",
        yaxis_title="Macro-F1 Score",
        yaxis=dict(range=[0, 1.05]),
        barmode="group",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig


def plot_sentiment_distribution(df: pd.DataFrame) -> go.Figure:
    """Distribution of sentiment labels in the dataset."""
    counts = df["sentiment"].value_counts().reset_index()
    counts.columns = ["Sentiment", "Count"]
    color_map = {"Positive": "#28a745", "Negative": "#dc3545", "Neutral": "#ffc107"}

    fig = px.bar(
        counts,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        color_discrete_map=color_map,
        text="Count",
        title="Sentiment Distribution Across Aspect Instances",
        template="plotly_white",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig


def plot_language_pair_distribution(df: pd.DataFrame) -> go.Figure:
    """Distribution of aspect-opinion language combinations."""
    combos = (df["aspect_language"].astype(str) + " (Aspect) → " + df["opinion_language"].astype(str) + " (Opinion)").value_counts().reset_index()
    combos.columns = ["Language Combination", "Count"]

    fig = px.bar(
        combos,
        x="Count",
        y="Language Combination",
        orientation="h",
        text="Count",
        title="Language Association Distribution (Aspect → Opinion)",
        template="plotly_white",
        color="Count",
        color_continuous_scale="Teal",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig


def plot_error_categories(category_counts: Dict[str, int]) -> go.Figure:
    """Bar chart breakdown of diagnosed error types."""
    df_err = pd.DataFrame(list(category_counts.items()), columns=["Error Category", "Count"])
    df_err = df_err.sort_values(by="Count", ascending=True)

    fig = px.bar(
        df_err,
        x="Count",
        y="Error Category",
        orientation="h",
        text="Count",
        title="Diagnosed Error Categories Breakdown",
        template="plotly_white",
        color="Count",
        color_continuous_scale="Reds",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(margin=dict(l=40, r=40, t=60, b=40))
    return fig


def plot_confusion_matrix_figure(cm: List[List[int]], labels: List[str]) -> go.Figure:
    """Interactive heatmap confusion matrix."""
    z = np.array(cm)
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=labels,
        y=labels,
        hoverongaps=False,
        colorscale="Blues",
        text=z,
        texttemplate="%{text}",
        textfont={"size": 14},
    ))
    fig.update_layout(
        title="Aspect Sentiment Confusion Matrix",
        xaxis_title="Predicted Polarity",
        yaxis_title="True Polarity",
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40),
    )
    return fig
