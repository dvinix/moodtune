"""
components/mood_timeline.py
---------------------------
Builds a Plotly line+scatter chart to display the user's emotion history over time.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from components.radar_chart import EMOTION_COLORS


def render_mood_timeline(history: list[dict]) -> go.Figure:
    """
    Construct a Plotly timeline chart from a list of history records.

    Parameters
    ----------
    history : list[dict]
        List of dictionaries, each containing:
            "timestamp"  : datetime object
            "emotion"    : str (e.g., "happy")
            "confidence" : float [0, 1]

    Returns
    -------
    plotly.graph_objects.Figure
        The constructed timeline figure (not yet rendered to Streamlit).
    """
    if not history:
        # Return an empty figure with a placeholder message if no history
        fig = go.Figure()
        fig.update_layout(
            title="Awaiting Emotion Data...",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
        )
        return fig

    # Convert to DataFrame for easier plotting with Plotly Express
    df = pd.DataFrame(history)
    
    # Create the base line chart
    fig = px.line(
        df,
        x="timestamp",
        y="confidence",
        title="Mood History",
        markers=True,
    )

    # We want to color the individual scatter points based on the emotion.
    # px.line doesn't easily let us color markers independently of the line
    # when keeping a single connected line. So we update the traces manually.

    # 1. Style the connecting line (make it neutral and semi-transparent)
    fig.update_traces(
        line=dict(color="rgba(255, 255, 255, 0.3)", width=2),
        marker=dict(size=0)  # Hide the default markers
    )

    # 2. Add custom colored scatter points over the line
    marker_colors = [EMOTION_COLORS.get(em.lower(), "gray") for em in df["emotion"]]
    
    # Create custom hover text
    hover_text = [
        f"Time: {ts.strftime('%H:%M:%S')}<br>Emotion: {em.capitalize()}<br>Confidence: {conf:.0%}"
        for ts, em, conf in zip(df["timestamp"], df["emotion"], df["confidence"])
    ]

    fig.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["confidence"],
        mode="markers",
        marker=dict(
            color=marker_colors,
            size=10,
            line=dict(color="white", width=1),
        ),
        text=hover_text,
        hoverinfo="text",
        showlegend=False,
    ))

    # 3. Layout tweaks
    fig.update_layout(
        xaxis=dict(
            title="",
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.1)",
            type="date",
            tickformat="%H:%M:%S",
        ),
        yaxis=dict(
            title="Confidence",
            range=[0, 1.05],
            showgrid=True,
            gridcolor="rgba(255, 255, 255, 0.1)",
            tickformat=".0%",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        margin=dict(l=40, r=40, t=40, b=20),
        hovermode="x unified",
        uirevision="constant",  # prevents zoom reset on redraw during live feed
    )

    return fig
