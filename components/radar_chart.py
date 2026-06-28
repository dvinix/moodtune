"""
components/radar_chart.py
-------------------------
Builds a Plotly filled radar/spider chart for the 7-class emotion distribution.
"""

from __future__ import annotations

import plotly.graph_objects as go


# Shared color palette across all UI components
EMOTION_COLORS: dict[str, str] = {
    "happy":    "gold",
    "sad":      "blue",
    "angry":    "red",
    "neutral":  "gray",
    "fear":     "purple",
    "disgust":  "green",
    "surprise": "orange",
}


def build_radar_chart(emotion_probs: dict[str, float]) -> go.Figure:
    """
    Construct a Plotly radar chart from a dictionary of emotion probabilities.

    Parameters
    ----------
    emotion_probs : dict
        Mapping of emotion string to float [0, 1] confidence.

    Returns
    -------
    plotly.graph_objects.Figure
        The constructed radar chart (not yet rendered to Streamlit).
    """
    # Ensure consistent order of axes for the radar chart
    categories = ["happy", "surprise", "fear", "angry", "disgust", "sad", "neutral"]
    
    # Handle missing keys gracefully just in case
    values = [emotion_probs.get(c, 0.0) for c in categories]

    # Radar charts in Plotly need to be closed by repeating the first value
    plot_categories = categories + [categories[0]]
    plot_values     = values + [values[0]]

    # Determine dominant emotion for fill color
    dominant_emotion = max(emotion_probs, key=emotion_probs.get) if emotion_probs else "neutral"
    fill_color = EMOTION_COLORS.get(dominant_emotion.lower(), "gray")

    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=plot_values,
        theta=[c.capitalize() for c in plot_categories],
        fill='toself',
        name='Confidence',
        line=dict(color=fill_color, width=2),
        fillcolor=fill_color,
        opacity=0.6,
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],       # Probabilities are strictly 0 to 1
                showticklabels=False,
                gridcolor="rgba(255, 255, 255, 0.2)",
            ),
            angularaxis=dict(
                gridcolor="rgba(255, 255, 255, 0.2)",
                direction="clockwise",
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(color="white"),
        # Built-in smooth transitions for Streamlit auto-refresh
        uirevision="constant",  # prevents zoom reset on redraw
    )
    
    # Adding animation properties for smooth frame transitions
    fig.layout.updatemenus = [dict(
        type="buttons",
        showactive=False,
        buttons=[dict(
            label="Play",
            method="animate",
            args=[None, {"frame": {"duration": 300, "redraw": True}, "transition": {"duration": 300, "easing": "quadratic-in-out"}}]
        )]
    )]

    return fig
