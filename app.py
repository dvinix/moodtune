"""
app.py
------
Main Streamlit application for MoodTune AI.
Ties together emotion detection, Spotify recommendations, and Plotly UI components.
"""

from __future__ import annotations

import time
from datetime import datetime
from collections import Counter

import cv2
import streamlit as st

from emotion.detector import EmotionDetector
from recommender.spotify import get_recommendations, is_spotify_available
from components.radar_chart import build_radar_chart
from components.album_cards import render_album_cards
from components.mood_timeline import render_mood_timeline


# ---------------------------------------------------------------------------
# Setup & Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    layout="wide",
    page_title="MoodTune AI",
    page_icon="🎵",
    initial_sidebar_state="expanded"
)


def init_session_state() -> None:
    """Initialize required session state variables."""
    defaults = {
        "webcam_active": False,
        "mood_history": [],       # list of dict: timestamp, emotion, confidence
        "last_emotion": None,
        "last_fetch_time": 0.0,
        "cached_tracks": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session_state()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🎵 MoodTune AI")
    st.markdown("*Real-time emotion detection to curate your perfect Spotify vibe.*")
    
    st.divider()
    
    # Webcam Control
    if st.session_state.webcam_active:
        if st.button("⏹ Stop Webcam", use_container_width=True, type="secondary"):
            st.session_state.webcam_active = False
            st.rerun()
    else:
        if st.button("📷 Start Webcam", use_container_width=True, type="primary"):
            st.session_state.webcam_active = True
            st.rerun()

    st.divider()

    # Threshold slider
    conf_thresh = st.slider(
        "Confidence Threshold",
        min_value=0.50,
        max_value=0.95,
        value=0.65,
        step=0.05,
        help="Minimum ViT confidence required to register a mood change."
    )

    st.divider()

    # Status indicators
    st.subheader("System Status")
    
    spotify_ok = is_spotify_available()
    if spotify_ok:
        st.markdown("🟢 **Spotify API:** Connected")
    else:
        st.markdown("🔴 **Spotify API:** Mock Mode (Missing `.env`)")

    st.markdown("🟢 **Model:** `vit-face-expression`")
    
    st.divider()

    if st.button("🗑 Clear History", use_container_width=True):
        st.session_state.mood_history = []
        st.session_state.last_emotion = None
        st.session_state.cached_tracks = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main Panel Layout Placeholders
# ---------------------------------------------------------------------------

# We create static containers/placeholders so we can inject content into them 
# from inside the webcam while-loop without appending infinitely to the page.

st.header("Live Analysis")

# Section A: Live Feed + Radar
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    feed_placeholder = st.empty()
    st.markdown("<br>", unsafe_allow_html=True)
    metric_placeholder = st.empty()
    progress_placeholder = st.empty()

with col_right:
    radar_placeholder = st.empty()

st.divider()

# Section B & C placeholders
music_header_placeholder = st.empty()
music_cards_placeholder = st.empty()

st.divider()

timeline_header_placeholder = st.empty()
timeline_chart_placeholder = st.empty()
stats_placeholder = st.empty()


# ---------------------------------------------------------------------------
# Static / Initial Render 
# (Fills placeholders when webcam is OFF or on first load)
# ---------------------------------------------------------------------------

def render_static_ui():
    """Populate UI from session state when the camera isn't actively looping."""
    if not st.session_state.webcam_active:
        feed_placeholder.info("📷 Webcam is currently stopped. Click 'Start Webcam' in the sidebar.")
        metric_placeholder.metric("Current Mood", "—", delta=None)
        progress_placeholder.progress(0, text="Confidence: 0%")
        
        # Empty radar
        radar_placeholder.plotly_chart(build_radar_chart({}), width="stretch")

    # Music Section
    if st.session_state.last_emotion and st.session_state.cached_tracks:
        music_header_placeholder.subheader(f"🎧 Songs for your {st.session_state.last_emotion.capitalize()} mood")
        with music_cards_placeholder.container():
            render_album_cards(st.session_state.cached_tracks)
    else:
        music_header_placeholder.subheader("🎧 Waiting for mood detection...")

    # Timeline Section
    timeline_header_placeholder.subheader("📈 Mood History")
    timeline_chart_placeholder.plotly_chart(
        render_mood_timeline(st.session_state.mood_history), 
        width="stretch"
    )
    
    # Stats
    if st.session_state.mood_history:
        emotions = [h["emotion"] for h in st.session_state.mood_history]
        most_common = Counter(emotions).most_common(1)[0][0]
        total_detections = len(st.session_state.mood_history)
        
        with stats_placeholder.container():
            sc1, sc2 = st.columns(2)
            sc1.metric("Most Frequent Mood Today", most_common.capitalize())
            sc2.metric("Total Detections", total_detections)


render_static_ui()


# ---------------------------------------------------------------------------
# Active Webcam Loop
# ---------------------------------------------------------------------------

if st.session_state.webcam_active:
    
    detector = EmotionDetector(confidence_threshold=conf_thresh)
    
    try:
        # We use a while loop to keep grabbing frames as fast as Streamlit allows
        loop_idx = 0
        while st.session_state.webcam_active:
            loop_idx += 1
            
            frame = detector.capture_frame()
            if frame is None:
                st.error("Failed to capture from webcam. Is it in use by another app?")
                st.session_state.webcam_active = False
                break
                
            # 1. Run inference
            result = detector.detect_from_frame(frame)
            
            # 2. Annotate & Display frame (convert BGR to RGB for Streamlit)
            annotated = detector.annotate_frame(frame, result)
            rgb_frame = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            feed_placeholder.image(rgb_frame, channels="RGB", width="stretch")
            
            # 3. Update Radar
            radar_fig = build_radar_chart(result["distribution"])
            radar_placeholder.plotly_chart(radar_fig, width="stretch", key=f"radar_live_{loop_idx}")

            # 4. Handle Valid Detections
            if result["face_detected"] and result["above_threshold"]:
                current_emotion = result["emotion"]
                confidence      = result["confidence"]
                
                # Update Metrics
                metric_placeholder.metric(
                    "Current Mood", 
                    f"{current_emotion.capitalize()}", 
                    delta=f"{confidence:.1%} confidence"
                )
                progress_placeholder.progress(
                    int(confidence * 100), 
                    text=f"Confidence: {confidence:.1%}"
                )
                
                # Update History
                st.session_state.mood_history.append({
                    "timestamp": datetime.now(),
                    "emotion": current_emotion,
                    "confidence": confidence
                })
                # Cap history at 50 to prevent memory blowup and lag
                if len(st.session_state.mood_history) > 50:
                    st.session_state.mood_history.pop(0)

                # Re-render Timeline
                timeline_fig = render_mood_timeline(st.session_state.mood_history)
                timeline_chart_placeholder.plotly_chart(timeline_fig, width="stretch", key=f"timeline_live_{loop_idx}")
                
                # Update Stats
                emotions = [h["emotion"] for h in st.session_state.mood_history]
                most_common = Counter(emotions).most_common(1)[0][0]
                with stats_placeholder.container():
                    sc1, sc2 = st.columns(2)
                    sc1.metric("Most Frequent Mood Today", most_common.capitalize())
                    sc2.metric("Total Detections", len(st.session_state.mood_history))

                # Handle Music Recommendation trigger
                if current_emotion != st.session_state.last_emotion:
                    st.session_state.last_emotion = current_emotion
                    
                    music_header_placeholder.subheader(f"🎧 Songs for your {current_emotion.capitalize()} mood")
                    
                    with music_cards_placeholder.container():
                        with st.spinner(f"Finding your {current_emotion} vibe..."):
                            tracks = get_recommendations(current_emotion, limit=6)
                            st.session_state.cached_tracks = tracks
                            render_album_cards(tracks)

            # Throttle loop to ~5 FPS to keep UI responsive and prevent browser crash
            time.sleep(0.2)

    finally:
        # Always release the camera when the loop breaks
        detector.release()

