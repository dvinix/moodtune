"""
components/album_cards.py
-------------------------
Renders a 3-column grid of Spotify track recommendation cards.
"""

from __future__ import annotations

import streamlit as st

def _render_track_card(track: dict) -> None:
    """
    Renders a single track card using HTML/CSS for strict layout control,
    wrapping it in a Streamlit container.
    """
    # Fallback to grey square if no album art
    img_url = track.get("album_art_url")
    if not img_url:
        img_url = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='150' height='150'><rect width='100%' height='100%' fill='gray'/></svg>"

    name    = track.get("name", "Unknown Track")
    artist  = track.get("artist", "Unknown Artist")
    val     = track.get("valence", 0.0)
    nrg     = track.get("energy", 0.0)
    dnc     = track.get("danceability", 0.0)
    url     = track.get("spotify_url", "#")

    # Construct the HTML structure
    # Using simple inline styles to guarantee it looks like a "card"
    html = f"""
    <div style="
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        height: 100%;
    ">
        <img src="{img_url}" alt="Album Art" style="width: 150px; height: 150px; border-radius: 8px; object-fit: cover; margin-bottom: 15px;">
        
        <div style="flex-grow: 1;">
            <h4 style="margin: 0 0 5px 0; font-size: 1.1rem; color: #ffffff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 200px;" title="{name}">{name}</h4>
            <p style="margin: 0 0 15px 0; font-size: 0.9rem; color: #b3b3b3; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 200px;" title="{artist}">{artist}</p>
            
            <div style="display: flex; justify-content: center; gap: 8px; margin-bottom: 20px;">
                <span style="background-color: rgba(255, 215, 0, 0.2); color: gold; padding: 3px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: bold;">Val: {val:.2f}</span>
                <span style="background-color: rgba(255, 69, 0, 0.2); color: #ff4500; padding: 3px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: bold;">Nrg: {nrg:.2f}</span>
                <span style="background-color: rgba(30, 144, 255, 0.2); color: #1e90ff; padding: 3px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: bold;">Dnc: {dnc:.2f}</span>
            </div>
        </div>
        
        <a href="{url}" target="_blank" style="
            display: inline-block;
            background-color: #1DB954;
            color: white;
            text-decoration: none;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: bold;
            width: 100%;
            transition: transform 0.2s;
        ">
            ▶ Open in Spotify
        </a>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def render_album_cards(tracks: list[dict]) -> None:
    """
    Render exactly 6 track cards in a 3-column layout.
    
    Parameters
    ----------
    tracks : list[dict]
        List of track dictionaries as returned by spotify.get_recommendations().
    """
    # Ensure we only try to render up to 6 tracks to maintain the 2x3 grid
    display_tracks = tracks[:6]
    
    if not display_tracks:
        st.info("No tracks available to display.")
        return

    # Create rows of 3 columns
    for i in range(0, len(display_tracks), 3):
        cols = st.columns(3)
        chunk = display_tracks[i:i+3]
        
        for col, track in zip(cols, chunk):
            with col:
                _render_track_card(track)
