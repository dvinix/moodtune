"""
recommender/spotify.py
----------------------
Spotify music recommendation engine for MoodTune.

Responsibilities
----------------
1. Authenticate with Spotify via SpotifyClientCredentials (env-var driven).
2. Call sp.recommendations() with emotion-specific audio-feature targets from
   EMOTION_AUDIO_MAP.
3. Fetch real audio features (valence, energy, danceability) for each returned
   track via sp.audio_features().
4. Cache results for 10 minutes per (emotion, limit) pair to avoid hammering
   the Spotify API on every webcam frame update.
5. Fall back gracefully to 6 mock track dicts when credentials are absent or
   the API call fails — so the Streamlit UI always has something to render.

Environment variables (place in a .env file at project root)
------------------------------------------------------------
    SPOTIPY_CLIENT_ID      = <your Spotify app client ID>
    SPOTIPY_CLIENT_SECRET  = <your Spotify app client secret>
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Optional

from dotenv import load_dotenv

from recommender.emotion_map import (
    EMOTION_AUDIO_MAP,
    get_recommendation_params,
    all_emotions,
)

# Load .env before importing spotipy so credentials are in the environment
load_dotenv()

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Typing alias
# ---------------------------------------------------------------------------

TrackDict = dict  # see _make_track_dict() for the schema

# ---------------------------------------------------------------------------
# Cache configuration
# ---------------------------------------------------------------------------

_CACHE_TTL_SECONDS = 600          # 10 minutes
_cache: dict[tuple, tuple[float, list[TrackDict]]] = {}
#  key   → (emotion, limit)
#  value → (timestamp_fetched, list_of_track_dicts)


# ---------------------------------------------------------------------------
# Spotify client — lazy singleton
# ---------------------------------------------------------------------------

_sp = None          # spotipy.Spotify instance (None until first use)
_sp_available = None  # bool — True once credentials confirmed good


def _get_client():
    """
    Return a cached Spotipy client, or None if credentials are missing/invalid.
    Uses a module-level singleton so we don't re-authenticate on every call.
    """
    global _sp, _sp_available

    if _sp_available is True:
        return _sp
    if _sp_available is False:
        return None

    client_id     = os.getenv("SPOTIPY_CLIENT_ID",     "").strip()
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET", "").strip()

    if not client_id or not client_secret:
        log.warning(
            "SPOTIPY_CLIENT_ID / SPOTIPY_CLIENT_SECRET not set. "
            "Running in mock mode."
        )
        _sp_available = False
        return None

    try:
        import spotipy
        from spotipy.oauth2 import SpotifyClientCredentials

        auth_manager = SpotifyClientCredentials(
            client_id=client_id,
            client_secret=client_secret,
        )
        _sp = spotipy.Spotify(auth_manager=auth_manager)

        # Smoke-test — lightweight call to verify credentials
        _sp.search(q="test", limit=1, type="track")

        _sp_available = True
        log.info("Spotify client authenticated successfully.")
        return _sp

    except Exception as exc:  # noqa: BLE001
        log.error("Spotify auth failed: %s", exc)
        _sp_available = False
        return None


# ---------------------------------------------------------------------------
# Mock data — per-emotion, so the fallback is also mood-appropriate
# ---------------------------------------------------------------------------

_MOCK_TRACKS_BY_EMOTION: dict[str, list[TrackDict]] = {
    "happy": [
        {"name": "Happy", "artist": "Pharrell Williams", "album": "G I R L",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b273e8107e6d9214baa81bb79bba",
         "spotify_url": "https://open.spotify.com/track/60nZcImufyMA1MKQY3dcCH",
         "preview_url": "", "valence": 0.96, "energy": 0.82, "danceability": 0.66},
        {"name": "Can't Stop the Feeling!", "artist": "Justin Timberlake", "album": "Trolls",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b273d4e9bc25d53c04cd04c68943",
         "spotify_url": "https://open.spotify.com/track/6JV2fgDPiRBMQO9YbgF5m4",
         "preview_url": "", "valence": 0.93, "energy": 0.80, "danceability": 0.74},
        {"name": "Uptown Funk", "artist": "Mark Ronson ft. Bruno Mars", "album": "Uptown Special",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b273e419ccba0baa8bd3f3d7abf2",
         "spotify_url": "https://open.spotify.com/track/32OlwWuMpZ6b0aN2RZOeMS",
         "preview_url": "", "valence": 0.90, "energy": 0.87, "danceability": 0.85},
        {"name": "Good as Hell", "artist": "Lizzo", "album": "Cuz I Love You",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b273f2c9acad4dfb96f39bae98bb",
         "spotify_url": "https://open.spotify.com/track/7C6MeKKpqGnRFq5FbPdNOR",
         "preview_url": "", "valence": 0.89, "energy": 0.71, "danceability": 0.77},
        {"name": "Walking on Sunshine", "artist": "Katrina & The Waves", "album": "Walking on Sunshine",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b27352c8ebb36afe43617a0ef20d",
         "spotify_url": "https://open.spotify.com/track/05wIrZSwuaVWhcv5FfqeH0",
         "preview_url": "", "valence": 0.97, "energy": 0.85, "danceability": 0.67},
        {"name": "Don't Stop Me Now", "artist": "Queen", "album": "Jazz",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b2736861f94fe2bfac9d87ca99fe",
         "spotify_url": "https://open.spotify.com/track/7hQJA50XrCWABAu5v6QZ4i",
         "preview_url": "", "valence": 0.97, "energy": 0.86, "danceability": 0.59},
    ],
    "sad": [
        {"name": "Someone Like You", "artist": "Adele", "album": "21",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b2732118bf9b198b05a95ded6300",
         "spotify_url": "https://open.spotify.com/track/1zwMYTA5nlNjZxYrvBB2pV",
         "preview_url": "", "valence": 0.16, "energy": 0.32, "danceability": 0.24},
        {"name": "Fix You", "artist": "Coldplay", "album": "X&Y",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b273de3c04b5fc750b68899b20a9",
         "spotify_url": "https://open.spotify.com/track/7LVHVU3tWfcxj5aiPFEW4Q",
         "preview_url": "", "valence": 0.25, "energy": 0.41, "danceability": 0.32},
        {"name": "Skinny Love", "artist": "Bon Iver", "album": "For Emma, Forever Ago",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b273756d98576de36e40baa2a0fe",
         "spotify_url": "https://open.spotify.com/track/4sSrSBBEBhtDp35bFsMqLZ",
         "preview_url": "", "valence": 0.20, "energy": 0.26, "danceability": 0.37},
        {"name": "The Night We Met", "artist": "Lord Huron", "album": "Strange Trails",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b27317cccb40eb70b1e46dd9d018",
         "spotify_url": "https://open.spotify.com/track/3hRV0jL3zgFJB7hQzJUCRP",
         "preview_url": "", "valence": 0.28, "energy": 0.38, "danceability": 0.30},
        {"name": "Liability", "artist": "Lorde", "album": "Melodrama",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b273174d85f5f3ed4ac0aca49ef1",
         "spotify_url": "https://open.spotify.com/track/1UGD3lW3tDmgZfAVDh6w7r",
         "preview_url": "", "valence": 0.11, "energy": 0.27, "danceability": 0.28},
        {"name": "Hurt", "artist": "Johnny Cash", "album": "The Man Comes Around",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b273b46b7b4e3d74a4d21db4d8f4",
         "spotify_url": "https://open.spotify.com/track/28cngnOh9GuTCOoUwFCj5H",
         "preview_url": "", "valence": 0.05, "energy": 0.20, "danceability": 0.21},
    ],
    "angry": [
        {"name": "Break Stuff", "artist": "Limp Bizkit", "album": "Significant Other",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b273d3f7dfeaece6d5f35c5e0ed6",
         "spotify_url": "https://open.spotify.com/track/3gMaNLQm7D9MornNILzdSl",
         "preview_url": "", "valence": 0.28, "energy": 0.96, "danceability": 0.53},
        {"name": "Given Up", "artist": "Linkin Park", "album": "Minutes to Midnight",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b273b4bfee2c47be0a6abe9be8e7",
         "spotify_url": "https://open.spotify.com/track/2FsaKoCEMz4qNTiGHLaS8T",
         "preview_url": "", "valence": 0.24, "energy": 0.97, "danceability": 0.46},
        {"name": "Killing in the Name", "artist": "Rage Against the Machine", "album": "Rage Against the Machine",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b273254f23dc4a97e7fc53c8dc01",
         "spotify_url": "https://open.spotify.com/track/59WN2psjkt1tyaxjspN8fp",
         "preview_url": "", "valence": 0.37, "energy": 0.98, "danceability": 0.47},
        {"name": "Chop Suey!", "artist": "System of a Down", "album": "Toxicity",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b2737e3c5b0ea7bcb09e9dc4256d",
         "spotify_url": "https://open.spotify.com/track/2DlHlPMa4M17kufBvI2lEN",
         "preview_url": "", "valence": 0.31, "energy": 0.97, "danceability": 0.40},
        {"name": "Lose Yourself", "artist": "Eminem", "album": "8 Mile",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b2736ca5c90113b30c3c43ffb8f4",
         "spotify_url": "https://open.spotify.com/track/5Z01UMMf7V1o0MzF86s6WJ",
         "preview_url": "", "valence": 0.25, "energy": 0.93, "danceability": 0.52},
        {"name": "Master of Puppets", "artist": "Metallica", "album": "Master of Puppets",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b27346ce47e3bcd887d3e75cb4e5",
         "spotify_url": "https://open.spotify.com/track/4bz63rFGCqBxnkl7MNmBce",
         "preview_url": "", "valence": 0.23, "energy": 0.98, "danceability": 0.39},
    ],
    "neutral": [
        {"name": "Blinding Lights", "artist": "The Weeknd", "album": "After Hours",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b2738863bc11d2aa12b54f5aeb21",
         "spotify_url": "https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b",
         "preview_url": "", "valence": 0.33, "energy": 0.80, "danceability": 0.51},
        {"name": "Levitating", "artist": "Dua Lipa", "album": "Future Nostalgia",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b2734bc66095f8a70bc4e6593f4f",
         "spotify_url": "https://open.spotify.com/track/463CkQjx2Zk1yXoBuierM9",
         "preview_url": "", "valence": 0.82, "energy": 0.81, "danceability": 0.70},
        {"name": "drivers license", "artist": "Olivia Rodrigo", "album": "SOUR",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b2737f53a91efe7a6a5b8b4adeaa",
         "spotify_url": "https://open.spotify.com/track/5wANPM4fQCJwkGd4rN57mH",
         "preview_url": "", "valence": 0.29, "energy": 0.43, "danceability": 0.59},
        {"name": "Golden Hour", "artist": "JVKE", "album": "this is what ____ feels like",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b273f35b6a3dd8459c90dc86bc01",
         "spotify_url": "https://open.spotify.com/track/5odlY52u43F5BjByhxg7wg",
         "preview_url": "", "valence": 0.71, "energy": 0.53, "danceability": 0.61},
        {"name": "Starboy", "artist": "The Weeknd ft. Daft Punk", "album": "Starboy",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b2734718e2b124f79258be7bc452",
         "spotify_url": "https://open.spotify.com/track/5aAx2yezTd8zXrkmtKl66Z",
         "preview_url": "", "valence": 0.49, "energy": 0.59, "danceability": 0.68},
        {"name": "Watermelon Sugar", "artist": "Harry Styles", "album": "Fine Line",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b2732e8ed79e177ff6011076f5f0",
         "spotify_url": "https://open.spotify.com/track/6UelLqGlWMcVH1E5c4H7lY",
         "preview_url": "", "valence": 0.56, "energy": 0.82, "danceability": 0.55},
    ],
    "fear": [
        {"name": "Weightless", "artist": "Marconi Union", "album": "Weightless",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b27344b40b4b99b9b88c02d20b8d",
         "spotify_url": "https://open.spotify.com/track/7ygpwy2qP3NbrxVkHvIhqP",
         "preview_url": "", "valence": 0.06, "energy": 0.07, "danceability": 0.18},
        {"name": "Burn", "artist": "Nine Inch Nails", "album": "Further Down the Spiral",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b273a8c93e254ea4e11f66efe38d",
         "spotify_url": "https://open.spotify.com/track/2JHFMD7tHmXFRPUKKw3cFf",
         "preview_url": "", "valence": 0.10, "energy": 0.72, "danceability": 0.29},
        {"name": "Creep", "artist": "Radiohead", "album": "Pablo Honey",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b27307021d1506d1fc60e46a93b4",
         "spotify_url": "https://open.spotify.com/track/70LcF31zb1H0PyJoS1Sx1r",
         "preview_url": "", "valence": 0.10, "energy": 0.44, "danceability": 0.51},
        {"name": "In the Air Tonight", "artist": "Phil Collins", "album": "Face Value",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b27381b0be4b9f6c0b96bd51e5e0",
         "spotify_url": "https://open.spotify.com/track/6FDttCCaGRmFkrVcS99uYL",
         "preview_url": "", "valence": 0.18, "energy": 0.55, "danceability": 0.32},
        {"name": "Black", "artist": "Pearl Jam", "album": "Ten",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b2733e83e0a36a7caef2e5b1b01e",
         "spotify_url": "https://open.spotify.com/track/4xcMnRVFV2JZYSr4B7OLzd",
         "preview_url": "", "valence": 0.14, "energy": 0.41, "danceability": 0.35},
        {"name": "Mad World", "artist": "Gary Jules", "album": "Trading Snakeoil for Wolftickets",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b27335bdb5b1edf3cbdef7a28d3f",
         "spotify_url": "https://open.spotify.com/track/3JOVTQ5h8HyvI8kka4nF1d",
         "preview_url": "", "valence": 0.04, "energy": 0.23, "danceability": 0.25},
    ],
    "disgust": [
        {"name": "Come As You Are", "artist": "Nirvana", "album": "Nevermind",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b273fbc71c99f9a1296c56dd51be",
         "spotify_url": "https://open.spotify.com/track/0MHT7EMudhMI4aSABjKwBT",
         "preview_url": "", "valence": 0.37, "energy": 0.67, "danceability": 0.59},
        {"name": "Black Hole Sun", "artist": "Soundgarden", "album": "Superunknown",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b27374f024eb95fb20e4a41ff50b",
         "spotify_url": "https://open.spotify.com/track/5VoZJMFOqmAVTFgZcZb0bh",
         "preview_url": "", "valence": 0.27, "energy": 0.63, "danceability": 0.40},
        {"name": "Lithium", "artist": "Nirvana", "album": "Nevermind",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b273fbc71c99f9a1296c56dd51be",
         "spotify_url": "https://open.spotify.com/track/2RlgNHKcydI9sayD2Df2xp",
         "preview_url": "", "valence": 0.52, "energy": 0.84, "danceability": 0.47},
        {"name": "Smells Like Teen Spirit", "artist": "Nirvana", "album": "Nevermind",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b273fbc71c99f9a1296c56dd51be",
         "spotify_url": "https://open.spotify.com/track/5ghIJDpPoe3CfHMGu71E6T",
         "preview_url": "", "valence": 0.42, "energy": 0.92, "danceability": 0.50},
        {"name": "Sabotage", "artist": "Beastie Boys", "album": "Ill Communication",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b2730a8f2f9d8c6fd93be0ff01b8",
         "spotify_url": "https://open.spotify.com/track/3HkWMrqmO5vAV8eqJRiHEk",
         "preview_url": "", "valence": 0.53, "energy": 0.97, "danceability": 0.58},
        {"name": "When I Come Around", "artist": "Green Day", "album": "Dookie",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b273e72643edd8e15e3c4cb77abe",
         "spotify_url": "https://open.spotify.com/track/5cI00q2JeKOPFHEJEqWKYD",
         "preview_url": "", "valence": 0.51, "energy": 0.84, "danceability": 0.60},
    ],
    "surprise": [
        {"name": "Superstition", "artist": "Stevie Wonder", "album": "Talking Book",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b273463f3e5b04b1e3c7e29efbc8",
         "spotify_url": "https://open.spotify.com/track/1h0UNCschHMOAqXGNKGypk",
         "preview_url": "", "valence": 0.83, "energy": 0.85, "danceability": 0.79},
        {"name": "Get Lucky", "artist": "Daft Punk ft. Pharrell Williams", "album": "Random Access Memories",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b273b33d46dfa2f5c3a37ab2c ede",
         "spotify_url": "https://open.spotify.com/track/69bp2EbF7Q2rqc5N3ylezZ",
         "preview_url": "", "valence": 0.79, "energy": 0.79, "danceability": 0.83},
        {"name": "September", "artist": "Earth, Wind & Fire", "album": "The Best of Earth Wind & Fire Vol. 1",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b273b8db7a1c8ef37fc0b03df38e",
         "spotify_url": "https://open.spotify.com/track/2grjqo0Frpf2okIBiifQKs",
         "preview_url": "", "valence": 0.98, "energy": 0.79, "danceability": 0.77},
        {"name": "Treasure", "artist": "Bruno Mars", "album": "Unorthodox Jukebox",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b27396ce26c5c98a16e959cf38c7",
         "spotify_url": "https://open.spotify.com/track/55h7vJchibLdUkxdlX3fK7",
         "preview_url": "", "valence": 0.94, "energy": 0.81, "danceability": 0.85},
        {"name": "Shake It Off", "artist": "Taylor Swift", "album": "1989",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b273f6b58e5b4c1d1d7dd45a4f48",
         "spotify_url": "https://open.spotify.com/track/0cqRj7pUJDkTCEsJkx8snD",
         "preview_url": "", "valence": 0.94, "energy": 0.80, "danceability": 0.65},
        {"name": "Jump Around", "artist": "House of Pain", "album": "House of Pain",
         "album_art_url": "https://i.scdn.co/image/ab67616d0000b273b7a18e7bdb2b67c5d42dbc02",
         "spotify_url": "https://open.spotify.com/track/4fBqiEnruLTMJSZujzYeF1",
         "preview_url": "", "valence": 0.76, "energy": 0.94, "danceability": 0.79},
    ],
}


def _get_mock_tracks(emotion: str = "neutral", limit: int = 6) -> list[TrackDict]:
    """Return mood-appropriate mock tracks. Falls back to neutral if key missing."""
    tracks = _MOCK_TRACKS_BY_EMOTION.get(emotion, _MOCK_TRACKS_BY_EMOTION["neutral"])
    cycled = tracks * ((limit // len(tracks)) + 1)
    return cycled[:limit]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_track_dict(
    track_item: dict,
    audio_feat: Optional[dict],
) -> TrackDict:
    """
    Flatten a Spotify track object + its audio features into our schema.

    Parameters
    ----------
    track_item  : one element of ``sp.recommendations()["tracks"]``
    audio_feat  : one element of ``sp.audio_features([ids])`` (may be None)

    Returns
    -------
    dict with keys: name, artist, album, album_art_url, spotify_url,
                    preview_url, valence, energy, danceability
    """
    artists = ", ".join(a["name"] for a in track_item.get("artists", []))

    album     = track_item.get("album", {})
    album_name = album.get("name", "Unknown Album")

    images = album.get("images", [])
    # Prefer the 300×300 thumbnail (index 1); fall back to largest (0)
    art_url = ""
    if images:
        art_url = images[1]["url"] if len(images) > 1 else images[0]["url"]

    ext_urls    = track_item.get("external_urls", {})
    spotify_url = ext_urls.get("spotify", "")
    preview_url = track_item.get("preview_url") or ""

    valence      = 0.0
    energy       = 0.0
    danceability = 0.0
    if audio_feat:
        valence      = float(audio_feat.get("valence",      0.0))
        energy       = float(audio_feat.get("energy",       0.0))
        danceability = float(audio_feat.get("danceability", 0.0))

    return {
        "name":          track_item.get("name", "Unknown Track"),
        "artist":        artists,
        "album":         album_name,
        "album_art_url": art_url,
        "spotify_url":   spotify_url,
        "preview_url":   preview_url,
        "valence":       valence,
        "energy":        energy,
        "danceability":  danceability,
    }


# ---------------------------------------------------------------------------
# Emotion → search query variants
# ---------------------------------------------------------------------------
# Each emotion has multiple query strings. A random one is chosen every call
# so the user always gets a fresh, varied batch of songs.
# sp.recommendations() is restricted for new Spotify apps (deprecated Nov 2024).

_EMOTION_QUERY_VARIANTS: dict[str, list[str]] = {
    "happy": [
        'genre:pop mood:happy year:2020-2024',
        'genre:dance-pop mood:euphoric year:2018-2024',
        'genre:indie-pop genre:feel-good year:2015-2024',
        'genre:funk genre:soul upbeat year:2010-2024',
        'genre:k-pop mood:cheerful year:2019-2024',
    ],
    "sad": [
        'genre:singer-songwriter mood:melancholy year:2018-2024',
        'genre:indie mood:heartbreak year:2015-2024',
        'genre:acoustic mood:sad year:2010-2024',
        'genre:alt-rock mood:lonely year:2005-2020',
        'genre:piano mood:emotional year:2012-2024',
    ],
    "angry": [
        'genre:metal genre:hardcore year:2015-2024',
        'genre:hard-rock mood:intense year:2000-2020',
        'genre:punk-rock mood:aggressive year:1995-2015',
        'genre:nu-metal genre:rap-rock year:1998-2010',
        'genre:industrial genre:heavy year:2010-2024',
    ],
    "neutral": [
        'genre:indie-pop genre:chill year:2020-2024',
        'genre:lo-fi genre:chillhop year:2018-2024',
        'genre:alternative mood:relaxed year:2019-2024',
        'genre:synth-pop mood:dreamy year:2015-2024',
        'genre:bedroom-pop genre:mellow year:2019-2024',
    ],
    "fear": [
        'genre:dark-ambient mood:tense year:2015-2024',
        'genre:atmospheric genre:cinematic dark year:2010-2024',
        'genre:post-rock mood:eerie year:2012-2022',
        'genre:gothic mood:suspense year:2005-2020',
        'genre:electronic mood:haunting year:2010-2024',
    ],
    "disgust": [
        'genre:grunge genre:alternative year:1990-2010',
        'genre:noise-rock genre:experimental year:2000-2020',
        'genre:post-punk mood:dark year:1980-2015',
        'genre:alternative-rock mood:raw year:1990-2005',
        'genre:indie-rock genre:abrasive year:2005-2020',
    ],
    "surprise": [
        'genre:funk genre:disco mood:energetic year:2015-2024',
        'genre:electro-swing genre:upbeat year:2010-2024',
        'genre:jazz-funk mood:playful year:2012-2024',
        'genre:afrobeats mood:festive year:2018-2024',
        'genre:brazilian genre:tropical mood:vibrant year:2015-2024',
    ],
}


def _fetch_from_spotify(emotion: str, limit: int) -> list[TrackDict]:
    """
    Search Spotify for tracks matching the emotion using a randomly selected
    query variant. Fetches 50 results then shuffles and slices `limit` tracks
    so every call to the same emotion yields a different set of songs.
    """
    sp = _get_client()
    if sp is None:
        raise RuntimeError("No Spotify client available.")

    # Pick a random query variant for this emotion
    variants = _EMOTION_QUERY_VARIANTS.get(emotion, ['genre:pop'])
    query = random.choice(variants)
    log.info("Spotify search | emotion='%s' | query=%r", emotion, query)

    # Always fetch 50 so we have a large pool to shuffle from
    search_result = sp.search(q=query, type="track", limit=50)
    items = search_result.get("tracks", {}).get("items", [])

    if not items:
        log.warning("Spotify search returned 0 tracks for emotion='%s'.", emotion)
        return []

    # Deduplicate by track ID
    seen_ids: set = set()
    unique_tracks = []
    for t in items:
        tid = t.get("id")
        if tid and tid not in seen_ids:
            seen_ids.add(tid)
            unique_tracks.append(t)

    # Shuffle so we don't always return the top-ranked songs
    random.shuffle(unique_tracks)
    unique_tracks = unique_tracks[:limit]

    # Batch-fetch audio features (single API call)
    track_ids   = [t["id"] for t in unique_tracks if t.get("id")]
    audio_feats = sp.audio_features(track_ids) or []
    feat_by_id  = {f["id"]: f for f in audio_feats if f and f.get("id")}

    result: list[TrackDict] = []
    for track_item in unique_tracks:
        tid  = track_item.get("id")
        feat = feat_by_id.get(tid)
        result.append(_make_track_dict(track_item, feat))

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_recommendations(
    emotion: str,
    limit: int = 6,
) -> list[TrackDict]:
    """
    Return ``limit`` fresh track dicts tuned to ``emotion``.

    Every call hits the live Spotify search API with a randomly chosen query
    variant and returns a shuffled subset, so the user always sees different
    songs when the same emotion is detected again.

    Fallback
    --------
    If Spotify credentials are absent or the API call raises, mood-appropriate
    mock tracks are returned so the UI always has something to render.

    Parameters
    ----------
    emotion : str
        One of the 7 canonical emotion labels (case-insensitive).
    limit   : int
        Number of tracks to return (1–20).

    Returns
    -------
    list[dict]  — each dict has keys:
        name, artist, album, album_art_url, spotify_url, preview_url,
        valence, energy, danceability
    """
    emotion = emotion.lower().strip()

    if emotion not in all_emotions():
        raise ValueError(
            f"Unknown emotion: '{emotion}'. "
            f"Valid options: {all_emotions()}"
        )

    limit = max(1, min(limit, 20))

    # ── Live Spotify fetch (always fresh — no cache) ──────────────────────
    try:
        tracks = _fetch_from_spotify(emotion, limit)
        if tracks:
            log.info(
                "Fetched %d fresh tracks from Spotify for emotion='%s'.",
                len(tracks), emotion,
            )
            return tracks
        log.warning("Spotify returned no tracks; using mock fallback.")
    except Exception as exc:  # noqa: BLE001
        log.warning("Spotify API error (%s); using mock fallback.", exc)

    # ── Mock fallback — mood-appropriate & shuffled ───────────────────────
    pool = list(_MOCK_TRACKS_BY_EMOTION.get(emotion, _MOCK_TRACKS_BY_EMOTION["neutral"]))
    random.shuffle(pool)
    return pool[:limit]


def clear_cache() -> None:
    """Manually invalidate the entire recommendation cache."""
    _cache.clear()
    log.info("Recommendation cache cleared.")


def cache_info() -> dict:
    """
    Return a summary of the current cache state.

    Returns
    -------
    dict with keys:
        entries : int   — number of cached (emotion, limit) pairs
        keys    : list  — the cached keys
        ages_s  : dict  — age in seconds for each key
    """
    now = time.monotonic()
    return {
        "entries": len(_cache),
        "keys":    list(_cache.keys()),
        "ages_s":  {
            str(k): round(now - ts, 1)
            for k, (ts, _) in _cache.items()
        },
    }


def is_spotify_available() -> bool:
    """
    Return True if Spotify credentials are present and the client authenticated.
    Triggers lazy authentication on first call.
    """
    return _get_client() is not None


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json
    import sys

    logging.basicConfig(level=logging.INFO)

    target_emotion = sys.argv[1] if len(sys.argv) > 1 else "happy"
    print(f"\nFetching recommendations for emotion: {target_emotion!r}\n")

    tracks = get_recommendations(target_emotion, limit=6)

    for i, t in enumerate(tracks, 1):
        print(f"  {i}. {t['name']} — {t['artist']}")
        print(f"     valence={t['valence']:.2f}  energy={t['energy']:.2f}  "
              f"danceability={t['danceability']:.2f}")
        print(f"     {t['spotify_url']}")
        print()

    print(f"Cache info: {cache_info()}")
    print(f"Spotify available: {is_spotify_available()}")
