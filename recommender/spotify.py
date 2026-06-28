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
# Mock data — used when Spotify credentials are unavailable
# ---------------------------------------------------------------------------

_MOCK_TRACKS: list[TrackDict] = [
    {
        "name":          "Blinding Lights",
        "artist":        "The Weeknd",
        "album":         "After Hours",
        "album_art_url": "https://i.scdn.co/image/ab67616d0000b2738863bc11d2aa12b54f5aeb21",
        "spotify_url":   "https://open.spotify.com/track/0VjIjW4GlUZAMYd2vXMi3b",
        "preview_url":   "https://p.scdn.co/mp3-preview/sample1",
        "valence":       0.33,
        "energy":        0.80,
        "danceability":  0.51,
    },
    {
        "name":          "Happy",
        "artist":        "Pharrell Williams",
        "album":         "G I R L",
        "album_art_url": "https://i.scdn.co/image/ab67616d0000b273e8107e6d9214baa81bb79bba",
        "spotify_url":   "https://open.spotify.com/track/60nZcImufyMA1MKQY3dcCH",
        "preview_url":   "https://p.scdn.co/mp3-preview/sample2",
        "valence":       0.96,
        "energy":        0.82,
        "danceability":  0.66,
    },
    {
        "name":          "Lose Yourself",
        "artist":        "Eminem",
        "album":         "8 Mile",
        "album_art_url": "https://i.scdn.co/image/ab67616d0000b2736ca5c90113b30c3c43ffb8f4",
        "spotify_url":   "https://open.spotify.com/track/5Z01UMMf7V1o0MzF86s6WJ",
        "preview_url":   "https://p.scdn.co/mp3-preview/sample3",
        "valence":       0.25,
        "energy":        0.93,
        "danceability":  0.52,
    },
    {
        "name":          "Someone Like You",
        "artist":        "Adele",
        "album":         "21",
        "album_art_url": "https://i.scdn.co/image/ab67616d0000b2732118bf9b198b05a95ded6300",
        "spotify_url":   "https://open.spotify.com/track/1zwMYTA5nlNjZxYrvBB2pV",
        "preview_url":   "https://p.scdn.co/mp3-preview/sample4",
        "valence":       0.16,
        "energy":        0.32,
        "danceability":  0.24,
    },
    {
        "name":          "Weightless",
        "artist":        "Marconi Union",
        "album":         "Weightless",
        "album_art_url": "https://i.scdn.co/image/ab67616d0000b27344b40b4b99b9b88c02d20b8d",
        "spotify_url":   "https://open.spotify.com/track/7ygpwy2qP3NbrxVkHvIhqP",
        "preview_url":   "https://p.scdn.co/mp3-preview/sample5",
        "valence":       0.06,
        "energy":        0.07,
        "danceability":  0.18,
    },
    {
        "name":          "Eye of the Tiger",
        "artist":        "Survivor",
        "album":         "Eye of the Tiger",
        "album_art_url": "https://i.scdn.co/image/ab67616d0000b273f5b0e9b0b6c0ee8f2fb0b0db",
        "spotify_url":   "https://open.spotify.com/track/2HHtWyy5CgaQbC7XSoOb0e",
        "preview_url":   "https://p.scdn.co/mp3-preview/sample6",
        "valence":       0.54,
        "energy":        0.97,
        "danceability":  0.61,
    },
]


def _get_mock_tracks(limit: int = 6) -> list[TrackDict]:
    """Return up to ``limit`` mock tracks (cycles if limit > 6)."""
    mocks = _MOCK_TRACKS * ((limit // len(_MOCK_TRACKS)) + 1)
    return mocks[:limit]


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


def _fetch_from_spotify(emotion: str, limit: int) -> list[TrackDict]:
    """
    Hit the Spotify Recommendations + Audio Features APIs and return a list
    of track dicts.  Raises on any Spotify error so the caller can fall back.
    """
    sp = _get_client()
    if sp is None:
        raise RuntimeError("No Spotify client available.")

    params = get_recommendation_params(emotion)
    params["limit"] = max(limit, 1)

    rec_response = sp.recommendations(**params)
    tracks = rec_response.get("tracks", [])

    if not tracks:
        log.warning("Spotify returned 0 tracks for emotion='%s'.", emotion)
        return []

    # Batch-fetch audio features (single API call for all track IDs)
    track_ids   = [t["id"] for t in tracks if t.get("id")]
    audio_feats = sp.audio_features(track_ids) or []  # list[dict | None]
    feat_by_id  = {
        f["id"]: f
        for f in audio_feats
        if f and f.get("id")
    }

    result: list[TrackDict] = []
    for track_item in tracks:
        tid  = track_item.get("id")
        feat = feat_by_id.get(tid)
        result.append(_make_track_dict(track_item, feat))

    return result[:limit]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_recommendations(
    emotion: str,
    limit: int = 6,
) -> list[TrackDict]:
    """
    Return ``limit`` track dicts tuned to ``emotion``.

    Caching
    -------
    Results are cached for 10 minutes per ``(emotion, limit)`` key.
    Call ``clear_cache()`` to invalidate manually (e.g. on user request).

    Fallback
    --------
    If Spotify credentials are absent or the API call raises, mock tracks are
    returned so the UI continues to render.

    Parameters
    ----------
    emotion : str
        One of the 7 canonical emotion labels (case-insensitive).
    limit   : int
        Number of tracks to return (1–20; Spotify max is 100 but we cap at 20).

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

    limit = max(1, min(limit, 20))   # clamp to [1, 20]

    # ── Cache lookup ──────────────────────────────────────────────────────
    cache_key = (emotion, limit)
    if cache_key in _cache:
        ts, cached_tracks = _cache[cache_key]
        age = time.monotonic() - ts
        if age < _CACHE_TTL_SECONDS:
            log.debug(
                "Cache hit for emotion='%s' limit=%d (age=%.0fs).",
                emotion, limit, age,
            )
            return cached_tracks
        else:
            log.debug("Cache expired for key=%s.", cache_key)
            del _cache[cache_key]

    # ── Live Spotify fetch ────────────────────────────────────────────────
    try:
        tracks = _fetch_from_spotify(emotion, limit)
        if tracks:
            _cache[cache_key] = (time.monotonic(), tracks)
            log.info(
                "Fetched %d tracks from Spotify for emotion='%s'.",
                len(tracks), emotion,
            )
            return tracks
        # API returned empty list — fall through to mock
        log.warning("Spotify returned no tracks; using mock fallback.")
    except Exception as exc:  # noqa: BLE001
        log.warning("Spotify API error (%s); using mock fallback.", exc)

    # ── Mock fallback ─────────────────────────────────────────────────────
    return _get_mock_tracks(limit)


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
