"""
recommender/emotion_map.py
--------------------------
Maps each of the 7 detected emotion classes to:
  - Spotify audio feature targets  (valence, energy, danceability, tempo, acousticness)
  - 2–3 Spotify seed genres        (used for recommendations API call)

Audio Feature Reference
-----------------------
valence      : 0.0 = very negative/sad mood  →  1.0 = very positive/happy mood
energy       : 0.0 = calm/slow               →  1.0 = loud/fast/intense
danceability : 0.0 = not danceable           →  1.0 = highly danceable
tempo        : beats per minute (BPM)
acousticness : 0.0 = electric/synthesised    →  1.0 = purely acoustic

Design rationale
----------------
Values are grounded in Spotify's own audio-feature distributions for each
mood cluster (per Spotify developer documentation and published music-emotion
research such as Thayer's 2D valence–arousal model).

Each emotion gets a "tolerance" dict as well, which defines the ± range the
recommender uses when filtering tracks from the Spotify catalog — wider for
emotions like "neutral" where many tracks qualify, narrower for emotional
extremes to preserve character.
"""

from __future__ import annotations

from typing import TypedDict


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------

class AudioFeatureTarget(TypedDict):
    valence:         float
    energy:          float
    danceability:    float
    tempo:           float
    acousticness:    float


class AudioFeatureTolerance(TypedDict):
    valence:         float
    energy:          float
    danceability:    float
    tempo:           float
    acousticness:    float


class EmotionProfile(TypedDict):
    target:      AudioFeatureTarget
    tolerance:   AudioFeatureTolerance
    seed_genres: list[str]
    description: str          # human-readable rationale


# ---------------------------------------------------------------------------
# Core mapping
# ---------------------------------------------------------------------------

EMOTION_AUDIO_MAP: dict[str, EmotionProfile] = {

    # ── HAPPY ──────────────────────────────────────────────────────────────
    # High arousal + high valence → upbeat, energetic, danceable pop/dance.
    "happy": {
        "target": {
            "valence":      0.85,
            "energy":       0.80,
            "danceability": 0.75,
            "tempo":        128.0,
            "acousticness": 0.15,
        },
        "tolerance": {
            "valence":      0.15,
            "energy":       0.15,
            "danceability": 0.15,
            "tempo":        20.0,
            "acousticness": 0.20,
        },
        "seed_genres": ["pop", "dance", "happy"],
        "description": (
            "Bright, danceable tracks with high valence and punchy energy. "
            "Think Top-40 pop, upbeat funk, or EDM drops."
        ),
    },

    # ── SAD ────────────────────────────────────────────────────────────────
    # Low arousal + low valence → slow, introspective, often acoustic.
    "sad": {
        "target": {
            "valence":      0.20,
            "energy":       0.30,
            "danceability": 0.25,
            "tempo":        72.0,
            "acousticness": 0.70,
        },
        "tolerance": {
            "valence":      0.15,
            "energy":       0.15,
            "danceability": 0.15,
            "tempo":        18.0,
            "acousticness": 0.25,
        },
        "seed_genres": ["sad", "singer-songwriter", "indie"],
        "description": (
            "Soft, melancholic tracks with low energy and high acousticness. "
            "Slow ballads, indie folk, and minimal piano works."
        ),
    },

    # ── ANGRY ──────────────────────────────────────────────────────────────
    # High arousal + low valence → aggressive, intense, fast-tempo.
    "angry": {
        "target": {
            "valence":      0.30,
            "energy":       0.90,
            "danceability": 0.55,
            "tempo":        160.0,
            "acousticness": 0.05,
        },
        "tolerance": {
            "valence":      0.20,
            "energy":       0.10,
            "danceability": 0.20,
            "tempo":        25.0,
            "acousticness": 0.10,
        },
        "seed_genres": ["metal", "hard-rock", "punk"],
        "description": (
            "Intense, loud, fast tracks — very high energy, low valence. "
            "Heavy metal, hardcore punk, aggressive hip-hop."
        ),
    },

    # ── NEUTRAL ────────────────────────────────────────────────────────────
    # Mid arousal + mid valence → relaxed, versatile background listening.
    "neutral": {
        "target": {
            "valence":      0.50,
            "energy":       0.50,
            "danceability": 0.50,
            "tempo":        110.0,
            "acousticness": 0.40,
        },
        "tolerance": {
            "valence":      0.25,
            "energy":       0.25,
            "danceability": 0.25,
            "tempo":        30.0,
            "acousticness": 0.35,
        },
        "seed_genres": ["indie-pop", "chill", "alternative"],
        "description": (
            "Balanced, mid-tempo tracks suitable for focused work or casual "
            "listening. Wide tolerance to capture diverse styles."
        ),
    },

    # ── FEAR ───────────────────────────────────────────────────────────────
    # High arousal + very low valence → tense, dark, atmospheric.
    "fear": {
        "target": {
            "valence":      0.15,
            "energy":       0.70,
            "danceability": 0.30,
            "tempo":        140.0,
            "acousticness": 0.10,
        },
        "tolerance": {
            "valence":      0.15,
            "energy":       0.20,
            "danceability": 0.20,
            "tempo":        25.0,
            "acousticness": 0.15,
        },
        "seed_genres": ["ambient", "dark-ambient", "industrial"],
        "description": (
            "Tense, dark, atmospheric tracks with very low valence and "
            "high energy. Cinematic scores, dark electronic, industrial."
        ),
    },

    # ── DISGUST ────────────────────────────────────────────────────────────
    # Moderate arousal + low valence → cynical, edgy, mid-tempo.
    "disgust": {
        "target": {
            "valence":      0.25,
            "energy":       0.55,
            "danceability": 0.40,
            "tempo":        105.0,
            "acousticness": 0.20,
        },
        "tolerance": {
            "valence":      0.15,
            "energy":       0.20,
            "danceability": 0.20,
            "tempo":        20.0,
            "acousticness": 0.20,
        },
        "seed_genres": ["grunge", "alternative", "rock"],
        "description": (
            "Edgy, mid-tempo tracks with low valence. Captures the raw, "
            "cynical quality of grunge and alternative rock."
        ),
    },

    # ── SURPRISE ───────────────────────────────────────────────────────────
    # High arousal + moderately high valence → energetic, unexpected, fun.
    "surprise": {
        "target": {
            "valence":      0.70,
            "energy":       0.75,
            "danceability": 0.70,
            "tempo":        125.0,
            "acousticness": 0.15,
        },
        "tolerance": {
            "valence":      0.20,
            "energy":       0.20,
            "danceability": 0.20,
            "tempo":        20.0,
            "acousticness": 0.20,
        },
        "seed_genres": ["pop", "electro", "funk"],
        "description": (
            "Upbeat, groove-heavy tracks with an element of playful energy. "
            "Funky pop, electro, feel-good dance music."
        ),
    },
}


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def get_target(emotion: str) -> AudioFeatureTarget:
    """Return the audio feature targets for a given emotion."""
    profile = EMOTION_AUDIO_MAP.get(emotion.lower())
    if profile is None:
        raise KeyError(
            f"Unknown emotion: '{emotion}'. "
            f"Valid options: {list(EMOTION_AUDIO_MAP)}"
        )
    return profile["target"]


def get_tolerance(emotion: str) -> AudioFeatureTolerance:
    """Return the per-feature tolerance ranges for a given emotion."""
    profile = EMOTION_AUDIO_MAP.get(emotion.lower())
    if profile is None:
        raise KeyError(f"Unknown emotion: '{emotion}'.")
    return profile["tolerance"]


def get_seed_genres(emotion: str) -> list[str]:
    """Return the Spotify seed genres for a given emotion."""
    profile = EMOTION_AUDIO_MAP.get(emotion.lower())
    if profile is None:
        raise KeyError(f"Unknown emotion: '{emotion}'.")
    return profile["seed_genres"]


def get_recommendation_params(emotion: str) -> dict:
    """
    Build the full kwargs dict ready to pass to Spotipy's
    ``recommendations()`` call.

    Returns
    -------
    dict with keys:
        seed_genres          : list[str]
        target_valence       : float
        target_energy        : float
        target_danceability  : float
        target_tempo         : float
        target_acousticness  : float
        min_valence          : float
        max_valence          : float
        min_energy           : float
        max_energy           : float
        min_danceability     : float
        max_danceability     : float
        min_tempo            : float
        max_tempo            : float
        min_acousticness     : float
        max_acousticness     : float
    """
    profile   = EMOTION_AUDIO_MAP.get(emotion.lower())
    if profile is None:
        raise KeyError(f"Unknown emotion: '{emotion}'.")

    target    = profile["target"]
    tolerance = profile["tolerance"]
    genres    = profile["seed_genres"]

    params: dict = {"seed_genres": genres}

    for feature in ("valence", "energy", "danceability", "acousticness"):
        t   = target[feature]       # type: ignore[literal-required]
        tol = tolerance[feature]    # type: ignore[literal-required]
        params[f"target_{feature}"] = t
        params[f"min_{feature}"]    = max(0.0, t - tol)
        params[f"max_{feature}"]    = min(1.0, t + tol)

    # Tempo: unbounded above 0
    t_tempo   = target["tempo"]
    tol_tempo = tolerance["tempo"]
    params["target_tempo"] = t_tempo
    params["min_tempo"]    = max(40.0, t_tempo - tol_tempo)
    params["max_tempo"]    = t_tempo + tol_tempo

    return params


def all_emotions() -> list[str]:
    """Return the list of all supported emotion labels."""
    return list(EMOTION_AUDIO_MAP.keys())


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import json

    for emotion in all_emotions():
        params = get_recommendation_params(emotion)
        print(f"\n{'-' * 56}")
        print(f"  {emotion.upper()}")
        print(f"{'-' * 56}")
        print(json.dumps(params, indent=4))
