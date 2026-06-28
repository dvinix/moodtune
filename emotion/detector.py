"""
emotion/detector.py
-------------------
Real-time emotion detection pipeline using:
  - Vision Transformer (ViT)  : trpakov/vit-face-expression (HuggingFace)
  - OpenCV                    : webcam capture + Haar-cascade face detection
  - transformers pipeline     : image-classification interface

Design goals:
  - Inference latency < 180 ms per frame (measured with time.perf_counter())
  - Confidence threshold: 0.65 — predictions below this are discarded
  - Returns top emotion label, confidence, and full 7-class distribution
"""

from __future__ import annotations

import time
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from transformers import pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "trpakov/vit-face-expression"

EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

CONFIDENCE_THRESHOLD = 0.65
LATENCY_BUDGET_MS    = 180.0   # milliseconds

# HuggingFace label aliases → canonical class names
_LABEL_MAP: dict[str, str] = {
    "angry":    "angry",
    "disgust":  "disgust",
    "fear":     "fear",
    "happy":    "happy",
    "neutral":  "neutral",
    "sad":      "sad",
    "surprise": "surprise",
    # Some checkpoints use 0-6 integer labels — guard against that
    "0": "angry",
    "1": "disgust",
    "2": "fear",
    "3": "happy",
    "4": "neutral",
    "5": "sad",
    "6": "surprise",
}

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper — locate Haar cascade XML
# ---------------------------------------------------------------------------

def _find_cascade() -> str:
    """
    Resolve the path to haarcascade_frontalface_default.xml.
    Tries the OpenCV data directory first, then the current file's directory.
    """
    # Standard OpenCV install path
    cv_data = Path(cv2.data.haarcascades)
    cascade_path = cv_data / "haarcascade_frontalface_default.xml"
    if cascade_path.exists():
        return str(cascade_path)

    # Fallback: check alongside this file (for bundled deployments)
    local = Path(__file__).parent / "haarcascade_frontalface_default.xml"
    if local.exists():
        return str(local)

    raise FileNotFoundError(
        "haarcascade_frontalface_default.xml not found. "
        "Install opencv-python or place the file in emotion/"
    )


# ---------------------------------------------------------------------------
# EmotionResult dataclass-like dict schema
# ---------------------------------------------------------------------------
# detect_from_frame() returns:
# {
#   "emotion"       : str   | None,   # top predicted class (canonical)
#   "confidence"    : float | None,   # score for top class  (0–1)
#   "distribution"  : dict[str, float],  # full 7-class scores
#   "face_detected" : bool,
#   "above_threshold": bool,
#   "latency_ms"    : float,
# }


class EmotionDetector:
    """
    End-to-end real-time emotion detection.

    Usage
    -----
    >>> detector = EmotionDetector()
    >>> cap = cv2.VideoCapture(0)
    >>> ret, frame = cap.read()
    >>> result = detector.detect_from_frame(frame)
    >>> print(result["emotion"], result["confidence"])
    >>> detector.release()
    """

    # ------------------------------------------------------------------
    def __init__(
        self,
        model_id: str = MODEL_ID,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        device: Optional[int] = None,
        camera_index: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        model_id              : HuggingFace model repo id.
        confidence_threshold  : Minimum score to accept a prediction.
        device                : torch device index (0 = first GPU, -1 = CPU).
                                None → auto-select (GPU if available).
        camera_index          : OpenCV camera device index.
        """
        self.model_id             = model_id
        self.confidence_threshold = confidence_threshold
        self.camera_index         = camera_index

        log.info("Loading ViT model: %s", model_id)
        _device = device if device is not None else self._auto_device()
        self._pipe = pipeline(
            "image-classification",
            model=model_id,
            top_k=None,          # return all 7 class scores
            device=_device,
        )
        log.info("Model loaded on device=%s", _device)

        # Haar cascade for face detection
        cascade_path = _find_cascade()
        self._face_cascade = cv2.CascadeClassifier(cascade_path)
        if self._face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
        log.info("Haar cascade loaded: %s", cascade_path)

        # OpenCV capture (lazy — only opened if capture() is called directly)
        self._cap: Optional[cv2.VideoCapture] = None

    # ------------------------------------------------------------------
    @staticmethod
    def _auto_device() -> int:
        """Return 0 (first GPU) if CUDA is available, else -1 (CPU)."""
        try:
            import torch
            return 0 if torch.cuda.is_available() else -1
        except ImportError:
            return -1

    # ------------------------------------------------------------------
    def _open_capture(self) -> cv2.VideoCapture:
        """Open the webcam if not already open."""
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.camera_index)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"Cannot open camera index {self.camera_index}. "
                    "Check that the webcam is connected and not in use."
                )
        return self._cap

    # ------------------------------------------------------------------
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Read a single frame from the webcam.

        Returns
        -------
        BGR numpy array (H×W×3) or None if capture fails.
        """
        cap = self._open_capture()
        ret, frame = cap.read()
        return frame if ret else None

    # ------------------------------------------------------------------
    def detect_from_frame(self, frame: np.ndarray) -> dict:
        """
        Run the full detection pipeline on a BGR OpenCV frame.

        Steps
        -----
        1. Detect faces with Haar cascade.
        2. Crop the largest face ROI.
        3. Convert BGR → RGB → PIL Image.
        4. Run ViT image-classification pipeline.
        5. Normalise scores, apply confidence filter.
        6. Return structured result dict.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (as returned by cv2.VideoCapture.read()).

        Returns
        -------
        dict with keys:
            emotion         : str | None
            confidence      : float | None
            distribution    : dict[str, float]
            face_detected   : bool
            above_threshold : bool
            latency_ms      : float
        """
        t0 = time.perf_counter()

        empty_dist = {e: 0.0 for e in EMOTION_CLASSES}

        result: dict = {
            "emotion":         None,
            "confidence":      None,
            "distribution":    empty_dist.copy(),
            "face_detected":   False,
            "above_threshold": False,
            "latency_ms":      0.0,
        }

        # ── 1. Face detection ──────────────────────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(48, 48),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces) == 0:
            result["latency_ms"] = (time.perf_counter() - t0) * 1000
            return result

        result["face_detected"] = True

        # Pick the largest face by area
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])

        # ── 2. Crop + colour-convert ───────────────────────────────────
        face_bgr = frame[y : y + h, x : x + w]
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_img  = Image.fromarray(face_rgb)

        # ── 3. ViT inference ───────────────────────────────────────────
        raw_preds: list[dict] = self._pipe(pil_img)   # list of {label, score}

        # ── 4. Build distribution dict ─────────────────────────────────
        distribution: dict[str, float] = empty_dist.copy()
        for pred in raw_preds:
            label     = pred["label"].lower().strip()
            canonical = _LABEL_MAP.get(label, label)
            if canonical in distribution:
                distribution[canonical] = float(pred["score"])

        # Normalise so scores sum to 1.0 (guard against fp drift)
        total = sum(distribution.values())
        if total > 0:
            distribution = {k: v / total for k, v in distribution.items()}

        # ── 5. Top class ───────────────────────────────────────────────
        top_emotion   = max(distribution, key=distribution.get)  # type: ignore[arg-type]
        top_confidence = distribution[top_emotion]

        result["distribution"] = distribution
        result["emotion"]      = top_emotion
        result["confidence"]   = top_confidence
        result["above_threshold"] = top_confidence >= self.confidence_threshold

        # ── 6. Latency logging ─────────────────────────────────────────
        elapsed_ms = (time.perf_counter() - t0) * 1000
        result["latency_ms"] = elapsed_ms

        if elapsed_ms > LATENCY_BUDGET_MS:
            log.warning(
                "Inference latency %.1f ms exceeded budget of %.1f ms",
                elapsed_ms,
                LATENCY_BUDGET_MS,
            )
        else:
            log.debug("Inference latency: %.1f ms", elapsed_ms)

        return result

    # ------------------------------------------------------------------
    def annotate_frame(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """
        Draw bounding-box overlay and emotion label on the frame.
        Returns a copy of the frame with annotations.
        """
        annotated = frame.copy()

        if not result["face_detected"]:
            cv2.putText(
                annotated,
                "No face detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            return annotated

        # We re-detect to draw boxes; for live display pass faces separately
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )
        for x, y, w, h in faces:
            color = (0, 255, 0) if result["above_threshold"] else (0, 165, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        if result["emotion"] and result["confidence"]:
            label = f"{result['emotion'].upper()}  {result['confidence']:.0%}"
            cv2.putText(
                annotated, label, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
            )

        lat = f"{result['latency_ms']:.0f} ms"
        cv2.putText(
            annotated, lat, (10, annotated.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1,
        )
        return annotated

    # ------------------------------------------------------------------
    def release(self) -> None:
        """Release the OpenCV VideoCapture resource."""
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            self._cap = None
        log.info("Camera released.")

    # ------------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
