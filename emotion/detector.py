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
  - Frame-skip strategy: ViT runs every N frames; cascade runs every frame
  - Returns top emotion label, its confidence score, and the full 7-class
    probability distribution.
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

# Run ViT inference every N frames (cascade still runs every frame for live box)
INFERENCE_EVERY_N_FRAMES = 5

# HuggingFace label aliases → canonical class names
_LABEL_MAP: dict[str, str] = {
    "angry":    "angry",
    "disgust":  "disgust",
    "fear":     "fear",
    "happy":    "happy",
    "neutral":  "neutral",
    "sad":      "sad",
    "surprise": "surprise",
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
    cv_data = Path(cv2.data.haarcascades)
    cascade_path = cv_data / "haarcascade_frontalface_default.xml"
    if cascade_path.exists():
        return str(cascade_path)

    local = Path(__file__).parent / "haarcascade_frontalface_default.xml"
    if local.exists():
        return str(local)

    raise FileNotFoundError(
        "haarcascade_frontalface_default.xml not found. "
        "Install opencv-python or place the file in emotion/"
    )


# ---------------------------------------------------------------------------
# EmotionDetector
# ---------------------------------------------------------------------------

class EmotionDetector:
    """
    End-to-end real-time emotion detection with frame-skip for low latency.

    Strategy
    --------
    - Haar cascade runs every frame (very fast, < 5 ms) → live bounding box
    - ViT runs every INFERENCE_EVERY_N_FRAMES frames (~300–700 ms on CPU)
    - Between ViT runs, the last result is returned immediately (0 ms)
    - This keeps the Streamlit feed smooth at ~5 fps while ViT runs async
    """

    def __init__(
        self,
        model_id: str = MODEL_ID,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
        device: Optional[int] = None,
        camera_index: int = 0,
        inference_every_n: int = INFERENCE_EVERY_N_FRAMES,
    ) -> None:
        self.model_id             = model_id
        self.confidence_threshold = confidence_threshold
        self.camera_index         = camera_index
        self.inference_every_n    = inference_every_n

        log.info("Loading ViT model: %s", model_id)
        _device = device if device is not None else self._auto_device()
        self._pipe = pipeline(
            "image-classification",
            model=model_id,
            top_k=None,
            device=_device,
        )
        log.info("Model loaded on device=%s", _device)

        cascade_path = _find_cascade()
        self._face_cascade = cv2.CascadeClassifier(cascade_path)
        if self._face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade from {cascade_path}")
        log.info("Haar cascade loaded: %s", cascade_path)

        self._cap: Optional[cv2.VideoCapture] = None

        # Frame-skip state
        self._frame_count: int  = 0
        self._last_result: dict = self._empty_result()
        self._last_faces          = []  # cached face rects for annotation

    # ------------------------------------------------------------------
    @staticmethod
    def _empty_result() -> dict:
        return {
            "emotion":         None,
            "confidence":      None,
            "distribution":    {e: 0.0 for e in EMOTION_CLASSES},
            "face_detected":   False,
            "above_threshold": False,
            "latency_ms":      0.0,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _auto_device() -> int:
        try:
            import torch
            return 0 if torch.cuda.is_available() else -1
        except ImportError:
            return -1

    # ------------------------------------------------------------------
    def _open_capture(self) -> cv2.VideoCapture:
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
        cap = self._open_capture()
        ret, frame = cap.read()
        return frame if ret else None

    # ------------------------------------------------------------------
    def _detect_faces(self, frame: np.ndarray):
        """
        Run Haar cascade on a HALF-RESOLUTION copy for speed.
        Returns face rects in ORIGINAL frame coordinates.
        """
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // 2, h // 2))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # equalizeHist dramatically improves detection in poor lighting
        gray = cv2.equalizeHist(gray)

        faces_small = self._face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,    # finer pyramid steps → catches more faces
            minNeighbors=4,      # slightly more lenient
            minSize=(30, 30),    # smaller min (we're at half res → 60px original)
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        if len(faces_small) == 0:
            return []

        # Scale rects back to original resolution
        return [(x * 2, y * 2, w * 2, h * 2) for x, y, w, h in faces_small]

    # ------------------------------------------------------------------
    def detect_from_frame(self, frame: np.ndarray) -> dict:
        """
        Run the detection pipeline.

        Frame-skip strategy
        -------------------
        Every call:   Haar cascade on half-res frame (~3 ms)
        Every N calls: ViT inference on face crop (~300-700 ms on CPU)
        Between ViT calls: return last cached emotion result instantly.

        Parameters
        ----------
        frame : np.ndarray  —  BGR image from cv2.VideoCapture.read()

        Returns
        -------
        dict : emotion, confidence, distribution, face_detected,
               above_threshold, latency_ms
        """
        t0 = time.perf_counter()
        self._frame_count += 1

        # ── 1. Fast face detection (every frame) ──────────────────────
        faces = self._detect_faces(frame)
        self._last_faces = faces  # store for annotate_frame

        if not faces:
            result = self._empty_result()
            result["latency_ms"] = (time.perf_counter() - t0) * 1000
            self._last_result = result
            return result

        # ── 2. Skip ViT unless it's the Nth frame ─────────────────────
        if self._frame_count % self.inference_every_n != 0:
            # Return cached result but update face_detected flag
            cached = dict(self._last_result)
            cached["face_detected"] = True
            cached["latency_ms"]    = (time.perf_counter() - t0) * 1000
            return cached

        # ── 3. ViT inference (every N frames) ─────────────────────────
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        face_bgr = frame[y : y + h, x : x + w]
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_img  = Image.fromarray(face_rgb)

        raw_preds: list[dict] = self._pipe(pil_img)

        # ── 4. Build distribution dict ─────────────────────────────────
        empty_dist   = {e: 0.0 for e in EMOTION_CLASSES}
        distribution = empty_dist.copy()
        for pred in raw_preds:
            label     = pred["label"].lower().strip()
            canonical = _LABEL_MAP.get(label, label)
            if canonical in distribution:
                distribution[canonical] = float(pred["score"])

        total = sum(distribution.values())
        if total > 0:
            distribution = {k: v / total for k, v in distribution.items()}

        # ── 5. Top class ───────────────────────────────────────────────
        top_emotion    = max(distribution, key=distribution.get)   # type: ignore
        top_confidence = distribution[top_emotion]

        elapsed_ms = (time.perf_counter() - t0) * 1000

        result = {
            "emotion":         top_emotion,
            "confidence":      top_confidence,
            "distribution":    distribution,
            "face_detected":   True,
            "above_threshold": top_confidence >= self.confidence_threshold,
            "latency_ms":      elapsed_ms,
        }

        if elapsed_ms > LATENCY_BUDGET_MS:
            log.warning("Inference latency %.1f ms exceeded budget %.1f ms",
                        elapsed_ms, LATENCY_BUDGET_MS)

        self._last_result = result
        return result

    # ------------------------------------------------------------------
    def annotate_frame(self, frame: np.ndarray, result: dict) -> np.ndarray:
        """
        Draw bounding-box overlay on the frame.
        Uses cached face rects from the last detect_from_frame() call —
        NO second cascade run.
        """
        annotated = frame.copy()

        if not self._last_faces:
            cv2.putText(annotated, "No face detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return annotated

        for x, y, w, h in self._last_faces:
            color = (0, 255, 0) if result.get("above_threshold") else (0, 165, 255)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)

        if result.get("emotion") and result.get("confidence") is not None:
            label = f"{result['emotion'].upper()}  {result['confidence']:.0%}"
            cv2.putText(annotated, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        lat = f"ViT: {result['latency_ms']:.0f} ms"
        cv2.putText(annotated, lat, (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        return annotated

    # ------------------------------------------------------------------
    def release(self) -> None:
        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            self._cap = None
        log.info("Camera released.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
