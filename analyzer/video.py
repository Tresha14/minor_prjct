"""Thin wrapper around MediaPipe's FaceLandmarker (Tasks API).

Kept small and separate from video_features.py, same split as
transcribe.py / audio_features.py: this is the part that touches the model
and a camera frame, so it's the part most likely to need machine-specific
tweaks. video_features.py never imports this module, which keeps its tests
fast and camera-independent.

Needs the model file at config.VIDEO.model_path -- download it once with:
    curl -L -o analyzer/models/face_landmarker.task \\
        https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
(also documented in README.md's Setup section). The file isn't committed to
git (it's a few MB of binary model weights); every dev downloads their own.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from analyzer.config import VIDEO
from analyzer.video_features import HeadPose, matrix_to_euler

logger = logging.getLogger(__name__)


@dataclass
class FaceReading:
    """One frame's worth of face-tracking output. `pose`/`blendshapes` are
    None when no face was detected in the frame (candidate stepped out of
    frame, bad lighting, etc.) -- callers should treat that as its own
    signal, not crash on it.
    """
    face_detected: bool
    pose: HeadPose | None
    blendshapes: dict[str, float] | None


class FaceTracker:
    """Wraps MediaPipe's FaceLandmarker in VIDEO running mode (i.e. fed a
    sequence of frames with increasing timestamps from one continuous
    stream -- a webcam or a captured call window, not independent images).
    """

    def __init__(self, model_path: str = VIDEO.model_path):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Face landmarker model not found at '{model_path}'. "
                "See the download command in this module's docstring / README.md Setup."
            )
        self._landmarker = self._build(str(path))
        self._last_timestamp_ms = -1

    def _build(self, model_path: str):
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def process_frame(self, frame_bgr: np.ndarray, timestamp_ms: int) -> FaceReading:
        """`frame_bgr`: an OpenCV-style HxWx3 BGR frame (as cv2.VideoCapture
        returns). `timestamp_ms`: must strictly increase between calls --
        use the session clock, not wall-clock time, if frames can be
        processed slightly out of real time.
        """
        import mediapipe as mp

        if timestamp_ms <= self._last_timestamp_ms:
            # MediaPipe's VIDEO mode requires strictly increasing timestamps;
            # nudge forward by 1ms rather than crashing on a duplicate/out-of-order frame.
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        rgb = frame_bgr[:, :, ::-1]  # BGR (OpenCV) -> RGB (MediaPipe)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)

        if not result.face_landmarks:
            return FaceReading(face_detected=False, pose=None, blendshapes=None)

        pose = None
        if result.facial_transformation_matrixes:
            pose = matrix_to_euler(result.facial_transformation_matrixes[0])

        blendshapes = None
        if result.face_blendshapes:
            blendshapes = {b.category_name: b.score for b in result.face_blendshapes[0]}

        return FaceReading(face_detected=True, pose=pose, blendshapes=blendshapes)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
