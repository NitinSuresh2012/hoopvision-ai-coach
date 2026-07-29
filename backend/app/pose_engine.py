"""OpenCV frame decoding and MediaPipe Pose inference."""

from __future__ import annotations

import base64
import binascii
from typing import Any

from .analysis import Landmark


class PoseEngineError(RuntimeError):
    """Raised when a frame cannot be decoded or the pose engine cannot start."""


class MediaPipePoseEngine:
    """One MediaPipe Pose tracker.

    A separate instance is created for each WebSocket connection because the
    MediaPipe tracking graph is stateful and should not mix different cameras.
    """

    LANDMARK_INDEX = {
        "left_ear": 7,
        "right_ear": 8,
        "left_shoulder": 11,
        "right_shoulder": 12,
        "left_hip": 23,
        "right_hip": 24,
    }

    def __init__(self) -> None:
        try:
            import cv2
            import mediapipe as mp
            import numpy as np
        except ImportError as error:
            raise PoseEngineError(
                "OpenCV and MediaPipe are required. Install backend/requirements.txt."
            ) from error

        self.cv2 = cv2
        self.np = np
        try:
            pose_module = mp.solutions.pose
        except AttributeError as error:
            raise PoseEngineError(
                "This service requires the MediaPipe Pose Solutions API. "
                "Install the pinned MediaPipe version from backend/requirements.txt."
            ) from error

        self._pose = pose_module.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65,
        )

    def close(self) -> None:
        self._pose.close()

    def process_data_url(
        self,
        image_payload: str,
        _timestamp_ms: int,
    ) -> dict[str, Landmark]:
        """Decode one browser frame and return required normalized landmarks."""

        if not isinstance(image_payload, str):
            raise PoseEngineError("Frame image must be a base64 string.")
        if len(image_payload) > 8_000_000:
            raise PoseEngineError("Frame is larger than the 8 MB limit.")

        encoded = image_payload
        if image_payload.startswith("data:"):
            try:
                header, encoded = image_payload.split(",", 1)
            except ValueError as error:
                raise PoseEngineError("Malformed image data URL.") from error
            allowed = (
                "data:image/jpeg;base64",
                "data:image/png;base64",
                "data:image/webp;base64",
            )
            if header.lower() not in allowed:
                raise PoseEngineError(
                    "Frame must be a JPEG, PNG, or WebP data URL."
                )

        try:
            image_bytes = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as error:
            raise PoseEngineError("Frame is not valid base64.") from error
        if not image_bytes:
            raise PoseEngineError("Frame is empty.")

        buffer = self.np.frombuffer(image_bytes, dtype=self.np.uint8)
        frame = self.cv2.imdecode(buffer, self.cv2.IMREAD_COLOR)
        if frame is None:
            raise PoseEngineError("OpenCV could not decode the frame.")

        height, width = frame.shape[:2]
        if height < 64 or width < 64:
            raise PoseEngineError("Frame resolution is too small.")
        if height * width > 16_000_000:
            raise PoseEngineError("Frame resolution exceeds 16 megapixels.")

        max_side = max(height, width)
        if max_side > 1280:
            scale = 1280.0 / max_side
            frame = self.cv2.resize(
                frame,
                (int(width * scale), int(height * scale)),
                interpolation=self.cv2.INTER_AREA,
            )

        rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result: Any = self._pose.process(rgb)
        if result.pose_landmarks is None:
            return {}

        output: dict[str, Landmark] = {}
        for name, index in self.LANDMARK_INDEX.items():
            raw = result.pose_landmarks.landmark[index]
            # The Pose Solutions graph reports `visibility` but does not
            # populate the optional `presence` field. Protobuf exposes an
            # unset scalar as 0.0, which must not turn every real pose into a
            # low-confidence result.
            presence = 1.0
            try:
                if raw.HasField("presence"):
                    presence = float(raw.presence)
            except (AttributeError, ValueError):
                presence = 1.0
            output[name] = Landmark(
                x=float(raw.x),
                y=float(raw.y),
                z=float(raw.z),
                visibility=float(getattr(raw, "visibility", 1.0)),
                presence=presence,
            )
        return output
