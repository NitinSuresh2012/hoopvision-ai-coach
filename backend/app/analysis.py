"""Pure posture-analysis and per-session state.

This module intentionally has no OpenCV, MediaPipe, or FastAPI dependency.  It
contains the deterministic part of the posture coach so it can be tested with
explicit pose-landmark fixtures.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from math import atan2, degrees, hypot
from statistics import fmean, median
from typing import Deque, Iterable, Mapping, Optional


LANDMARK_NAMES = (
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_hip",
    "right_hip",
)


@dataclass(frozen=True)
class Landmark:
    """A normalized MediaPipe pose landmark."""

    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0
    presence: float = 1.0

    @property
    def confidence(self) -> float:
        return min(self.visibility, self.presence)

    def public_dict(self) -> dict[str, float]:
        return {
            "x": round(self.x, 5),
            "y": round(self.y, 5),
            "z": round(self.z, 5),
            "visibility": round(self.visibility, 4),
        }


@dataclass(frozen=True)
class Point:
    x: float
    y: float
    z: float = 0.0

    def public_dict(self) -> dict[str, float]:
        return {
            "x": round(self.x, 5),
            "y": round(self.y, 5),
            "z": round(self.z, 5),
        }


@dataclass(frozen=True)
class PostureMetrics:
    """Camera-normalized measurements derived from ears, shoulders, and hips."""

    neck_angle_deg: float
    head_offset_ratio: float
    shoulder_tilt_deg: float
    upper_back_angle_deg: float
    hip_tilt_deg: float
    torso_length: float
    shoulder_width: float

    def public_dict(self) -> dict[str, float]:
        return {key: round(value, 3) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class CameraWarning:
    code: str
    message: str

    def public_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class FrameObservation:
    valid: bool
    metrics: Optional[PostureMetrics]
    landmarks: dict[str, Landmark]
    points: dict[str, Point]
    confidence: float
    missing: tuple[str, ...]
    camera_warning: Optional[CameraWarning]


@dataclass(frozen=True)
class CalibrationBaseline:
    metrics: PostureMetrics
    variability: PostureMetrics
    sample_count: int
    created_at_ms: int

    def public_dict(self) -> dict[str, object]:
        return {
            "sample_count": self.sample_count,
            "created_at_ms": self.created_at_ms,
            "metrics": self.metrics.public_dict(),
        }


@dataclass(frozen=True)
class Correction:
    key: str
    instruction: str
    severity: float


def _average_point(points: Iterable[Landmark]) -> Point:
    values = tuple(points)
    return Point(
        x=fmean(point.x for point in values),
        y=fmean(point.y for point in values),
        z=fmean(point.z for point in values),
    )


def _distance(a: Point, b: Point) -> float:
    return hypot(a.x - b.x, a.y - b.y)


def _line_angle_from_horizontal(a: Point, b: Point) -> float:
    angle = degrees(atan2(b.y - a.y, b.x - a.x))
    # Anatomical left/right swap screen sides in a front-facing camera. Treat
    # a horizontal line as 0 degrees regardless of point ordering and retain
    # only its signed tilt.
    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0
    return angle


def _line_angle_from_vertical(lower: Point, upper: Point) -> float:
    """Signed degrees between an upward vertical line and lower->upper."""

    return degrees(atan2(upper.x - lower.x, lower.y - upper.y))


def _camera_warning(
    landmarks: Mapping[str, Landmark],
    ear: Point,
    shoulder: Point,
    hip: Point,
    torso_length: float,
    shoulder_width: float,
) -> Optional[CameraWarning]:
    xs = [landmark.x for landmark in landmarks.values()]
    ys = [landmark.y for landmark in landmarks.values()]
    body_width = max(xs) - min(xs)
    body_height = max(ys) - min(ys)

    # These are framing safeguards rather than a claim to recover true camera
    # pitch from a monocular image. They catch the camera positions that most
    # often make seated-pose measurements unreliable.
    if (
        shoulder_width > 0.58
        or torso_length > 0.56
        or body_width > 0.78
        or body_height > 0.93
    ):
        return CameraWarning(
            "camera_too_close",
            "Camera is too close — move it farther away.",
        )
    if ear.y < 0.06 or shoulder.y < 0.16 or hip.y < 0.46:
        return CameraWarning(
            "camera_too_low",
            "Camera is too low — raise it to shoulder height.",
        )
    if ear.y > 0.40 or shoulder.y > 0.64 or hip.y > 0.94:
        return CameraWarning(
            "camera_too_high",
            "Camera is too high — lower it to shoulder height.",
        )
    return None


def observe_landmarks(
    landmarks: Mapping[str, Landmark],
    confidence_threshold: float = 0.60,
) -> FrameObservation:
    """Validate required landmarks and calculate raw posture metrics."""

    received = {
        name: landmarks[name]
        for name in LANDMARK_NAMES
        if name in landmarks
    }
    def usable(name: str) -> bool:
        point = received.get(name)
        return bool(
            point
            and point.confidence >= confidence_threshold
            and 0.0 <= point.x <= 1.0
            and 0.0 <= point.y <= 1.0
        )

    torso_names = (
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
    )
    usable_ears = tuple(
        name for name in ("left_ear", "right_ear") if usable(name)
    )
    missing_torso = tuple(name for name in torso_names if not usable(name))
    missing_ears = (
        ()
        if usable_ears
        else ("left_ear", "right_ear")
    )
    missing = missing_torso + missing_ears
    def raw_confidence(name: str) -> float:
        point = received.get(name)
        if (
            point is None
            or not (0.0 <= point.x <= 1.0)
            or not (0.0 <= point.y <= 1.0)
        ):
            return 0.0
        return point.confidence

    required_confidences = [
        *(raw_confidence(name) for name in torso_names),
        max(
            raw_confidence("left_ear"),
            raw_confidence("right_ear"),
        ),
    ]
    confidence = min(required_confidences)

    if missing:
        return FrameObservation(
            valid=False,
            metrics=None,
            landmarks=received,
            points={},
            confidence=confidence,
            missing=missing,
            camera_warning=None,
        )

    left_shoulder = received["left_shoulder"]
    right_shoulder = received["right_shoulder"]
    left_hip = received["left_hip"]
    right_hip = received["right_hip"]

    ear_mid = _average_point(received[name] for name in usable_ears)
    shoulder_mid = _average_point((left_shoulder, right_shoulder))
    hip_mid = _average_point((left_hip, right_hip))
    points = {
        "ear_mid": ear_mid,
        "shoulder_mid": shoulder_mid,
        "hip_mid": hip_mid,
    }

    torso_length = max(_distance(shoulder_mid, hip_mid), 0.001)
    shoulder_width = _distance(
        Point(left_shoulder.x, left_shoulder.y),
        Point(right_shoulder.x, right_shoulder.y),
    )
    metrics = PostureMetrics(
        neck_angle_deg=_line_angle_from_vertical(shoulder_mid, ear_mid),
        head_offset_ratio=(ear_mid.x - shoulder_mid.x) / torso_length,
        shoulder_tilt_deg=_line_angle_from_horizontal(
            Point(left_shoulder.x, left_shoulder.y),
            Point(right_shoulder.x, right_shoulder.y),
        ),
        upper_back_angle_deg=_line_angle_from_vertical(hip_mid, shoulder_mid),
        hip_tilt_deg=_line_angle_from_horizontal(
            Point(left_hip.x, left_hip.y),
            Point(right_hip.x, right_hip.y),
        ),
        torso_length=torso_length,
        shoulder_width=shoulder_width,
    )
    warning = _camera_warning(
        {
            name: received[name]
            for name in (*torso_names, *usable_ears)
        },
        ear_mid,
        shoulder_mid,
        hip_mid,
        torso_length,
        shoulder_width,
    )
    return FrameObservation(
        valid=True,
        metrics=metrics,
        landmarks=received,
        points=points,
        confidence=confidence,
        missing=(),
        camera_warning=warning,
    )


class CalibrationAccumulator:
    """Collect a stable, median baseline over a short calibration period."""

    def __init__(
        self,
        started_at_ms: int,
        duration_ms: int = 3_000,
        minimum_samples: int = 12,
    ) -> None:
        self.started_at_ms = started_at_ms
        self.duration_ms = duration_ms
        self.minimum_samples = minimum_samples
        self.samples: list[PostureMetrics] = []

    def add(self, metrics: PostureMetrics) -> None:
        self.samples.append(metrics)

    def restart(self, timestamp_ms: int) -> None:
        """Require a new continuous valid hold after tracking is interrupted."""

        self.started_at_ms = timestamp_ms
        self.samples.clear()

    def progress(self, now_ms: int) -> float:
        elapsed_progress = max(
            0.0,
            min(1.0, (now_ms - self.started_at_ms) / self.duration_ms),
        )
        sample_progress = min(1.0, len(self.samples) / self.minimum_samples)
        return min(elapsed_progress, sample_progress)

    def ready(self, now_ms: int) -> bool:
        return (
            now_ms - self.started_at_ms >= self.duration_ms
            and len(self.samples) >= self.minimum_samples
        )

    def finish(self, now_ms: int, force: bool = False) -> CalibrationBaseline:
        if not force and not self.ready(now_ms):
            raise ValueError("Calibration does not yet have enough valid frames.")
        if len(self.samples) < 3:
            raise ValueError("At least three valid frames are required.")

        fields = tuple(PostureMetrics.__dataclass_fields__)
        medians: dict[str, float] = {}
        deviations: dict[str, float] = {}
        for field in fields:
            values = [getattr(sample, field) for sample in self.samples]
            center = median(values)
            medians[field] = center
            deviations[field] = median(abs(value - center) for value in values)
        return CalibrationBaseline(
            metrics=PostureMetrics(**medians),
            variability=PostureMetrics(**deviations),
            sample_count=len(self.samples),
            created_at_ms=now_ms,
        )


class TimeWindowSmoother:
    """Two-second rolling metric mean used before colors and scoring."""

    def __init__(self, window_ms: int = 2_000) -> None:
        self.window_ms = window_ms
        self._samples: Deque[tuple[int, PostureMetrics]] = deque()

    def clear(self) -> None:
        self._samples.clear()

    def add(self, timestamp_ms: int, metrics: PostureMetrics) -> None:
        self._samples.append((timestamp_ms, metrics))
        cutoff = timestamp_ms - self.window_ms
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def mean(self) -> PostureMetrics:
        if not self._samples:
            raise ValueError("No measurements are available to smooth.")
        fields = tuple(PostureMetrics.__dataclass_fields__)
        return PostureMetrics(
            **{
                field: fmean(
                    getattr(metrics, field) for _, metrics in self._samples
                )
                for field in fields
            }
        )

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    @property
    def span_ms(self) -> int:
        if len(self._samples) < 2:
            return 0
        return self._samples[-1][0] - self._samples[0][0]


class PostureSession:
    """Calibration, smoothing, feedback, and alert state for one camera."""

    DEFAULT_ENTER_THRESHOLDS = {
        "head": 7.0,
        "shoulders": 4.0,
        "back": 6.0,
        "hips": 5.0,
    }
    DEFAULT_EXIT_THRESHOLDS = {
        "head": 4.5,
        "shoulders": 2.5,
        "back": 4.0,
        "hips": 3.0,
    }
    INSTRUCTIONS = {
        "head": "Move your head back",
        "shoulders": "Level your shoulders",
        "back": "Sit up",
        "hips": "Sit up",
    }

    def __init__(
        self,
        smoothing_ms: int = 2_000,
        alert_after_ms: int = 3_000,
    ) -> None:
        self.baseline: Optional[CalibrationBaseline] = None
        self.calibration: Optional[CalibrationAccumulator] = None
        self.smoother = TimeWindowSmoother(smoothing_ms)
        self.alert_after_ms = alert_after_ms
        self._issues = {
            "head": False,
            "shoulders": False,
            "back": False,
            "hips": False,
        }
        self._bad_since_ms: Optional[int] = None
        self._active_instruction: Optional[str] = None
        self._alerted = False

    def start_calibration(self, timestamp_ms: int) -> dict[str, object]:
        self.calibration = CalibrationAccumulator(timestamp_ms)
        self.baseline = None
        self.smoother.clear()
        self._reset_correction_episode()
        return self.calibration_status(timestamp_ms)

    def finish_calibration(
        self,
        timestamp_ms: int,
        force: bool = False,
    ) -> CalibrationBaseline:
        if self.calibration is None:
            raise ValueError("Calibration has not been started.")
        self.baseline = self.calibration.finish(timestamp_ms, force=force)
        self.calibration = None
        self.smoother.clear()
        self._issues = {key: False for key in self._issues}
        self._reset_correction_episode()
        return self.baseline

    def reset_calibration(self) -> None:
        self.baseline = None
        self.calibration = None
        self.smoother.clear()
        self._issues = {key: False for key in self._issues}
        self._reset_correction_episode()

    def calibration_status(self, timestamp_ms: int) -> dict[str, object]:
        if self.calibration is not None:
            return {
                "phase": "capturing",
                "progress": round(self.calibration.progress(timestamp_ms), 3),
                "valid_frames": len(self.calibration.samples),
                "message": "Hold your best sitting posture",
            }
        if self.baseline is not None:
            return {
                "phase": "complete",
                "progress": 1.0,
                "valid_frames": self.baseline.sample_count,
                "message": "Correct posture saved",
            }
        return {
            "phase": "required",
            "progress": 0.0,
            "valid_frames": 0,
            "message": "Save your correct posture to begin",
        }

    def process(
        self,
        landmarks: Mapping[str, Landmark],
        timestamp_ms: int,
    ) -> dict[str, object]:
        observation = observe_landmarks(landmarks)
        if not observation.valid:
            self._interrupt_tracking(timestamp_ms)
            return self._paused_result(observation, timestamp_ms)

        assert observation.metrics is not None
        if observation.camera_warning is not None:
            self._interrupt_tracking(timestamp_ms)
            return self._camera_warning_result(observation, timestamp_ms)

        if self.calibration is not None:
            self.calibration.add(observation.metrics)
            if self.calibration.ready(timestamp_ms):
                self.finish_calibration(timestamp_ms)
                return self._calibration_result(
                    observation,
                    timestamp_ms,
                    just_completed=True,
                )
            return self._calibration_result(
                observation,
                timestamp_ms,
                just_completed=False,
            )

        if self.baseline is None:
            return self._needs_calibration_result(observation, timestamp_ms)

        self.smoother.add(timestamp_ms, observation.metrics)
        if self.smoother.span_ms < self.smoother.window_ms:
            self._reset_correction_episode()
            return self._stabilizing_result(observation, timestamp_ms)
        smoothed = self.smoother.mean()
        errors = self._errors(smoothed)
        thresholds = self._thresholds()
        self._update_issue_hysteresis(errors, thresholds)
        correction = self._choose_correction(errors, thresholds)
        play_alert = self._update_alert(correction, timestamp_ms)
        score = self._score(errors, thresholds)
        area_scores = self._area_scores(errors, thresholds)
        areas = self._areas()
        return {
            "type": "posture_result",
            "timestamp_ms": timestamp_ms,
            "scoring_active": True,
            "score": score,
            "instruction": correction.instruction if correction else None,
            "status_message": (
                correction.instruction if correction else "Posture locked"
            ),
            "play_alert": play_alert,
            "confidence": round(observation.confidence, 4),
            "camera_warning": (
                observation.camera_warning.public_dict()
                if observation.camera_warning
                else None
            ),
            "calibration": self.calibration_status(timestamp_ms),
            "areas": areas,
            "area_scores": area_scores,
            "segments": self._segments(areas),
            "landmarks": self._public_landmarks(observation),
            "metrics": smoothed.public_dict(),
            "smoothing_window_ms": self.smoother.window_ms,
            "smoothed_frame_count": self.smoother.sample_count,
        }

    def _stabilizing_result(
        self,
        observation: FrameObservation,
        timestamp_ms: int,
    ) -> dict[str, object]:
        neutral_areas = {
            "head_neck": "neutral",
            "shoulders": "neutral",
            "upper_back": "neutral",
            "hips": "neutral",
        }
        return {
            "type": "posture_result",
            "timestamp_ms": timestamp_ms,
            "scoring_active": False,
            "score": None,
            "instruction": None,
            "status_message": "Reading your alignment",
            "play_alert": False,
            "confidence": round(observation.confidence, 4),
            "camera_warning": None,
            "calibration": self.calibration_status(timestamp_ms),
            "areas": neutral_areas,
            "area_scores": None,
            "segments": self._segments(neutral_areas),
            "landmarks": self._public_landmarks(observation),
            "metrics": (
                observation.metrics.public_dict()
                if observation.metrics
                else None
            ),
            "smoothing_window_ms": self.smoother.window_ms,
            "smoothing_span_ms": self.smoother.span_ms,
            "smoothed_frame_count": self.smoother.sample_count,
        }

    def _paused_result(
        self,
        observation: FrameObservation,
        timestamp_ms: int,
    ) -> dict[str, object]:
        return {
            "type": "posture_result",
            "timestamp_ms": timestamp_ms,
            "scoring_active": False,
            "score": None,
            "instruction": "Move fully into view",
            "status_message": "Move fully into view",
            "play_alert": False,
            "confidence": round(observation.confidence, 4),
            "missing_landmarks": list(observation.missing),
            "camera_warning": None,
            "calibration": self.calibration_status(timestamp_ms),
            "areas": {
                "head_neck": "unavailable",
                "shoulders": "unavailable",
                "upper_back": "unavailable",
                "hips": "unavailable",
            },
            "area_scores": None,
            "segments": [],
            "landmarks": self._public_landmarks(observation),
            "metrics": None,
            "smoothing_window_ms": self.smoother.window_ms,
            "smoothed_frame_count": self.smoother.sample_count,
        }

    def _camera_warning_result(
        self,
        observation: FrameObservation,
        timestamp_ms: int,
    ) -> dict[str, object]:
        assert observation.camera_warning is not None
        warning = observation.camera_warning.public_dict()
        neutral_areas = {
            "head_neck": "neutral",
            "shoulders": "neutral",
            "upper_back": "neutral",
            "hips": "neutral",
        }
        return {
            "type": "posture_result",
            "timestamp_ms": timestamp_ms,
            "scoring_active": False,
            "score": None,
            "instruction": warning["message"],
            "status_message": warning["message"],
            "play_alert": False,
            "confidence": round(observation.confidence, 4),
            "camera_warning": warning,
            "calibration": self.calibration_status(timestamp_ms),
            "areas": neutral_areas,
            "area_scores": None,
            "segments": self._segments(neutral_areas),
            "landmarks": self._public_landmarks(observation),
            "metrics": (
                observation.metrics.public_dict()
                if observation.metrics
                else None
            ),
            "smoothing_window_ms": self.smoother.window_ms,
            "smoothed_frame_count": self.smoother.sample_count,
        }

    def _calibration_result(
        self,
        observation: FrameObservation,
        timestamp_ms: int,
        just_completed: bool,
    ) -> dict[str, object]:
        calibration = self.calibration_status(timestamp_ms)
        return {
            "type": "posture_result",
            "timestamp_ms": timestamp_ms,
            "scoring_active": False,
            "score": None,
            "instruction": None,
            "status_message": (
                "Correct posture saved"
                if just_completed
                else "Hold your best sitting posture"
            ),
            "play_alert": False,
            "confidence": round(observation.confidence, 4),
            "camera_warning": (
                observation.camera_warning.public_dict()
                if observation.camera_warning
                else None
            ),
            "calibration": calibration,
            "areas": {
                "head_neck": "green",
                "shoulders": "green",
                "upper_back": "green",
                "hips": "green",
            },
            "area_scores": None,
            "segments": self._segments(
                {
                    "head_neck": "green",
                    "shoulders": "green",
                    "upper_back": "green",
                    "hips": "green",
                }
            ),
            "landmarks": self._public_landmarks(observation),
            "metrics": observation.metrics.public_dict()
            if observation.metrics
            else None,
            "smoothing_window_ms": self.smoother.window_ms,
            "smoothed_frame_count": 0,
        }

    def _needs_calibration_result(
        self,
        observation: FrameObservation,
        timestamp_ms: int,
    ) -> dict[str, object]:
        return {
            "type": "posture_result",
            "timestamp_ms": timestamp_ms,
            "scoring_active": False,
            "score": None,
            "instruction": None,
            "status_message": "Save your correct posture to begin",
            "play_alert": False,
            "confidence": round(observation.confidence, 4),
            "camera_warning": (
                observation.camera_warning.public_dict()
                if observation.camera_warning
                else None
            ),
            "calibration": self.calibration_status(timestamp_ms),
            "areas": {
                "head_neck": "neutral",
                "shoulders": "neutral",
                "upper_back": "neutral",
                "hips": "neutral",
            },
            "area_scores": None,
            "segments": self._segments(
                {
                    "head_neck": "neutral",
                    "shoulders": "neutral",
                    "upper_back": "neutral",
                    "hips": "neutral",
                }
            ),
            "landmarks": self._public_landmarks(observation),
            "metrics": observation.metrics.public_dict()
            if observation.metrics
            else None,
            "smoothing_window_ms": self.smoother.window_ms,
            "smoothed_frame_count": 0,
        }

    def _errors(self, metrics: PostureMetrics) -> dict[str, float]:
        assert self.baseline is not None
        reference = self.baseline.metrics
        neck_angle_error = abs(
            metrics.neck_angle_deg - reference.neck_angle_deg
        )
        head_ratio_error = abs(
            metrics.head_offset_ratio - reference.head_offset_ratio
        )
        return {
            "head": max(neck_angle_error, head_ratio_error * 45.0),
            "shoulders": abs(
                metrics.shoulder_tilt_deg - reference.shoulder_tilt_deg
            ),
            "back": abs(
                metrics.upper_back_angle_deg
                - reference.upper_back_angle_deg
            ),
            "hips": abs(metrics.hip_tilt_deg - reference.hip_tilt_deg),
        }

    def _thresholds(self) -> dict[str, tuple[float, float]]:
        assert self.baseline is not None
        variability = self.baseline.variability
        variability_by_issue = {
            "head": max(
                variability.neck_angle_deg,
                variability.head_offset_ratio * 45.0,
            ),
            "shoulders": variability.shoulder_tilt_deg,
            "back": variability.upper_back_angle_deg,
            "hips": variability.hip_tilt_deg,
        }
        output: dict[str, tuple[float, float]] = {}
        for key, default_enter in self.DEFAULT_ENTER_THRESHOLDS.items():
            enter = max(default_enter, variability_by_issue[key] * 3.0)
            exit_value = min(
                enter * 0.75,
                max(
                    self.DEFAULT_EXIT_THRESHOLDS[key],
                    variability_by_issue[key] * 2.0,
                ),
            )
            output[key] = (enter, exit_value)
        return output

    def _update_issue_hysteresis(
        self,
        errors: Mapping[str, float],
        thresholds: Mapping[str, tuple[float, float]],
    ) -> None:
        for key in self._issues:
            enter, exit_value = thresholds[key]
            if self._issues[key]:
                self._issues[key] = errors[key] >= exit_value
            else:
                self._issues[key] = errors[key] > enter

    def _choose_correction(
        self,
        errors: Mapping[str, float],
        thresholds: Mapping[str, tuple[float, float]],
    ) -> Optional[Correction]:
        candidates = [
            Correction(
                key=key,
                instruction=self.INSTRUCTIONS[key],
                severity=errors[key] / thresholds[key][0],
            )
            for key, active in self._issues.items()
            if active
        ]
        return max(candidates, key=lambda item: item.severity, default=None)

    def _score(
        self,
        errors: Mapping[str, float],
        thresholds: Mapping[str, tuple[float, float]],
    ) -> int:
        weights = {
            "head": 34.0,
            "shoulders": 20.0,
            "back": 36.0,
            "hips": 10.0,
        }
        severe_multiplier = {
            "head": 4.0,
            "shoulders": 4.0,
            "back": 4.0,
            "hips": 3.0,
        }
        penalty = 0.0
        for key, weight in weights.items():
            enter = thresholds[key][0]
            # A score can begin moving before the red threshold while the line
            # remains green, making the 0-100 value responsive but the visual
            # state resistant to flicker.
            start = enter * 0.35
            severe = enter * severe_multiplier[key]
            fraction = (errors[key] - start) / max(severe - start, 0.001)
            penalty += weight * max(0.0, min(1.0, fraction))
        return int(round(max(0.0, min(100.0, 100.0 - penalty))))

    @staticmethod
    def _area_scores(
        errors: Mapping[str, float],
        thresholds: Mapping[str, tuple[float, float]],
    ) -> dict[str, int]:
        output: dict[str, int] = {}
        severe_multiplier = {
            "head": 4.0,
            "shoulders": 4.0,
            "back": 4.0,
            "hips": 3.0,
        }
        for key, error in errors.items():
            enter = thresholds[key][0]
            start = enter * 0.35
            severe = enter * severe_multiplier[key]
            fraction = (error - start) / max(severe - start, 0.001)
            output[key] = int(
                round(100.0 * (1.0 - max(0.0, min(1.0, fraction))))
            )
        return output

    def _areas(self) -> dict[str, str]:
        return {
            "head_neck": "red" if self._issues["head"] else "green",
            "shoulders": (
                "red" if self._issues["shoulders"] else "green"
            ),
            "upper_back": "red" if self._issues["back"] else "green",
            "hips": "red" if self._issues["hips"] else "green",
        }

    @staticmethod
    def _segments(areas: Mapping[str, str]) -> list[dict[str, str]]:
        return [
            {
                "from": "ear_mid",
                "to": "shoulder_mid",
                "area": "head_neck",
                "status": areas["head_neck"],
            },
            {
                "from": "left_shoulder",
                "to": "right_shoulder",
                "area": "shoulders",
                "status": areas["shoulders"],
            },
            {
                "from": "shoulder_mid",
                "to": "hip_mid",
                "area": "upper_back",
                "status": areas["upper_back"],
            },
            {
                "from": "left_shoulder",
                "to": "left_hip",
                "area": "upper_back",
                "status": areas["upper_back"],
            },
            {
                "from": "right_shoulder",
                "to": "right_hip",
                "area": "upper_back",
                "status": areas["upper_back"],
            },
            {
                "from": "left_hip",
                "to": "right_hip",
                "area": "hips",
                "status": areas["hips"],
            },
        ]

    @staticmethod
    def _public_landmarks(
        observation: FrameObservation,
    ) -> dict[str, dict[str, float]]:
        output = {
            name: landmark.public_dict()
            for name, landmark in observation.landmarks.items()
        }
        output.update(
            {
                name: point.public_dict()
                for name, point in observation.points.items()
            }
        )
        return output

    def _update_alert(
        self,
        correction: Optional[Correction],
        timestamp_ms: int,
    ) -> bool:
        if correction is None:
            self._reset_correction_episode()
            return False
        if self._active_instruction != correction.instruction:
            self._active_instruction = correction.instruction
            self._bad_since_ms = timestamp_ms
            self._alerted = False
            return False
        if self._bad_since_ms is None:
            self._bad_since_ms = timestamp_ms
        if (
            not self._alerted
            and timestamp_ms - self._bad_since_ms >= self.alert_after_ms
        ):
            self._alerted = True
            return True
        return False

    def _reset_correction_episode(self) -> None:
        self._bad_since_ms = None
        self._active_instruction = None
        self._alerted = False

    def _interrupt_tracking(self, timestamp_ms: int) -> None:
        """Discard stale smoothing and partial calibration after an invalid view."""

        self.smoother.clear()
        self._issues = {key: False for key in self._issues}
        self._reset_correction_episode()
        if self.calibration is not None:
            self.calibration.restart(timestamp_ms)
