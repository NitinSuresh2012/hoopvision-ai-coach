from __future__ import annotations

import unittest

from backend.app.analysis import Landmark, PostureSession, observe_landmarks


def seated_pose(
    *,
    ear_shift_x: float = 0.0,
    shoulder_shift_x: float = 0.0,
    shoulder_tilt: float = 0.0,
    confidence: float = 0.99,
) -> dict[str, Landmark]:
    """Explicit front/three-quarter seated landmark fixture."""

    return {
        "left_ear": Landmark(
            0.48 + ear_shift_x + shoulder_shift_x,
            0.18,
            visibility=confidence,
            presence=confidence,
        ),
        "right_ear": Landmark(
            0.52 + ear_shift_x + shoulder_shift_x,
            0.18,
            visibility=confidence,
            presence=confidence,
        ),
        "left_shoulder": Landmark(
            0.42 + shoulder_shift_x,
            0.36 - shoulder_tilt / 2,
            visibility=confidence,
            presence=confidence,
        ),
        "right_shoulder": Landmark(
            0.58 + shoulder_shift_x,
            0.36 + shoulder_tilt / 2,
            visibility=confidence,
            presence=confidence,
        ),
        "left_hip": Landmark(
            0.44,
            0.72,
            visibility=confidence,
            presence=confidence,
        ),
        "right_hip": Landmark(
            0.56,
            0.72,
            visibility=confidence,
            presence=confidence,
        ),
    }


def calibrated_session() -> tuple[PostureSession, int]:
    session = PostureSession()
    session.start_calibration(0)
    result: dict[str, object] = {}
    for timestamp_ms in range(0, 3_001, 250):
        result = session.process(seated_pose(), timestamp_ms)
    if result["status_message"] != "Correct posture saved":
        raise AssertionError("Fixture did not complete calibration.")
    return session, 3_000


def warm_scoring(
    session: PostureSession,
    timestamp_ms: int,
    pose: dict[str, Landmark],
) -> tuple[dict[str, object], int]:
    """Fill the complete two-second smoothing window."""

    result: dict[str, object] = {}
    for offset in range(250, 2_251, 250):
        result = session.process(pose, timestamp_ms + offset)
    return result, timestamp_ms + 2_250


class ObservationTests(unittest.TestCase):
    def test_detects_all_requested_metrics(self) -> None:
        observation = observe_landmarks(seated_pose())

        self.assertTrue(observation.valid)
        self.assertIsNotNone(observation.metrics)
        assert observation.metrics is not None
        self.assertAlmostEqual(observation.metrics.neck_angle_deg, 0.0)
        self.assertAlmostEqual(observation.metrics.shoulder_tilt_deg, 0.0)
        self.assertAlmostEqual(
            observation.metrics.upper_back_angle_deg,
            0.0,
        )
        self.assertAlmostEqual(observation.metrics.hip_tilt_deg, 0.0)
        self.assertIn("ear_mid", observation.points)
        self.assertIn("shoulder_mid", observation.points)
        self.assertIn("hip_mid", observation.points)

    def test_level_lines_stay_zero_when_anatomical_sides_are_mirrored(self) -> None:
        pose = seated_pose()
        pose["left_shoulder"] = Landmark(0.58, 0.36)
        pose["right_shoulder"] = Landmark(0.42, 0.36)
        pose["left_hip"] = Landmark(0.56, 0.72)
        pose["right_hip"] = Landmark(0.44, 0.72)

        observation = observe_landmarks(pose)

        self.assertTrue(observation.valid)
        assert observation.metrics is not None
        self.assertAlmostEqual(observation.metrics.shoulder_tilt_deg, 0.0)
        self.assertAlmostEqual(observation.metrics.hip_tilt_deg, 0.0)

    def test_low_confidence_pauses_observation(self) -> None:
        pose = seated_pose()
        pose["left_shoulder"] = Landmark(
            0.42,
            0.36,
            visibility=0.2,
            presence=0.2,
        )

        observation = observe_landmarks(pose)

        self.assertFalse(observation.valid)
        self.assertIn("left_shoulder", observation.missing)
        self.assertIsNone(observation.metrics)

    def test_one_visible_ear_is_enough_for_a_side_view(self) -> None:
        pose = seated_pose()
        pose["right_ear"] = Landmark(
            0.52,
            0.18,
            visibility=0.1,
            presence=0.1,
        )

        observation = observe_landmarks(pose)

        self.assertTrue(observation.valid)
        self.assertIsNotNone(observation.metrics)
        self.assertGreaterEqual(observation.confidence, 0.60)

    def test_both_low_confidence_ears_pause_observation(self) -> None:
        pose = seated_pose()
        for name, x in (("left_ear", 0.48), ("right_ear", 0.52)):
            pose[name] = Landmark(
                x,
                0.18,
                visibility=0.1,
                presence=0.1,
            )

        observation = observe_landmarks(pose)

        self.assertFalse(observation.valid)
        self.assertIn("left_ear", observation.missing)
        self.assertIn("right_ear", observation.missing)

    def test_camera_too_close_warning(self) -> None:
        pose = seated_pose()
        pose["left_shoulder"] = Landmark(0.15, 0.36)
        pose["right_shoulder"] = Landmark(0.85, 0.36)

        warning = observe_landmarks(pose).camera_warning

        self.assertIsNotNone(warning)
        assert warning is not None
        self.assertEqual(warning.code, "camera_too_close")

    def test_camera_too_low_warning(self) -> None:
        pose = seated_pose()
        pose["left_ear"] = Landmark(0.48, 0.03)
        pose["right_ear"] = Landmark(0.52, 0.03)

        warning = observe_landmarks(pose).camera_warning

        self.assertIsNotNone(warning)
        assert warning is not None
        self.assertEqual(warning.code, "camera_too_low")

    def test_camera_too_high_warning(self) -> None:
        pose = {
            "left_ear": Landmark(0.48, 0.46),
            "right_ear": Landmark(0.52, 0.46),
            "left_shoulder": Landmark(0.42, 0.66),
            "right_shoulder": Landmark(0.58, 0.66),
            "left_hip": Landmark(0.44, 0.94),
            "right_hip": Landmark(0.56, 0.94),
        }

        warning = observe_landmarks(pose).camera_warning

        self.assertIsNotNone(warning)
        assert warning is not None
        self.assertEqual(warning.code, "camera_too_high")


class SessionTests(unittest.TestCase):
    def test_calibration_saves_median_baseline_per_session(self) -> None:
        session = PostureSession()
        session.start_calibration(0)
        for timestamp_ms in range(0, 3_001, 250):
            shift = 0.002 if timestamp_ms % 500 else -0.002
            result = session.process(
                seated_pose(ear_shift_x=shift),
                timestamp_ms,
            )

        self.assertIsNotNone(session.baseline)
        assert session.baseline is not None
        self.assertGreaterEqual(session.baseline.sample_count, 12)
        self.assertEqual(result["calibration"]["phase"], "complete")
        self.assertFalse(result["scoring_active"])

    def test_missing_landmark_pauses_existing_score(self) -> None:
        session, timestamp_ms = calibrated_session()
        pose = seated_pose()
        pose["right_hip"] = Landmark(
            0.56,
            0.72,
            visibility=0.1,
            presence=0.1,
        )

        result = session.process(pose, timestamp_ms + 100)

        self.assertFalse(result["scoring_active"])
        self.assertIsNone(result["score"])
        self.assertEqual(result["instruction"], "Move fully into view")
        self.assertFalse(result["play_alert"])

    def test_camera_warning_pauses_scoring_and_calibration(self) -> None:
        session = PostureSession()
        session.start_calibration(0)
        for timestamp_ms in range(0, 1_751, 250):
            session.process(seated_pose(), timestamp_ms)
        pose = seated_pose()
        pose["left_shoulder"] = Landmark(0.15, 0.36)
        pose["right_shoulder"] = Landmark(0.85, 0.36)

        result = session.process(pose, 2_000)

        self.assertFalse(result["scoring_active"])
        self.assertIsNone(result["score"])
        self.assertEqual(
            result["camera_warning"]["code"],
            "camera_too_close",
        )
        self.assertEqual(result["calibration"]["valid_frames"], 0)
        self.assertEqual(result["calibration"]["progress"], 0)
        self.assertFalse(result["play_alert"])

    def test_missing_view_restarts_partial_calibration_hold(self) -> None:
        session = PostureSession()
        session.start_calibration(0)
        for timestamp_ms in range(0, 2_501, 250):
            session.process(seated_pose(), timestamp_ms)
        missing = seated_pose()
        missing["right_hip"] = Landmark(
            0.56,
            0.72,
            visibility=0.1,
            presence=0.1,
        )

        interrupted = session.process(missing, 2_750)

        self.assertEqual(interrupted["calibration"]["valid_frames"], 0)
        self.assertEqual(interrupted["calibration"]["progress"], 0)
        for timestamp_ms in range(3_000, 5_501, 250):
            result = session.process(seated_pose(), timestamp_ms)
        self.assertEqual(result["calibration"]["phase"], "capturing")
        self.assertIsNone(session.baseline)
        result = session.process(seated_pose(), 5_750)
        self.assertEqual(result["calibration"]["phase"], "complete")

    def test_score_waits_for_complete_two_second_window(self) -> None:
        session, timestamp_ms = calibrated_session()

        for offset in range(250, 2_001, 250):
            result = session.process(seated_pose(), timestamp_ms + offset)
            self.assertFalse(result["scoring_active"])
            self.assertIsNone(result["score"])
            self.assertEqual(result["status_message"], "Reading your alignment")

        result = session.process(seated_pose(), timestamp_ms + 2_250)
        self.assertTrue(result["scoring_active"])
        self.assertEqual(result["score"], 100)

    def test_head_forward_only_marks_head_neck_red(self) -> None:
        session, timestamp_ms = calibrated_session()

        result, _ = warm_scoring(
            session,
            timestamp_ms,
            seated_pose(ear_shift_x=0.16),
        )

        self.assertEqual(result["instruction"], "Move your head back")
        self.assertEqual(result["areas"]["head_neck"], "red")
        self.assertEqual(result["areas"]["shoulders"], "green")
        self.assertEqual(result["areas"]["upper_back"], "green")
        self.assertLess(result["area_scores"]["head"], 100)
        self.assertEqual(result["area_scores"]["shoulders"], 100)
        self.assertEqual(
            {
                segment["area"]
                for segment in result["segments"]
                if segment["status"] == "red"
            },
            {"head_neck"},
        )
        self.assertGreaterEqual(result["score"], 0)
        self.assertLessEqual(result["score"], 100)

    def test_uneven_shoulders_only_marks_shoulders_red(self) -> None:
        session, timestamp_ms = calibrated_session()

        result, _ = warm_scoring(
            session,
            timestamp_ms,
            seated_pose(shoulder_tilt=0.10),
        )

        self.assertEqual(result["instruction"], "Level your shoulders")
        self.assertEqual(result["areas"]["shoulders"], "red")
        self.assertEqual(result["areas"]["head_neck"], "green")

    def test_slouch_marks_upper_back_and_says_sit_up(self) -> None:
        session, timestamp_ms = calibrated_session()

        result, _ = warm_scoring(
            session,
            timestamp_ms,
            seated_pose(shoulder_shift_x=0.13),
        )

        self.assertEqual(result["instruction"], "Sit up")
        self.assertEqual(result["areas"]["upper_back"], "red")
        self.assertEqual(result["areas"]["head_neck"], "green")

    def test_two_second_smoothing_rejects_single_bad_frame(self) -> None:
        session, timestamp_ms = calibrated_session()
        result, timestamp_ms = warm_scoring(
            session,
            timestamp_ms,
            seated_pose(),
        )
        self.assertEqual(result["areas"]["head_neck"], "green")

        result = session.process(
            seated_pose(ear_shift_x=0.16),
            timestamp_ms + 250,
        )

        self.assertEqual(result["areas"]["head_neck"], "green")
        self.assertIsNone(result["instruction"])
        self.assertEqual(result["smoothing_window_ms"], 2_000)

    def test_alert_fires_once_after_three_continuous_bad_seconds(self) -> None:
        session, timestamp_ms = calibrated_session()
        result, timestamp_ms = warm_scoring(
            session,
            timestamp_ms,
            seated_pose(ear_shift_x=0.16),
        )
        self.assertEqual(result["instruction"], "Move your head back")
        alert_results: list[tuple[int, bool]] = []
        for offset in range(250, 4_001, 250):
            result = session.process(
                seated_pose(ear_shift_x=0.16),
                timestamp_ms + offset,
            )
            alert_results.append((offset, bool(result["play_alert"])))

        before_three_seconds = [
            played for offset, played in alert_results if offset < 3_000
        ]
        self.assertFalse(any(before_three_seconds))
        self.assertEqual(
            sum(1 for _, played in alert_results if played),
            1,
        )

    def test_tracking_interruption_restarts_bad_posture_alert_timer(
        self,
    ) -> None:
        session, timestamp_ms = calibrated_session()
        _, timestamp_ms = warm_scoring(
            session,
            timestamp_ms,
            seated_pose(ear_shift_x=0.16),
        )
        for offset in range(250, 2_751, 250):
            result = session.process(
                seated_pose(ear_shift_x=0.16),
                timestamp_ms + offset,
            )
            self.assertFalse(result["play_alert"])

        missing = seated_pose()
        missing["right_hip"] = Landmark(
            0.56,
            0.72,
            visibility=0.1,
            presence=0.1,
        )
        timestamp_ms += 3_000
        paused = session.process(missing, timestamp_ms)
        self.assertFalse(paused["play_alert"])

        _, timestamp_ms = warm_scoring(
            session,
            timestamp_ms,
            seated_pose(ear_shift_x=0.16),
        )
        for offset in range(250, 3_000, 250):
            result = session.process(
                seated_pose(ear_shift_x=0.16),
                timestamp_ms + offset,
            )
            self.assertFalse(result["play_alert"])
        result = session.process(
            seated_pose(ear_shift_x=0.16),
            timestamp_ms + 3_000,
        )
        self.assertTrue(result["play_alert"])

    def test_good_posture_is_green_and_scores_100(self) -> None:
        session, timestamp_ms = calibrated_session()

        result, _ = warm_scoring(session, timestamp_ms, seated_pose())

        self.assertEqual(result["score"], 100)
        self.assertTrue(
            all(status == "green" for status in result["areas"].values())
        )
        self.assertTrue(
            all(score == 100 for score in result["area_scores"].values())
        )
        self.assertIsNone(result["instruction"])

    def test_missing_view_clears_score_and_requires_new_smoothing_window(
        self,
    ) -> None:
        session, timestamp_ms = calibrated_session()
        active, timestamp_ms = warm_scoring(
            session,
            timestamp_ms,
            seated_pose(),
        )
        self.assertTrue(active["scoring_active"])
        missing = seated_pose()
        missing["left_shoulder"] = Landmark(
            0.42,
            0.36,
            visibility=0.1,
            presence=0.1,
        )
        paused = session.process(missing, timestamp_ms + 250)
        self.assertFalse(paused["scoring_active"])

        for offset in range(500, 2_251, 250):
            reading = session.process(seated_pose(), timestamp_ms + offset)
            self.assertFalse(reading["scoring_active"])
        resumed = session.process(seated_pose(), timestamp_ms + 2_500)
        self.assertTrue(resumed["scoring_active"])


if __name__ == "__main__":
    unittest.main()
