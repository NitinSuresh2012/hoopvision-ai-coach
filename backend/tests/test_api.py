from __future__ import annotations

import importlib.util
import unittest


FASTAPI_TESTS_AVAILABLE = (
    importlib.util.find_spec("fastapi") is not None
    and importlib.util.find_spec("httpx") is not None
)


@unittest.skipUnless(
    FASTAPI_TESTS_AVAILABLE,
    "FastAPI/httpx test dependencies are not installed.",
)
class WebSocketProtocolTests(unittest.TestCase):
    def test_ready_calibration_and_missing_pose_protocol(self) -> None:
        from fastapi.testclient import TestClient

        from backend.app import main

        class ExplicitNoPoseEngine:
            def process_data_url(self, image: str, timestamp_ms: int):
                self.last_input = (image, timestamp_ms)
                return {}

            def close(self) -> None:
                pass

        original_factory = main.pose_engine_factory
        main.pose_engine_factory = ExplicitNoPoseEngine
        try:
            with TestClient(main.app) as client:
                with client.websocket_connect(
                    "/api/posture/live"
                ) as websocket:
                    ready = websocket.receive_json()
                    self.assertEqual(ready["type"], "ready")
                    self.assertEqual(
                        ready["calibration"]["phase"],
                        "required",
                    )

                    websocket.send_json({"action": "start_calibration"})
                    calibration = websocket.receive_json()
                    self.assertEqual(
                        calibration["calibration"]["phase"],
                        "capturing",
                    )

                    websocket.send_json(
                        {
                            "action": "frame",
                            "image": "data:image/jpeg;base64,fixture",
                        }
                    )
                    result = websocket.receive_json()
                    self.assertEqual(
                        result["instruction"],
                        "Move fully into view",
                    )
                    self.assertFalse(result["scoring_active"])
                    self.assertIsNone(result["score"])
        finally:
            main.pose_engine_factory = original_factory

    def test_live_protocol_calibrates_warms_scores_and_alerts(self) -> None:
        from fastapi.testclient import TestClient

        from backend.app import main
        from backend.tests.test_analysis import seated_pose

        class FixturePoseEngine:
            def __init__(self) -> None:
                self.landmarks = seated_pose()

            def process_data_url(self, image: str, timestamp_ms: int):
                self.last_input = (image, timestamp_ms)
                return self.landmarks

            def close(self) -> None:
                pass

        clock = {"now": 0}
        engine = FixturePoseEngine()
        original_factory = main.pose_engine_factory
        original_clock = main._now_ms
        main.pose_engine_factory = lambda: engine
        main._now_ms = lambda: clock["now"]
        try:
            with TestClient(main.app) as client:
                with client.websocket_connect(
                    "/api/posture/live"
                ) as websocket:
                    self.assertEqual(
                        websocket.receive_json()["type"],
                        "ready",
                    )
                    websocket.send_json({"action": "start_calibration"})
                    self.assertEqual(
                        websocket.receive_json()["calibration"]["phase"],
                        "capturing",
                    )

                    for clock["now"] in range(0, 3_001, 250):
                        websocket.send_json(
                            {"action": "frame", "image": "fixture"}
                        )
                        result = websocket.receive_json()
                    self.assertEqual(
                        result["calibration"]["phase"],
                        "complete",
                    )
                    self.assertFalse(result["scoring_active"])

                    engine.landmarks = seated_pose(ear_shift_x=0.16)
                    for clock["now"] in range(3_250, 5_501, 250):
                        websocket.send_json(
                            {"action": "frame", "image": "fixture"}
                        )
                        result = websocket.receive_json()
                    self.assertTrue(result["scoring_active"])
                    self.assertEqual(
                        result["instruction"],
                        "Move your head back",
                    )
                    self.assertEqual(
                        {
                            segment["area"]
                            for segment in result["segments"]
                            if segment["status"] == "red"
                        },
                        {"head_neck"},
                    )
                    self.assertFalse(result["play_alert"])

                    alerts = 0
                    for clock["now"] in range(5_750, 8_501, 250):
                        websocket.send_json(
                            {"action": "frame", "image": "fixture"}
                        )
                        result = websocket.receive_json()
                        alerts += int(bool(result["play_alert"]))
                    self.assertEqual(alerts, 1)
        finally:
            main.pose_engine_factory = original_factory
            main._now_ms = original_clock


if __name__ == "__main__":
    unittest.main()
