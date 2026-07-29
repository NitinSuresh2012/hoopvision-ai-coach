"""FastAPI WebSocket entry point for live sitting-posture correction."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import time
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from .analysis import PostureSession
from .pose_engine import MediaPipePoseEngine, PoseEngineError


app = FastAPI(
    title="HoopVision AI Posture Service",
    version="1.0.0",
    description="Real-time MediaPipe sitting-posture analysis.",
)

# Kept as a replaceable callable so API/protocol tests do not need a camera or
# a MediaPipe graph. Production always uses MediaPipePoseEngine.
pose_engine_factory: Callable[[], MediaPipePoseEngine] = MediaPipePoseEngine


def _now_ms() -> int:
    return int(time.monotonic() * 1_000)


def _origin_allowed(websocket: WebSocket) -> bool:
    configured = {
        item.strip()
        for item in os.getenv("HOOPVISION_ALLOWED_ORIGINS", "").split(",")
        if item.strip()
    }
    if not configured:
        return True
    origin = websocket.headers.get("origin")
    return origin in configured


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "opencv_installed": importlib.util.find_spec("cv2") is not None,
        "mediapipe_installed": (
            importlib.util.find_spec("mediapipe") is not None
        ),
        "protocol": "websocket",
        "endpoint": "/api/posture/live",
    }


async def _send_error(
    websocket: WebSocket,
    code: str,
    message: str,
) -> None:
    await websocket.send_json(
        {"type": "error", "code": code, "message": message}
    )


@app.websocket("/api/posture/live")
async def live_posture(websocket: WebSocket) -> None:
    if not _origin_allowed(websocket):
        await websocket.close(code=1008, reason="Origin is not allowed.")
        return

    await websocket.accept()
    try:
        engine = pose_engine_factory()
    except PoseEngineError as error:
        await _send_error(
            websocket,
            "pose_engine_unavailable",
            str(error),
        )
        await websocket.close(code=1011)
        return

    session = PostureSession()
    await websocket.send_json(
        {
            "type": "ready",
            "calibration": session.calibration_status(_now_ms()),
            "message": "Camera analysis ready",
        }
    )

    try:
        while True:
            try:
                message: dict[str, Any] = await websocket.receive_json()
            except ValueError:
                await _send_error(
                    websocket,
                    "invalid_json",
                    "Send a JSON object.",
                )
                continue

            action = message.get("action")
            timestamp_ms = _now_ms()

            if action == "start_calibration":
                await websocket.send_json(
                    {
                        "type": "calibration",
                        "calibration": session.start_calibration(timestamp_ms),
                    }
                )
                continue

            if action == "finish_calibration":
                try:
                    baseline = session.finish_calibration(timestamp_ms)
                except ValueError as error:
                    await _send_error(
                        websocket,
                        "calibration_incomplete",
                        str(error),
                    )
                else:
                    await websocket.send_json(
                        {
                            "type": "calibration",
                            "calibration": session.calibration_status(
                                timestamp_ms
                            ),
                            "baseline": baseline.public_dict(),
                        }
                    )
                continue

            if action == "reset_calibration":
                session.reset_calibration()
                await websocket.send_json(
                    {
                        "type": "calibration",
                        "calibration": session.calibration_status(
                            timestamp_ms
                        ),
                    }
                )
                continue

            if action != "frame":
                await _send_error(
                    websocket,
                    "unknown_action",
                    "Use start_calibration, finish_calibration, "
                    "reset_calibration, or frame.",
                )
                continue

            image = message.get("image")
            if not isinstance(image, str):
                await _send_error(
                    websocket,
                    "missing_frame",
                    "The frame action requires an image data URL.",
                )
                continue
            try:
                landmarks = await asyncio.to_thread(
                    engine.process_data_url,
                    image,
                    timestamp_ms,
                )
            except PoseEngineError as error:
                await _send_error(
                    websocket,
                    "invalid_frame",
                    str(error),
                )
                continue
            await websocket.send_json(
                session.process(landmarks, timestamp_ms)
            )
    except WebSocketDisconnect:
        pass
    finally:
        engine.close()

