# HoopVision sitting-posture service

This is the real-time Python analysis service for the existing **Sitting
Posture Correction** mode. It does not generate demo scores. Every score,
colored segment, and correction comes from MediaPipe pose landmarks detected
in the current camera frame.

## Run

Use Python 3.11, then from the repository root:

```powershell
python -m venv backend/.venv
backend/.venv/Scripts/Activate.ps1
python -m pip install -r backend/requirements.txt
python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000
```

For a deployed frontend, set `HOOPVISION_ALLOWED_ORIGINS` to a comma-separated
list of the exact HTTPS origins allowed to open the WebSocket. If the variable
is omitted, origins are not filtered (convenient for local development).

## Browser protocol

Connect the existing camera UI to:

```text
ws://127.0.0.1:8000/api/posture/live
```

Use `wss://` when the site is on HTTPS. The server sends a `ready` event. Start
the three-second calibration and stream frames:

```json
{"action": "start_calibration"}
{"action": "frame", "image": "data:image/jpeg;base64,..."}
```

Send frames continuously during and after calibration (10–15 FPS is enough).
Calibration auto-saves after three seconds and at least twelve valid frames.
It is intentionally kept only in the current WebSocket session.

Optional controls:

```json
{"action": "finish_calibration"}
{"action": "reset_calibration"}
```

`finish_calibration` succeeds only when the normal duration/frame minimum is
met. A frame response includes:

- `score`: 0–100, or `null` while scoring is paused.
- `instruction`: at most one of `Sit up`, `Move your head back`, or
  `Level your shoulders`.
- `areas` and `segments`: independently green/red regions for the overlay.
- `area_scores`: real 0–100 head, shoulder, upper-back, and hip sub-scores.
- `landmarks`: normalized coordinates for the current skeleton.
- `camera_warning`: too low, too high, or too close when detected.
- `play_alert`: becomes `true` once after the same bad-posture correction has
  remained active for three seconds. The browser should play one quiet chime
  only on that event; audio playback must begin after a user gesture.
- `calibration`: current calibration phase and progress.

If required landmarks are missing or below 0.60 confidence, `score` is `null`,
`scoring_active` is `false`, and the instruction is `Move fully into view`.
Both shoulders and both hips must be reliable; one reliable ear is sufficient
for a side-facing seated view. After calibration—or after tracking is
interrupted—the service collects a fresh two-second window before activating
the score.

## Analysis behavior

The baseline is the median of the calibration frames. Neck angle/head offset,
shoulder level, upper-back (shoulder-to-hip) angle, and hip level are measured
relative to that baseline. Red/green decisions use a two-second rolling mean
plus hysteresis to prevent flicker. Only the body area whose measurement is
outside its calibrated tolerance turns red.
Missing landmarks or unsafe camera framing discard a partial calibration, so
the saved baseline always comes from one continuous valid hold.

Camera height warnings are upper-body framing safeguards derived from the
normalized ear/shoulder/hip placement. Monocular pose landmarks cannot recover
the camera's physical pitch exactly. Upper-back alignment is likewise an
estimate from the shoulder-to-hip line; MediaPipe Pose has no thoracic-spine
landmarks.

The bundled browser runtime automatically uses this service when the site is
opened on `localhost` or `127.0.0.1`. For a hosted Python service, set
`window.HOOPVISION_POSTURE_WS_URL` to its exact `wss://.../api/posture/live`
URL before `scripts/posture-runtime.js` runs. If no Python endpoint is
configured, the current UI falls back to real on-device MediaPipe Tasks
inference; it never substitutes demo scores. The static Sites host cannot
execute native Python/OpenCV itself, so the Python service must run on a
separate Python container or VM.

## Test

```powershell
python -m unittest discover -s backend/tests -v
```

The pure tests use explicit landmarks and do not need OpenCV, MediaPipe, a
camera, or synthetic runtime scores. The optional WebSocket test runs when the
FastAPI test dependencies are installed.
