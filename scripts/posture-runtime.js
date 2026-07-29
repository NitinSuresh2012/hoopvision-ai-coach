(() => {
  "use strict";

  const MEDIAPIPE_VERSION = "0.10.35";
  const VISION_MODULE =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@" +
    MEDIAPIPE_VERSION +
    "/vision_bundle.mjs";
  const WASM_ROOT =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@" +
    MEDIAPIPE_VERSION +
    "/wasm";
  const MODEL_URL =
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
  const CALIBRATION_KEY = "hoopvision-posture-baseline-v1";
  const REQUIRED_CONFIDENCE = 0.5;
  const HIP_CONFIDENCE = 0.4;
  const PARTIAL_CONFIDENCE = 0.28;
  const INFERENCE_INTERVAL_MS = 90;
  const SMOOTHING_WINDOW_MS = 2000;
  const CALIBRATION_HOLD_MS = 2000;
  const ALERT_AFTER_MS = 3000;
  const INVALID_GRACE_MS = 650;
  const PYTHON_BACKEND_URL =
    String(window.HOOPVISION_POSTURE_WS_URL || "").trim() ||
    (location.hostname === "127.0.0.1" || location.hostname === "localhost"
      ? "ws://127.0.0.1:8000/api/posture/live"
      : "");

  const state = {
    mode: "basketball",
    cameraOn: false,
    landmarker: null,
    loadPromise: null,
    runToken: 0,
    frameRequest: 0,
    lastInferenceAt: 0,
    invalidSince: 0,
    validSince: 0,
    baseline: null,
    calibrating: false,
    calibrationStartedAt: 0,
    calibrationValidMs: 0,
    calibrationLastAt: 0,
    calibrationSamples: [],
    metricWindow: [],
    smoothLandmarks: null,
    lastValidLandmarks: null,
    lastValidMetrics: null,
    lastWarning: "",
    lastMissing: "",
    failState: { head: false, shoulders: false, back: false },
    activeGroup: null,
    activeGroupSince: 0,
    badSince: 0,
    alerted: false,
    soundEnabled: true,
    audioContext: null,
    uiReady: false,
    backendSocket: null,
    backendReady: false,
    backendAwaitingFrame: false,
    backendCalibrated: false,
    backendCaptureCanvas: null,
    backendLastFrameAt: 0,
    backendFailed: false,
  };

  const $ = (id) => document.getElementById(id);
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  const distance = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  const midpoint = (a, b) => ({
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    z: ((a.z || 0) + (b.z || 0)) / 2,
  });
  const degrees = (radians) => (radians * 180) / Math.PI;
  const median = (values) => {
    const sorted = values.filter(Number.isFinite).sort((a, b) => a - b);
    if (!sorted.length) return 0;
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2
      ? sorted[middle]
      : (sorted[middle - 1] + sorted[middle]) / 2;
  };
  const ramp = (value, good, bad) =>
    clamp((value - good) / Math.max(0.0001, bad - good), 0, 1);
  const confidence = (landmark) =>
    Math.min(landmark?.visibility ?? 1, landmark?.presence ?? 1);

  function ensureUi() {
    if (state.uiReady) return;
    const stage = document.querySelector(".camera-stage");
    const cameraButton = $("cameraButton");
    if (!stage || !cameraButton) return;

    let canvas = $("postureCanvas");
    if (!canvas) {
      canvas = document.createElement("canvas");
      canvas.id = "postureCanvas";
      canvas.className = "posture-canvas";
      canvas.hidden = true;
      canvas.setAttribute("aria-hidden", "true");
      stage.appendChild(canvas);
    }

    let setup = $("postureSetup");
    if (!setup) {
      setup = document.createElement("div");
      setup.id = "postureSetup";
      setup.className = "posture-setup";
      setup.hidden = true;
      setup.innerHTML =
        '<i aria-hidden="true"></i><span id="postureSetupText">Positioning camera</span>';
      stage.appendChild(setup);
    }

    let calibrate = $("calibrateButton");
    if (!calibrate) {
      calibrate = document.createElement("button");
      calibrate.id = "calibrateButton";
      calibrate.className = "button button-ghost posture-calibrate";
      calibrate.type = "button";
      calibrate.hidden = true;
      calibrate.disabled = true;
      calibrate.textContent = "Save correct posture";
      cameraButton.insertAdjacentElement("afterend", calibrate);
    }
    calibrate.addEventListener("click", beginCalibration);
    cameraButton.addEventListener("click", unlockAudio, { capture: true });

    const voiceButton = $("voiceButton");
    if (voiceButton) {
      voiceButton.addEventListener("click", () => {
        window.setTimeout(() => {
          state.soundEnabled = voiceButton.getAttribute("aria-pressed") !== "false";
          if (state.soundEnabled) unlockAudio();
        }, 0);
      });
    }

    state.uiReady = true;
  }

  function loadSavedCalibration() {
    state.baseline = null;
    try {
      const saved = JSON.parse(localStorage.getItem(CALIBRATION_KEY) || "null");
      const stored = saved?.metrics || saved;
      const metrics = stored && Number.isFinite(stored.torsoAngle) && !Number.isFinite(stored.torsoLean)
        ? { ...stored, torsoLean: stored.torsoAngle }
        : stored;
      if (
        metrics &&
        ["neckAngle", "headDepth", "shoulderTilt", "torsoLean", "torsoRatio"].every(
          (key) => Number.isFinite(metrics[key]),
        )
      ) {
        state.baseline = metrics;
      }
    } catch {
      state.baseline = null;
    }
  }

  function saveCalibration(metrics) {
    try {
      localStorage.setItem(
        CALIBRATION_KEY,
        JSON.stringify({ ...metrics, torsoAngle: metrics.torsoLean }),
      );
    } catch {
      // Coaching still works if private browsing prevents local persistence.
    }
  }

  function enterPosture(cameraOn) {
    ensureUi();
    state.mode = "posture";
    state.cameraOn = Boolean(cameraOn);
    state.soundEnabled = $("voiceButton")?.getAttribute("aria-pressed") !== "false";
    document.body.classList.add("posture-mode");
    loadSavedCalibration();
    resetAnalysis();
    setPostureLabels();
    const skeleton = document.querySelector(".skeleton-overlay");
    if (skeleton) skeleton.hidden = true;
    const canvas = $("postureCanvas");
    if (canvas) canvas.hidden = false;
    const setup = $("postureSetup");
    if (setup) setup.hidden = false;
    const calibrate = $("calibrateButton");
    if (calibrate) {
      calibrate.hidden = !state.cameraOn;
      calibrate.disabled = true;
      calibrate.textContent = state.cameraOn
        ? "Show ears, shoulders & hips"
        : state.baseline
          ? "Recalibrate posture"
          : "Save correct posture";
    }
    pauseScore(
      state.cameraOn
        ? "Step 1 of 3 · Loading posture tracking…"
        : state.baseline
          ? "Calibration saved · Start the camera"
          : "Step 1 of 3 · Start the camera and show your upper body",
      "setup",
    );
    document.body.dataset.postureModel = state.landmarker ? "ready" : "loading";
    if (state.cameraOn) {
      startPoseLoop();
    } else if (!state.landmarker) {
      loadLandmarker()
        .then(() => {
          document.body.dataset.postureModel = "ready";
        })
        .catch(() => {
          document.body.dataset.postureModel = "error";
        });
    }
  }

  function leavePosture() {
    state.mode = "basketball";
    state.cameraOn = false;
    state.runToken += 1;
    closeBackend();
    if (state.frameRequest) cancelAnimationFrame(state.frameRequest);
    state.frameRequest = 0;
    resetAnalysis();
    document.body.classList.remove("posture-mode");
    restoreLabels();
    const skeleton = document.querySelector(".skeleton-overlay");
    if (skeleton) skeleton.hidden = false;
    const canvas = $("postureCanvas");
    if (canvas) {
      canvas.hidden = true;
      clearCanvas();
    }
    const setup = $("postureSetup");
    if (setup) setup.hidden = true;
    const calibrate = $("calibrateButton");
    if (calibrate) calibrate.hidden = true;
  }

  function setPostureLabels() {
    const names = ["Neck", "Shoulders", "Upper back", "Alignment"];
    document.querySelectorAll(".metrics > div").forEach((row, index) => {
      const label = row.querySelector("span");
      if (!label) return;
      if (!label.dataset.originalLabel) label.dataset.originalLabel = label.textContent || "";
      label.textContent = names[index] || label.textContent;
    });
    const repStat = $("repCount")?.closest("article");
    const repLabel = repStat?.querySelector("span");
    if (repLabel) {
      if (!repLabel.dataset.originalLabel) repLabel.dataset.originalLabel = repLabel.textContent || "";
      repLabel.textContent = "STATUS";
    }
    const voiceLabel = $("voiceButton")?.querySelector("span");
    if (voiceLabel) {
      if (!voiceLabel.dataset.originalLabel) voiceLabel.dataset.originalLabel = voiceLabel.textContent || "";
      voiceLabel.textContent = "ALERT";
    }
    const scoreLabel = document.querySelector(".score-ring > span");
    if (scoreLabel) {
      if (!scoreLabel.dataset.originalLabel) scoreLabel.dataset.originalLabel = scoreLabel.textContent || "";
      scoreLabel.textContent = "POSTURE SCORE";
    }
    const sectionCopy = document.querySelector(".live-section .section-heading > p");
    if (sectionCopy) {
      if (!sectionCopy.dataset.originalCopy) sectionCopy.dataset.originalCopy = sectionCopy.textContent || "";
      sectionCopy.textContent =
        "MediaPipe tracks your ears, shoulders, hips, neck, and upper back in real time. Camera frames are processed privately and are never saved.";
    }
    const placeholderTitle = document.querySelector("#cameraPlaceholder h3");
    const placeholderCopy = document.querySelector("#cameraPlaceholder p");
    if (placeholderTitle) {
      if (!placeholderTitle.dataset.originalCopy) placeholderTitle.dataset.originalCopy = placeholderTitle.textContent || "";
      placeholderTitle.textContent = "Posture Coach Ready";
    }
    if (placeholderCopy) {
      if (!placeholderCopy.dataset.originalCopy) placeholderCopy.dataset.originalCopy = placeholderCopy.textContent || "";
      placeholderCopy.textContent = "Start the camera, sit fully in view, then save your correct posture.";
    }
    const tracking = document.querySelector(".skill-tracking");
    if (tracking) tracking.hidden = true;
    const hearCue = $("hearCue");
    if (hearCue) hearCue.hidden = true;
    const tips = document.querySelectorAll(".tip-stack p");
    const copies = [
      "Green means your saved alignment is on target.",
      "Only the body area that needs correction turns red.",
    ];
    tips.forEach((tip, index) => {
      if (!tip.dataset.originalCopy) tip.dataset.originalCopy = tip.textContent || "";
      if (copies[index]) tip.textContent = copies[index];
    });
  }

  function restoreLabels() {
    document.querySelectorAll(".metrics > div span").forEach((label) => {
      if (label.dataset.originalLabel) label.textContent = label.dataset.originalLabel;
    });
    const repStat = $("repCount")?.closest("article");
    const repLabel = repStat?.querySelector("span");
    if (repLabel?.dataset.originalLabel) repLabel.textContent = repLabel.dataset.originalLabel;
    const voiceLabel = $("voiceButton")?.querySelector("span");
    if (voiceLabel?.dataset.originalLabel) voiceLabel.textContent = voiceLabel.dataset.originalLabel;
    const scoreLabel = document.querySelector(".score-ring > span");
    if (scoreLabel?.dataset.originalLabel) scoreLabel.textContent = scoreLabel.dataset.originalLabel;
    const sectionCopy = document.querySelector(".live-section .section-heading > p");
    if (sectionCopy?.dataset.originalCopy) sectionCopy.textContent = sectionCopy.dataset.originalCopy;
    const placeholderTitle = document.querySelector("#cameraPlaceholder h3");
    const placeholderCopy = document.querySelector("#cameraPlaceholder p");
    if (placeholderTitle?.dataset.originalCopy) placeholderTitle.textContent = placeholderTitle.dataset.originalCopy;
    if (placeholderCopy?.dataset.originalCopy) placeholderCopy.textContent = placeholderCopy.dataset.originalCopy;
    const tracking = document.querySelector(".skill-tracking");
    if (tracking) tracking.hidden = false;
    const hearCue = $("hearCue");
    if (hearCue) hearCue.hidden = false;
    document.querySelectorAll(".tip-stack p").forEach((tip) => {
      if (tip.dataset.originalCopy) tip.textContent = tip.dataset.originalCopy;
    });
    const ring = document.querySelector(".score-ring");
    ring?.classList.remove("posture-good", "posture-bad", "posture-paused");
  }

  function resetAnalysis() {
    state.calibrating = false;
    state.calibrationValidMs = 0;
    state.calibrationLastAt = 0;
    state.calibrationSamples = [];
    state.metricWindow = [];
    state.smoothLandmarks = null;
    state.lastValidLandmarks = null;
    state.lastValidMetrics = null;
    state.lastWarning = "";
    state.lastMissing = "";
    state.invalidSince = 0;
    state.validSince = 0;
    state.failState = { head: false, shoulders: false, back: false };
    resetBadEpisode();
  }

  async function loadLandmarker() {
    if (state.landmarker) return state.landmarker;
    if (state.loadPromise) return state.loadPromise;
    state.loadPromise = (async () => {
      const vision = await import(VISION_MODULE);
      const files = await vision.FilesetResolver.forVisionTasks(WASM_ROOT);
      const options = {
        baseOptions: {
          modelAssetPath: MODEL_URL,
          delegate: "GPU",
        },
        runningMode: "VIDEO",
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
        outputSegmentationMasks: false,
      };
      try {
        state.landmarker = await vision.PoseLandmarker.createFromOptions(files, options);
      } catch {
        options.baseOptions.delegate = "CPU";
        state.landmarker = await vision.PoseLandmarker.createFromOptions(files, options);
      }
      document.body.dataset.postureModel = "ready";
      return state.landmarker;
    })();
    try {
      return await state.loadPromise;
    } finally {
      state.loadPromise = null;
    }
  }

  function connectBackend(token) {
    return new Promise((resolve) => {
      let settled = false;
      let socket;
      const finish = (connected) => {
        if (settled) return;
        settled = true;
        window.clearTimeout(timeout);
        resolve(connected);
      };
      const timeout = window.setTimeout(() => {
        state.backendFailed = true;
        try {
          socket?.close();
        } catch {
          // The browser fallback starts below.
        }
        finish(false);
      }, 2200);
      try {
        socket = new WebSocket(PYTHON_BACKEND_URL);
      } catch {
        state.backendFailed = true;
        finish(false);
        return;
      }
      socket.addEventListener("open", () => {
        if (token !== state.runToken || !state.cameraOn) {
          socket.close();
          finish(false);
          return;
        }
        state.backendSocket = socket;
        state.backendFailed = false;
        finish(true);
      });
      socket.addEventListener("message", (event) => {
        let message;
        try {
          message = JSON.parse(event.data);
        } catch {
          return;
        }
        handleBackendMessage(message);
      });
      socket.addEventListener("error", () => {
        state.backendFailed = true;
        finish(false);
      });
      socket.addEventListener("close", () => {
        const wasActive = state.backendSocket === socket;
        if (wasActive) {
          state.backendSocket = null;
          state.backendReady = false;
          state.backendAwaitingFrame = false;
        }
        if (
          wasActive &&
          state.cameraOn &&
          state.mode === "posture" &&
          token === state.runToken
        ) {
          state.backendFailed = true;
          pauseScore("Python service disconnected. Switching to on-device tracking…", "loading");
          startPoseLoop();
        }
      });
    });
  }

  function closeBackend() {
    const socket = state.backendSocket;
    state.backendSocket = null;
    state.backendReady = false;
    state.backendAwaitingFrame = false;
    state.backendCalibrated = false;
    if (socket && socket.readyState < WebSocket.CLOSING) {
      socket.close(1000, "Camera stopped");
    }
  }

  function startBackendFrames(token) {
    state.backendLastFrameAt = 0;
    const frame = (now) => {
      if (
        token !== state.runToken ||
        state.mode !== "posture" ||
        !state.cameraOn ||
        !state.backendSocket
      ) {
        return;
      }
      const video = $("cameraVideo");
      const socket = state.backendSocket;
      if (
        state.backendReady &&
        !state.backendAwaitingFrame &&
        socket.readyState === WebSocket.OPEN &&
        socket.bufferedAmount < 250_000 &&
        video?.readyState >= 2 &&
        video.videoWidth > 0 &&
        now - state.backendLastFrameAt >= 110
      ) {
        state.backendLastFrameAt = now;
        const maximumWidth = 720;
        const scale = Math.min(1, maximumWidth / video.videoWidth);
        const width = Math.max(64, Math.round(video.videoWidth * scale));
        const height = Math.max(64, Math.round(video.videoHeight * scale));
        if (!state.backendCaptureCanvas) {
          state.backendCaptureCanvas = document.createElement("canvas");
        }
        const capture = state.backendCaptureCanvas;
        if (capture.width !== width || capture.height !== height) {
          capture.width = width;
          capture.height = height;
        }
        const context = capture.getContext("2d", { alpha: false });
        context.drawImage(video, 0, 0, width, height);
        state.backendAwaitingFrame = true;
        socket.send(
          JSON.stringify({
            action: "frame",
            image: capture.toDataURL("image/jpeg", 0.72),
          }),
        );
      }
      state.frameRequest = requestAnimationFrame(frame);
    };
    state.frameRequest = requestAnimationFrame(frame);
  }

  function handleBackendMessage(message) {
    if (message.type === "ready") {
      state.backendReady = true;
      state.backendCalibrated = false;
      state.backendAwaitingFrame = false;
      state.baseline = null;
      const calibrate = $("calibrateButton");
      if (calibrate) {
        calibrate.disabled = true;
        calibrate.textContent = "Show ears, shoulders & hips";
      }
      pauseScore("Step 1 of 3 · Move fully into view", "ready");
      setSetup("Step 1 of 3 · Python posture tracking ready", "ready");
      return;
    }
    if (message.type === "error") {
      state.backendAwaitingFrame = false;
      console.error("HoopVision Python posture service:", message.message || message.code);
      return;
    }
    if (message.type === "calibration") {
      renderBackendCalibration(message.calibration);
      return;
    }
    if (message.type !== "posture_result") return;
    state.backendAwaitingFrame = false;
    const calibration = message.calibration || {};
    if (calibration.phase === "capturing") {
      state.calibrating = true;
      renderBackendCalibration(calibration);
    } else if (calibration.phase === "complete") {
      state.calibrating = false;
      state.backendCalibrated = true;
      const calibrate = $("calibrateButton");
      if (calibrate) {
        calibrate.disabled = false;
        calibrate.textContent = "Recalibrate posture";
      }
    }

    drawBackendSkeleton(message.landmarks || {}, message.segments || []);
    if (!message.scoring_active || message.score == null) {
      const warning = message.camera_warning?.message;
      const detected = message.landmarks || {};
      const fullPoseReady =
        detected.left_shoulder &&
        detected.right_shoulder &&
        detected.left_hip &&
        detected.right_hip &&
        (detected.left_ear || detected.right_ear);
      const calibrate = $("calibrateButton");
      if (calibrate && calibration.phase !== "capturing") {
        calibrate.disabled = Boolean(warning) || !fullPoseReady;
        calibrate.textContent = fullPoseReady
          ? calibration.phase === "complete"
            ? "Recalibrate posture"
            : "Save correct posture"
          : message.missing_landmarks?.includes("left_hip") ||
              message.missing_landmarks?.includes("right_hip")
            ? "Move back to show hips"
            : "Show ears, shoulders & hips";
      }
      const setupMessage = warning
        ? warning
        : calibration.phase === "complete" && fullPoseReady
          ? "Step 3 of 3 · Reading alignment"
        : fullPoseReady && calibration.phase !== "complete"
          ? "Step 2 of 3 · Pose found · Save your posture"
          : message.instruction || message.status_message || "Step 1 of 3 · Move fully into view";
      pauseScore(
        setupMessage,
        warning
          ? "warning"
          : calibration.phase === "capturing"
            ? "calibrating"
            : fullPoseReady
              ? calibration.phase === "complete"
                ? "reading"
                : "ready"
              : "missing",
      );
      if (calibration.phase === "capturing") {
        renderBackendCalibration(calibration);
      }
      return;
    }

    const instruction = message.instruction || "Posture aligned";
    const cue = $("liveCue");
    if (cue) cue.textContent = instruction;
    const score = $("overallScore");
    if (score) score.textContent = String(message.score);
    const rows = Array.from(document.querySelectorAll(".metrics > div"));
    const areaScores = message.area_scores || {};
    [
      areaScores.head,
      areaScores.shoulders,
      areaScores.back,
      message.score,
    ].forEach((value, index) => setMetricValue(rows[index], Number.isFinite(value) ? value : message.score));
    const hasCorrection = Boolean(message.instruction);
    const ring = document.querySelector(".score-ring");
    ring?.classList.remove("posture-paused", "posture-good", "posture-bad");
    ring?.classList.add(hasCorrection ? "posture-bad" : "posture-good");
    const liveState = $("liveState");
    if (liveState) {
      liveState.textContent = hasCorrection ? "ADJUST" : "ALIGNED";
      liveState.classList.toggle("online", !hasCorrection);
      liveState.classList.toggle("needs-adjustment", hasCorrection);
    }
    const rep = $("repCount");
    if (rep) rep.textContent = "LIVE";
    setSetup(hasCorrection ? "Correction detected" : "Alignment locked", hasCorrection ? "bad" : "good");
    if (message.play_alert && state.soundEnabled) playSoftAlert();
  }

  function renderBackendCalibration(calibration) {
    if (!calibration) return;
    const progress = clamp(Math.round((calibration.progress || 0) * 100), 0, 100);
    const calibrate = $("calibrateButton");
    if (calibration.phase === "capturing") {
      state.calibrating = true;
      if (calibrate) {
        calibrate.disabled = true;
        calibrate.textContent = "Hold still · " + progress + "%";
      }
      setSetup("Calibrating · " + progress + "%", "calibrating");
      return;
    }
    if (calibration.phase === "complete") {
      state.calibrating = false;
      state.backendCalibrated = true;
      if (calibrate) {
        calibrate.disabled = false;
        calibrate.textContent = "Recalibrate posture";
      }
      setSetup("Calibration saved", "good");
    }
  }

  function drawBackendSkeleton(landmarks, segments) {
    const canvas = $("postureCanvas");
    const video = $("cameraVideo");
    if (!canvas || !video || canvas.hidden) return;
    const stage = canvas.parentElement;
    const width = stage.clientWidth;
    const height = stage.clientHeight;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      canvas.style.width = width + "px";
      canvas.style.height = height + "px";
    }
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    if (!video.videoWidth || !video.videoHeight) return;
    const scale = Math.max(width / video.videoWidth, height / video.videoHeight);
    const renderedWidth = video.videoWidth * scale;
    const renderedHeight = video.videoHeight * scale;
    const cropX = (renderedWidth - width) / 2;
    const cropY = (renderedHeight - height) / 2;
    const map = (point) => ({
      x: point.x * renderedWidth - cropX,
      y: point.y * renderedHeight - cropY,
    });
    const colors = {
      green: "#35ff8b",
      red: "#ff3a3a",
      neutral: "#35ff8b",
      unavailable: "rgba(150, 157, 166, .45)",
    };
    const seen = new Set();
    segments.forEach((segment) => {
      const startRaw = landmarks[segment.from];
      const endRaw = landmarks[segment.to];
      if (!startRaw || !endRaw) return;
      const start = map(startRaw);
      const end = map(endRaw);
      const color = colors[segment.status] || colors.neutral;
      context.save();
      context.strokeStyle = color;
      context.lineWidth = 4;
      context.lineCap = "round";
      context.shadowColor = color;
      context.shadowBlur = 14;
      context.beginPath();
      context.moveTo(start.x, start.y);
      context.lineTo(end.x, end.y);
      context.stroke();
      context.restore();
      [segment.from, segment.to].forEach((name) => {
        if (seen.has(name)) return;
        seen.add(name);
        const point = map(landmarks[name]);
        context.save();
        context.fillStyle = "#090a0b";
        context.strokeStyle = color;
        context.lineWidth = 3;
        context.shadowColor = color;
        context.shadowBlur = 12;
        context.beginPath();
        context.arc(point.x, point.y, 6, 0, Math.PI * 2);
        context.fill();
        context.stroke();
        context.restore();
      });
    });
  }

  async function startPoseLoop() {
    if (state.mode !== "posture" || !state.cameraOn) return;
    const token = ++state.runToken;
    if (PYTHON_BACKEND_URL && !state.backendFailed) {
      pauseScore("Connecting Python posture service…", "loading");
      const connected = await connectBackend(token);
      if (
        connected &&
        token === state.runToken &&
        state.mode === "posture" &&
        state.cameraOn
      ) {
        document.body.dataset.postureEngine = "python-opencv-mediapipe";
        startBackendFrames(token);
        return;
      }
    }
    document.body.dataset.postureEngine = "mediapipe-web";
    pauseScore("Loading posture model…", "loading");
    try {
      await loadLandmarker();
    } catch (error) {
      console.error("HoopVision posture model failed to load", error);
      document.body.dataset.postureModel = "error";
      if (token === state.runToken) {
        pauseScore("Posture model could not load. Check your connection and try again.", "error");
        const liveState = $("liveState");
        if (liveState) {
          liveState.textContent = "ERROR";
          liveState.classList.remove("online");
        }
      }
      return;
    }
    if (token !== state.runToken || state.mode !== "posture" || !state.cameraOn) return;
    state.lastInferenceAt = 0;

    const frame = (now) => {
      if (token !== state.runToken || state.mode !== "posture" || !state.cameraOn) return;
      const video = $("cameraVideo");
      if (
        video &&
        video.readyState >= 2 &&
        video.videoWidth > 0 &&
        now - state.lastInferenceAt >= INFERENCE_INTERVAL_MS
      ) {
        state.lastInferenceAt = now;
        try {
          const result = state.landmarker.detectForVideo(video, performance.now());
          processPoseResult(result, now);
        } catch (error) {
          console.error("HoopVision posture frame failed", error);
        }
      }
      state.frameRequest = requestAnimationFrame(frame);
    };
    state.frameRequest = requestAnimationFrame(frame);
  }

  function trackedPoint(landmarks, index, minimum = REQUIRED_CONFIDENCE) {
    const point = landmarks?.[index];
    return Boolean(
      point &&
        Number.isFinite(point.x) &&
        Number.isFinite(point.y) &&
        confidence(point) >= minimum &&
        point.x >= -0.04 &&
        point.x <= 1.04 &&
        point.y >= -0.04 &&
        point.y <= 1.04
    );
  }

  function poseReadiness(landmarks) {
    if (!landmarks || landmarks.length < 25) {
      return {
        ready: false,
        missing: "pose",
        message: "Step 1 of 3 · Move fully into view",
      };
    }
    const hasHead =
      trackedPoint(landmarks, 7) || trackedPoint(landmarks, 8);
    const hasShoulders =
      trackedPoint(landmarks, 11) && trackedPoint(landmarks, 12);
    const hasHips =
      trackedPoint(landmarks, 23, HIP_CONFIDENCE) &&
      trackedPoint(landmarks, 24, HIP_CONFIDENCE);
    if (!hasShoulders) {
      return {
        ready: false,
        missing: "shoulders",
        message: "Step 1 of 3 · Center both shoulders",
      };
    }
    if (!hasHead) {
      return {
        ready: false,
        missing: "head",
        message: "Step 1 of 3 · Keep one ear visible",
      };
    }
    if (!hasHips) {
      return {
        ready: false,
        missing: "hips",
        message: "Step 1 of 3 · Move back until hips are visible",
      };
    }
    return {
      ready: true,
      missing: "",
      message: "",
      hasHead,
      hasShoulders,
      hasHips,
    };
  }

  function metricsFrom(landmarks, worldLandmarks) {
    const leftEar = landmarks[7];
    const rightEar = landmarks[8];
    const leftShoulder = landmarks[11];
    const rightShoulder = landmarks[12];
    const leftHip = landmarks[23];
    const rightHip = landmarks[24];
    const ear =
      confidence(leftEar) >= REQUIRED_CONFIDENCE &&
      confidence(rightEar) >= REQUIRED_CONFIDENCE
        ? midpoint(leftEar, rightEar)
        : confidence(leftEar) >= confidence(rightEar)
          ? leftEar
          : rightEar;
    const shoulders = midpoint(leftShoulder, rightShoulder);
    const hips = midpoint(leftHip, rightHip);
    const shoulderWidth = Math.max(distance(leftShoulder, rightShoulder), 0.001);
    const torsoHeight = distance(shoulders, hips);

    let headDepth =
      (shoulders.z - ear.z) / shoulderWidth;
    if (worldLandmarks?.length >= 25) {
      const worldEar =
        confidence(leftEar) >= REQUIRED_CONFIDENCE &&
        confidence(rightEar) >= REQUIRED_CONFIDENCE
          ? midpoint(worldLandmarks[7], worldLandmarks[8])
          : confidence(leftEar) >= confidence(rightEar)
            ? worldLandmarks[7]
            : worldLandmarks[8];
      const worldShoulders = midpoint(worldLandmarks[11], worldLandmarks[12]);
      const worldWidth = Math.max(
        distance(worldLandmarks[11], worldLandmarks[12]),
        0.001,
      );
      headDepth = (worldShoulders.z - worldEar.z) / worldWidth;
    }

    return {
      neckAngle: Math.abs(
        degrees(Math.atan2(ear.x - shoulders.x, Math.max(0.001, shoulders.y - ear.y))),
      ),
      headDepth,
      shoulderTilt: Math.abs(
        degrees(
          Math.atan2(
            rightShoulder.y - leftShoulder.y,
            Math.max(0.001, rightShoulder.x - leftShoulder.x),
          ),
        ),
      ),
      torsoLean: Math.abs(
        degrees(
          Math.atan2(
            shoulders.x - hips.x,
            Math.max(0.001, hips.y - shoulders.y),
          ),
        ),
      ),
      torsoRatio: torsoHeight / shoulderWidth,
      shoulderWidth,
      upperBodyHeight: distance(ear, hips),
      centerX: (ear.x + hips.x) / 2,
      centerY: (ear.y + hips.y) / 2,
      earY: ear.y,
      hipY: hips.y,
    };
  }

  function cameraWarning(metrics) {
    if (metrics.shoulderWidth > 0.72 || metrics.upperBodyHeight > 0.9) {
      return "Camera too close · Move back slightly";
    }
    if (metrics.earY < -0.01 || metrics.centerY < 0.22) {
      return "Camera too low · Aim it higher";
    }
    if (metrics.hipY > 1.015 || metrics.centerY > 0.78) {
      return "Camera too high · Aim it lower";
    }
    return "";
  }

  function updateSmoothedLandmarks(landmarks) {
    if (!state.smoothLandmarks || state.smoothLandmarks.length !== landmarks.length) {
      state.smoothLandmarks = landmarks.map((point) => ({ ...point }));
      return;
    }
    const alpha = 0.34;
    state.smoothLandmarks = landmarks.map((point, index) => {
      const previous = state.smoothLandmarks[index] || point;
      return {
        ...point,
        x: previous.x + (point.x - previous.x) * alpha,
        y: previous.y + (point.y - previous.y) * alpha,
        z: previous.z + (point.z - previous.z) * alpha,
      };
    });
  }

  function processPoseResult(result, now) {
    const landmarks = result?.landmarks?.[0];
    const worldLandmarks = result?.worldLandmarks?.[0];
    const readiness = poseReadiness(landmarks);
    if (landmarks?.length) updateSmoothedLandmarks(landmarks);

    if (!readiness.ready) {
      state.validSince = 0;
      state.lastValidMetrics = null;
      state.lastMissing = readiness.missing;
      state.calibrationLastAt = 0;
      const calibrate = $("calibrateButton");
      if (calibrate) {
        calibrate.disabled = true;
        calibrate.textContent =
          readiness.missing === "hips"
            ? "Move back to show hips"
            : "Show ears, shoulders & hips";
      }
      if (!state.invalidSince) state.invalidSince = now;
      if (now - state.invalidSince >= INVALID_GRACE_MS) {
        if (state.calibrating) restartCalibration(readiness.message);
        state.metricWindow = [];
        pauseScore(readiness.message, "missing");
        resetBadEpisode();
      } else {
        pauseScore(readiness.message, state.calibrating ? "calibrating" : "missing");
      }
      drawSkeleton(state.smoothLandmarks || landmarks, null, readiness);
      return;
    }

    state.invalidSince = 0;
    state.lastMissing = "";
    if (!state.validSince) state.validSince = now;
    const metrics = metricsFrom(landmarks, worldLandmarks);
    const warning = cameraWarning(metrics);
    state.lastValidLandmarks = landmarks;

    if (warning) {
      state.lastWarning = warning;
      state.lastValidMetrics = null;
      state.calibrationLastAt = 0;
      state.validSince = now;
      if (state.calibrating) restartCalibration(warning);
      state.metricWindow = [];
      const calibrate = $("calibrateButton");
      if (calibrate) {
        calibrate.disabled = true;
        calibrate.textContent = "Adjust camera to continue";
      }
      pauseScore(warning, "warning");
      resetBadEpisode();
      drawSkeleton(state.smoothLandmarks, null, readiness);
      return;
    }
    state.lastWarning = "";
    state.lastValidMetrics = metrics;

    if (now - state.validSince < 350) {
      const calibrate = $("calibrateButton");
      if (calibrate) {
        calibrate.disabled = true;
        calibrate.textContent = "Pose found · Hold still";
      }
      pauseScore("Step 1 of 3 · Pose found · Hold still", "reading");
      drawSkeleton(state.smoothLandmarks, null, readiness);
      return;
    }

    if (state.calibrating) {
      collectCalibration(metrics, now);
      drawSkeleton(state.smoothLandmarks, null, readiness);
      return;
    }

    if (!state.baseline) {
      pauseScore("Step 2 of 3 · Sit tall, then save this posture", "ready");
      setSetup("Step 2 of 3 · Pose found · Save your posture", "ready");
      const calibrate = $("calibrateButton");
      if (calibrate) {
        calibrate.disabled = false;
        calibrate.textContent = "Save correct posture";
      }
      drawSkeleton(state.smoothLandmarks, null, readiness);
      return;
    }

    const calibrate = $("calibrateButton");
    if (calibrate) {
      calibrate.disabled = false;
      calibrate.textContent = "Recalibrate posture";
    }
    state.metricWindow.push({ time: now, metrics });
    state.metricWindow = state.metricWindow.filter(
      (sample) => now - sample.time <= SMOOTHING_WINDOW_MS,
    );
    const windowSpan =
      state.metricWindow.length > 1
        ? now - state.metricWindow[0].time
        : 0;
    if (windowSpan < 1600) {
      const progress = clamp(Math.round((windowSpan / 1600) * 100), 0, 100);
      pauseScore("Step 3 of 3 · Reading alignment · " + progress + "%", "reading");
      drawSkeleton(state.smoothLandmarks, null, readiness);
      return;
    }

    const smoothed = medianMetrics(state.metricWindow.map((sample) => sample.metrics));
    const evaluation = evaluatePosture(smoothed, now);
    renderEvaluation(evaluation, now);
    drawSkeleton(state.smoothLandmarks, evaluation);
  }

  function beginCalibration() {
    if (state.mode !== "posture") return;
    unlockAudio();
    if (!state.cameraOn) {
      pauseScore("Start the camera before calibration.", "setup");
      return;
    }
    if (
      state.backendSocket?.readyState === WebSocket.OPEN &&
      state.backendReady
    ) {
      state.calibrating = true;
      state.backendCalibrated = false;
      state.backendSocket.send(JSON.stringify({ action: "start_calibration" }));
      const calibrate = $("calibrateButton");
      if (calibrate) {
        calibrate.disabled = true;
        calibrate.textContent = "Hold still · 0%";
      }
      pauseScore("Hold your best sitting posture…", "calibrating");
      setSetup("Calibrating · 0%", "calibrating");
      return;
    }
    if (!state.lastValidMetrics || state.lastWarning) {
      pauseScore(state.lastWarning || "Move fully into view", "missing");
      return;
    }
    state.calibrating = true;
    state.calibrationStartedAt = performance.now();
    state.calibrationValidMs = 0;
    state.calibrationLastAt = 0;
    state.calibrationSamples = [];
    state.metricWindow = [];
    resetBadEpisode();
    const calibrate = $("calibrateButton");
    if (calibrate) {
      calibrate.disabled = true;
      calibrate.textContent = "Hold still · 0%";
    }
    pauseScore("Hold your best sitting posture…", "calibrating");
    setSetup("Calibrating · 0%", "calibrating");
  }

  function restartCalibration(message) {
    if (!state.calibrating) return;
    state.calibrationStartedAt = performance.now();
    state.calibrationValidMs = 0;
    state.calibrationLastAt = 0;
    state.calibrationSamples = [];
    const calibrate = $("calibrateButton");
    if (calibrate) calibrate.textContent = "Hold still · 0%";
    setSetup(message, "warning");
  }

  function collectCalibration(metrics, now) {
    const previous = state.calibrationSamples.at(-1);
    if (
      previous &&
      (Math.hypot(
        metrics.centerX - previous.centerX,
        metrics.centerY - previous.centerY,
      ) > 0.07 ||
        Math.abs(metrics.shoulderWidth - previous.shoulderWidth) > 0.08)
    ) {
      restartCalibration("Hold still to save this posture");
      return;
    }
    if (state.calibrationLastAt) {
      state.calibrationValidMs += clamp(now - state.calibrationLastAt, 0, 250);
    }
    state.calibrationLastAt = now;
    state.calibrationSamples.push(metrics);
    const progress = clamp(
      Math.round((state.calibrationValidMs / CALIBRATION_HOLD_MS) * 100),
      0,
      100,
    );
    pauseScore("Step 2 of 3 · Hold still · " + progress + "%", "calibrating");
    setSetup("Step 2 of 3 · Saving baseline · " + progress + "%", "calibrating");
    const calibrate = $("calibrateButton");
    if (calibrate) calibrate.textContent = "Hold still · " + progress + "%";

    if (
      state.calibrationValidMs < CALIBRATION_HOLD_MS ||
      state.calibrationSamples.length < 12
    ) {
      return;
    }
    const baseline = medianMetrics(state.calibrationSamples);
    state.baseline = {
      neckAngle: baseline.neckAngle,
      headDepth: baseline.headDepth,
      shoulderTilt: baseline.shoulderTilt,
      torsoLean: baseline.torsoLean,
      torsoRatio: baseline.torsoRatio,
    };
    saveCalibration(state.baseline);
    state.calibrating = false;
    state.calibrationValidMs = 0;
    state.calibrationLastAt = 0;
    state.calibrationSamples = [];
    state.metricWindow = [];
    state.validSince = now;
    if (calibrate) {
      calibrate.disabled = false;
      calibrate.textContent = "Recalibrate posture";
    }
    setSetup("Step 3 of 3 · Calibration saved", "good");
    pauseScore("Step 3 of 3 · Reading alignment · 0%", "reading");
  }

  function medianMetrics(samples) {
    const keys = [
      "neckAngle",
      "headDepth",
      "shoulderTilt",
      "torsoLean",
      "torsoRatio",
      "shoulderWidth",
      "upperBodyHeight",
      "centerX",
      "centerY",
      "earY",
      "hipY",
    ];
    return Object.fromEntries(
      keys.map((key) => [key, median(samples.map((sample) => sample[key]))]),
    );
  }

  function evaluatePosture(metrics, now) {
    const baseline = state.baseline;
    const severities = {
      head: Math.max(
        ramp(metrics.neckAngle - baseline.neckAngle, 5, 18),
        ramp(metrics.headDepth - baseline.headDepth, 0.08, 0.28),
      ),
      shoulders: ramp(
        Math.max(0, metrics.shoulderTilt - baseline.shoulderTilt),
        3,
        12,
      ),
      back: Math.max(
        ramp(metrics.torsoLean - baseline.torsoLean, 5, 20),
        ramp(baseline.torsoRatio - metrics.torsoRatio, 0.07, 0.25),
      ),
    };
    Object.keys(state.failState).forEach((group) => {
      state.failState[group] = state.failState[group]
        ? severities[group] > 0.12
        : severities[group] > 0.25;
    });
    const weights = { head: 0.35, shoulders: 0.25, back: 0.4 };
    const failing = Object.keys(state.failState).filter((group) => state.failState[group]);
    let candidate = null;
    if (failing.length) {
      candidate = failing.sort(
        (a, b) => severities[b] * weights[b] - severities[a] * weights[a],
      )[0];
    }
    if (!candidate) {
      state.activeGroup = null;
      state.activeGroupSince = 0;
    } else if (state.activeGroup !== candidate) {
      const currentSeverity = state.activeGroup
        ? severities[state.activeGroup] * weights[state.activeGroup]
        : 0;
      const nextSeverity = severities[candidate] * weights[candidate];
      if (
        !state.activeGroup ||
        !state.failState[state.activeGroup] ||
        now - state.activeGroupSince >= 2000 ||
        nextSeverity > currentSeverity + 0.18
      ) {
        state.activeGroup = candidate;
        state.activeGroupSince = now;
        resetBadEpisode();
      }
    }
    const penalty =
      severities.head * weights.head +
      severities.shoulders * weights.shoulders +
      severities.back * weights.back;
    return {
      score: clamp(Math.round(100 * (1 - penalty)), 0, 100),
      group: state.activeGroup,
      severities,
      bad: { ...state.failState },
      groupScores: {
        head: clamp(Math.round(100 * (1 - severities.head)), 0, 100),
        shoulders: clamp(Math.round(100 * (1 - severities.shoulders)), 0, 100),
        back: clamp(Math.round(100 * (1 - severities.back)), 0, 100),
      },
    };
  }

  function renderEvaluation(evaluation, now) {
    const instructions = {
      back: "Sit up",
      head: "Move your head back",
      shoulders: "Level your shoulders",
    };
    const instruction = evaluation.group ? instructions[evaluation.group] : "Posture aligned";
    const cue = $("liveCue");
    if (cue) cue.textContent = instruction;
    const score = $("overallScore");
    if (score) score.textContent = String(evaluation.score);
    const ring = document.querySelector(".score-ring");
    ring?.classList.remove("posture-paused", "posture-good", "posture-bad");
    ring?.classList.add(evaluation.group ? "posture-bad" : "posture-good");
    const rows = Array.from(document.querySelectorAll(".metrics > div"));
    const values = [
      evaluation.groupScores.head,
      evaluation.groupScores.shoulders,
      evaluation.groupScores.back,
      evaluation.score,
    ];
    rows.forEach((row, index) => setMetricValue(row, values[index]));
    const rep = $("repCount");
    if (rep) rep.textContent = "LIVE";
    const liveState = $("liveState");
    if (liveState) {
      liveState.textContent = evaluation.group ? "ADJUST" : "ALIGNED";
      liveState.classList.toggle("online", !evaluation.group);
      liveState.classList.toggle("needs-adjustment", Boolean(evaluation.group));
    }
    setSetup(
      evaluation.group ? "Correction detected" : "Alignment locked",
      evaluation.group ? "bad" : "good",
    );
    updateAlert(evaluation.group, now);
  }

  function setMetricValue(row, value) {
    if (!row) return;
    const bar = row.querySelector("em");
    const copy = row.querySelector("b");
    if (bar) bar.style.width = value == null ? "0%" : value + "%";
    if (copy) copy.textContent = value == null ? "—" : String(value);
  }

  function pauseScore(message, kind) {
    const cue = $("liveCue");
    if (cue) cue.textContent = message;
    const score = $("overallScore");
    if (score) score.textContent = "—";
    document.querySelectorAll(".metrics > div").forEach((row) => setMetricValue(row, null));
    const rep = $("repCount");
    if (rep) rep.textContent = state.calibrating ? "CAL" : "—";
    const ring = document.querySelector(".score-ring");
    ring?.classList.remove("posture-good", "posture-bad");
    ring?.classList.add("posture-paused");
    const liveState = $("liveState");
    if (liveState) {
      liveState.textContent =
        kind === "warning" || kind === "missing"
          ? "REFRAME"
          : kind === "calibrating"
            ? "CALIBRATE"
            : kind === "ready"
              ? "READY"
          : kind === "error"
            ? "ERROR"
            : "READING";
      liveState.classList.remove("online", "needs-adjustment");
    }
    setSetup(message, kind);
  }

  function setSetup(message, kind) {
    const setup = $("postureSetup");
    const copy = $("postureSetupText");
    if (copy) copy.textContent = message;
    if (setup) setup.dataset.state = kind || "setup";
    document.body.dataset.postureStage = kind || "setup";
    if (state.lastMissing) {
      document.body.dataset.postureMissing = state.lastMissing;
    } else {
      delete document.body.dataset.postureMissing;
    }
  }

  function resetBadEpisode() {
    state.badSince = 0;
    state.alerted = false;
  }

  function updateAlert(group, now) {
    if (!group) {
      resetBadEpisode();
      return;
    }
    if (!state.badSince) state.badSince = now;
    if (
      !state.alerted &&
      now - state.badSince >= ALERT_AFTER_MS &&
      state.soundEnabled
    ) {
      playSoftAlert();
      state.alerted = true;
    }
  }

  function unlockAudio() {
    if (!window.AudioContext && !window.webkitAudioContext) return;
    if (!state.audioContext) {
      const Context = window.AudioContext || window.webkitAudioContext;
      state.audioContext = new Context();
    }
    if (state.audioContext.state === "suspended") {
      state.audioContext.resume().catch(() => undefined);
    }
  }

  function playSoftAlert() {
    unlockAudio();
    const context = state.audioContext;
    if (!context || context.state !== "running") return;
    const oscillator = context.createOscillator();
    const gain = context.createGain();
    const now = context.currentTime;
    oscillator.type = "sine";
    oscillator.frequency.setValueAtTime(392, now);
    oscillator.frequency.exponentialRampToValueAtTime(520, now + 0.14);
    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.exponentialRampToValueAtTime(0.026, now + 0.025);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.17);
    oscillator.connect(gain);
    gain.connect(context.destination);
    oscillator.start(now);
    oscillator.stop(now + 0.18);
  }

  function drawSkeleton(landmarks, evaluation) {
    const canvas = $("postureCanvas");
    const video = $("cameraVideo");
    if (!canvas || !video || canvas.hidden) return;
    const stage = canvas.parentElement;
    const width = stage.clientWidth;
    const height = stage.clientHeight;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      canvas.style.width = width + "px";
      canvas.style.height = height + "px";
    }
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    if (!landmarks || !video.videoWidth || !video.videoHeight) return;

    const scale = Math.max(width / video.videoWidth, height / video.videoHeight);
    const renderedWidth = video.videoWidth * scale;
    const renderedHeight = video.videoHeight * scale;
    const cropX = (renderedWidth - width) / 2;
    const cropY = (renderedHeight - height) / 2;
    const map = (point) => ({
      // The canvas and video are both mirrored by CSS, so draw in source
      // coordinates and let the shared transform keep landmarks aligned.
      x: point.x * renderedWidth - cropX,
      y: point.y * renderedHeight - cropY,
    });
    const points = landmarks.map(map);
    const green = "#35ff8b";
    const red = "#ff3a3a";
    const unavailable = "rgba(150, 157, 166, .58)";
    const groupColor = (group) => {
      // Before calibration, green means a real joint was found—not that
      // posture quality has been scored yet.
      if (!evaluation) return green;
      if (evaluation.group === group) return red;
      if (evaluation.bad[group]) return unavailable;
      return green;
    };
    const line = (a, b, color) => {
      if (!a || !b) return;
      context.save();
      context.strokeStyle = color;
      context.lineWidth = 4;
      context.lineCap = "round";
      context.shadowColor = color;
      context.shadowBlur = 14;
      context.beginPath();
      context.moveTo(a.x, a.y);
      context.lineTo(b.x, b.y);
      context.stroke();
      context.restore();
    };
    const lineBetween = (startIndex, endIndex, color) => {
      if (
        !trackedPoint(landmarks, startIndex, PARTIAL_CONFIDENCE) ||
        !trackedPoint(landmarks, endIndex, PARTIAL_CONFIDENCE)
      ) {
        return;
      }
      line(points[startIndex], points[endIndex], color);
    };
    const dot = (point, color) => {
      if (!point) return;
      context.save();
      context.fillStyle = "#090a0b";
      context.strokeStyle = color;
      context.lineWidth = 3;
      context.shadowColor = color;
      context.shadowBlur = 13;
      context.beginPath();
      context.arc(point.x, point.y, 6, 0, Math.PI * 2);
      context.fill();
      context.stroke();
      context.restore();
    };

    const headColor = groupColor("head");
    const shoulderColor = groupColor("shoulders");
    const backColor = groupColor("back");
    if (trackedPoint(landmarks, 7, PARTIAL_CONFIDENCE)) {
      lineBetween(7, 11, headColor);
      dot(points[7], headColor);
    }
    if (trackedPoint(landmarks, 8, PARTIAL_CONFIDENCE)) {
      lineBetween(8, 12, headColor);
      dot(points[8], headColor);
    }
    lineBetween(11, 12, shoulderColor);
    lineBetween(11, 23, backColor);
    lineBetween(12, 24, backColor);
    lineBetween(23, 24, backColor);
    if (
      trackedPoint(landmarks, 11, PARTIAL_CONFIDENCE) &&
      trackedPoint(landmarks, 12, PARTIAL_CONFIDENCE) &&
      trackedPoint(landmarks, 23, PARTIAL_CONFIDENCE) &&
      trackedPoint(landmarks, 24, PARTIAL_CONFIDENCE)
    ) {
      line(
        midpoint(points[11], points[12]),
        midpoint(points[23], points[24]),
        backColor,
      );
    }
    if (trackedPoint(landmarks, 11, PARTIAL_CONFIDENCE)) {
      dot(points[11], evaluation?.group === "shoulders" ? shoulderColor : backColor);
    }
    if (trackedPoint(landmarks, 12, PARTIAL_CONFIDENCE)) {
      dot(points[12], evaluation?.group === "shoulders" ? shoulderColor : backColor);
    }
    if (trackedPoint(landmarks, 23, PARTIAL_CONFIDENCE)) dot(points[23], backColor);
    if (trackedPoint(landmarks, 24, PARTIAL_CONFIDENCE)) dot(points[24], backColor);
  }

  function clearCanvas() {
    const canvas = $("postureCanvas");
    if (!canvas) return;
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);
  }

  window.addEventListener("hoopvision:mode", (event) => {
    const nextMode = event.detail?.mode;
    if (nextMode === "posture") {
      enterPosture(event.detail?.cameraOn);
    } else if (state.mode === "posture") {
      leavePosture();
    } else {
      state.mode = nextMode || "basketball";
    }
  });

  window.addEventListener("hoopvision:camera", (event) => {
    state.cameraOn = Boolean(event.detail?.on);
    if (state.mode !== "posture") return;
    const calibrate = $("calibrateButton");
    if (calibrate) {
      calibrate.hidden = !state.cameraOn;
      if (state.cameraOn) {
        calibrate.disabled = true;
        calibrate.textContent = "Show ears, shoulders & hips";
      }
    }
    if (state.cameraOn) {
      state.backendFailed = false;
      resetAnalysis();
      startPoseLoop();
    } else {
      state.runToken += 1;
      closeBackend();
      resetAnalysis();
      clearCanvas();
      pauseScore(
        state.baseline
          ? "Calibration saved · Start the camera"
          : "Step 1 of 3 · Start the camera and show your upper body",
        "setup",
      );
    }
  });

  ensureUi();
})();
