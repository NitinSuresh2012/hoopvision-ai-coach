"use client";

import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";

type Mode = "basketball" | "posture" | "upload";
type IntelligenceView = "film" | "compare" | "report";
type PosturePhase = "idle" | "loading" | "ready" | "calibrating" | "active" | "error";
type PostureCorrection = "head" | "shoulders" | "back" | null;
type MissingPostureRegion = "pose" | "head" | "shoulders" | "hips" | null;
type PostureStatus =
  | "GOOD_POSTURE"
  | "SLOUCHING"
  | "LEANING"
  | "SHOULDERS_UNEVEN"
  | "TURNED_AWAY"
  | "TOO_CLOSE"
  | "TOO_FAR"
  | "LOW_CONFIDENCE"
  | "NO_PERSON";

type PosePoint = {
  x: number;
  y: number;
  z?: number;
  visibility?: number;
  presence?: number;
};

type PoseResult = {
  landmarks?: PosePoint[][];
  worldLandmarks?: PosePoint[][];
};

type PoseLandmarkerInstance = {
  detectForVideo: (video: HTMLVideoElement, timestamp: number) => PoseResult;
  close?: () => void;
};

type VisionModule = {
  FilesetResolver: {
    forVisionTasks: (wasmPath: string) => Promise<unknown>;
  };
  PoseLandmarker: {
    createFromOptions: (
      fileset: unknown,
      options: Record<string, unknown>,
    ) => Promise<PoseLandmarkerInstance>;
  };
};

type PostureMeasurement = {
  neckAngle: number;
  shoulderTilt: number;
  torsoAngle: number;
  torsoRatio: number;
  headDepth: number;
  headOffset: number;
  bodyVisibility: number;
  facingConfidence: number;
  cameraScale: number;
};

type PostureScores = {
  neck: number;
  head: number;
  shoulderLevel: number;
  upperBack: number;
  framing: number;
};

type PostureFlags = {
  head: boolean;
  shoulders: boolean;
  back: boolean;
};

type CalibrationRun = {
  startedAt: number;
  validMs: number;
  lastValidAt: number | null;
  samples: PostureMeasurement[];
};

const VISION_CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/vision_bundle.mjs";
const VISION_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm";
const POSE_MODEL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const POSTURE_BASELINE_KEY = "hoopvision-posture-baseline-v3";
const POSTURE_WINDOW_MS = 2000;
const POSTURE_LANDMARK_CONFIDENCE = 0.5;
const POSTURE_HIP_CONFIDENCE = 0.32;

const correctionCopy: Record<Exclude<PostureCorrection, null>, string> = {
  head: "Move your head back",
  shoulders: "Level your shoulders",
  back: "Sit up",
};

const postureStatusCopy: Record<PostureStatus, string> = {
  GOOD_POSTURE: "Posture aligned",
  SLOUCHING: "Sit up",
  LEANING: "Center your upper body",
  SHOULDERS_UNEVEN: "Level your shoulders",
  TURNED_AWAY: "Turn back toward the camera",
  TOO_CLOSE: "Move farther from the camera",
  TOO_FAR: "Move closer to the camera",
  LOW_CONFIDENCE: "Move fully into view",
  NO_PERSON: "Move fully into view",
};

type PostureCameraFeedback = {
  score: number | null;
  tone: "green" | "red" | "neutral";
  message:
    | "Posture aligned"
    | "Face the screen"
    | "Sit upright"
    | "Center your shoulders"
    | "Move back into frame"
    | "Move closer to the camera"
    | "Move farther from the camera"
    | "Sit upright while calibration completes"
    | "Posture unavailable — return to the camera";
};

function getPostureCameraFeedback({
  calibrated,
  cameraOn,
  confidence,
  facingConfidence,
  fullPoseReady,
  phase,
  score,
  status,
}: {
  calibrated: boolean;
  cameraOn: boolean;
  confidence: number;
  facingConfidence: number | null;
  fullPoseReady: boolean;
  phase: PosturePhase;
  score: number | null;
  status: PostureStatus;
}): PostureCameraFeedback {
  const rawScore = Number.isFinite(score) ? Number(score) : null;
  const calibrating = phase === "calibrating" || (!calibrated && phase === "ready");
  const invalidView = !fullPoseReady ||
    confidence < 50 ||
    facingConfidence === null ||
    facingConfidence < 0.5 ||
    status === "TURNED_AWAY" ||
    status === "TOO_CLOSE" ||
    status === "TOO_FAR" ||
    status === "LOW_CONFIDENCE";
  const scoringReady = phase === "active" || phase === "ready";
  const safeScore = !cameraOn || calibrating || rawScore === null
    ? null
    : status === "NO_PERSON"
      ? 0
      : invalidView
        ? clamp(rawScore, 0, 29)
        : clamp(rawScore, 0, 100);
  const green = cameraOn &&
    scoringReady &&
    !calibrating &&
    fullPoseReady &&
    confidence >= 50 &&
    facingConfidence !== null &&
    facingConfidence >= 0.5 &&
    status === "GOOD_POSTURE" &&
    safeScore !== null &&
    safeScore >= 85;

  let message: PostureCameraFeedback["message"];
  if (!cameraOn) {
    message = "Posture unavailable — return to the camera";
  } else if (calibrating && fullPoseReady) {
    message = "Sit upright while calibration completes";
  } else if (status === "NO_PERSON") {
    message = "Posture unavailable — return to the camera";
  } else if (status === "TURNED_AWAY") {
    message = "Face the screen";
  } else if (status === "TOO_CLOSE") {
    message = "Move farther from the camera";
  } else if (status === "TOO_FAR") {
    message = "Move closer to the camera";
  } else if (status === "LOW_CONFIDENCE") {
    message = "Move back into frame";
  } else if (status === "SHOULDERS_UNEVEN" || status === "LEANING") {
    message = "Center your shoulders";
  } else if (status === "SLOUCHING" || !green) {
    message = "Sit upright";
  } else {
    message = "Posture aligned";
  }

  return {
    score: safeScore === null ? null : Math.round(safeScore),
    tone: calibrating ? "neutral" : green ? "green" : "red",
    message,
  };
}

const modeCopy = {
  basketball: {
    label: "Live Basketball Training",
    eyebrow: "LIVE COURT MODE",
    title: "Move. Adjust. Score again.",
    cue: "Load your hips before the rise. Right knee is drifting inward on takeoff.",
  },
  posture: {
    label: "Sitting/Posture Correction",
    eyebrow: "DESK POSTURE MODE",
    title: "Sit tall. Stay aligned.",
    cue: "Bring shoulders over hips and raise your screen slightly. Neck angle is improving.",
  },
  upload: {
    label: "Upload Video",
    eyebrow: "SECONDARY VIDEO REVIEW",
    title: "Review any clip later.",
    cue: "Upload a saved video for a slower breakdown with drills and progress notes.",
  },
};

const drills = [
  {
    title: "Form Lock",
    cue: "Hold the finish until wrist and elbow stay green for two seconds.",
    focus: "Release path",
    duration: "04:00",
    reps: "12 reps",
    target: "92",
  },
  {
    title: "Balance Freeze",
    cue: "Land quiet. The rep counts only when both feet stabilize inside 0.4 seconds.",
    focus: "Landing control",
    duration: "05:00",
    reps: "10 reps",
    target: "88",
  },
  {
    title: "Footwork Sync",
    cue: "Attack left-right into the shot while the knees stay stacked over the toes.",
    focus: "Gather timing",
    duration: "06:00",
    reps: "16 reps",
    target: "90",
  },
];
const athletes = [
  { initials: "N", name: "Nitin", focus: "Live shooting", score: 94, status: "Training live", tone: "aligned", trend: "+8" },
  { initials: "T", name: "Theo", focus: "Movement form", score: 88, status: "Tracking", tone: "aligned", trend: "+11" },
];
const plans = [
  { name: "Rookie", price: 0, audience: "Explore the system", features: ["Live coach demo", "Core movement scores", "2 video reviews"] },
  { name: "Elite", price: 24, audience: "Build your game", features: ["Unlimited live coaching", "Voice cues + Film IQ", "Pro compare + reports"] },
  { name: "Coach", price: 69, audience: "Develop a roster", features: ["25 athlete profiles", "Team intelligence", "Assignments + exports"] },
];
const intelligenceTabs: { id: IntelligenceView; label: string }[] = [
  { id: "film", label: "Game Film IQ" },
  { id: "compare", label: "Pro Comparison" },
  { id: "report", label: "Scouting Report" },
];

function Mark() {
  return (
    <span className="mark" aria-hidden="true">
      <span />
      <span />
      <span />
    </span>
  );
}

function Arrow() {
  return <span aria-hidden="true">-&gt;</span>;
}

function formatTime(seconds: number) {
  const mins = Math.floor(seconds / 60).toString().padStart(2, "0");
  const secs = (seconds % 60).toString().padStart(2, "0");
  return `${mins}:${secs}`;
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}

function midpoint(a: PosePoint, b: PosePoint): PosePoint {
  return {
    x: (a.x + b.x) / 2,
    y: (a.y + b.y) / 2,
    z: ((a.z ?? 0) + (b.z ?? 0)) / 2,
  };
}

function pointDistance(a: PosePoint, b: PosePoint) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function angleFromVertical(top: PosePoint, bottom: PosePoint) {
  return Math.atan2(Math.abs(bottom.x - top.x), Math.max(0.001, Math.abs(bottom.y - top.y))) * (180 / Math.PI);
}

function pointConfidence(point: PosePoint | undefined) {
  if (!point || !Number.isFinite(point.x) || !Number.isFinite(point.y)) return 0;
  return clamp(Math.min(point.visibility ?? 1, point.presence ?? 1), 0, 1);
}

function usablePosturePoint(
  point: PosePoint | undefined,
  minimum = POSTURE_LANDMARK_CONFIDENCE,
): point is PosePoint {
  return Boolean(
    point &&
    Number.isFinite(point.x) &&
    Number.isFinite(point.y) &&
    pointConfidence(point) >= minimum,
  );
}

function invalidPostureFrame(
  landmarks: PosePoint[] | null,
  confidence: number,
  missing: MissingPostureRegion,
  status: Extract<PostureStatus, "NO_PERSON" | "LOW_CONFIDENCE" | "TURNED_AWAY" | "TOO_CLOSE" | "TOO_FAR">,
  score: number,
) {
  return {
    landmarks,
    measurement: null,
    confidence,
    missing,
    status,
    score: status === "NO_PERSON" ? 0 : clamp(Math.round(score), 5, 29),
  };
}

function extractPostureFrame(result: PoseResult) {
  const landmarks = result.landmarks?.[0];
  const world = result.worldLandmarks?.[0];
  if (!landmarks || landmarks.length < 25) {
    return invalidPostureFrame(null, 0, "pose", "NO_PERSON", 0);
  }

  const nose = landmarks[0];
  const leftEye = landmarks[2];
  const rightEye = landmarks[5];
  const leftEar = landmarks[7];
  const rightEar = landmarks[8];
  const leftShoulder = landmarks[11];
  const rightShoulder = landmarks[12];
  const leftHip = landmarks[23];
  const rightHip = landmarks[24];
  const faceSupportConfidence = Math.max(
    pointConfidence(leftEye),
    pointConfidence(rightEye),
    pointConfidence(leftEar),
    pointConfidence(rightEar),
  );
  const essentialConfidences = [
    pointConfidence(nose),
    faceSupportConfidence,
    pointConfidence(leftShoulder),
    pointConfidence(rightShoulder),
    pointConfidence(leftHip),
    pointConfidence(rightHip),
  ];
  const confidence = essentialConfidences.reduce(
    (total, value) => total + value,
    0,
  ) / essentialConfidences.length;
  const shoulderConfidence = Math.min(
    pointConfidence(leftShoulder),
    pointConfidence(rightShoulder),
  );
  const hipConfidence = Math.max(
    pointConfidence(leftHip),
    pointConfidence(rightHip),
  );
  const torsoConfidence = Math.min(shoulderConfidence, hipConfidence);
  const eyeConfidence = Math.max(pointConfidence(leftEye), pointConfidence(rightEye));
  const earConfidence = Math.max(pointConfidence(leftEar), pointConfidence(rightEar));
  const faceConfidence = Math.min(
    pointConfidence(nose),
    Math.max(eyeConfidence, earConfidence),
  );
  const bodyVisibility = clamp(
    torsoConfidence * 0.58 + faceConfidence * 0.3 + confidence * 0.12,
    0,
    1,
  );

  if (confidence < 0.14 && torsoConfidence < 0.18) {
    return invalidPostureFrame(landmarks, confidence, "pose", "NO_PERSON", 0);
  }

  const hasHead = usablePosturePoint(nose, 0.42) &&
    (
      usablePosturePoint(leftEye, 0.35) ||
      usablePosturePoint(rightEye, 0.35) ||
      usablePosturePoint(leftEar, 0.35) ||
      usablePosturePoint(rightEar, 0.35)
    );
  const hasShoulders = usablePosturePoint(leftShoulder, 0.45) &&
    usablePosturePoint(rightShoulder, 0.45);
  const hasLeftHip = usablePosturePoint(leftHip, POSTURE_HIP_CONFIDENCE);
  const hasRightHip = usablePosturePoint(rightHip, POSTURE_HIP_CONFIDENCE);
  const hasHips = hasLeftHip || hasRightHip;
  const missing: MissingPostureRegion = !hasShoulders
    ? "shoulders"
    : !hasHead
      ? "head"
      : !hasHips
        ? "hips"
        : null;
  if (missing) {
    const turnedAway = hasShoulders && hasHips && faceConfidence < 0.42;
    return invalidPostureFrame(
      landmarks,
      confidence,
      missing,
      turnedAway ? "TURNED_AWAY" : "LOW_CONFIDENCE",
      5 + bodyVisibility * 24,
    );
  }

  const headAnchor = usablePosturePoint(leftEar) && usablePosturePoint(rightEar)
    ? midpoint(leftEar, rightEar)
    : usablePosturePoint(leftEar, 0.5)
      ? leftEar
      : usablePosturePoint(rightEar, 0.5)
        ? rightEar
        : nose;
  const faceMid = usablePosturePoint(leftEye, 0.5) && usablePosturePoint(rightEye, 0.5)
    ? midpoint(leftEye, rightEye)
    : nose;
  const shoulderMid = midpoint(leftShoulder, rightShoulder);
  const hipMid = hasLeftHip && hasRightHip
    ? midpoint(leftHip, rightHip)
    : hasLeftHip
      ? { ...leftHip, x: shoulderMid.x }
      : { ...rightHip, x: shoulderMid.x };
  const shoulderWidth = pointDistance(leftShoulder, rightShoulder);
  const visibleHeight = hipMid.y - headAnchor.y;
  const torsoLength = pointDistance(shoulderMid, hipMid);

  if (confidence < 0.42 || shoulderConfidence < 0.45 || hipConfidence < POSTURE_HIP_CONFIDENCE || bodyVisibility < 0.42) {
    return invalidPostureFrame(
      landmarks,
      confidence,
      "pose",
      "LOW_CONFIDENCE",
      5 + bodyVisibility * 24,
    );
  }
  if (shoulderWidth > 0.72 || visibleHeight > 0.95 || torsoLength > 0.76) {
    return invalidPostureFrame(
      landmarks,
      confidence,
      null,
      "TOO_CLOSE",
      29 - Math.max(shoulderWidth - 0.72, visibleHeight - 0.95) * 70,
    );
  }
  if (shoulderWidth < 0.08 || visibleHeight < 0.18 || torsoLength < 0.11) {
    return invalidPostureFrame(
      landmarks,
      confidence,
      null,
      "TOO_FAR",
      8 + bodyVisibility * 20,
    );
  }

  let headDepth = 0;
  let shoulderYaw = 0;
  if (world && world.length >= 25) {
    const worldHead = usablePosturePoint(leftEar) && usablePosturePoint(rightEar)
      ? midpoint(world[7], world[8])
      : usablePosturePoint(leftEar, 0.5)
        ? world[7]
        : usablePosturePoint(rightEar, 0.5)
          ? world[8]
          : world[0];
    const worldShoulder = midpoint(world[11], world[12]);
    const worldShoulderWidth = Math.max(pointDistance(world[11], world[12]), 0.05);
    headDepth = (worldShoulder.z ?? 0) - (worldHead.z ?? 0);
    shoulderYaw = Math.abs((world[11].z ?? 0) - (world[12].z ?? 0)) / worldShoulderWidth;
  }

  const headOffset = Math.abs(faceMid.x - shoulderMid.x) / Math.max(shoulderWidth, 0.05);
  const facingConfidence = clamp(
    faceConfidence * 0.72 +
      (1 - clamp(headOffset / 0.9, 0, 1)) * 0.18 +
      (1 - clamp(shoulderYaw / 1.15, 0, 1)) * 0.1,
    0,
    1,
  );
  if (
    faceConfidence < 0.52 ||
    facingConfidence < 0.5 ||
    headOffset > 0.82 ||
    (shoulderYaw > 1.05 && pointConfidence(nose) < 0.7)
  ) {
    return invalidPostureFrame(
      landmarks,
      confidence,
      "head",
      "TURNED_AWAY",
      5 + facingConfidence * 24,
    );
  }

  const measurement: PostureMeasurement = {
    neckAngle: angleFromVertical(headAnchor, shoulderMid),
    shoulderTilt: Math.atan2(
      Math.abs(rightShoulder.y - leftShoulder.y),
      Math.max(0.001, Math.abs(rightShoulder.x - leftShoulder.x)),
    ) * (180 / Math.PI),
    torsoAngle: angleFromVertical(shoulderMid, hipMid),
    torsoRatio: torsoLength / Math.max(shoulderWidth, 0.05),
    headDepth,
    headOffset,
    bodyVisibility,
    facingConfidence,
    cameraScale: shoulderWidth,
  };

  return {
    landmarks,
    measurement,
    confidence,
    missing: null as MissingPostureRegion,
    status: "GOOD_POSTURE" as PostureStatus,
    score: null as number | null,
  };
}

function aggregateMeasurements(samples: PostureMeasurement[], useMedian = false): PostureMeasurement {
  const keys: (keyof PostureMeasurement)[] = [
    "neckAngle",
    "shoulderTilt",
    "torsoAngle",
    "torsoRatio",
    "headDepth",
    "headOffset",
    "bodyVisibility",
    "facingConfidence",
    "cameraScale",
  ];
  return keys.reduce((result, key) => {
    const values = samples
      .map((sample) => sample[key])
      .filter(Number.isFinite)
      .sort((a, b) => a - b);
    result[key] = values.length === 0
      ? 0
      : useMedian
        ? values[Math.floor(values.length / 2)]
        : values.reduce((total, value) => total + value, 0) / values.length;
    return result;
  }, {} as PostureMeasurement);
}

function validBaseline(value: unknown): value is PostureMeasurement {
  if (!value || typeof value !== "object") return false;
  return [
    "neckAngle",
    "shoulderTilt",
    "torsoAngle",
    "torsoRatio",
    "headDepth",
    "headOffset",
    "bodyVisibility",
    "facingConfidence",
    "cameraScale",
  ].every((key) => Number.isFinite((value as Record<string, unknown>)[key]));
}

function evaluatePosture(
  measurement: PostureMeasurement,
  baseline: PostureMeasurement | null,
  previous: PostureFlags,
) {
  const finite = Object.values(measurement).every(Number.isFinite);
  if (!finite) {
    const scores: PostureScores = {
      neck: 0,
      head: 0,
      shoulderLevel: 0,
      upperBack: 0,
      framing: 0,
    };
    return {
      correction: null as PostureCorrection,
      flags: { head: false, shoulders: false, back: false },
      scores,
      score: 0,
      status: "LOW_CONFIDENCE" as PostureStatus,
    };
  }

  const neckReference = baseline?.neckAngle ?? 6;
  const headOffsetReference = baseline?.headOffset ?? 0.08;
  const shoulderReference = baseline?.shoulderTilt ?? 1.5;
  const torsoReference = baseline?.torsoAngle ?? 4;
  const headAngleDelta = Math.max(0, measurement.neckAngle - neckReference - 3);
  const headDepthDelta = baseline
    ? Math.max(0, measurement.headDepth - baseline.headDepth - 0.025)
    : 0;
  const headOffsetDelta = Math.max(0, measurement.headOffset - headOffsetReference - 0.05);
  const shoulderDelta = Math.max(0, measurement.shoulderTilt - shoulderReference - 2);
  const torsoAngleDelta = Math.max(0, measurement.torsoAngle - torsoReference - 3);
  const torsoCompression = baseline
    ? Math.max(0, baseline.torsoRatio - measurement.torsoRatio - 0.05)
    : 0;

  const severities = {
    slouch: clamp(Math.max(headAngleDelta / 18, headDepthDelta / 0.13, torsoCompression / 0.25), 0, 1.5),
    lean: clamp(Math.max(torsoAngleDelta / 18, headOffsetDelta / 0.5), 0, 1.5),
    shoulders: clamp(shoulderDelta / 12, 0, 1.5),
  };
  const framingSeverity = clamp(
    Math.max(
      (0.58 - measurement.bodyVisibility) / 0.35,
      (0.5 - measurement.facingConfidence) / 0.35,
      measurement.cameraScale < 0.12 ? (0.12 - measurement.cameraScale) / 0.08 : 0,
      measurement.cameraScale > 0.58 ? (measurement.cameraScale - 0.58) / 0.18 : 0,
    ),
    0,
    1.5,
  );
  const scores: PostureScores = {
    neck: Math.round(clamp(100 - Math.max(headAngleDelta / 18, headDepthDelta / 0.13) * 70, 0, 100)),
    head: Math.round(clamp(100 - Math.max(headOffsetDelta / 0.5, headDepthDelta / 0.13) * 70, 0, 100)),
    shoulderLevel: Math.round(clamp(100 - severities.shoulders * 70, 0, 100)),
    upperBack: Math.round(clamp(100 - Math.max(torsoAngleDelta / 18, torsoCompression / 0.25) * 70, 0, 100)),
    framing: Math.round(clamp(100 - framingSeverity * 70, 0, 100)),
  };
  const score = Math.round(clamp(
    scores.neck * 0.22 +
      scores.head * 0.18 +
      scores.shoulderLevel * 0.18 +
      scores.upperBack * 0.27 +
      scores.framing * 0.15,
    0,
    100,
  ));
  const dominant = (Object.keys(severities) as (keyof typeof severities)[])
    .sort((a, b) => severities[b] - severities[a])[0];
  const status: PostureStatus = score >= 85
    ? "GOOD_POSTURE"
    : dominant === "shoulders"
      ? "SHOULDERS_UNEVEN"
      : dominant === "lean"
        ? "LEANING"
        : "SLOUCHING";
  const flags: PostureFlags = {
    head: status === "SLOUCHING" && (previous.head ? severities.slouch >= 0.28 : score < 85),
    shoulders: status === "SHOULDERS_UNEVEN" &&
      (previous.shoulders ? severities.shoulders >= 0.28 : score < 85),
    back: status === "LEANING" && (previous.back ? severities.lean >= 0.28 : score < 85),
  };
  const correction: PostureCorrection = status === "SLOUCHING"
    ? "head"
    : status === "SHOULDERS_UNEVEN"
      ? "shoulders"
      : status === "LEANING"
        ? "back"
        : null;
  return { correction, flags, scores, score, status };
}

function clearPostureCanvas(canvas: HTMLCanvasElement | null) {
  if (!canvas) return;
  canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
}

function drawPostureOverlay(
  canvas: HTMLCanvasElement,
  video: HTMLVideoElement,
  landmarks: PosePoint[],
  greenFeedback: boolean,
) {
  const box = canvas.getBoundingClientRect();
  if (!box.width || !box.height || !video.videoWidth || !video.videoHeight) return;
  const pixelRatio = window.devicePixelRatio || 1;
  const targetWidth = Math.round(box.width * pixelRatio);
  const targetHeight = Math.round(box.height * pixelRatio);
  if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
    canvas.width = targetWidth;
    canvas.height = targetHeight;
  }
  const context = canvas.getContext("2d");
  if (!context) return;
  context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
  context.clearRect(0, 0, box.width, box.height);

  const coverScale = Math.max(box.width / video.videoWidth, box.height / video.videoHeight);
  const offsetX = (box.width - video.videoWidth * coverScale) / 2;
  const offsetY = (box.height - video.videoHeight * coverScale) / 2;
  const project = (point: PosePoint) => ({
    x: offsetX + point.x * video.videoWidth * coverScale,
    y: offsetY + point.y * video.videoHeight * coverScale,
  });

  const leftEar = landmarks[7];
  const rightEar = landmarks[8];
  const leftShoulder = landmarks[11];
  const rightShoulder = landmarks[12];
  const leftHip = landmarks[23];
  const rightHip = landmarks[24];
  const leftEarReady = usablePosturePoint(leftEar, 0.5);
  const rightEarReady = usablePosturePoint(rightEar, 0.5);
  const earMid = leftEarReady && rightEarReady
    ? midpoint(leftEar, rightEar)
    : leftEarReady
      ? leftEar
      : rightEarReady
        ? rightEar
        : null;
  const shoulderMid = usablePosturePoint(leftShoulder, 0.5) && usablePosturePoint(rightShoulder, 0.5)
    ? midpoint(leftShoulder, rightShoulder)
    : null;
  const hipMid = usablePosturePoint(leftHip, 0.48) && usablePosturePoint(rightHip, 0.48)
    ? midpoint(leftHip, rightHip)
    : null;
  const good = "#35ff8b";
  const bad = "#ff3131";
  const areaColor = () => greenFeedback ? good : bad;

  const line = (
    a: PosePoint | null | undefined,
    b: PosePoint | null | undefined,
    color: string,
    width = 6,
    dashed = false,
  ) => {
    if (!a || !b || !usablePosturePoint(a, 0.48) || !usablePosturePoint(b, 0.48)) return;
    const start = project(a);
    const end = project(b);
    context.save();
    context.beginPath();
    context.moveTo(start.x, start.y);
    context.lineTo(end.x, end.y);
    context.lineCap = "round";
    context.lineWidth = width;
    context.setLineDash(dashed ? [12, 14] : []);
    context.strokeStyle = color;
    context.shadowBlur = 22;
    context.shadowColor = color;
    context.stroke();
    context.restore();
  };
  const joint = (point: PosePoint | undefined | null, color: string, radius = 8) => {
    if (!point || !usablePosturePoint(point, 0.48)) return;
    const projected = project(point);
    context.save();
    context.beginPath();
    context.arc(projected.x, projected.y, radius, 0, Math.PI * 2);
    context.fillStyle = "rgba(5, 8, 8, 0.88)";
    context.fill();
    context.lineWidth = 4;
    context.strokeStyle = color;
    context.shadowBlur = 26;
    context.shadowColor = color;
    context.stroke();
    context.restore();
  };

  const headColor = areaColor();
  const shoulderColor = areaColor();
  const backColor = areaColor();
  line(leftShoulder, rightShoulder, shoulderColor, 6);
  line(earMid, shoulderMid, headColor, 6);
  line(shoulderMid, hipMid, backColor, 6, true);
  if (hipMid) {
    const projected = project(hipMid);
    const halfWidth = Math.max(34, pointDistance(leftHip, rightHip) * video.videoWidth * coverScale * 0.42);
    context.save();
    context.beginPath();
    context.moveTo(projected.x - halfWidth, projected.y);
    context.lineTo(projected.x + halfWidth, projected.y);
    context.lineCap = "round";
    context.lineWidth = 5;
    context.strokeStyle = backColor;
    context.shadowBlur = 20;
    context.shadowColor = backColor;
    context.stroke();
    context.restore();
  }
  joint(earMid, headColor, 8);
  joint(shoulderMid, shoulderColor, 8);
  joint(hipMid, backColor, 8);
  joint(leftShoulder, greenFeedback ? good : bad, 7);
  joint(rightShoulder, greenFeedback ? good : bad, 7);
}

async function loadVisionModule() {
  return import(/* @vite-ignore */ VISION_CDN) as Promise<VisionModule>;
}

export default function Home() {
  const [mode, setMode] = useState<Mode>("basketball");
  const [cameraOn, setCameraOn] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const [reps, setReps] = useState(18);
  const [pulse, setPulse] = useState(0);
  const [videoName, setVideoName] = useState("");
  const [voiceOn, setVoiceOn] = useState(true);
  const [intelligenceView, setIntelligenceView] = useState<IntelligenceView>("film");
  const [reportReady, setReportReady] = useState(false);
  const [activeDrill, setActiveDrill] = useState(0);
  const [annualBilling, setAnnualBilling] = useState(true);
  const [chatOpen, setChatOpen] = useState(false);
  const [message, setMessage] = useState("");
  const [chat, setChat] = useState([
    { who: "coach", text: "Live coach online. I will call out form, balance, footwork, and speed in real time." },
  ]);
  const [posturePhase, setPosturePhase] = useState<PosturePhase>("idle");
  const [postureInstruction, setPostureInstruction] = useState("Start the camera to begin");
  const [postureScore, setPostureScore] = useState<number | null>(null);
  const [postureScores, setPostureScores] = useState<PostureScores>({
    neck: 0,
    head: 0,
    shoulderLevel: 0,
    upperBack: 0,
    framing: 0,
  });
  const [postureConfidence, setPostureConfidence] = useState(0);
  const [postureCorrection, setPostureCorrection] = useState<PostureCorrection>(null);
  const [postureCalibrated, setPostureCalibrated] = useState(false);
  const [calibrationProgress, setCalibrationProgress] = useState(0);
  const [postureCameraState, setPostureCameraState] = useState("Waiting for camera");
  const [postureStatus, setPostureStatus] = useState<PostureStatus>("NO_PERSON");
  const [postureAngles, setPostureAngles] = useState<PostureMeasurement | null>(null);
  const [postureFullPoseReady, setPostureFullPoseReady] = useState(false);
  const cameraRef = useRef<HTMLVideoElement>(null);
  const postureCanvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const poseLandmarkerRef = useRef<PoseLandmarkerInstance | null>(null);
  const poseLandmarkerPromiseRef = useRef<Promise<PoseLandmarkerInstance> | null>(null);
  const postureAnimationRef = useRef<number | null>(null);
  const postureBaselineRef = useRef<PostureMeasurement | null>(null);
  const postureWindowRef = useRef<{ timestamp: number; measurement: PostureMeasurement }[]>([]);
  const calibrationRef = useRef<CalibrationRun | null>(null);
  const autoCalibrationStartedRef = useRef(false);
  const postureFlagsRef = useRef<PostureFlags>({ head: false, shoulders: false, back: false });
  const postureDisplayedScoreRef = useRef<number | null>(null);
  const postureScoreUpdatedAtRef = useRef(0);
  const sustainedBadRef = useRef<{ correction: Exclude<PostureCorrection, null>; since: number; sounded: boolean } | null>(null);
  const postureInvalidSinceRef = useRef<number | null>(null);
  const lastVideoTimeRef = useRef(-1);
  const lastInferenceRef = useRef(0);
  const audioContextRef = useRef<AudioContext | null>(null);
  const voiceOnRef = useRef(voiceOn);

  useEffect(() => {
    if (!cameraOn) return;
    const tick = window.setInterval(() => {
      setSeconds((value) => value + 1);
      if (mode !== "posture") {
        setPulse((value) => value + 1);
        if (Math.random() > 0.68) setReps((value) => value + 1);
      }
    }, 1000);
    return () => window.clearInterval(tick);
  }, [cameraOn, mode]);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      try {
        const stored = window.localStorage.getItem(POSTURE_BASELINE_KEY);
        const baseline = stored ? JSON.parse(stored) : null;
        if (validBaseline(baseline)) {
          postureBaselineRef.current = baseline;
          setPostureCalibrated(true);
        }
      } catch {
        // Local storage can be unavailable in private browsing; live calibration still works.
      }
    });
    return () => window.cancelAnimationFrame(frame);
  }, []);

  useEffect(() => {
    voiceOnRef.current = voiceOn;
  }, [voiceOn]);

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      if (postureAnimationRef.current !== null) window.cancelAnimationFrame(postureAnimationRef.current);
      poseLandmarkerRef.current?.close?.();
      void audioContextRef.current?.close();
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const canvasNode = postureCanvasRef.current;

    if (!cameraOn || mode !== "posture") {
      if (postureAnimationRef.current !== null) {
        window.cancelAnimationFrame(postureAnimationRef.current);
        postureAnimationRef.current = null;
      }
      clearPostureCanvas(canvasNode);
      postureWindowRef.current = [];
      postureFlagsRef.current = { head: false, shoulders: false, back: false };
      sustainedBadRef.current = null;
      postureInvalidSinceRef.current = null;
      calibrationRef.current = null;
      autoCalibrationStartedRef.current = false;
      return;
    }

    const startTracking = async () => {
      await Promise.resolve();
      if (cancelled) return;
      setPosturePhase("loading");
      setPostureScore(0);
      setPostureStatus("NO_PERSON");
      setPostureCorrection(null);
      setPostureInstruction("Loading posture model");
      setPostureCameraState("Stage 1 · Loading posture tracking");
      setPostureFullPoseReady(false);
      try {
        if (!poseLandmarkerPromiseRef.current) {
          poseLandmarkerPromiseRef.current = (async () => {
            const vision = await loadVisionModule();
            const fileset = await vision.FilesetResolver.forVisionTasks(VISION_WASM);
            const options = {
              baseOptions: { modelAssetPath: POSE_MODEL, delegate: "GPU" },
              runningMode: "VIDEO",
              numPoses: 1,
              minPoseDetectionConfidence: 0.5,
              minPosePresenceConfidence: 0.5,
              minTrackingConfidence: 0.5,
              outputSegmentationMasks: false,
            };
            try {
              return await vision.PoseLandmarker.createFromOptions(fileset, options);
            } catch {
              return vision.PoseLandmarker.createFromOptions(fileset, {
                ...options,
                baseOptions: { modelAssetPath: POSE_MODEL, delegate: "CPU" },
              });
            }
          })();
        }
        const landmarker = await poseLandmarkerPromiseRef.current;
        poseLandmarkerRef.current = landmarker;
        if (cancelled) return;

        setPosturePhase(postureBaselineRef.current ? "active" : "ready");
        setPostureInstruction(
          postureBaselineRef.current
            ? "Posture tracking active"
            : "Sit or stand upright for automatic calibration",
        );
        setPostureCameraState("Looking for posture");
        setPostureStatus("NO_PERSON");
        setPostureFullPoseReady(false);
        lastVideoTimeRef.current = -1;
        lastInferenceRef.current = 0;

        const invalidatePostureFrame = (
          timestamp: number,
          canvas: HTMLCanvasElement | null,
        ) => {
          postureInvalidSinceRef.current = timestamp;
          postureWindowRef.current = [];
          postureFlagsRef.current = { head: false, shoulders: false, back: false };
          postureDisplayedScoreRef.current = 0;
          postureScoreUpdatedAtRef.current = 0;
          sustainedBadRef.current = null;
          setPostureScore(0);
          setPostureScores({ neck: 0, head: 0, shoulderLevel: 0, upperBack: 0, framing: 0 });
          setPostureStatus("NO_PERSON");
          setPostureCorrection(null);
          setPostureAngles(null);
          setPostureConfidence(0);
          setPostureFullPoseReady(false);
          setPostureInstruction("Posture unavailable — return to the camera");
          setPostureCameraState("NO_PERSON");
          clearPostureCanvas(canvas);
          if (calibrationRef.current) {
            calibrationRef.current = {
              startedAt: timestamp,
              validMs: 0,
              lastValidAt: null,
              samples: [],
            };
            setCalibrationProgress(0);
          }
        };

        const processFrame = () => {
          if (cancelled) return;
          postureAnimationRef.current = window.requestAnimationFrame(processFrame);
          const video = cameraRef.current;
          const canvas = postureCanvasRef.current;
          const now = performance.now();
          if (!video || !canvas || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
            if (now - lastInferenceRef.current >= 500) {
              lastInferenceRef.current = now;
              invalidatePostureFrame(now, canvas);
            }
            return;
          }
          if (now - lastInferenceRef.current < 80) return;
          if (video.currentTime === lastVideoTimeRef.current) {
            if (now - lastInferenceRef.current >= 500) {
              lastInferenceRef.current = now;
              invalidatePostureFrame(now, canvas);
            }
            return;
          }
          lastInferenceRef.current = now;
          lastVideoTimeRef.current = video.currentTime;

          let frame;
          try {
            frame = extractPostureFrame(landmarker.detectForVideo(video, now));
          } catch {
            invalidatePostureFrame(now, canvas);
            return;
          }

          setPostureConfidence(Math.round(frame.confidence * 100));
          if (!frame.landmarks || !frame.measurement) {
            const invalidScore = frame.status === "NO_PERSON"
              ? 0
              : clamp(Math.round(frame.score ?? 0), 0, 29);
            postureInvalidSinceRef.current = now;
            setPostureScore(invalidScore);
            setPostureScores({
              neck: invalidScore,
              head: invalidScore,
              shoulderLevel: invalidScore,
              upperBack: invalidScore,
              framing: invalidScore,
            });
            postureDisplayedScoreRef.current = invalidScore;
            postureScoreUpdatedAtRef.current = 0;
            setPostureCorrection(null);
            setPostureAngles(null);
            setPostureFullPoseReady(false);
            setPostureInstruction(postureStatusCopy[frame.status]);
            setPostureCameraState(frame.status);
            setPostureStatus(frame.status);
            sustainedBadRef.current = null;
            postureWindowRef.current = [];
            postureFlagsRef.current = { head: false, shoulders: false, back: false };
            if (frame.landmarks) {
              drawPostureOverlay(canvas, video, frame.landmarks, false);
            } else {
              clearPostureCanvas(canvas);
            }
            if (calibrationRef.current) {
              calibrationRef.current = {
                startedAt: now,
                validMs: 0,
                lastValidAt: null,
                samples: [],
              };
              setCalibrationProgress(0);
            }
            return;
          }

          postureInvalidSinceRef.current = null;
          setPostureFullPoseReady(true);
          setPostureStatus("GOOD_POSTURE");
          if (!postureBaselineRef.current && !calibrationRef.current) {
            autoCalibrationStartedRef.current = true;
            calibrationRef.current = {
              startedAt: now,
              validMs: 0,
              lastValidAt: now,
              samples: [frame.measurement],
            };
            postureWindowRef.current = [];
            postureFlagsRef.current = { head: false, shoulders: false, back: false };
            postureDisplayedScoreRef.current = null;
            postureScoreUpdatedAtRef.current = 0;
            setPosturePhase("calibrating");
            setPostureScore(null);
            setPostureScores({ neck: 0, head: 0, shoulderLevel: 0, upperBack: 0, framing: 0 });
            setPostureCorrection(null);
            setPostureAngles(frame.measurement);
            setCalibrationProgress(0);
            setPostureInstruction("Sit or stand upright while baseline calibrates");
            setPostureCameraState("CALIBRATING_BASELINE");
            drawPostureOverlay(canvas, video, frame.landmarks, false);
            return;
          }
          const calibration = calibrationRef.current;
          if (calibration) {
            setPostureCameraState("Stage 2 · Saving baseline");
            if (calibration.lastValidAt !== null) {
              calibration.validMs += clamp(now - calibration.lastValidAt, 0, 250);
            }
            calibration.lastValidAt = now;
            calibration.samples.push(frame.measurement);
            setCalibrationProgress(
              clamp(Math.round((calibration.validMs / POSTURE_WINDOW_MS) * 100), 0, 100),
            );
            setPostureScore(null);
            setPostureCorrection(null);
            setPostureAngles(frame.measurement);
            setPostureInstruction("Hold your correct sitting posture");
            drawPostureOverlay(canvas, video, frame.landmarks, false);

            if (calibration.validMs >= POSTURE_WINDOW_MS && calibration.samples.length >= 8) {
              const baseline = aggregateMeasurements(calibration.samples, true);
              postureBaselineRef.current = baseline;
              calibrationRef.current = null;
              autoCalibrationStartedRef.current = false;
              postureWindowRef.current = [];
              postureFlagsRef.current = { head: false, shoulders: false, back: false };
              setPostureCalibrated(true);
              setPosturePhase("active");
              setPostureInstruction("Calibration saved");
              setCalibrationProgress(100);
              try {
                window.localStorage.setItem(POSTURE_BASELINE_KEY, JSON.stringify(baseline));
              } catch {
                // The baseline remains available for this session if storage is blocked.
              }
            }
            return;
          }

          const baseline = postureBaselineRef.current;
          if (!baseline) {
            setPostureCameraState("Stage 2 · Ready to calibrate");
            setPosturePhase("ready");
            setPostureScore(null);
            setPostureCorrection(null);
            setPostureAngles(frame.measurement);
            setPostureInstruction("Sit correctly, then calibrate");
            drawPostureOverlay(canvas, video, frame.landmarks, false);
            return;
          }

          postureWindowRef.current.push({ timestamp: now, measurement: frame.measurement });
          postureWindowRef.current = postureWindowRef.current.filter(
            (sample) => now - sample.timestamp <= POSTURE_WINDOW_MS,
          );
          const smoothed = aggregateMeasurements(
            postureWindowRef.current.map((sample) => sample.measurement),
            true,
          );
          const assessment = evaluatePosture(smoothed, baseline, postureFlagsRef.current);
          setPostureCameraState(assessment.status);
          setPostureStatus(assessment.status);
          postureFlagsRef.current = assessment.flags;
          setPosturePhase("active");
          if (now - postureScoreUpdatedAtRef.current >= 200) {
            postureDisplayedScoreRef.current = assessment.score;
            postureScoreUpdatedAtRef.current = now;
            setPostureScore(assessment.score);
          }
          setPostureScores(assessment.scores);
          setPostureAngles(smoothed);
          setPostureCorrection(assessment.correction);
          setPostureInstruction(postureStatusCopy[assessment.status]);
          drawPostureOverlay(
            canvas,
            video,
            frame.landmarks,
            assessment.status === "GOOD_POSTURE" &&
              assessment.score >= 85 &&
              frame.confidence >= POSTURE_LANDMARK_CONFIDENCE,
          );

          if (!assessment.correction) {
            sustainedBadRef.current = null;
            return;
          }
          if (sustainedBadRef.current?.correction !== assessment.correction) {
            sustainedBadRef.current = { correction: assessment.correction, since: now, sounded: false };
            return;
          }
          const sustained = sustainedBadRef.current;
          if (sustained.sounded || now - sustained.since < 3000 || !voiceOnRef.current) return;
          const audio = audioContextRef.current;
          if (!audio || audio.state !== "running") return;
          const oscillator = audio.createOscillator();
          const gain = audio.createGain();
          oscillator.type = "sine";
          oscillator.frequency.setValueAtTime(440, audio.currentTime);
          oscillator.frequency.exponentialRampToValueAtTime(520, audio.currentTime + 0.16);
          gain.gain.setValueAtTime(0.0001, audio.currentTime);
          gain.gain.exponentialRampToValueAtTime(0.028, audio.currentTime + 0.025);
          gain.gain.exponentialRampToValueAtTime(0.0001, audio.currentTime + 0.2);
          oscillator.connect(gain).connect(audio.destination);
          oscillator.start();
          oscillator.stop(audio.currentTime + 0.21);
          sustained.sounded = true;
        };

        processFrame();
      } catch {
        poseLandmarkerPromiseRef.current = null;
        if (cancelled) return;
        setPosturePhase("error");
        setPostureScore(null);
        setPostureCorrection(null);
        setPostureInstruction("Posture model unavailable");
        setPostureCameraState("Check your connection");
        setPostureStatus("NO_PERSON");
        setPostureFullPoseReady(false);
        clearPostureCanvas(postureCanvasRef.current);
      }
    };

    void startTracking();
    return () => {
      cancelled = true;
      if (postureAnimationRef.current !== null) {
        window.cancelAnimationFrame(postureAnimationRef.current);
        postureAnimationRef.current = null;
      }
      clearPostureCanvas(canvasNode);
    };
  }, [cameraOn, mode]);

  const scores = useMemo(() => {
    const movement = cameraOn ? pulse : 2;
    const base = mode === "upload" ? 82 : 88;
    return [
      { label: "Overall", value: Math.min(99, base + ((movement * 3) % 9)) },
      { label: "Form", value: Math.min(99, base + 3 + ((movement * 5) % 7)) },
      { label: "Balance", value: Math.max(58, base - 8 + ((movement * 7) % 13)) },
      { label: "Footwork", value: Math.max(60, base - 10 + ((movement * 4) % 16)) },
      { label: "Speed", value: Math.max(60, base - 6 + ((movement * 6) % 14)) },
    ];
  }, [cameraOn, mode, pulse]);

  const postureCameraFeedback = getPostureCameraFeedback({
    calibrated: postureCalibrated,
    cameraOn,
    confidence: postureConfidence,
    facingConfidence: postureAngles?.facingConfidence ?? null,
    fullPoseReady: postureFullPoseReady,
    phase: posturePhase,
    score: postureScore,
    status: postureStatus,
  });
  const dashboardScores: { label: string; value: number | null }[] = mode === "posture"
    ? [
        { label: "Overall", value: cameraOn ? postureCameraFeedback.score : null },
        { label: "Neck alignment", value: !cameraOn || postureScore === null ? null : postureScores.neck },
        { label: "Head position", value: !cameraOn || postureScore === null ? null : postureScores.head },
        { label: "Shoulder level", value: !cameraOn || postureScore === null ? null : postureScores.shoulderLevel },
        { label: "Upper-back angle", value: !cameraOn || postureScore === null ? null : postureScores.upperBack },
        { label: "Body framing", value: !cameraOn || postureScore === null ? null : postureScores.framing },
        { label: "Confidence", value: cameraOn ? postureConfidence : null },
      ]
    : scores;
  const postureAlerting = mode === "posture" &&
    cameraOn &&
    posturePhase === "active" &&
    postureCameraFeedback.tone === "red";
  const correctionHot = mode === "posture"
    ? postureAlerting
    : scores[2].value < 86 || scores[3].value < 82;
  const tracking = useMemo(
    () => ({
      shotsMade: 11 + ((pulse * 2) % 5),
      shotsTaken: 16 + ((pulse * 2) % 6),
      passes: 24 + pulse,
      dribbles: 87 + pulse * 3,
    }),
    [pulse],
  );

  const ensurePostureAudio = () => {
    if (typeof window === "undefined" || !("AudioContext" in window)) return;
    if (!audioContextRef.current) audioContextRef.current = new AudioContext();
    if (audioContextRef.current.state === "suspended") void audioContextRef.current.resume();
  };

  const selectMode = (nextMode: Mode) => {
    if (nextMode === "posture") ensurePostureAudio();
    setPostureFullPoseReady(false);
    setMode(nextMode);
  };

  const beginCalibration = () => {
    if (!cameraOn || mode !== "posture") return;
    ensurePostureAudio();
    if (posturePhase === "loading") {
      setPostureInstruction("Loading posture model");
      return;
    }
    if (posturePhase === "error" || posturePhase === "calibrating") return;
    if (!postureFullPoseReady) {
      setPostureInstruction("Move back until your ears, shoulders, and hips are visible");
      return;
    }
    calibrationRef.current = {
      startedAt: performance.now(),
      validMs: 0,
      lastValidAt: null,
      samples: [],
    };
    autoCalibrationStartedRef.current = false;
    postureWindowRef.current = [];
    postureFlagsRef.current = { head: false, shoulders: false, back: false };
    postureDisplayedScoreRef.current = null;
    postureScoreUpdatedAtRef.current = 0;
    sustainedBadRef.current = null;
    setPosturePhase("calibrating");
    setPostureScore(null);
    setPostureCorrection(null);
    setCalibrationProgress(0);
    setPostureInstruction("Hold your correct sitting posture");
  };

  const coachSpeak = (text = modeCopy[mode].cue) => {
    if (mode === "posture" || !voiceOn || typeof window === "undefined" || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    const cue = new SpeechSynthesisUtterance(text);
    cue.rate = 1.04;
    cue.pitch = 0.92;
    window.speechSynthesis.speak(cue);
  };

  useEffect(() => {
    if (mode === "posture" || !cameraOn || !voiceOn || pulse === 0 || pulse % 12 !== 0) return;
    coachSpeak();
    // The changing pose pulse intentionally schedules brief live coaching cues.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraOn, mode, pulse, voiceOn]);

  const startCamera = async () => {
    if (cameraOn) {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      setCameraOn(false);
      calibrationRef.current = null;
      setPosturePhase("idle");
      setPostureScore(null);
      setPostureCorrection(null);
      setPostureAngles(null);
      setPostureConfidence(0);
      setPostureCameraState("Waiting for camera");
      setPostureInstruction("Start the camera to begin");
      setCalibrationProgress(0);
      setPostureFullPoseReady(false);
      setPostureStatus("NO_PERSON");
      return;
    }
    try {
      if (mode === "posture") ensurePostureAudio();
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: "user",
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      streamRef.current = stream;
      if (cameraRef.current) {
        cameraRef.current.srcObject = stream;
        await cameraRef.current.play();
      }
      setCameraOn(true);
      setSeconds(0);
      setChatOpen(true);
      if (mode === "posture") {
        postureBaselineRef.current = null;
        calibrationRef.current = null;
        autoCalibrationStartedRef.current = false;
        setPostureCalibrated(false);
        setPostureInstruction("Sit or stand upright for automatic calibration");
        setPostureCameraState("Finding face, shoulders, torso, and hips");
      } else {
        coachSpeak("HoopVision live coach active. Set your base and begin when ready.");
      }
    } catch {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      setCameraOn(false);
      if (mode === "posture") {
        setPostureInstruction("Camera access blocked");
        setPostureCameraState("Allow camera access");
        setPostureFullPoseReady(false);
        setPostureStatus("NO_PERSON");
      }
      setChatOpen(true);
      setChat((items) => [
        ...items,
        { who: "coach", text: "Camera access was blocked. Allow camera access to run live posture detection." },
      ]);
    }
  };

  const handleUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setMode("upload");
    setVideoName(file.name);
  };

  const sendMessage = (event: FormEvent) => {
    event.preventDefault();
    if (!message.trim()) return;
    const prompt = message.trim();
    setChat((items) => [...items, { who: "you", text: prompt }]);
    setMessage("");
    window.setTimeout(() => {
      setChat((items) => [
        ...items,
        {
          who: "coach",
          text:
            mode === "posture"
              ? postureScore === null
                ? postureInstruction
                : postureCorrection
                  ? `Current correction: ${correctionCopy[postureCorrection]}.`
                  : `Posture aligned at ${postureScore}.`
              : "Your base is strong. Fix the red knee drift first, then speed up the gather once the landing turns green.",
        },
      ]);
    }, 420);
  };

  const postureStage = !cameraOn
    ? { label: "01 / FRAME BODY", value: "START CAMERA" }
    : posturePhase === "loading"
      ? { label: "01 / FRAME BODY", value: "LOADING" }
      : posturePhase === "error"
        ? { label: "POSTURE TRACKING", value: "UNAVAILABLE" }
        : !postureFullPoseReady
          ? {
              label: "01 / FRAME BODY",
              value: postureInstruction.startsWith("Move back") ? "MOVE BACK" : "ADJUST",
            }
          : posturePhase === "calibrating"
            ? { label: "02 / SAVE BASELINE", value: `${calibrationProgress}%` }
            : !postureCalibrated
              ? { label: "02 / SAVE BASELINE", value: "READY" }
              : { label: "03 / LIVE COACH", value: postureScore === null ? "LOCKING" : "ACTIVE" };
  const calibrationDisabled = !cameraOn ||
    !postureFullPoseReady ||
    posturePhase === "loading" ||
    posturePhase === "error" ||
    posturePhase === "calibrating";
  const calibrationButtonText = !cameraOn
    ? "Start camera first"
    : posturePhase === "loading"
      ? "Loading posture tracking"
      : posturePhase === "error"
        ? "Tracking unavailable"
        : !postureFullPoseReady
          ? postureInstruction.startsWith("Move back")
            ? "Move back to calibrate"
            : posturePhase === "calibrating"
              ? "Reframe to continue"
              : "Full pose needed"
          : posturePhase === "calibrating"
            ? `Hold still · ${calibrationProgress}%`
            : postureCalibrated
              ? "Recalibrate posture"
              : "Save correct posture";
  const postureTrackingStatus = !cameraOn
    ? "Standby"
    : posturePhase === "loading"
      ? "Loading"
      : !postureFullPoseReady
        ? "Framing"
        : posturePhase === "calibrating"
          ? "Calibrating"
          : postureScore === null
            ? "Reading"
            : "Tracking";

  return (
    <main>
      <header className="nav-shell">
        <a className="brand" href="#home" aria-label="HoopVision AI home">
          <Mark />
          <span>HoopVision</span>
          <em>AI</em>
        </a>
        <nav aria-label="Primary navigation">
          <a href="#live">Live Coach</a>
          <a href="#capabilities">Platform</a>
          <a href="#intelligence">Film IQ</a>
          <a href="#training">Training Plans</a>
          <a href="#progress">Progress</a>
          <a href="#pricing">Pricing</a>
        </nav>
        <a className="sign-in" href="#profile">Sign In <Arrow /></a>
      </header>

      <section className="hero" id="home">
        <div className="hero-media" aria-hidden="true">
          <div className="arena-grid" />
          <div className="rim" />
          <div className="hero-player">
            <span className="limb l1" />
            <span className="limb l2" />
            <span className="limb l3" />
            <span className="limb l4" />
            <span className="torso" />
            <span className="head" />
            <span className="hud-ball" />
          </div>
        </div>
        <div className="hero-copy">
          <div className="eyebrow"><i /> Elite basketball coaching powered by advanced AI</div>
          <h1>HoopVision AI</h1>
          <p>One AI performance system for live form correction, skill tracking, game-film intelligence, pro comparison, and a development plan built around you.</p>
          <div className="hero-actions">
            <a className="button button-primary" href="#live">Start Live Coaching <Arrow /></a>
            <button className="button button-ghost" data-mode="posture" onClick={() => selectMode("posture")}>Try Posture Mode</button>
          </div>
        </div>
        <div className="hero-panel glass">
          <span>LIVE READINESS</span>
          <strong>97%</strong>
          <small>Live motion, voice cues, skill tracking, film intelligence</small>
        </div>
      </section>

      <section className="mode-strip" aria-label="Coaching modes">
        {(Object.keys(modeCopy) as Mode[]).map((item) => (
          <button className={mode === item ? "active" : ""} data-mode={item} onClick={() => selectMode(item)} key={item}>
            {modeCopy[item].label}
          </button>
        ))}
      </section>

      <section className="section live-section" id="live">
        <div className="section-heading">
          <div>
            <span className="section-no">01 / LIVE AI COACH</span>
            <h2 id="modeTitle">{modeCopy[mode].title}</h2>
          </div>
          <p>
            {mode === "posture"
              ? "MediaPipe tracks your ears, shoulders, hips, neck, and upper back on-device. Your camera stays in the browser."
              : "Real-time camera analysis uses a prototype coaching stream with scoring shaped after OpenCV and MediaPipe pose landmarks."}
          </p>
        </div>

        <div className="live-grid">
          <div className="camera-shell glass">
            <div className="panel-head">
              <span id="modeEyebrow">{modeCopy[mode].eyebrow}</span>
              <b
                id="liveState"
                className={cameraOn
                  ? mode === "posture"
                    ? postureCameraFeedback.tone === "green"
                      ? "posture-online-green"
                      : "posture-online-red"
                    : "online"
                  : ""}
              >
                {cameraOn ? "LIVE" : "STANDBY"}
              </b>
            </div>
            <div className={`camera-stage ${cameraOn ? "camera-live" : ""} ${correctionHot ? "has-correction" : "all-clear"} ${mode === "posture" ? `posture-mode ${cameraOn ? `posture-feedback-${postureCameraFeedback.tone}` : ""}` : ""}`}>
              <video id="cameraVideo" ref={cameraRef} autoPlay playsInline muted hidden={!cameraOn} />
              {mode === "posture" ? <canvas className="posture-canvas" ref={postureCanvasRef} aria-hidden="true" /> : null}
              {mode === "posture" && cameraOn ? (
                <div
                  className={`posture-score-overlay posture-${postureCameraFeedback.tone} ${postureAlerting ? "posture-alerting" : ""}`}
                  data-posture-status={postureStatus}
                  role="status"
                  aria-live="polite"
                >
                  <span>POSTURE SCORE</span>
                  <div><strong>{postureCameraFeedback.score}</strong><small>/ 100</small></div>
                  <p>{postureCameraFeedback.message}</p>
                </div>
              ) : null}
              <div className="camera-placeholder" id="cameraPlaceholder" hidden={cameraOn}>
                <Mark />
                <h3>Camera HUD Ready</h3>
                <p>
                  {mode === "posture"
                    ? "Start the camera, sit fully in view, then save your correct posture."
                    : "Start a session to see instant red and green body feedback."}
                </p>
              </div>
              {mode !== "posture" ? (
                <div className="skeleton-overlay" aria-hidden="true">
                  {["head", "neck", "chest", "hip", "l-shoulder", "r-shoulder", "l-elbow", "r-elbow", "l-wrist", "r-wrist", "l-knee", "r-knee", "l-ankle", "r-ankle"].map((joint) => (
                    <span className={`joint ${joint} ${joint.includes("r-") || joint === "hip" ? "red" : "green"}`} key={joint} />
                  ))}
                  <i className="bone spine" />
                  <i className="bone shoulders" />
                  <i className="bone hips" />
                  <i className="bone arm-left" />
                  <i className="bone arm-right redline" />
                  <i className="bone leg-left" />
                  <i className="bone leg-right redline" />
                </div>
              ) : null}
              {mode === "posture" && cameraOn ? (
                <div
                  className="posture-calibration"
                  data-phase={posturePhase}
                  data-state={postureCameraFeedback.tone === "green" ? "ready" : postureCameraFeedback.tone === "red" ? "bad" : "calibrating"}
                >
                  <div>
                    <span>{postureStage.label}</span>
                    <b>{postureStage.value}</b>
                  </div>
                  <i><em style={{ width: `${posturePhase === "calibrating" ? calibrationProgress : postureCalibrated && postureFullPoseReady ? 100 : 0}%` }} /></i>
                </div>
              ) : null}
              {mode !== "posture" ? <div className="scan" /> : null}
              <div className={`coach-caption ${mode === "posture" ? !cameraOn ? "paused" : postureCameraFeedback.tone === "green" ? "aligned" : "correction" : ""}`}>
                <b>Coach V</b>
                <span id="liveCue">
                  {mode === "posture"
                    ? cameraOn
                      ? postureCameraFeedback.message
                      : "Start the camera to begin"
                    : modeCopy[mode].cue}
                </span>
              </div>
            </div>
            <div className="control-row">
              <button className="button button-primary" id="cameraButton" onClick={startCamera}>
                {cameraOn ? mode === "posture" ? "Stop Posture Coach" : "Stop Live Coach" : "Start Camera"} <Arrow />
              </button>
              {mode === "posture" ? (
                <button
                  className="button button-ghost calibrate-button"
                  id="calibratePosture"
                  onClick={beginCalibration}
                  disabled={calibrationDisabled}
                >
                  {calibrationButtonText}
                </button>
              ) : null}
              <label className="button button-ghost">
                Upload Clip
                <input id="uploadInput" type="file" accept="video/mp4,video/quicktime,video/webm" onChange={handleUpload} />
              </label>
              <button className="delete-button" id="deleteVideo" hidden={!videoName} onClick={() => setVideoName("")}>Delete {videoName || "clip"}</button>
            </div>
          </div>

          <aside className="dashboard-shell glass">
            <div className="panel-head">
              <span>LIVE DASHBOARD</span>
              <span id="timerTop">{formatTime(seconds)}</span>
            </div>
            <div className={`score-ring ${mode === "posture" ? !cameraOn || postureCameraFeedback.tone === "neutral" ? "posture-paused" : postureCameraFeedback.tone === "green" ? "posture-good" : "posture-bad" : ""}`}>
              <strong id="overallScore">{dashboardScores[0].value ?? "—"}</strong>
              <span>{mode === "posture" ? "POSTURE SCORE" : "OVERALL SCORE"}</span>
            </div>
            <div className="metrics">
              {dashboardScores.slice(1).map((score) => (
                <div data-metric={score.label} key={score.label}>
                  <span>{score.label}</span>
                  <i><em style={{ width: `${score.value ?? 0}%` }} /></i>
                  <b>{score.value ?? "—"}</b>
                </div>
              ))}
            </div>
            <div className="session-stats">
              <article>
                <span>{mode === "posture" ? "BASELINE" : "REPS"}</span>
                <strong id="repCount">{mode === "posture" ? postureCalibrated ? "SAVED" : "NEEDED" : reps}</strong>
              </article>
              <article><span>TIMER</span><strong id="sessionTimer">{formatTime(seconds)}</strong></article>
              <button
                id="voiceButton"
                className={voiceOn ? "voice-on" : ""}
                onClick={() => {
                  if (!voiceOn && mode === "posture") ensurePostureAudio();
                  setVoiceOn((value) => !value);
                }}
                aria-pressed={voiceOn}
              >
                <span>{mode === "posture" ? "SOFT ALERT" : "VOICE"}</span><strong id="voiceState">{voiceOn ? "ON" : "OFF"}</strong>
              </button>
            </div>
            {mode === "posture" ? (
              <div className="skill-tracking posture-tracking">
                <div className="tracking-head">
                  <h3>Live posture tracking</h3>
                  <span className={postureFullPoseReady ? "tracking-ready" : "tracking-paused"}><i /> {postureTrackingStatus}</span>
                </div>
                <div className="tracking-grid">
                  <article>
                    <span>CAMERA</span>
                    <strong>{postureConfidence}%</strong>
                    <small>{postureCameraState}</small>
                  </article>
                  <article>
                    <span>NECK</span>
                    <strong>{postureAngles ? `${Math.round(postureAngles.neckAngle)}°` : "—"}</strong>
                    <small>2-second smoothing</small>
                  </article>
                  <article>
                    <span>SHOULDERS</span>
                    <strong>{postureAngles ? `${Math.round(postureAngles.shoulderTilt)}°` : "—"}</strong>
                    <small>Live level angle</small>
                  </article>
                </div>
              </div>
            ) : (
              <div className="skill-tracking">
                <div className="tracking-head">
                  <h3>Auto Skill Tracking</h3>
                  <span><i /> Recording</span>
                </div>
                <div className="tracking-grid">
                  <article><span>SHOTS</span><strong id="shotCount">{tracking.shotsMade}/{tracking.shotsTaken}</strong><small id="shotAccuracy">{Math.round((tracking.shotsMade / tracking.shotsTaken) * 100)}% accuracy</small></article>
                  <article><span>PASSES</span><strong id="passCount">{tracking.passes}</strong><small>92% on target</small></article>
                  <article><span>DRIBBLES</span><strong id="dribbleCount">{tracking.dribbles}</strong><small>3.8 per second</small></article>
                </div>
              </div>
            )}
            <div className="tip-stack">
              <h3>{mode === "posture" ? "Current instruction" : "Instant Coaching Tips"}</h3>
              {mode === "posture" ? (
                <p className={!cameraOn ? "neutral-tip" : postureCameraFeedback.tone === "green" ? "green-tip" : "red-tip"}>
                  {cameraOn ? postureCameraFeedback.message : "Start the camera to begin"}
                </p>
              ) : (
                <>
                  <p className="green-tip">Green: shoulders stacked, wrist path clean, base stable.</p>
                  <p className="red-tip">Red: right knee drift and late hip load need correction.</p>
                  <button className="hear-cue" id="hearCue" onClick={() => coachSpeak()}>Hear live cue <Arrow /></button>
                </>
              )}
            </div>
          </aside>
        </div>
      </section>

      <section className="section capabilities-section" id="capabilities">
        <div className="story-heading">
          <div>
            <span className="section-no">02 / FROM MISTAKES TO MUSCLE MEMORY</span>
            <h2>See it. Fix it. Master it.</h2>
          </div>
          <p>Every practice becomes smarter with live AI coaching.</p>
        </div>
        <div className="performance-loop glass">
          {[
            ["01", "Analyze", "Every movement is tracked."],
            ["02", "Correct", "Fix mistakes as they happen."],
            ["03", "Repeat", "Practice exactly what you need."],
            ["04", "Master", "Watch your game level up."],
          ].map((step) => (
            <article key={step[1]}>
              <span>{step[0]}</span>
              <h3>{step[1]}</h3>
              <p>{step[2]}</p>
            </article>
          ))}
          <i className="loop-signal" />
        </div>
      </section>

      <section className="section training-section" id="training">
        <div className="story-heading">
          <div>
            <span className="section-no">03 / ADAPTIVE TRAINING</span>
            <h2>Your next workout writes itself.</h2>
          </div>
          <p>Computer vision finds the leak. Coach V builds the fix.</p>
        </div>

        <div className="training-system glass">
          <aside className="plan-rail">
            <div className="plan-status">
              <span>SESSION 024</span>
              <b><i /> AI plan ready</b>
            </div>
            <div className="plan-list" role="tablist" aria-label="AI generated workout">
              {drills.map((drill, index) => (
                <button
                  className={activeDrill === index ? "active" : ""}
                  data-drill={index}
                  key={drill.title}
                  onClick={() => setActiveDrill(index)}
                  role="tab"
                  aria-selected={activeDrill === index}
                >
                  <span>0{index + 1}</span>
                  <div><strong>{drill.title}</strong><small>{drill.focus}</small></div>
                  <em>{drill.duration}</em>
                </button>
              ))}
            </div>
            <div className="plan-summary">
              <span>15 MIN</span>
              <span>38 REPS</span>
              <span>1 PRIORITY</span>
            </div>
          </aside>

          <div className="drill-workspace">
            <div className="workspace-bar">
              <span>LIVE DRILL PREVIEW</span>
              <b>Generated from your last 42 reps</b>
            </div>
            <div className="drill-stage">
              <div className="drill-visual" aria-label={`${drills[activeDrill].title} computer vision preview`}>
                <div className="vision-orbit orbit-one" />
                <div className="vision-orbit orbit-two" />
                <div className="drill-pose" aria-hidden="true">
                  <i className="pose-head" />
                  <i className="pose-body" />
                  <i className="pose-arm-a" />
                  <i className="pose-arm-b" />
                  <i className="pose-leg-a" />
                  <i className="pose-leg-b" />
                  <b className="pose-joint joint-good" />
                  <b className="pose-joint joint-fix" />
                </div>
                <span className="vision-label"><i /> Tracking 33 landmarks</span>
                <span className="correction-label" id="drillFocus">Fix: {drills[activeDrill].focus}</span>
              </div>
              <div className="drill-brief">
                <span className="ai-kicker">COACH V / PRIORITY 01</span>
                <h3 id="drillTitle">{drills[activeDrill].title}</h3>
                <p id="drillCue">{drills[activeDrill].cue}</p>
                <div className="drill-specs">
                  <article><span>TIME</span><strong id="drillDuration">{drills[activeDrill].duration}</strong></article>
                  <article><span>VOLUME</span><strong id="drillReps">{drills[activeDrill].reps}</strong></article>
                  <article><span>TARGET</span><strong id="drillTarget">{drills[activeDrill].target}</strong></article>
                </div>
                <div className="target-line">
                  <div><span>Movement match</span><b id="drillMatch">74 → {drills[activeDrill].target}</b></div>
                  <i><em id="drillProgress" style={{ width: `${drills[activeDrill].target}%` }} /></i>
                </div>
                <button className="start-workout">Start live workout <Arrow /></button>
              </div>
            </div>
            <div className="training-loop">
              <span><b>01</b> Detect</span><i />
              <span><b>02</b> Correct</span><i />
              <span><b>03</b> Re-test</span><i />
              <span><b>04</b> Progress</span>
            </div>
          </div>
        </div>
      </section>

      <section className="section intelligence-section" id="intelligence">
        <div className="story-heading">
          <div>
            <span className="section-no">04 / VISION LAB</span>
            <h2>Every frame has an answer.</h2>
          </div>
          <p>See the read. Compare the move. Build the report.</p>
        </div>
        <div className="intelligence-tabs" role="tablist" aria-label="Basketball intelligence tools">
          {intelligenceTabs.map((tab, index) => (
            <button
              className={intelligenceView === tab.id ? "active" : ""}
              key={tab.id}
              onClick={() => setIntelligenceView(tab.id)}
              onKeyDown={(event) => {
                if (event.key !== "ArrowRight" && event.key !== "ArrowLeft") return;
                event.preventDefault();
                const direction = event.key === "ArrowRight" ? 1 : -1;
                const next = intelligenceTabs[(index + direction + intelligenceTabs.length) % intelligenceTabs.length];
                setIntelligenceView(next.id);
                window.requestAnimationFrame(() => document.getElementById(`intelligence-tab-${next.id}`)?.focus());
              }}
              id={`intelligence-tab-${tab.id}`}
              aria-controls={`intelligence-panel-${tab.id}`}
              role="tab"
              aria-selected={intelligenceView === tab.id}
              tabIndex={intelligenceView === tab.id ? 0 : -1}
            >
              {tab.label}
            </button>
          ))}
        </div>
        <div className="intelligence-stage glass">
          <div className="intelligence-bar">
            <span>ATHLETE / JORDAN MILES</span>
            <b><i /> 6 sessions synced</b>
          </div>

          <div className="film-layout" id="intelligence-panel-film" role="tabpanel" aria-labelledby="intelligence-tab-film" hidden={intelligenceView !== "film"}>
              <div className="film-screen" aria-label="Game film analysis preview">
                <div className="court-map">
                  <i className="court-arc" />
                  <span className="player-marker ball-handler">JM</span>
                  <span className="player-marker teammate-one">1</span>
                  <span className="player-marker teammate-two">2</span>
                  <span className="player-marker defender-one">D</span>
                  <span className="player-marker defender-two">D</span>
                  <em className="passing-lane" />
                </div>
                <div className="possession-chip"><span>Q3 · 04:18</span><b>Missed corner read</b></div>
                <div className="film-timeline">
                  <i /><i className="good" /><i className="warning" /><i /><i className="good" />
                  <span style={{ left: "54%" }} />
                </div>
              </div>
              <aside className="decision-panel">
                <span className="analysis-label">POSSESSION INTELLIGENCE</span>
                <strong>82</strong>
                <small>Decision score</small>
                <article><b>+0.8s</b><p>Ball held too long after the help defender committed.</p></article>
                <article className="positive"><b>92%</b><p>Corner pass was the highest-value read on this frame.</p></article>
                <button onClick={() => setIntelligenceView("compare")}>Compare movement <Arrow /></button>
              </aside>
          </div>

          <div className="compare-layout" id="intelligence-panel-compare" role="tabpanel" aria-labelledby="intelligence-tab-compare" hidden={intelligenceView !== "compare"}>
              <article className="compare-frame athlete-frame">
                <div className="compare-label"><span>YOU</span><b>0.54s release</b></div>
                <div className="motion-figure" aria-hidden="true"><i /><i /><i /><i /><i /></div>
                <div className="angle-tag red-angle">ELBOW 71°</div>
              </article>
              <div className="comparison-delta">
                <span>KEY GAP</span>
                <strong>-11°</strong>
                <p>Raise the set point before the elbow extends.</p>
                <em>87% movement match</em>
              </div>
              <article className="compare-frame pro-frame">
                <div className="compare-label"><span>ELITE MODEL</span><b>0.47s release</b></div>
                <div className="motion-figure pro-motion" aria-hidden="true"><i /><i /><i /><i /><i /></div>
                <div className="angle-tag green-angle">ELBOW 82°</div>
              </article>
          </div>

          <div className="report-layout" id="intelligence-panel-report" role="tabpanel" aria-labelledby="intelligence-tab-report" hidden={intelligenceView !== "report"}>
              <div className="report-score">
                <span>PLAYER IMPACT</span>
                <strong id="reportGrade">{reportReady ? "A−" : "—"}</strong>
                <small id="reportStatus">{reportReady ? "Development trajectory: rising" : "Ready to analyze 6 sessions"}</small>
              </div>
              <div className="report-findings">
                <article>
                  <span>STRENGTHS</span>
                  <h3>Stable base · Quick first read · High motor</h3>
                  <p>Shot preparation improved 14% across the last three sessions.</p>
                </article>
                <article>
                  <span>DEVELOPMENT PRIORITY</span>
                  <h3>Right-knee alignment under speed</h3>
                  <p>Correcting the inward drift should improve balance and finishing efficiency.</p>
                </article>
              </div>
              <button className="report-button" id="reportButton" onClick={() => setReportReady(true)}>
                {reportReady ? "Scouting report ready" : "Generate AI scouting report"} <Arrow />
              </button>
          </div>
        </div>

      </section>

      <section className="section progress-section" id="progress">
        <div className="story-heading">
          <div>
            <span className="section-no">05 / VERIFIED PROGRESS</span>
            <h2>Proof in every frame.</h2>
          </div>
          <p>No vanity stats. Just movement getting better.</p>
        </div>

        <div className="progress-command glass">
          <div className="progress-topbar">
            <div><span>PERFORMANCE INDEX</span><b>Last 30 days</b></div>
            <div className="progress-status"><i /> Model confidence 98.4%</div>
          </div>
          <div className="progress-main">
            <div className="progress-score">
              <span>OVERALL FORM</span>
              <strong>91</strong>
              <b>+29 <small>since baseline</small></b>
              <p>Release, balance, and landing control are now consistently green.</p>
            </div>
            <div className="trajectory-chart" aria-label="Form score increased from 62 to 91 over nine sessions">
              <div className="trajectory-grid"><i /><i /><i /><i /></div>
              <div className="trajectory-area" />
              {[62, 69, 66, 74, 78, 82, 80, 88, 91].map((value, index) => (
                <span
                  className={index === 8 ? "current" : ""}
                  style={{ left: `${index * 12.5}%`, bottom: `${((value - 55) / 40) * 100}%` }}
                  key={value + index}
                >
                  <i />
                  {index === 0 || index === 8 ? <b>{value}</b> : null}
                </span>
              ))}
              <div className="chart-axis"><span>BASELINE</span><span>SESSION 09</span></div>
            </div>
          </div>
          <div className="progress-signals">
            <article><span>RELEASE</span><strong>0.47s</strong><small>0.08s faster</small></article>
            <article><span>BALANCE</span><strong>93</strong><small>+22 points</small></article>
            <article><span>GREEN REPS</span><strong>87%</strong><small>Last 100 reps</small></article>
            <article><span>STREAK</span><strong>12 days</strong><small>Best: 18</small></article>
          </div>
          <div className="progress-compare">
            <div>
              <span>BEFORE / SESSION 01</span>
              <strong>Late load. Knee drift.</strong>
              <em className="compare-state red-state">62</em>
            </div>
            <i><span /></i>
            <div>
              <span>NOW / SESSION 09</span>
              <strong>Stable base. Clean release.</strong>
              <em className="compare-state green-state">91</em>
            </div>
          </div>
        </div>
      </section>

      <section className="section coach-section" id="coach">
        <div className="story-heading">
          <div>
            <span className="section-no">06 / COACH OS</span>
            <h2>Know who needs you.</h2>
          </div>
          <p>AI triages the roster. You make the call.</p>
        </div>

        <div className="legacy-roster-shell glass">
          <div className="legacy-roster-bar">
            <div><span>HOOPVISION AI</span><b><i /> Live roster</b></div>
            <button>View team <Arrow /></button>
          </div>
          <div className="legacy-roster-head" aria-hidden="true">
            <span>Player</span><span>Current focus</span><span>AI score</span><span>Live status</span><span />
          </div>
          <div className="legacy-player-list">
            {athletes.map((athlete) => (
              <article className={athlete.tone} key={athlete.name}>
                <div className="legacy-player">
                  <b>{athlete.initials}</b>
                  <span><strong>{athlete.name}</strong><small>Trend {athlete.trend}</small></span>
                </div>
                <div className="legacy-focus"><strong>{athlete.focus}</strong><small>AI session</small></div>
                <div className="legacy-score"><strong>{athlete.score}</strong><small>/ 100</small></div>
                <div className="legacy-status"><i /><span>{athlete.status}</span></div>
                <button aria-label={`Open ${athlete.name} player profile`}><Arrow /></button>
              </article>
            ))}
          </div>
          <div className="legacy-roster-footer">
            <span>HoopVision AI performance team</span>
            <div><b>2</b> live now <i /> <b>0</b> need review</div>
          </div>
        </div>
      </section>

      <section className="section research-section" id="research">
        <div className="story-heading">
          <div>
            <span className="section-no">07 / REAL-TIME SYSTEM</span>
            <h2>Camera to cue. One motion.</h2>
          </div>
          <p>Computer vision that sees movement—not identity.</p>
        </div>

        <div className="system-shell glass">
          <div className="system-bar">
            <span>VISION PIPELINE / LIVE</span>
            <b><i /> &lt;120ms response target</b>
          </div>
          <div className="vision-pipeline">
            {[
              ["01", "Capture", "WebRTC", "Private camera stream"],
              ["02", "Track", "MediaPipe", "33 body landmarks"],
              ["03", "Reason", "Python + OpenCV", "Biomechanics scored"],
              ["04", "Coach", "FastAPI + WebSockets", "Cue delivered live"],
            ].map((step) => (
              <article key={step[1]}>
                <span>{step[0]}</span>
                <div><h3>{step[1]}</h3><b>{step[2]}</b></div>
                <p>{step[3]}</p>
              </article>
            ))}
            <i className="pipeline-signal" />
          </div>
          <div className="system-proof">
            <article><strong>33</strong><span>pose landmarks</span></article>
            <article><strong>&lt;120ms</strong><span>response target</span></article>
            <article><strong>0</strong><span>face templates</span></article>
          </div>
        </div>

        <div className="trust-panel">
          <div>
            <span className="section-no">PRIVATE BY DESIGN</span>
            <h3>Your movement data is yours.</h3>
          </div>
          <div className="trust-list">
            <span><i>✓</i> No facial recognition</span>
            <span><i>✓</i> Private by default</span>
            <span><i>✓</i> Delete any session</span>
            <span><i>✓</i> Parent consent for minors</span>
          </div>
        </div>
      </section>

      <section className="section pricing-section" id="pricing">
        <div className="story-heading pricing-heading">
          <div>
            <span className="section-no">08 / MEMBERSHIP</span>
            <h2>Start with one rep.</h2>
          </div>
          <div>
            <p>Go live free. Upgrade when coaching becomes your habit.</p>
            <div className="billing-toggle" aria-label="Billing period">
              <button data-billing="monthly" className={!annualBilling ? "active" : ""} onClick={() => setAnnualBilling(false)}>Monthly</button>
              <button data-billing="annual" className={annualBilling ? "active" : ""} onClick={() => setAnnualBilling(true)}>Annual <span>Save 20%</span></button>
            </div>
          </div>
        </div>

        <div className="pricing-shell">
          <article className="elite-plan">
            <div className="elite-glow" />
            <div className="plan-topline"><span>MOST POPULAR</span><b>7 days free</b></div>
            <div className="elite-copy">
              <span>{plans[1].name}</span>
              <h3>Your AI coach. Every session.</h3>
              <div className="plan-price">
                <strong id="elitePrice">${annualBilling ? "19" : plans[1].price}</strong>
                <span>/ month<small id="eliteBilling">{annualBilling ? "billed annually" : "cancel anytime"}</small></span>
              </div>
            </div>
            <div className="elite-features">
              {["Unlimited live coaching", "Instant voice corrections", "Adaptive training plans", "Film IQ + pro comparison", "AI scouting reports", "Full progress history"].map((feature) => (
                <span key={feature}><i>✓</i>{feature}</span>
              ))}
            </div>
            <button>Start Elite free <Arrow /></button>
          </article>

          <div className="secondary-plans">
            {[plans[0], plans[2]].map((plan) => (
              <article className="secondary-plan glass" key={plan.name}>
                <div><span>{plan.name}</span><small>{plan.audience}</small></div>
                <strong>${plan.price}<small>{plan.price ? "/ mo" : " forever"}</small></strong>
                <ul>
                  {plan.features.map((feature) => <li key={feature}>{feature}</li>)}
                </ul>
                <button>{plan.price ? "Open Coach OS" : "Train free"} <Arrow /></button>
              </article>
            ))}
          </div>
        </div>

        <div className="pricing-trust">
          <span>No credit card to start</span><i />
          <span>Private athlete profile</span><i />
          <span>Cancel anytime</span>
        </div>
        <div className="launch-banner" id="profile">
          <div><span>HOOPVISION AI</span><h3>Your live coach is ready.</h3></div>
          <button>Start your first session <Arrow /></button>
        </div>
      </section>

      <footer>
        <a className="brand" href="#home"><Mark /><span>HoopVision</span><em>AI</em></a>
        <p>Elite basketball coaching powered by advanced AI. From the live rep to the film room, development plan, and scouting report.</p>
      </footer>

      <button className="chat-trigger" id="chatTrigger" onClick={() => setChatOpen(!chatOpen)} aria-label="Open AI coach chat">
        <Mark /> Ask Coach V
      </button>
      <aside className="chat-panel glass" id="chatPanel" hidden={!chatOpen} aria-label="AI coach chat">
          <div className="chat-head">
            <span><b>Coach V</b><small>AI live coach</small></span>
            <button id="closeChat" onClick={() => setChatOpen(false)} aria-label="Close chat">x</button>
          </div>
          <div className="chat-messages" id="chatMessages">
            {chat.map((item, index) => (
              <div className={item.who} key={index}>
                <span>{item.who === "coach" ? "V" : "YOU"}</span>
                <p>{item.text}</p>
              </div>
            ))}
          </div>
          <form id="chatForm" onSubmit={sendMessage}>
            <input id="chatInput" value={message} onChange={(event) => setMessage(event.target.value)} placeholder="Ask for a correction..." />
            <button>Send</button>
          </form>
      </aside>
    </main>
  );
}
