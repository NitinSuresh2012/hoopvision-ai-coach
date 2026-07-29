import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders the production HoopVision experience", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>HoopVision AI - Live AI Coach<\/title>/i);
  assert.match(html, /id="live"/);
  assert.match(html, /Sitting\/Posture Correction/);
  assert.match(html, /id="cameraButton"/);
  assert.match(html, /id="overallScore"/);
  assert.match(html, /id="training"/);
  assert.match(html, /id="progress"/);
  assert.match(html, /id="coach"/);
  assert.match(html, /id="pricing"/);
  assert.match(html, /href="#coach">Coach Dashboard/);
  assert.doesNotMatch(html, /codex-preview|Building your site|Starter Project/i);
});

test("posture source contains release safety guardrails", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");

  assert.match(page, /setPostureScore\(null\);\s*setPostureStatus\("NO_PERSON"\)/);
  assert.match(page, /status === "NO_PERSON"\s*\?\s*0/);
  assert.match(page, /phase === "calibrating"/);
  assert.match(page, /POSTURE_WINDOW_MS = 2000/);
  assert.match(page, /calibration\.validMs >= POSTURE_WINDOW_MS/);
  assert.match(page, /assessment\.score >= 85/);
  assert.match(page, /frame\.confidence >= POSTURE_LANDMARK_CONFIDENCE/);
  assert.match(page, /streamRef\.current\?\.getTracks\(\)\.forEach\(\(track\) => track\.stop\(\)\)/);
  assert.match(page, /poseLandmarkerRef\.current\?\.close\?\.\(\)/);
  assert.doesNotMatch(page, /href="#profile"/);
});
