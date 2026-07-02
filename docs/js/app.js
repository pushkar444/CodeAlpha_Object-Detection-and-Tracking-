// Live object detection + tracking through the deployed Flask/YOLO server.
//
// The visitor only grants camera permission. Frames are captured in the browser,
// sent to /api/detect, and the server returns boxes with stable tracking IDs.

const els = {
  video: document.getElementById("video"),
  overlay: document.getElementById("overlay"),
  placeholder: document.getElementById("placeholder"),
  toggle: document.getElementById("toggle"),
  camera: document.getElementById("camera"),
  conf: document.getElementById("conf"),
  confVal: document.getElementById("confVal"),
  fps: document.getElementById("fps"),
  objects: document.getElementById("objects"),
  status: document.getElementById("status"),
};

const ctx = els.overlay.getContext("2d");
const captureCanvas = document.createElement("canvas");
const captureCtx = captureCanvas.getContext("2d");

let stream = null;
let running = false;
let sessionId = null;
let confThreshold = 0.5;
let inFlight = false;
let lastTime = performance.now();
let lastRequestTime = 0;
let smoothedFps = 0;
const detectIntervalMs = 500;

window.addEventListener("load", async () => {
  try {
    setStatus("Connecting to server...");
    const health = await fetch("/api/health");
    if (!health.ok) throw new Error("health check failed");

    const session = await fetch("/api/session", { method: "POST" });
    const data = await session.json();
    sessionId = data.sessionId;

    setStatus("Ready");
    els.toggle.disabled = false;
  } catch (err) {
    setStatus("Server unavailable. Deploy/run the Flask app.");
    console.error(err);
  }
});

els.toggle.addEventListener("click", () => (running ? stop() : start()));

els.conf.addEventListener("input", () => {
  confThreshold = els.conf.value / 100;
  els.confVal.textContent = `${els.conf.value}%`;
});

els.camera.addEventListener("change", () => {
  if (running) {
    stop();
    start(els.camera.value);
  }
});

async function start(deviceId) {
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      setStatus("Camera needs HTTPS or localhost.");
      return;
    }

    setStatus("Starting camera...");
    const constraints = {
      video: deviceId ? { deviceId: { exact: deviceId } } : { facingMode: "environment" },
      audio: false,
    };
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    els.video.srcObject = stream;
    await els.video.play();

    await populateCameras();
    sizeCanvas();
    els.placeholder.style.display = "none";
    els.toggle.textContent = "Stop camera";
    running = true;
    inFlight = false;
    lastTime = performance.now();
    smoothedFps = 0;
    setStatus("Detecting");
    requestAnimationFrame(loop);
  } catch (err) {
    setStatus("Camera blocked or unavailable. Allow access and try again.");
    console.error(err);
  }
}

function stop() {
  running = false;
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  els.placeholder.style.display = "grid";
  els.toggle.textContent = "Start camera";
  els.fps.textContent = "FPS: -";
  els.objects.textContent = "Objects: 0";
  setStatus("Ready");
}

async function populateCameras() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cams = devices.filter((device) => device.kind === "videoinput");
  const current = stream.getVideoTracks()[0].getSettings().deviceId;

  els.camera.innerHTML = "";
  cams.forEach((cam, index) => {
    const opt = document.createElement("option");
    opt.value = cam.deviceId;
    opt.textContent = cam.label || `Camera ${index + 1}`;
    if (cam.deviceId === current) opt.selected = true;
    els.camera.appendChild(opt);
  });
  els.camera.disabled = cams.length < 2;
}

function sizeCanvas() {
  els.overlay.width = els.video.videoWidth;
  els.overlay.height = els.video.videoHeight;
}

async function loop() {
  if (!running) return;

  const now = performance.now();
  if (
    !inFlight &&
    now - lastRequestTime >= detectIntervalMs &&
    els.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA
  ) {
    inFlight = true;
    lastRequestTime = now;
    try {
      const image = captureFrame();
      const res = await fetch("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sessionId,
          confidence: confThreshold,
          image,
        }),
      });

      if (!res.ok) {
        const message = await readError(res);
        throw new Error(`detect failed: ${res.status} ${message}`);
      }
      const data = await res.json();
      draw(data.detections, data.width, data.height);
      updateStats(data.detections.length);
      setStatus("Detecting");
    } catch (err) {
      setStatus("Detection server error.");
      console.error(err);
    } finally {
      inFlight = false;
    }
  }

  requestAnimationFrame(loop);
}

function captureFrame() {
  const maxWidth = 416;
  const scale = Math.min(1, maxWidth / els.video.videoWidth);
  captureCanvas.width = Math.round(els.video.videoWidth * scale);
  captureCanvas.height = Math.round(els.video.videoHeight * scale);
  captureCtx.drawImage(els.video, 0, 0, captureCanvas.width, captureCanvas.height);
  return captureCanvas.toDataURL("image/jpeg", 0.6);
}

async function readError(res) {
  try {
    const data = await res.json();
    return data.error || data.detail || "";
  } catch {
    return "";
  }
}

function updateStats(count) {
  els.objects.textContent = `Objects: ${count}`;
  const now = performance.now();
  const fps = 1000 / (now - lastTime);
  smoothedFps = smoothedFps ? smoothedFps * 0.85 + fps * 0.15 : fps;
  lastTime = now;
  els.fps.textContent = `FPS: ${smoothedFps.toFixed(1)}`;
}

function draw(items, sourceWidth, sourceHeight) {
  ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  const xScale = els.overlay.width / sourceWidth;
  const yScale = els.overlay.height / sourceHeight;
  ctx.lineWidth = Math.max(2, els.overlay.width / 400);
  ctx.font = `${Math.max(14, els.overlay.width / 45)}px Inter, sans-serif`;
  ctx.textBaseline = "top";

  for (const item of items) {
    const [rawX, rawY, rawW, rawH] = item.bbox;
    const x = rawX * xScale;
    const y = rawY * yScale;
    const w = rawW * xScale;
    const h = rawH * yScale;
    const color = colorForId(item.id);
    const label = `${item.class} #${item.id} ${Math.round(item.score * 100)}%`;

    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);

    const padding = 4;
    const tw = ctx.measureText(label).width;
    const th = parseInt(ctx.font, 10) + padding;
    const ly = y > th ? y - th : y;

    ctx.fillStyle = color;
    ctx.fillRect(x, ly, tw + padding * 2, th);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x + padding, ly + 2);
  }
}

function colorForId(id) {
  const hue = (id * 47) % 360;
  return `hsl(${hue}, 65%, 45%)`;
}

function setStatus(text) {
  els.status.textContent = text;
}
