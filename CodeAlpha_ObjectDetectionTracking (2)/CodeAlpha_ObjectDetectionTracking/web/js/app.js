// Live object detection + tracking, fully in the browser.
//
// - getUserMedia for the camera (the visitor's own — works for anyone)
// - TensorFlow.js COCO-SSD for detection (80 classes, same as YOLO/COCO)
// - a small IoU tracker (below) to give each object a persistent ID
//
// Nothing is sent to a server; detection happens on the user's device.

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

let model = null;
let stream = null;
let running = false;
let confThreshold = 0.5;
const tracker = new IoUTracker();

// ---- model loads as soon as the CDN scripts are ready ----
window.addEventListener("load", async () => {
  try {
    setStatus("Loading model…");
    model = await cocoSsd.load();           // default mobilenet_v2 base
    setStatus("Ready");
    els.toggle.disabled = false;
  } catch (err) {
    setStatus("Could not load the model — check your connection.");
    console.error(err);
  }
});

// ---- controls ----
els.toggle.addEventListener("click", () => (running ? stop() : start()));

els.conf.addEventListener("input", () => {
  confThreshold = els.conf.value / 100;
  els.confVal.textContent = `${els.conf.value}%`;
});

els.camera.addEventListener("change", () => {
  if (running) { stop(); start(els.camera.value); }
});

// ---- camera ----
async function start(deviceId) {
  try {
    setStatus("Starting camera…");
    const constraints = {
      video: deviceId ? { deviceId: { exact: deviceId } } : { facingMode: "environment" },
      audio: false,
    };
    stream = await navigator.mediaDevices.getUserMedia(constraints);
    els.video.srcObject = stream;
    await els.video.play();

    await populateCameras();          // labels are available once permission is granted
    sizeCanvas();
    els.placeholder.style.display = "none";
    els.toggle.textContent = "Stop camera";
    running = true;
    tracker.reset();
    setStatus("Detecting");
    loop();
  } catch (err) {
    setStatus("Camera blocked or unavailable. Allow access and try again.");
    console.error(err);
  }
}

function stop() {
  running = false;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  els.placeholder.style.display = "grid";
  els.toggle.textContent = "Start camera";
  els.fps.textContent = "FPS: —";
  els.objects.textContent = "Objects: 0";
  setStatus("Ready");
}

async function populateCameras() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cams = devices.filter((d) => d.kind === "videoinput");
  const current = stream.getVideoTracks()[0].getSettings().deviceId;

  els.camera.innerHTML = "";
  cams.forEach((cam, i) => {
    const opt = document.createElement("option");
    opt.value = cam.deviceId;
    opt.textContent = cam.label || `Camera ${i + 1}`;
    if (cam.deviceId === current) opt.selected = true;
    els.camera.appendChild(opt);
  });
  els.camera.disabled = cams.length < 2;
}

function sizeCanvas() {
  els.overlay.width = els.video.videoWidth;
  els.overlay.height = els.video.videoHeight;
}

// ---- detection loop ----
let lastTime = performance.now();
let smoothedFps = 0;

async function loop() {
  if (!running) return;

  const predictions = await model.detect(els.video);
  const kept = predictions.filter((p) => p.score >= confThreshold);
  const tracked = tracker.update(kept);

  draw(tracked);
  els.objects.textContent = `Objects: ${tracked.length}`;

  const now = performance.now();
  const fps = 1000 / (now - lastTime);
  smoothedFps = smoothedFps ? smoothedFps * 0.85 + fps * 0.15 : fps;
  lastTime = now;
  els.fps.textContent = `FPS: ${smoothedFps.toFixed(1)}`;

  requestAnimationFrame(loop);
}

function draw(items) {
  ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  ctx.lineWidth = Math.max(2, els.overlay.width / 400);
  ctx.font = `${Math.max(14, els.overlay.width / 45)}px Inter, sans-serif`;
  ctx.textBaseline = "top";

  for (const item of items) {
    const [x, y, w, h] = item.bbox;
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

// =====================================================================
// A minimal IoU tracker — links detections across frames so each object
// keeps the same ID. (The Python app uses BoT-SORT; this is the browser
// equivalent in spirit: greedy matching by box overlap.)
// =====================================================================
function IoUTracker(iouThreshold = 0.3, maxMissed = 15) {
  let tracks = [];   // { id, bbox, cls, missed }
  let nextId = 1;

  this.reset = () => { tracks = []; nextId = 1; };

  this.update = (detections) => {
    const used = new Set();
    const results = [];

    // try to extend existing tracks first (highest IoU wins)
    for (const track of tracks) {
      let best = -1, bestIoU = iouThreshold;
      detections.forEach((det, i) => {
        if (used.has(i) || det.class !== track.cls) return;
        const score = iou(track.bbox, det.bbox);
        if (score > bestIoU) { bestIoU = score; best = i; }
      });

      if (best >= 0) {
        used.add(best);
        track.bbox = detections[best].bbox;
        track.missed = 0;
        results.push({ ...detections[best], id: track.id });
      } else {
        track.missed += 1;
      }
    }

    // anything left over is a new object
    detections.forEach((det, i) => {
      if (used.has(i)) return;
      const track = { id: nextId++, bbox: det.bbox, cls: det.class, missed: 0 };
      tracks.push(track);
      results.push({ ...det, id: track.id });
    });

    // drop tracks that have been missing too long
    tracks = tracks.filter((t) => t.missed <= maxMissed);
    return results;
  };

  function iou(a, b) {
    const [ax, ay, aw, ah] = a, [bx, by, bw, bh] = b;
    const x1 = Math.max(ax, bx), y1 = Math.max(ay, by);
    const x2 = Math.min(ax + aw, bx + bw), y2 = Math.min(ay + ah, by + bh);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const union = aw * ah + bw * bh - inter;
    return union > 0 ? inter / union : 0;
  }
}
