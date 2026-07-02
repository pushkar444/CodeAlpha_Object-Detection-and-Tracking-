// Live object detection + tracking, fully from the deployed website.
//
// Visitors do not install anything. TensorFlow.js and COCO-SSD load as normal
// web assets, the camera runs through getUserMedia, and a small IoU tracker
// keeps stable IDs across frames.

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
let lastTime = performance.now();
let smoothedFps = 0;
const tracker = new IoUTracker();

window.addEventListener("load", async () => {
  try {
    setStatus("Loading model...");
    model = await cocoSsd.load();
    setStatus("Ready");
    els.toggle.disabled = false;
  } catch (err) {
    setStatus("Could not load model. Check your internet connection.");
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
    tracker.reset();
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

  try {
    const predictions = await model.detect(els.video);
    const kept = predictions.filter((prediction) => prediction.score >= confThreshold);
    const tracked = tracker.update(kept);

    draw(tracked);
    updateStats(tracked.length);
    setStatus("Detecting");
  } catch (err) {
    setStatus("Detection error.");
    console.error(err);
  }

  requestAnimationFrame(loop);
}

function updateStats(count) {
  els.objects.textContent = `Objects: ${count}`;
  const now = performance.now();
  const fps = 1000 / (now - lastTime);
  smoothedFps = smoothedFps ? smoothedFps * 0.85 + fps * 0.15 : fps;
  lastTime = now;
  els.fps.textContent = `FPS: ${smoothedFps.toFixed(1)}`;
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

function IoUTracker(iouThreshold = 0.3, maxMissed = 15) {
  let tracks = [];
  let nextId = 1;

  this.reset = () => {
    tracks = [];
    nextId = 1;
  };

  this.update = (detections) => {
    const used = new Set();
    const results = [];

    for (const track of tracks) {
      let best = -1;
      let bestIoU = iouThreshold;
      detections.forEach((detection, index) => {
        if (used.has(index) || detection.class !== track.cls) return;
        const score = iou(track.bbox, detection.bbox);
        if (score > bestIoU) {
          bestIoU = score;
          best = index;
        }
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

    detections.forEach((detection, index) => {
      if (used.has(index)) return;
      const track = { id: nextId++, bbox: detection.bbox, cls: detection.class, missed: 0 };
      tracks.push(track);
      results.push({ ...detection, id: track.id });
    });

    tracks = tracks.filter((track) => track.missed <= maxMissed);
    return results;
  };

  function iou(a, b) {
    const [ax, ay, aw, ah] = a;
    const [bx, by, bw, bh] = b;
    const x1 = Math.max(ax, bx);
    const y1 = Math.max(ay, by);
    const x2 = Math.min(ax + aw, bx + bw);
    const y2 = Math.min(ay + ah, by + bh);
    const intersection = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const union = aw * ah + bw * bh - intersection;
    return union > 0 ? intersection / union : 0;
  }
}
