"""Server-backed object detection and tracking for the web demo."""

import base64
import os
import threading
import time
import uuid
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/Ultralytics")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "docs"
LOCAL_MODEL = BASE_DIR / "yolov8n.pt"
DEFAULT_MODEL = str(LOCAL_MODEL) if LOCAL_MODEL.exists() else "yolov8n.pt"
MODEL_PATH = os.environ.get("MODEL_PATH", DEFAULT_MODEL)
CONFIDENCE = float(os.environ.get("MODEL_CONFIDENCE", "0.3"))
TRACK_TTL_SECONDS = 120

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")

model_lock = threading.Lock()
model = YOLO(MODEL_PATH)
trackers = {}
trackers_lock = threading.Lock()


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "model": Path(MODEL_PATH).name})


@app.post("/api/session")
def create_session():
    session_id = uuid.uuid4().hex
    with trackers_lock:
        trackers[session_id] = IoUTracker()
    return jsonify({"sessionId": session_id})


@app.post("/api/detect")
def detect():
    payload = request.get_json(silent=True) or {}
    image_data = payload.get("image")
    session_id = payload.get("sessionId")
    threshold = float(payload.get("confidence", CONFIDENCE))

    if not image_data:
        return jsonify({"error": "image is required"}), 400

    frame = decode_data_url(image_data)
    if frame is None:
        return jsonify({"error": "invalid image"}), 400

    with model_lock:
        results = model.predict(frame, conf=threshold, verbose=False)

    detections = result_to_detections(results[0])
    tracker = get_tracker(session_id)
    tracked = tracker.update(detections)

    return jsonify({
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "detections": tracked,
    })


def decode_data_url(data_url):
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    try:
        raw = base64.b64decode(data_url)
    except Exception:
        return None
    image = np.frombuffer(raw, dtype=np.uint8)
    return cv2.imdecode(image, cv2.IMREAD_COLOR)


def result_to_detections(result):
    detections = []
    boxes = result.boxes
    if boxes is None:
        return detections

    for box in boxes:
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        cls_id = int(box.cls[0])
        detections.append({
            "class": result.names[cls_id],
            "score": float(box.conf[0]),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
        })
    return detections


def get_tracker(session_id):
    now = time.time()
    with trackers_lock:
        stale = [
            key for key, tracker in trackers.items()
            if now - tracker.last_seen > TRACK_TTL_SECONDS
        ]
        for key in stale:
            del trackers[key]

        if not session_id or session_id not in trackers:
            session_id = uuid.uuid4().hex
            trackers[session_id] = IoUTracker()

        tracker = trackers[session_id]
        tracker.last_seen = now
        return tracker


class IoUTracker:
    def __init__(self, iou_threshold=0.3, max_missed=15):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks = []
        self.next_id = 1
        self.last_seen = time.time()

    def update(self, detections):
        used = set()
        results = []

        for track in self.tracks:
            best_index = -1
            best_iou = self.iou_threshold
            for index, detection in enumerate(detections):
                if index in used or detection["class"] != track["class"]:
                    continue
                score = iou(track["bbox"], detection["bbox"])
                if score > best_iou:
                    best_iou = score
                    best_index = index

            if best_index >= 0:
                used.add(best_index)
                detection = detections[best_index]
                track["bbox"] = detection["bbox"]
                track["missed"] = 0
                results.append({**detection, "id": track["id"]})
            else:
                track["missed"] += 1

        for index, detection in enumerate(detections):
            if index in used:
                continue
            track = {
                "id": self.next_id,
                "class": detection["class"],
                "bbox": detection["bbox"],
                "missed": 0,
            }
            self.next_id += 1
            self.tracks.append(track)
            results.append({**detection, "id": track["id"]})

        self.tracks = [
            track for track in self.tracks
            if track["missed"] <= self.max_missed
        ]
        return results


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0 else 0


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
