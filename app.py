"""Lightweight web host for the object detection demo.

Render's free tier does not have enough memory for PyTorch/YOLO inference.
The public demo therefore serves a browser-based detector so visitors only need
to open the website and allow camera access.
"""

import os
from pathlib import Path

from flask import Flask, jsonify, send_from_directory


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "docs"

app = Flask(__name__, static_folder=str(STATIC_DIR), static_url_path="")


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "mode": "browser-tensorflowjs",
        "reason": "Render free tier cannot reliably run PyTorch/YOLO inference",
    })


@app.post("/api/detect")
def detect_disabled():
    return jsonify({
        "error": "server-side detection is disabled on this free deployment",
        "mode": "browser-tensorflowjs",
    }), 410


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
