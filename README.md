# CodeAlpha - Object Detection and Tracking

**CodeAlpha Artificial Intelligence Internship - Task 4**

This project now has two entry points:

- **Deployable website:** visitors open a normal HTTPS website, allow camera
  permission, and the deployed Python server runs YOLO object detection.
- **Local CLI tool:** `detect_track.py` still runs YOLOv8 + tracking directly on
  your own machine for webcam/video-file experiments.

## Root Cause Fixed

The old web demo was static. It loaded TensorFlow.js and COCO-SSD from a CDN and
ran detection in each visitor's browser. That meant another device needed to
download the model, support the browser AI runtime, and have enough local
processing power. The Python YOLO tracker existed only as a command-line program,
so GitHub Pages could not run it for website visitors.

The website now calls a Flask backend:

```
Browser camera -> /api/detect -> Flask + YOLOv8 on the server -> boxes + IDs
```

Visitors do not install Python, OpenCV, TensorFlow.js, YOLO, or any command-line
tool. They only need a modern browser and camera permission.

## Project Structure

```
.
├── app.py              # Flask web server + YOLO detection API + server tracker
├── detect_track.py     # optional local command-line YOLO/OpenCV tool
├── docs/               # frontend served by app.py and usable as static assets
├── web/                # same frontend copy for easy inspection/editing
├── requirements.txt    # Python server dependencies
├── Procfile            # gunicorn start command for deployment platforms
└── yolov8n.pt          # optional local YOLOv8 nano weights
```

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
python app.py
```

Open `http://localhost:8000`. Browsers allow camera access on `localhost`.

## Deploy the Website

Use a Python web host, not GitHub Pages. GitHub Pages only serves static files
and cannot run `app.py`, YOLO, or OpenCV.

### Render

1. Push this repo to GitHub.
2. Create a new **Blueprint** on Render from the repo, or create a normal
   **Web Service**.
3. Render can read `render.yaml` automatically. If you create the service
   manually, use these settings:
   - Build command: `pip install -r requirements.txt`
   - Start command: `gunicorn app:app`
   - Python version: 3.12.3
4. Deploy and open the Render HTTPS URL.
5. Click **Start camera** and allow camera permission.

### Railway / VPS / Other Python Host

Use the same commands:

```bash
pip install -r requirements.txt
gunicorn app:app
```

The app reads `PORT` automatically when the platform provides it. You can also
set `MODEL_PATH` if you want to use another YOLO weights file. If `yolov8n.pt`
is not present on the deployed server, Ultralytics downloads the standard nano
weights on first start.

`requirements.txt` pins CPU-only PyTorch wheels so deployment hosts do not waste
time or disk space downloading CUDA/GPU packages.

The server also sets Ultralytics and Matplotlib config directories to `/tmp` so
read-only home directories on deployment hosts do not break startup.

## Browser Camera Requirements

Camera access works only on:

- `https://...` deployed URLs
- `http://localhost...` for local testing

Opening `docs/index.html` directly as a `file://` page will not work because
the browser blocks camera access and there is no backend API.

## API

- `GET /api/health` checks that the server is running.
- `POST /api/session` creates a tracking session.
- `POST /api/detect` accepts a JPEG data URL and returns detections:

```json
{
  "width": 640,
  "height": 360,
  "detections": [
    { "id": 1, "class": "person", "score": 0.91, "bbox": [10, 20, 120, 240] }
  ]
}
```

## Optional Local CLI

The original command-line script is still available:

```bash
python detect_track.py --source 0
python detect_track.py --source traffic.mp4 --save --no-show
```

The server uses `opencv-python-headless` for deployment. If you want OpenCV GUI
windows for the CLI preview on your own computer, install `opencv-python`
locally in your virtual environment.

## Notes

- The deployed server does the AI work, so server CPU/GPU affects speed.
- `yolov8n.pt` is the small YOLOv8 nano model; it is a good default for web
  demos.
- The browser sends small throttled frames to `/api/detect` so Render's free CPU
  service has a better chance of keeping up.
- The browser sends resized JPEG frames to your server for processing. Deploy
  only on HTTPS and use a trusted host.

**Author:** Pushkar Kumar - pushkar.kumar.cs28@iilm.edu
