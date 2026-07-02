# CodeAlpha - Object Detection and Tracking

**CodeAlpha Artificial Intelligence Internship - Task 4**

This project has two entry points:

- **Deployable website:** visitors open the Render URL, allow camera permission,
  and object detection/tracking runs in the browser with TensorFlow.js COCO-SSD.
  They do not install Python, OpenCV, YOLO, VS Code, or any command-line tool.
- **Local CLI tool:** `detect_track.py` still runs YOLOv8 + tracking directly on
  your own machine for webcam/video-file experiments.

## Why The Deployable Site Uses Browser AI

The first deployed server-side YOLO version loaded PyTorch/Ultralytics on Render
free. Render logs showed:

```text
Worker was sent SIGKILL! Perhaps out of memory?
POST /api/detect HTTP/1.1 500
```

That means Render's free instance does not have enough memory for reliable
PyTorch/YOLO inference. To make the website work for normal users on the free
deployment, the public site now uses TensorFlow.js in the browser. This still
requires no setup from visitors: they open the HTTPS link and allow the camera.

## Project Structure

```text
.
├── app.py              # lightweight Flask host for the website
├── detect_track.py     # optional local command-line YOLO/OpenCV tool
├── docs/               # frontend served by app.py
├── web/                # same frontend copy for easy inspection/editing
├── requirements.txt    # lightweight deployment dependencies
├── Procfile            # gunicorn start command
├── render.yaml         # Render blueprint config
└── runtime.txt         # Python version for deployment
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

## Deploy On Render

1. Push this repo to GitHub.
2. In Render, create a new **Blueprint** from the repo.
3. Render reads `render.yaml` automatically.
4. Deploy and open the HTTPS URL.
5. Click **Start camera** and allow camera permission.

Manual settings, if needed:

```text
Build command: pip install -r requirements.txt
Start command: gunicorn app:app --workers 1 --threads 2 --timeout 180
Python version: 3.12.3
```

## Browser Camera Requirements

Camera access works only on:

- `https://...` deployed URLs
- `http://localhost...` for local testing

Opening `docs/index.html` directly as a `file://` page may block camera access.

## Optional Local YOLO CLI

The original command-line script is still available for local use:

```bash
pip install ultralytics opencv-python numpy
python detect_track.py --source 0
python detect_track.py --source traffic.mp4 --save --no-show
```

This local CLI is separate from the deployed website.

## Notes

- The deployed site downloads TensorFlow.js/COCO-SSD as web assets, just like a
  normal website loading JavaScript.
- Speed depends on the visitor's browser/device, but no manual install is
  required.
- Use **Mode -> Accuracy** on laptops or stronger tablets for better detection.
  Use **Speed** on weaker phones if the camera feels laggy.
- Server-side YOLO can be restored on a paid server with enough RAM/CPU by
  adding the PyTorch/Ultralytics dependencies back and using the Flask detection
  API version.

**Author:** Pushkar Kumar - pushkar.kumar.cs28@iilm.edu
