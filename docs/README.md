# Live Object Detection and Tracking - Frontend

This is the frontend served by `app.py`. The page uses the visitor's browser
camera, sends resized frames to `/api/detect`, and draws the YOLO results
returned by the deployed server.

Run from the repo root:

```bash
pip install -r requirements.txt
python app.py
```

Open `http://localhost:8000` locally, or deploy the full repo to a Python host
with:

```bash
gunicorn app:app
```

GitHub Pages can host these static files, but it cannot run the YOLO backend.
For the complete app, deploy the Flask server.
