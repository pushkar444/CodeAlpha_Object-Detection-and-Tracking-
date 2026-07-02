# Live Object Detection and Tracking - Web Frontend

This folder contains the frontend used by the Flask app in the project root.
It is no longer a standalone AI demo: detection runs on the deployed server via
`/api/detect`.

Run the full app from the repo root:

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://localhost:8000`.

For deployment, use a Python web host with:

```bash
gunicorn app:app
```

Do not deploy this folder alone to GitHub Pages if you want YOLO tracking to run
on the server. Static hosting cannot run the backend API.
