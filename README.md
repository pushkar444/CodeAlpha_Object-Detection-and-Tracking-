# CodeAlpha — Object Detection and Tracking

**CodeAlpha Artificial Intelligence Internship · Task 4**

A real-time program that watches a webcam (or a video file), **detects objects**
with a pre-trained **YOLOv8** model, and **tracks** each one across frames so it
keeps a stable ID. Every object is drawn with a bounding box, its class label, a
confidence score and a tracking ID — live, with an FPS readout.

---

## ✨ Features

- **Real-time video input** from a webcam or a video file (OpenCV)
- **Pre-trained YOLOv8** detector (auto-downloads on first run)
- **Multi-object tracking** with **BoT-SORT** (a SORT-family tracker), so IDs
  persist frame to frame
- **Bounding boxes + labels + confidence + tracking ID**, each ID in its own colour
- **Live FPS** and an active-object count overlay
- **Class filtering** — track only what you care about (e.g. `person car`)
- **Save mode** — write the annotated result to `output.mp4`
- Clean command-line options via `argparse`

---

## 🧠 How it works

```
Webcam / video  ──►  YOLOv8 detects objects in each frame
                ──►  BoT-SORT links detections across frames → stable IDs
                ──►  OpenCV draws box + "class #id conf" + FPS overlay
                ──►  shown live (and optionally saved to output.mp4)
```

- **Detection:** YOLO (You Only Look Once) is a single-pass detector — fast
  enough for real time. The `yolov8n` (nano) weights keep it light.
- **Tracking:** detection alone has no memory — box positions don't tell you that
  "the car in frame 2 is the same car from frame 1." A tracker (BoT-SORT / SORT)
  uses motion (a Kalman filter) and overlap (IoU) to associate detections over
  time and assign each object a persistent ID.

> Ultralytics bundles **BoT-SORT** (default) and **ByteTrack** — both are modern
> members of the SORT family the task mentions. Pick one with `--tracker`.

---

## 🛠 Tech stack

| Part | Tool |
|------|------|
| Detection | Ultralytics YOLOv8 (PyTorch under the hood) |
| Tracking | BoT-SORT / ByteTrack (bundled with Ultralytics) |
| Video & drawing | OpenCV |
| Maths / colours | numpy |

---

## 📁 Structure

```
CodeAlpha_ObjectDetectionTracking/
├── detect_track.py     # the whole program (detection + tracking + display)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ▶️ Run it

```bash
# 1. (recommended) a virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

# 2. install dependencies (this also pulls in PyTorch — a large download)
pip install -r requirements.txt

# 3. run on your webcam
python detect_track.py --source 0
```

### Choosing a camera (e.g. Camo instead of the built-in webcam)

If you have more than one camera — laptop webcam, a phone-as-webcam app like
**Camo**, OBS Virtual Camera, etc. — you can list them by name and pick one:

```bash
# list every camera with its name and resolution
python detect_track.py --list-cameras
#   Detected cameras:
#     [0] Integrated Webcam  -  640x480
#     [1] Camo               -  1280x720
#     [2] OBS Virtual Camera -  640x480

# select by name (case-insensitive, partial match works)
python detect_track.py --source camo

# select interactively from a menu
python detect_track.py --source select

# or jump straight to a known index
python detect_track.py --source 1
```

> Camera **names** need `pygrabber` (Windows-only, in `requirements.txt`). Without
> it the tool still works — it just labels cameras `camera 0`, `camera 1`, … and
> you select by index.

The first run downloads the `yolov8n.pt` weights (~6 MB) automatically. Press
**`q`** in the preview window to quit.

### More examples

```bash
# a video file, and save the annotated output
python detect_track.py --source traffic.mp4 --save

# only track people and cars, higher confidence
python detect_track.py --source 0 --classes person car --conf 0.4

# use the ByteTrack tracker instead of BoT-SORT
python detect_track.py --source 0 --tracker bytetrack.yaml

# save without opening a window (e.g. on a server)
python detect_track.py --source traffic.mp4 --save --no-show
```

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--source` | `0` | camera index (`0`/`1`/…), camera name (`camo`), `select` to pick interactively, or a video path/URL |
| `--list-cameras` | off | list detected cameras (with names) and exit |
| `--model` | `yolov8n.pt` | YOLO weights (`n`/`s`/`m`/`l`/`x` — bigger = more accurate, slower) |
| `--conf` | `0.3` | minimum detection confidence |
| `--tracker` | `botsort.yaml` | `botsort.yaml` or `bytetrack.yaml` |
| `--classes` | all | class names to keep, e.g. `person car dog` |
| `--save` | off | write `output.mp4` |
| `--no-show` | off | don't open a preview window |

---

## 📝 Notes

- A **GPU** makes this much faster, but it runs on CPU too (use `yolov8n` for
  smoother CPU performance).
- The 80 detectable classes come from the COCO dataset (person, car, dog, bottle,
  laptop, …).
- For the demo video, point it at street/traffic footage — lots of objects moving
  through frame shows the tracking IDs off nicely.

---

**Author:** Pushkar Kumar · pushkar.kumar.cs28@iilm.edu

`#codealpha` `#artificialintelligence` `#internship` `#computervision` `#yolo` `#python`
