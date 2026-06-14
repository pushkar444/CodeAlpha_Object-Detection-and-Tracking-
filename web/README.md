# Live Object Detection &amp; Tracking — Web Demo

A browser version of Task 4 that **anyone can use from a link** — no install, no
server. It uses the visitor's own camera and runs detection **entirely in their
browser** with TensorFlow.js, so the video never leaves their device.

> This is the shareable web demo. The full Python version (YOLOv8 + BoT-SORT,
> with file/RTSP input and saving) lives in the parent folder.

---

## ✨ What it does

- **Real-time detection** with TensorFlow.js **COCO-SSD** (80 object classes — the
  same COCO set YOLO uses)
- **Object tracking** — a small IoU tracker gives each object a persistent **ID**
- **Pick your camera** in the browser (built-in webcam, **Camo**, etc.)
- **Confidence slider**, live **FPS** and object count
- **Private** — runs on-device; nothing is uploaded
- Works on desktop and mobile, responsive

---

## ⚠️ One requirement: HTTPS

Browsers only allow camera access on **`https://`** or **`localhost`**. So:

- ✅ Hosted on Netlify / GitHub Pages / Vercel → HTTPS → works
- ✅ Opened via `localhost` for testing → works
- ❌ Opening `index.html` directly as a `file://` → camera blocked

---

## 🚀 Make it live (easiest first)

### Option A — Netlify Drop (no account setup, ~1 minute)
1. Go to **https://app.netlify.com/drop**
2. Drag this **`web`** folder onto the page.
3. You get a public HTTPS link like `https://your-name.netlify.app` — share it
   with anyone.

### Option B — GitHub Pages
1. Put the contents of this `web` folder in a GitHub repo (e.g. its own repo, or
   move them into a `/docs` folder of your repo).
2. Repo **Settings → Pages → Build and deployment → Deploy from a branch**.
3. Choose the branch and the folder (root or `/docs`), Save.
4. Your site goes live at `https://<username>.github.io/<repo>/`.

### Option C — Vercel
1. Import the repo at **https://vercel.com/new**.
2. Set the root/output directory to this `web` folder. Deploy.

---

## 🧪 Test it locally first

```bash
cd web
python -m http.server 8000
# open http://localhost:8000  (localhost counts as secure, so the camera works)
```

Click **Start camera**, allow permission, then pick **Camo** from the camera
dropdown if you want to use your phone.

---

## 🛠 How it works

```
Visitor's camera (getUserMedia)
        │  video frames
        ▼
TensorFlow.js COCO-SSD  →  [ {class, score, bbox}, … ]
        │  filtered by confidence
        ▼
IoU tracker  →  assigns a persistent #ID to each object
        ▼
Canvas overlay  →  boxes + "class #id score" + FPS, drawn over the video
```

All of this runs client-side, which is exactly why it can be hosted for free as a
static site and used by anyone.

---

## 📁 Files

```
web/
├── index.html
├── css/style.css
├── js/app.js        # camera, detection loop, IoU tracker, drawing
└── README.md
```

---

## 📝 Notes & limits

- **First load** downloads the model from the CDN (a few MB) — give it a moment.
- Speed depends on the visitor's device; COCO-SSD is light and runs on the GPU via
  TensorFlow.js's WebGL backend. For heavier accuracy you'd export a YOLO model to
  TF.js / ONNX, but COCO-SSD keeps the demo simple and fast.
- The tracker is intentionally simple (greedy IoU). The Python app uses BoT-SORT,
  which also models motion with a Kalman filter for tougher scenes.

---

**Author:** Pushkar Kumar · pushkar.kumar.cs28@iilm.edu

`#codealpha` `#artificialintelligence` `#computervision` `#tensorflowjs`
