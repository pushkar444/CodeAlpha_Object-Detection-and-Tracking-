"""Real-time object detection and tracking — CodeAlpha AI Task 4.

Reads video from a webcam or a file, detects objects with a pre-trained YOLO
model (Ultralytics YOLOv8), and tracks them across frames with BoT-SORT (a
SORT-family tracker bundled with Ultralytics). Each object is drawn with a
bounding box, a class label, a confidence score and a persistent tracking ID.

Examples
--------
    # webcam
    python detect_track.py --source 0

    # a video file, saving the annotated result
    python detect_track.py --source traffic.mp4 --save

    # only track people and cars, with a higher confidence threshold
    python detect_track.py --source 0 --classes person car --conf 0.4
"""

import argparse
import os
import time

import cv2
import numpy as np
# NOTE: `ultralytics` (and PyTorch) is imported lazily inside main() so that
# --help and --list-cameras work without the heavy deps installed.


def parse_args():
    p = argparse.ArgumentParser(description="Object detection and tracking with YOLOv8.")
    p.add_argument("--source", default="0",
                   help="Camera index (0, 1, …), a camera name like 'camo', "
                        "'select' to pick interactively, or a path to a video file.")
    p.add_argument("--list-cameras", action="store_true",
                   help="List the cameras detected on this machine and exit.")
    p.add_argument("--model", default="yolov8n.pt",
                   help="YOLO weights. 'yolov8n.pt' (nano) is small and fast; "
                        "downloads automatically on first run.")
    p.add_argument("--conf", type=float, default=0.3,
                   help="Minimum confidence to keep a detection (0-1).")
    p.add_argument("--tracker", default="botsort.yaml",
                   help="Tracker config: 'botsort.yaml' (default) or 'bytetrack.yaml'.")
    p.add_argument("--classes", nargs="*", default=None,
                   help="Only track these class names, e.g. --classes person car.")
    p.add_argument("--save", action="store_true",
                   help="Save the annotated video to output.mp4.")
    p.add_argument("--no-show", action="store_true",
                   help="Don't open a preview window (useful when only saving).")
    return p.parse_args()


def camera_names():
    """Friendly device names by index (Windows/DirectShow), e.g. 'Camo'.

    Uses pygrabber if it's installed; otherwise returns an empty list and the
    rest of the program falls back to plain 'camera <n>' labels.
    """
    try:
        from pygrabber.dshow_graph import FilterGraph
        return FilterGraph().get_input_devices()
    except Exception:
        return []


def available_cameras(max_index=6):
    """Probe camera indices 0..max_index-1 and return the ones that work.

    Returns a list of (index, width, height, name). On Windows we use the
    DirectShow backend, which opens cameras faster and avoids long MSMF timeouts.
    """
    backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
    names = camera_names()
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, backend)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                name = names[i] if i < len(names) else f"camera {i}"
                found.append((i, w, h, name))
        cap.release()
    return found


def choose_camera():
    """Interactively pick a camera from the ones detected. Returns an index."""
    cams = available_cameras()
    if not cams:
        print("[error] no cameras detected. Plug one in, or pass --source <file>.")
        return None
    print("\nDetected cameras:")
    for i, w, h, name in cams:
        print(f"  [{i}] {name}  -  {w}x{h}")

    valid = {i for i, _, _, _ in cams}
    while True:
        choice = input("\nSelect a camera index (or 'q' to cancel): ").strip()
        if choice.lower() == "q":
            return None
        if choice.isdigit() and int(choice) in valid:
            return int(choice)
        print("  that index isn't in the list - try again.")


def resolve_source(arg):
    """Turn the --source argument into something YOLO can open.

    Accepts a camera index ('0'), 'select' for the interactive picker, a camera
    name or substring ('camo'), or a file path / URL / stream.
    """
    if arg.lower() == "select":
        return choose_camera()
    if arg.isdigit():
        return int(arg)

    # treat as a camera name only when it doesn't look like a path/URL
    if not any(c in arg for c in "/\\.:"):
        for i, name in enumerate(camera_names()):
            if arg.lower() in name.lower():
                print(f"[info] using camera [{i}] {name}")
                return i

    if os.path.exists(arg):
        return arg
    # could still be a URL / RTSP stream — let YOLO try to open it
    return arg


def color_for(track_id):
    """A stable, distinct colour per tracking ID (BGR for OpenCV)."""
    if track_id is None:
        return (120, 120, 120)
    # spread hues around the wheel using a large step, then convert HSV->BGR
    hue = (int(track_id) * 37) % 180
    hsv = np.uint8([[[hue, 200, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)


def class_ids_from_names(model, names):
    """Map requested class names to the model's numeric class ids."""
    if not names:
        return None
    lookup = {name.lower(): idx for idx, name in model.names.items()}
    ids, unknown = [], []
    for name in names:
        key = name.lower()
        (ids if key in lookup else unknown).append(lookup.get(key, name))
    if unknown:
        print(f"[warn] ignoring unknown class names: {unknown}")
    return ids or None


def draw_box(frame, xyxy, label, color):
    x1, y1, x2, y2 = xyxy
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # label chip above the box (or inside it near the top edge)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    ly = max(y1, th + 6)
    cv2.rectangle(frame, (x1, ly - th - 6), (x1 + tw + 6, ly), color, -1)
    cv2.putText(frame, label, (x1 + 3, ly - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    args = parse_args()

    # just list cameras and exit
    if args.list_cameras:
        cams = available_cameras()
        if cams:
            print("Detected cameras:")
            for i, w, h, name in cams:
                print(f"  [{i}] {name}  -  {w}x{h}")
        else:
            print("No cameras detected.")
        return

    source = resolve_source(args.source)
    if source is None:  # user cancelled the picker
        return

    from ultralytics import YOLO  # imported here so --list-cameras/--help stay light

    print(f"[info] loading model '{args.model}' ...")
    model = YOLO(args.model)
    wanted = class_ids_from_names(model, args.classes)

    writer = None
    frame_count = 0
    fps = 0.0
    t_prev = time.time()

    print("[info] starting — press 'q' in the window to quit.")
    # stream=True yields results frame-by-frame; persist=True keeps IDs alive
    results = model.track(
        source=source,
        stream=True,
        persist=True,
        conf=args.conf,
        classes=wanted,
        tracker=args.tracker,
        verbose=False,
    )

    for result in results:
        frame = result.orig_img
        boxes = result.boxes

        live_ids = set()
        if boxes is not None and boxes.id is not None:
            for box in boxes:
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                track_id = int(box.id[0])
                live_ids.add(track_id)

                name = model.names[cls_id]
                label = f"{name} #{track_id} {conf:.2f}"
                draw_box(frame, (x1, y1, x2, y2), label, color_for(track_id))

        # simple FPS estimate (smoothed)
        frame_count += 1
        now = time.time()
        if now - t_prev >= 0.5:
            fps = frame_count / (now - t_prev)
            frame_count = 0
            t_prev = now

        overlay = f"FPS: {fps:4.1f}   tracking: {len(live_ids)}"
        cv2.putText(frame, overlay, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, overlay, (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        if args.save:
            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter("output.mp4", fourcc, 25, (w, h))
            writer.write(frame)

        if not args.no_show:
            cv2.imshow("Object Detection & Tracking — press 'q' to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    if writer is not None:
        writer.release()
        print("[info] saved annotated video to output.mp4")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
