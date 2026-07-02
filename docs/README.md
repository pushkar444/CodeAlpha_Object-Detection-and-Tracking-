# Live Object Detection and Tracking - Frontend

This frontend is served by `app.py`. It uses the visitor's browser camera and
TensorFlow.js COCO-SSD for object detection, plus a small IoU tracker for stable
object IDs.

Visitors do not install anything. They open the HTTPS site and allow camera
permission.
