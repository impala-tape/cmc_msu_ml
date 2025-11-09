# фикс Qt на Wayland (до import cv2)
import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")

import cv2
from pathlib import Path
import time

# --------- ПАРАМЕТРЫ ----------
VIDEO_PATH = "Финиш Московского марафона 2025. Мужчины (online-video-cutter.com).mp4"
INIT_BBOX = None  # (x, y, w, h). Если None — выбери мышкой в окне.
# ------------------------------

BASE = Path(__file__).parent
cap = cv2.VideoCapture(str(BASE / VIDEO_PATH))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {BASE / VIDEO_PATH}")

# GOTURN требует файлы с ИМЕНАМИ ровно 'goturn.prototxt' и 'goturn.caffemodel'
proto = BASE / "goturn.prototxt"
model = BASE / "goturn.caffemodel"
if not (proto.exists() and model.exists()):
    raise FileNotFoundError("Нужны goturn.prototxt и goturn.caffemodel рядом со скриптом")

# многие сборки OpenCV ищут их в текущем каталоге — сменим cwd
os.chdir(BASE)

if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerGOTURN_create"):
    tracker = cv2.legacy.TrackerGOTURN_create()
elif hasattr(cv2, "TrackerGOTURN_create"):
    tracker = cv2.TrackerGOTURN_create()
else:
    raise RuntimeError("В твоём OpenCV нет GOTURN. Установи opencv-contrib-python.")

win = "Person tracking (GOTURN)"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)

ok, frame = cap.read()
if not ok:
    raise RuntimeError("Empty first frame")

if INIT_BBOX is None:
    sel = cv2.selectROI(win, frame, fromCenter=False, showCrosshair=True)
    INIT_BBOX = tuple(map(int, sel))

if INIT_BBOX is None or len(INIT_BBOX) != 4:
    raise ValueError("INIT_BBOX должен быть (x, y, w, h) или выбери ROI мышкой")

tracker.init(frame, INIT_BBOX)

fps_avg, alpha = 0.0, 0.1
while True:
    t0 = time.time()
    ok, frame = cap.read()
    if not ok:
        break

    ok, box = tracker.update(frame)
    if ok:
        x, y, w, h = map(int, box)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking lost", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    dt = time.time() - t0
    fps = 1.0 / dt if dt > 0 else 0.0
    fps_avg = (1 - alpha) * fps_avg + alpha * fps if fps_avg else fps
    cv2.putText(frame, f"FPS: {fps_avg:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow(win, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Esc
        break
    if key == ord('r'):  # перевыбор ROI
        sel = cv2.selectROI(win, frame, fromCenter=False, showCrosshair=True)
        if sel and len(sel) == 4:
            tracker = (cv2.legacy.TrackerGOTURN_create()
                       if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerGOTURN_create")
                       else cv2.TrackerGOTURN_create())
            tracker.init(frame, tuple(map(int, sel)))

cap.release()
cv2.destroyWindow(win)
