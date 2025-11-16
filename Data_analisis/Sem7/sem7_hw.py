# Wayland fix — до import cv2
import numpy as np
import cv2
from pathlib import Path
import time
import os
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")


# --------- НАСТРОЙКИ ----------
VIDEO_PATH = "girl_desert.mp4"
INIT_BBOX = None            # (x, y, w, h) или None для интерактивного выбора
# расширение области анализа в каждую сторону (доля w/h)
SEARCH_PAD = 0.20

# Лицо (Res10 SSD)
FACE_CONF_THR = 0.1
FACE_INPUT_WH = (300, 300)
FACE_MEAN_BGR = (104, 177, 123)  # корректный mean для Res10

# Поза (OpenPose COCO)
POSE_INPUT_WH = (368, 368)
POSE_THR = 0.1
# ------------------------------

BASE = Path(__file__).parent
cap = cv2.VideoCapture(str(BASE / VIDEO_PATH))
if not cap.isOpened():
    raise RuntimeError(f"Cannot open video: {BASE / VIDEO_PATH}")

# --- модели рядом со скриптом ---
proto_goturn = BASE / "goturn.prototxt"
model_goturn = BASE / "goturn.caffemodel"
if not (proto_goturn.exists() and model_goturn.exists()):
    raise FileNotFoundError("goturn.prototxt/goturn.caffemodel не найдены")

proto_face = BASE / "deploy.prototxt"
model_face = BASE / "res10_300x300_ssd_iter_140000_fp16.caffemodel"
if not (proto_face.exists() and model_face.exists()):
    raise FileNotFoundError(
        "deploy.prototxt/res10_300x300_ssd_iter_140000_fp16.caffemodel не найдены")

pose_proto = BASE / "pose_deploy_linevec_faster_4_stages.prototxt"
pose_model = BASE / "pose_iter_160000.caffemodel"
if not (pose_proto.exists() and pose_model.exists()):
    raise FileNotFoundError(
        "pose_deploy_linevec_faster_4_stages.prototxt/pose_iter_160000.caffemodel не найдены")

# Некоторые сборки OpenCV ищут модели в cwd
os.chdir(BASE)

# --- окно и первый кадр ---
ok, first = cap.read()
if not ok:
    raise RuntimeError("Empty first frame")
win = "GOTURN + Face-in-ROI + OpenPose"
cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
cv2.imshow(win, first)
cv2.waitKey(1)

# --- трекер GOTURN ---
if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerGOTURN_create"):
    tracker = cv2.legacy.TrackerGOTURN_create()
elif hasattr(cv2, "TrackerGOTURN_create"):
    tracker = cv2.TrackerGOTURN_create()
else:
    raise RuntimeError("OpenCV без GOTURN. Установи opencv-contrib-python.")

# --- детектор лица ---
net_face = cv2.dnn.readNetFromCaffe(str(proto_face), str(model_face))
# net_face.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net_face.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# --- поза (OpenPose COCO) ---
net_pose = cv2.dnn.readNetFromCaffe(str(pose_proto), str(pose_model))
# net_pose.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# net_pose.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

POSE_PAIRS = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)
]
N_POINTS = 18  # без background


def draw_skeleton(img, pts):
    for a, b in POSE_PAIRS:
        if pts[a] is not None and pts[b] is not None:
            cv2.line(img, pts[a], pts[b], (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(img, pts[a], 3, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(img, pts[b], 3, (0, 0, 255), -1, cv2.LINE_AA)


# --- init bbox ---
if INIT_BBOX is None:
    sel = cv2.selectROI(win, first, fromCenter=False, showCrosshair=True)
    INIT_BBOX = tuple(map(int, sel))
if INIT_BBOX is None or len(INIT_BBOX) != 4:
    raise ValueError("INIT_BBOX должен быть (x, y, w, h) или выбери ROI")

tracker.init(first, INIT_BBOX)

fps_avg, alpha = 0.0, 0.1

while True:
    t0 = time.time()
    ok, frame = cap.read()
    if not ok:
        break

    H, W = frame.shape[:2]
    ok_tr, box = tracker.update(frame)
    if ok_tr:
        x, y, w, h = map(int, box)
        x = max(0, x)
        y = max(0, y)
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))

        # --- расширенная область для анализа (лицо + поза) ---
        pad_w = int(w * SEARCH_PAD)
        pad_h = int(h * SEARCH_PAD)
        x0 = max(0, x - pad_w)
        y0 = max(0, y - pad_h)
        x1 = min(W, x + w + pad_w)
        y1 = min(H, y + h + pad_h)

        # ====== ЛИЦО (Res10 SSD) только по masked-кадру ======
        masked = np.zeros_like(frame)
        masked[y0:y1, x0:x1] = frame[y0:y1, x0:x1]
        blob_face = cv2.dnn.blobFromImage(masked, 1.0, FACE_INPUT_WH, FACE_MEAN_BGR,
                                          swapRB=False, crop=False)
        net_face.setInput(blob_face)
        det = net_face.forward()

        faces = []
        for i in range(det.shape[2]):
            conf = float(det[0, 0, i, 2])
            if conf < FACE_CONF_THR:
                continue
            fx1 = int(det[0, 0, i, 3] * W)
            fy1 = int(det[0, 0, i, 4] * H)
            fx2 = int(det[0, 0, i, 5] * W)
            fy2 = int(det[0, 0, i, 6] * H)
            # центр должен лежать внутри расширенной области
            cx = (fx1 + fx2) // 2
            cy = (fy1 + fy2) // 2
            if not (x0 <= cx <= x1 and y0 <= cy <= y1):
                continue
            faces.append((fx1, fy1, fx2, fy2, conf))

        # ====== ПОЗА (OpenPose) по кропу ROI ======
        roi = frame[y0:y1, x0:x1]
        if roi.size != 0:
            blob_pose = cv2.dnn.blobFromImage(roi, 1.0/255.0, POSE_INPUT_WH, (0, 0, 0),
                                              swapRB=False, crop=False)
            net_pose.setInput(blob_pose)
            out = net_pose.forward()[0]       # (C, Hout, Wout)
            Hout, Wout = out.shape[1], out.shape[2]

            pts = [None] * N_POINTS
            for i in range(N_POINTS):
                probMap = out[i, :, :]
                _, prob, _, point = cv2.minMaxLoc(probMap)
                if prob > POSE_THR:
                    px = int(point[0] * roi.shape[1] / Wout) + x0
                    py = int(point[1] * roi.shape[0] / Hout) + y0
                    pts[i] = (px, py)
            draw_skeleton(frame, pts)

        # --- рисуем трек и лица ---
        cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 220, 60), 2)
        for fx1, fy1, fx2, fy2, conf in faces:
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)
            cv2.putText(frame, f"{conf:.2f}", (fx1, max(0, fy1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # контур расширенной области (если надо дебажить — разкомментируй)
        # cv2.rectangle(frame, (x0, y0), (x1, y1), (120, 120, 120), 1)

    else:
        cv2.putText(frame, "Tracking lost", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # FPS
    dt = time.time() - t0
    fps = 1.0 / dt if dt > 0 else 0.0
    fps_avg = (1 - alpha) * fps_avg + alpha * fps if fps_avg else fps
    cv2.putText(frame, f"FPS: {fps_avg:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.imshow(win, frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Esc
        break
    if key == ord('r') and ok_tr:
        sel = cv2.selectROI(win, frame, fromCenter=False, showCrosshair=True)
        if sel and len(sel) == 4:
            if hasattr(cv2, "legacy") and hasattr(cv2.legacy, "TrackerGOTURN_create"):
                tracker = cv2.legacy.TrackerGOTURN_create()
            else:
                tracker = cv2.TrackerGOTURN_create()
            tracker.init(frame, tuple(map(int, sel)))

cap.release()
cv2.destroyWindow(win)
