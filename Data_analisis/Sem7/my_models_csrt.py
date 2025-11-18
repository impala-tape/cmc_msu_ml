from pathlib import Path
import time
import cv2
import numpy as np
import mediapipe as mp

VIDEO_PATH = "girl_desert.mp4"
INIT_BBOX = None          # (x, y, w, h) или None — выбрать мышкой
SEARCH_PAD = 0.10         # расширение области анализа (в долях w/h)

# YuNet (лицо)
YUNET_PATH = "models/face_detection_yunet_2023mar.onnx"
FACE_CONF_THR = 0.4
FACE_NMS_THR = 0.3

# MediaPipe Pose (скелет)
POSE_MIN_DET_CONF = 0.5
POSE_MIN_TRACK_CONF = 0.5

BASE = Path(__file__).parent
mp_pose = mp.solutions.pose


def expand_clip(x, y, w, h, H, W, pad):
    pw, ph = int(w * pad), int(h * pad)
    x0 = max(0, x - pw)
    y0 = max(0, y - ph)
    x1 = min(W, x + w + pw)
    y1 = min(H, y + h + ph)
    return x0, y0, x1, y1


def bbox_from_landmarks(lms, offx, offy, roi_w, roi_h):
    xs, ys = [], []
    for p in lms:
        xs.append(p.x * roi_w + offx)
        ys.append(p.y * roi_h + offy)
    if not xs:
        return None
    x1, y1 = int(max(0, min(xs))), int(max(0, min(ys)))
    x2, y2 = int(max(xs)), int(max(ys))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def draw_pose(img, results, offx, offy, roi_w, roi_h):
    if not results or not results.pose_landmarks:
        return
    lm = results.pose_landmarks.landmark
    for a, b in mp_pose.POSE_CONNECTIONS:
        pa, pb = lm[a], lm[b]
        xa, ya = int(pa.x * roi_w + offx), int(pa.y * roi_h + offy)
        xb, yb = int(pb.x * roi_w + offx), int(pb.y * roi_h + offy)
        cv2.line(img, (xa, ya), (xb, yb), (0, 255, 255), 2, cv2.LINE_AA)
    for p in lm:
        xg, yg = int(p.x * roi_w + offx), int(p.y * roi_h + offy)
        cv2.circle(img, (xg, yg), 2, (0, 255, 255), -1, cv2.LINE_AA)


def create_csrt_tracker():
    """Создает CSRT трекер"""
    try:
        return cv2.TrackerCSRT_create()
    except AttributeError:
        try:
            return cv2.legacy.TrackerCSRT_create()
        except AttributeError:
            raise RuntimeError("CSRT трекер не доступен")


def main():
    cap = cv2.VideoCapture(str(BASE / VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {BASE / VIDEO_PATH}")

    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Empty first frame")

    H, W = first.shape[:2]
    win = "CSRT-Tracker + YuNet(face) + Pose(skeleton)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win, first)
    cv2.waitKey(1)

    # YuNet (FaceDetectorYN)
    if not hasattr(cv2, "FaceDetectorYN_create"):
        raise RuntimeError(
            "Нужен OpenCV 4.8+ с FaceDetectorYN (opencv-contrib-python).")
    face_det = cv2.FaceDetectorYN_create(
        model=str(BASE / YUNET_PATH),
        config="",
        input_size=(W, H),
        score_threshold=FACE_CONF_THR,
        nms_threshold=FACE_NMS_THR,
        top_k=5000
    )

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=POSE_MIN_DET_CONF,
        min_tracking_confidence=POSE_MIN_TRACK_CONF
    )

    # Инициализация CSRT трекера
    global INIT_BBOX
    if INIT_BBOX is not None:
        cur_bbox = tuple(map(int, INIT_BBOX))
    else:
        # Выбор области для трекинга
        sel = cv2.selectROI(win, first, fromCenter=False, showCrosshair=True)
        cur_bbox = tuple(map(int, sel))
    
    if cur_bbox is None or len(cur_bbox) != 4:
        raise ValueError("INIT_BBOX должен быть (x,y,w,h) или выбери ROI")

    # Создаем CSRT трекер
    tracker = create_csrt_tracker()
    tracker.init(first, cur_bbox)

    fps_avg, alpha = 0.0, 0.1
    tracking_lost = False

    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]

        # Трекинг через CSRT вместо MediaPipe
        success, cur_bbox = tracker.update(frame)
        
        if not success:
            tracking_lost = True
            # Если трекинг потерян, используем последний известный bbox
            # или можно добавить логику для повторной инициализации
        else:
            tracking_lost = False

        x, y, w, h = map(int, cur_bbox)
        x = max(0, x)
        y = max(0, y)
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))

        # расширенная область для анализа
        x0, y0, x1, y1 = expand_clip(x, y, w, h, H, W, SEARCH_PAD)
        roi = frame[y0:y1, x0:x1]
        roi_h, roi_w = roi.shape[:2]

        # MediaPipe Pose только в ROI
        if roi_h > 0 and roi_w > 0:
            rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb_roi)
            
            # рисуем скелет в ROI
            if results.pose_landmarks:
                draw_pose(frame, results, offx=x0, offy=y0, roi_w=roi_w, roi_h=roi_h)
        else:
            results = None

        # Лицо: YuNet только в расширенной области (маска всего кадра)
        masked = np.zeros_like(frame)
        masked[y0:y1, x0:x1] = roi
        face_det.setInputSize((W, H))

        res = face_det.detect(masked)
        faces = []

        a, b, *_ = res
        if isinstance(a, (np.ndarray, list)) or a is None:
            faces = a
        else:
            faces = b

        faces = [] if faces is None else faces

        # Рисуем основной bbox трекера
        if tracking_lost:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Красный если потерян
            cv2.putText(frame, "TRACKING LOST", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 220, 60), 2)  # Зеленый если ок

        # Рисуем расширенную область
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 0), 1)  # Голубая рамка для ROI

        # Рисуем лицо(а)
        for f in faces:
            fx, fy, fw, fh, score = int(f[0]), int(f[1]), int(f[2]), int(f[3]), float(f[-1])
            cx, cy = fx + fw // 2, fy + fh // 2
            if not (x0 <= cx <= x1 and y0 <= cy <= y1):
                continue
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)
            cv2.putText(frame, f"{score:.2f}", (fx, max(0, fy - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        # FPS
        dt = time.time() - t0
        fps = 1.0 / dt if dt > 0 else 0.0
        fps_avg = (1 - alpha) * fps_avg + alpha * fps if fps_avg else fps
        
        # Статус трекинга
        # status = "LOST" if tracking_lost else "OK"
        cv2.putText(frame, f"FPS: {fps_avg:.1f}", (10, 30),
                     cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()