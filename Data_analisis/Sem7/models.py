from pathlib import Path
import time
import cv2
import numpy as np

VIDEO_PATH = "girl_desert.mp4"
INIT_BBOX = None          # (x, y, w, h) или None — выбрать мышкой
SEARCH_PAD = 0.10         # расширение области анализа (в долях w/h)

# YuNet (лицо)
YUNET_PATH = "models/face_detection_yunet_2023mar.onnx"
FACE_CONF_THR = 0.4
FACE_NMS_THR = 0.3

# OpenPose (Caffe-модель позы, COCO 18 точек)
POSE_PROTO = "models/pose_deploy_linevec_faster_4_stages.prototxt"
POSE_MODEL = "models/pose_iter_160000.caffemodel"
POSE_IN_WIDTH = 368
POSE_IN_HEIGHT = 368

POSE_SCORE_THR = 0.3
# ↑ порог уверенности точки позы:
#   - поднять (0.4–0.5), если много мусора
#   - опустить (0.15–0.2), если пропадают нужные суставы

POSE_POINTS_NUM = 18

# Индексы соответствуют COCO-конфигурации OpenPose:
BODY_PARTS = {
    0: "nose",        # 0  - нос
    1: "neck",        # 1  - шея
    2: "r_shoulder",  # 2  - правое плечо
    3: "r_elbow",     # 3  - правый локоть
    4: "r_wrist",     # 4  - правое запястье
    5: "l_shoulder",  # 5  - левое плечо
    6: "l_elbow",     # 6  - левый локоть
    7: "l_wrist",     # 7  - левое запястье
    8: "r_hip",       # 8  - правое бедро
    9: "r_knee",      # 9  - правое колено
    10: "r_ankle",    # 10 - правая лодыжка
    11: "l_hip",      # 11 - левое бедро
    12: "l_knee",     # 12 - левое колено
    13: "l_ankle",    # 13 - левая лодыжка
    14: "r_eye",      # 14 - правый глаз
    15: "l_eye",      # 15 - левый глаз
    16: "r_ear",      # 16 - правое ухо
    17: "l_ear",      # 17 - левое ухо
}

# Пары точек для отрисовки скелета (стикмен):
POSE_PAIRS = [
    (1, 2), (2, 3), (3, 4),       # шея → правая рука
    (1, 5), (5, 6), (6, 7),       # шея → левая рука
    (1, 8), (8, 9), (9, 10),      # шея → правая нога
    (1, 11), (11, 12), (12, 13),  # шея → левая нога
    (1, 0),                       # шея → нос
    (0, 14), (14, 16),            # нос → правый глаз → правое ухо
    (0, 15), (15, 17),            # нос → левый глаз → левое ухо
]

# Цвета (BGR):
COLOR_HEAD = (0, 255, 255)       # голова/лицо (nose + глаза + уши)
COLOR_RIGHT_ARM = (255, 0, 0)    # правая рука
COLOR_LEFT_ARM = (0, 255, 0)     # левая рука
COLOR_RIGHT_LEG = (0, 0, 255)    # правая нога
COLOR_LEFT_LEG = (255, 0, 255)   # левая нога
COLOR_TORSO = (255, 255, 0)      # шея↔нос / общая связка

# Цвет для каждой кости
PAIR_COLORS = {
    # правая рука
    (1, 2): COLOR_RIGHT_ARM,
    (2, 3): COLOR_RIGHT_ARM,
    (3, 4): COLOR_RIGHT_ARM,
    # левая рука
    (1, 5): COLOR_LEFT_ARM,
    (5, 6): COLOR_LEFT_ARM,
    (6, 7): COLOR_LEFT_ARM,
    # правая нога
    (1, 8): COLOR_RIGHT_LEG,
    (8, 9): COLOR_RIGHT_LEG,
    (9, 10): COLOR_RIGHT_LEG,
    # левая нога
    (1, 11): COLOR_LEFT_LEG,
    (11, 12): COLOR_LEFT_LEG,
    (12, 13): COLOR_LEFT_LEG,
    # голова/лицо
    (1, 0): COLOR_TORSO,
    (0, 14): COLOR_HEAD,
    (14, 16): COLOR_HEAD,
    (0, 15): COLOR_HEAD,
    (15, 17): COLOR_HEAD,
}

# базовая папка — рядом со скриптом
BASE = Path(__file__).parent


def expand_clip(x, y, w, h, H, W, pad):
    # ROI вокруг трека — для лица; позу считаем по всему кадру
    pw, ph = int(w * pad), int(h * pad)
    x0 = max(0, x - pw)
    y0 = max(0, y - ph)
    x1 = min(W, x + w + pw)
    y1 = min(H, y + h + ph)
    return x0, y0, x1, y1


def create_csrt_tracker():
    try:
        return cv2.TrackerCSRT_create()
    except AttributeError:
        try:
            return cv2.legacy.TrackerCSRT_create()
        except AttributeError:
            raise RuntimeError("CSRT трекер не доступен")


def infer_pose_openpose_full(frame_bgr, net):
    """
    Поза "в тупую" по всему кадру:
    возвращает список длины 18:
      (x, y, conf) или None для каждой точки.
    """
    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame_bgr,
        scalefactor=1.0 / 255,
        size=(POSE_IN_WIDTH, POSE_IN_HEIGHT),
        mean=(0, 0, 0),
        swapRB=False,
        crop=False,
    )
    net.setInput(blob)
    out = net.forward()  # (1, N, H_out, W_out), первые 18 каналов — наши точки

    H_out = out.shape[2]
    W_out = out.shape[3]

    points = [None] * POSE_POINTS_NUM
    for i in range(POSE_POINTS_NUM):
        heatmap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatmap)

        if conf > POSE_SCORE_THR:
            x = int(point[0] * w / W_out)
            y = int(point[1] * h / H_out)
            points[i] = (x, y, float(conf))
        else:
            points[i] = None

    return points


def draw_pose_openpose_colored(img, keypoints):
    """
    Рисует:
      - цветные “палочки” по POSE_PAIRS
      - точки суставов
      - подписи: для каждой палки "i-j", для каждой точки "idx:name:conf"
    """
    # сначала кости
    for (a, b) in POSE_PAIRS:
        if a >= len(keypoints) or b >= len(keypoints):
            continue
        pa = keypoints[a]
        pb = keypoints[b]
        if pa is None or pb is None:
            continue

        xa, ya, _ = pa
        xb, yb, _ = pb

        color = PAIR_COLORS.get((a, b), (0, 255, 255))  # дефолт: жёлтый
        cv2.line(img, (xa, ya), (xb, yb), color, 2, cv2.LINE_AA)

        # подпись кости посередине
        mx = (xa + xb) // 2
        my = (ya + yb) // 2
        bone_label = f"{a}-{b}"
        cv2.putText(
            img, bone_label, (mx + 4, my - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            color, 1, cv2.LINE_AA
        )

    # точки + подписи
    for idx, kp in enumerate(keypoints):
        if kp is None:
            continue
        x, y, conf = kp

        cv2.circle(img, (x, y), 3, (255, 255, 255), -1, cv2.LINE_AA)

        name = BODY_PARTS.get(idx, str(idx))
        label = f"{idx}:{name}:{conf:.2f}"
        cv2.putText(
            img, label, (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4,
            (255, 255, 255), 1, cv2.LINE_AA
        )


def main():
    # ВИДЕО — относительно папки скрипта
    video_path = BASE / VIDEO_PATH
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    ok, first = cap.read()
    if not ok:
        raise RuntimeError("Empty first frame")

    H, W = first.shape[:2]
    win = "CSRT-Tracker + YuNet(face) + OpenPose(skeleton)"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(win, first)
    cv2.waitKey(1)

    # YuNet (лицо) — модель тоже относительно скрипта
    if not hasattr(cv2, "FaceDetectorYN_create"):
        raise RuntimeError(
            "Нужен OpenCV 4.8+ с FaceDetectorYN (opencv-contrib-python)."
        )
    face_det = cv2.FaceDetectorYN_create(
        model=str(BASE / YUNET_PATH),
        config="",
        input_size=(W, H),
        score_threshold=FACE_CONF_THR,
        nms_threshold=FACE_NMS_THR,
        top_k=5000
    )

    # OpenPose (DNN) — модель/прототxt относительно скрипта
    pose_net = cv2.dnn.readNetFromCaffe(
        str(BASE / POSE_PROTO),
        str(BASE / POSE_MODEL)
    )
    # при желании CUDA:
    # pose_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # pose_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    global INIT_BBOX
    if INIT_BBOX is not None:
        cur_bbox = tuple(map(int, INIT_BBOX))
    else:
        sel = cv2.selectROI(win, first, fromCenter=False, showCrosshair=True)
        cur_bbox = tuple(map(int, sel))

    if cur_bbox is None or len(cur_bbox) != 4:
        raise ValueError("INIT_BBOX должен быть (x,y,w,h) или выбери ROI")

    tracker = create_csrt_tracker()
    tracker.init(first, cur_bbox)

    fps_avg, alpha = 0.1, 0.1
    tracking_lost = False

    while True:
        t0 = time.time()
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]

        # трекер
        success, cur_bbox = tracker.update(frame)
        tracking_lost = not success

        x, y, w, h = map(int, cur_bbox)
        x = max(0, x)
        y = max(0, y)
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))

        # ROI только для лица (YuNet), позу считаем по всему кадру
        x0, y0, x1, y1 = expand_clip(x, y, w, h, H, W, SEARCH_PAD)
        roi = frame[y0:y1, x0:x1]

        # поза по всему кадру
        pose_points = infer_pose_openpose_full(frame, pose_net)
        draw_pose_openpose_colored(frame, pose_points)

        # лицо: YuNet в ROI
        masked = np.zeros_like(frame)
        masked[y0:y1, x0:x1] = roi
        face_det.setInputSize((W, H))
        res = face_det.detect(masked)

        faces = []
        if res is not None:
            a, b, *_ = res
            if isinstance(a, (np.ndarray, list)) or a is None:
                faces = a
            else:
                faces = b
        faces = [] if faces is None else faces

        # bbox трекера
        if tracking_lost:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "TRACKING LOST", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (60, 220, 60), 2)

        # рамка ROI
        cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 0), 1)

        # лица
        for f in faces:
            fx, fy, fw, fh, score = int(f[0]), int(f[1]), int(f[2]), int(f[3]), float(f[-1])
            cx, cy = fx + fw // 2, fy + fh // 2
            if not (x0 <= cx <= x1 and y0 <= cy <= y1):
                continue
            cv2.rectangle(frame, (fx, fy), (fx + fw, fy + fh), (0, 0, 255), 2)
            cv2.putText(frame, f"{score:.2f}", (fx, max(0, fy - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        dt = time.time() - t0
        fps = 1.0 / dt if dt > 0 else 0.0
        fps_avg = (1 - alpha) * fps_avg + alpha * fps

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
