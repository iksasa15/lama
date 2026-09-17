"""Wrong-way person direction detection with YOLOv8 + ByteTrack (OpenCV window)."""

import sys

import cv2
import numpy as np
from ultralytics import YOLO

# 1. تحميل النموذج
print("جاري تحميل YOLOv8n...", flush=True)
model = YOLO("yolov8n.pt")
print("الموديل جاهز", flush=True)

# اتجاهات الدخول المسموحة — Tab للتبديل
DIRECTIONS = ["LTR", "RTL", "TTB", "BTT"]
DIRECTION_LABELS = {
    "LTR": "Left -> Right",
    "RTL": "Right -> Left",
    "TTB": "Top -> Bottom",
    "BTT": "Bottom -> Top",
}
dir_index = 0
ALLOWED_DIRECTION = DIRECTIONS[dir_index]

# قاموس لتخزين نقطة البداية لكل شخص حسب معرفه (ID)
track_history = {}


def score_frame(frame: np.ndarray) -> float:
    if frame is None or frame.size == 0:
        return -1.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) + float(gray.std()) * 0.15


def pick_camera_index(max_index: int = 6) -> int:
    """Prefer real PC webcam over Iriun/virtual black screens."""
    best_i = None
    best_score = -1.0
    print("البحث عن كاميرات الجهاز...", flush=True)
    for i in range(max_index):
        cap_try = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap_try.isOpened():
            continue
        ok, frame = cap_try.read()
        cap_try.release()
        if not ok or frame is None:
            continue
        sc = score_frame(frame)
        print(f"  [{i}] score={sc:.1f}", flush=True)
        if sc > best_score:
            best_score = sc
            best_i = i
    if best_i is None:
        raise RuntimeError("لم يتم العثور على أي كاميرا")
    print(f"استخدام الكاميرا رقم [{best_i}]", flush=True)
    return best_i


def check_wrong_direction(start_pos, current_pos, allowed):
    """التحقق مما إذا كانت الحركة عكس الاتجاه المسموح"""
    dx = current_pos[0] - start_pos[0]
    dy = current_pos[1] - start_pos[1]

    # حد أدنى من التحرك بالبكسل قبل الحكم على الاتجاه
    if abs(dx) < 30 and abs(dy) < 30:
        return False

    if allowed == "LTR" and dx < -30:  # يتحرك لليسار وهو ممنوع
        return True
    if allowed == "RTL" and dx > 30:  # يتحرك لليمين وهو ممنوع
        return True
    if allowed == "TTB" and dy < -30:  # يتحرك للأعلى وهو ممنوع
        return True
    if allowed == "BTT" and dy > 30:  # يتحرك للأسفل وهو ممنوع
        return True

    return False


# 2. فتح كاميرا الجهاز (يتجنب Iriun إن أمكن)
cam_index = pick_camera_index()
cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("تعذر فتح الكاميرا", flush=True)
    sys.exit(1)

print(
    f"الاتجاه المسموح: {ALLOWED_DIRECTION} ({DIRECTION_LABELS[ALLOWED_DIRECTION]})",
    flush=True,
)
print("Tab = تغيير الاتجاه | q = إغلاق", flush=True)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # استخدام خوارزمية التتبع ByteTrack للحفاظ على ID لكل شخص
    results = model.track(
        frame,
        persist=True,
        classes=[0],
        tracker="bytetrack.yaml",
        verbose=False,
    )

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xywh.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            current_center = (float(x), float(y))

            # تسجيل نقطة البداية للشخص عند ظهوره لأول مرة
            if track_id not in track_history:
                track_history[track_id] = current_center

            start_center = track_history[track_id]

            # فحص الاتجاه العكسي
            is_wrong_way = check_wrong_direction(
                start_center, current_center, ALLOWED_DIRECTION
            )

            # تحديد اللون (أحمر للعكسي، أخضر للطبيعي)
            color = (0, 0, 255) if is_wrong_way else (0, 255, 0)
            label = f"ID: {track_id} {'[WRONG WAY!]' if is_wrong_way else 'OK'}"

            # رسم مربع الشخص والتنبيه
            top_left = (int(x - w / 2), int(y - h / 2))
            bottom_right = (int(x + w / 2), int(y + h / 2))

            cv2.rectangle(frame, top_left, bottom_right, color, 2)
            cv2.putText(
                frame,
                label,
                (top_left[0], top_left[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    cv2.putText(
        frame,
        f"Allowed: {ALLOWED_DIRECTION} ({DIRECTION_LABELS[ALLOWED_DIRECTION]})",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Cam [{cam_index}] | Tab=change dir | q=quit",
        (30, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (200, 200, 200),
        2,
    )

    # عرض النتيجة
    cv2.imshow("Direction & Wrong Way Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    # Tab (Windows OpenCV often sends 9)
    if key == 9:
        dir_index = (dir_index + 1) % len(DIRECTIONS)
        ALLOWED_DIRECTION = DIRECTIONS[dir_index]
        track_history.clear()  # إعادة قياس الاتجاه من الصفر
        print(
            f"الاتجاه الجديد: {ALLOWED_DIRECTION} ({DIRECTION_LABELS[ALLOWED_DIRECTION]})",
            flush=True,
        )

cap.release()
cv2.destroyAllWindows()
print("تم الإغلاق", flush=True)
