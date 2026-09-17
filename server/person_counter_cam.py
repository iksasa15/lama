"""Live person counter — prefers the built-in PC camera over Iriun/virtual cams."""

import sys

import cv2
import numpy as np
from ultralytics import YOLO

SKIP_NAME_HINTS = ("iriun", "obs", "virtual", "droidcam", "epoccam", "manycam")


def score_frame(frame: np.ndarray) -> float:
    """Real webcam scenes score higher than black Iriun placeholder."""
    if frame is None or frame.size == 0:
        return -1.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) + float(gray.std()) * 0.15


def pick_camera_index(max_index: int = 6) -> int:
    """Pick the best local camera (skip dead / Iriun black screens when possible)."""
    best_i = None
    best_score = -1.0
    print("البحث عن كاميرات الجهاز...", flush=True)
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            continue
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            print(f"  [{i}] لا يوجد إطار", flush=True)
            continue
        sc = score_frame(frame)
        print(f"  [{i}] جاهزة — score={sc:.1f}", flush=True)
        if sc > best_score:
            best_score = sc
            best_i = i

    if best_i is None:
        raise RuntimeError("لم يتم العثور على أي كاميرا")
    # If index 0 looks like a black placeholder and another cam exists, prefer the other
    if best_i == 0 and best_score < 25:
        for i in range(1, max_index):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ok, frame = cap.read()
                cap.release()
                if ok and frame is not None and score_frame(frame) >= 25:
                    print(f"تجاهل الكاميرا الافتراضية السوداء — استخدام [{i}]", flush=True)
                    return i
    print(f"استخدام الكاميرا رقم [{best_i}]", flush=True)
    return best_i


print("جاري تحميل YOLOv8n...", flush=True)
model = YOLO("yolov8n.pt")
print("الموديل جاهز", flush=True)

cam_index = pick_camera_index()
print(f"فتح كاميرا الجهاز [{cam_index}] ...", flush=True)
cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("تعذر فتح كاميرا الجهاز", flush=True)
    sys.exit(1)

# Confirm we are not stuck on Iriun placeholder
ok, probe = cap.read()
if ok and probe is not None and score_frame(probe) < 12:
    print(
        "تحذير: الصورة شبه سوداء (غالباً Iriun). جرّب إغلاق Iriun أو غيّر الرقم يدوياً.",
        flush=True,
    )

print("الكاميرا تعمل — اضغط q في نافذة الفيديو للإغلاق", flush=True)

while True:
    ret, frame = cap.read()
    if not ret:
        print("تعذر استلام الفريم من الكاميرا", flush=True)
        break

    results = model(frame, classes=[0], verbose=False)
    person_count = len(results[0].boxes)
    annotated_frame = results[0].plot()

    cv2.putText(
        annotated_frame,
        f"People Count: {person_count}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        3,
    )
    cv2.putText(
        annotated_frame,
        f"Cam [{cam_index}]",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
    )

    cv2.imshow("Person Counter - PC Camera", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("تم الإغلاق", flush=True)
