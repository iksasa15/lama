"""Live person counter — auto-picks any working camera on this PC."""

import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

VIRTUAL_HINTS = ("iriun", "obs", "virtual", "droidcam", "epoccam", "manycam", "ndi")
MIN_LIVE_SCORE = 20.0  # below this ≈ black / placeholder (e.g. Iriun offline)


def score_frame(frame: np.ndarray) -> float:
    if frame is None or frame.size == 0:
        return -1.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) + float(gray.std()) * 0.15


def list_camera_names() -> dict[int, str]:
    """Best-effort Windows DirectShow names (optional)."""
    names: dict[int, str] = {}
    try:
        from pygrabber.dshow_graph import FilterGraph

        for i, name in enumerate(FilterGraph().get_input_devices()):
            names[i] = str(name)
    except Exception:
        pass
    return names


def is_virtual_name(name: str) -> bool:
    low = (name or "").lower()
    return any(h in low for h in VIRTUAL_HINTS)


def find_working_cameras(max_index: int = 8) -> list[dict]:
    """
    Scan indexes and return cameras that open + deliver a live-looking frame.
    """
    names = list_camera_names()
    working: list[dict] = []

    print("======= فحص الكاميرات =======", flush=True)
    for i in range(max_index):
        label = names.get(i, f"Camera {i}")
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"  [{i}] ✗ غير متاحة — {label}", flush=True)
            continue

        ok, frame = cap.read()
        # read a second frame (some cams need warmup)
        if ok:
            ok2, frame2 = cap.read()
            if ok2 and frame2 is not None:
                frame = frame2
        cap.release()

        if not ok or frame is None:
            print(f"  [{i}] ✗ مفتوحة لكن بدون صورة — {label}", flush=True)
            continue

        sc = score_frame(frame)
        virtual = is_virtual_name(label)
        live = sc >= MIN_LIVE_SCORE and not virtual

        status = "✓ شغالة" if live else "✗ ضعيفة/وهمية"
        print(f"  [{i}] {status} — {label} (score={sc:.1f})", flush=True)

        working.append(
            {
                "index": i,
                "name": label,
                "score": sc,
                "virtual": virtual,
                "live": live,
            }
        )

    print("=============================", flush=True)
    return working


def pick_camera_index() -> int:
    """Use CAM_INDEX env if set, else first/best live camera on this device."""
    env = os.environ.get("CAM_INDEX")
    if env is not None and env.strip().isdigit():
        idx = int(env.strip())
        print(f"استخدام CAM_INDEX من البيئة: [{idx}]", flush=True)
        return idx

    cams = find_working_cameras()
    live = [c for c in cams if c["live"]]
    if not live:
        # fallback: any opened cam with highest score
        if not cams:
            raise RuntimeError("لا توجد كاميرا شغالة على الجهاز")
        cams.sort(key=lambda c: c["score"], reverse=True)
        chosen = cams[0]
        print(
            f"تحذير: لا توجد كاميرا حية قوية — استخدام الأفضل المتاح [{chosen['index']}] {chosen['name']}",
            flush=True,
        )
        return chosen["index"]

    # Prefer highest score among live (real) cameras
    live.sort(key=lambda c: c["score"], reverse=True)
    chosen = live[0]
    print(
        f"تم اختيار الكاميرا الشغّالة: [{chosen['index']}] {chosen['name']} (score={chosen['score']:.1f})",
        flush=True,
    )
    return chosen["index"]


print("جاري تحميل YOLOv8n...", flush=True)
model = YOLO("yolov8n.pt")
print("الموديل جاهز", flush=True)

cam_index = pick_camera_index()
print(f"فتح الكاميرا [{cam_index}] ...", flush=True)
cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("تعذر فتح الكاميرا المختارة", flush=True)
    sys.exit(1)

ok, probe = cap.read()
if not ok or probe is None:
    print("الكاميرا لا ترسل إطارات", flush=True)
    cap.release()
    sys.exit(1)

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
