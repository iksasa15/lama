"""Detection helpers: people, fire/smoke, fall, wrong-way tracking."""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from models_loader import get_coco_model, get_fall_model, get_fire_smoke_model

PERSON_CLASS = 0  # COCO
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck
CONF = 0.25
FIRE_SMOKE_CONF = 0.15
FIRE_HSV_MIN_AREA = 180
FIRE_HSV_MAX_AREA_RATIO = 0.035
SMOKE_HSV_MIN_AREA_RATIO = 0.03

WRONG_WAY_MIN_MOVE = 8.0
WRONG_WAY_MATCH_DIST = 120.0
VALID_DIRECTIONS = {"down", "up", "right", "left"}

_session_lock = threading.Lock()
_sessions: Dict[str, List[Dict[str, float]]] = {}


def _centroid(box: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _resolve_name(names: dict, cls_id: int) -> str:
    cid = int(cls_id)
    raw = names.get(cid, names.get(str(cid), cid))
    return str(raw).strip().lower()


def _boxes_from_result(result, model_key: str, label_filter=None) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    names = result.names or {}
    if result.boxes is None or len(result.boxes) == 0:
        return out
    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(int)
    for box, conf, cls_id in zip(xyxy, confs, clss):
        label = _resolve_name(names, int(cls_id))
        if label_filter is not None and label not in label_filter:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        out.append(
            {
                "model": model_key,
                "label": label,
                "score": float(conf),
                "box": [x1, y1, x2, y2],
            }
        )
    return out


def detect_people(image: np.ndarray) -> Tuple[List[Dict[str, Any]], int]:
    model = get_coco_model()
    result = model.predict(source=image, conf=CONF, verbose=False)[0]
    detections: List[Dict[str, Any]] = []
    names = result.names or {}
    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        for box, conf, cls_id in zip(xyxy, confs, clss):
            if int(cls_id) != PERSON_CLASS:
                continue
            label = str(names.get(int(cls_id), "person"))
            x1, y1, x2, y2 = [float(v) for v in box]
            detections.append(
                {
                    "model": "people",
                    "label": label,
                    "score": float(conf),
                    "box": [x1, y1, x2, y2],
                }
            )
    return detections, len(detections)


def _normalize_fire_smoke_label(label: str, cls_id: int) -> str:
    lab = label.lower().strip()
    if "smoke" in lab or "دخان" in lab:
        return "smoke"
    if "fire" in lab or "flame" in lab or "حريق" in lab or "نار" in lab:
        return "fire"
    if cls_id == 0:
        return "smoke"
    if cls_id == 1:
        return "fire"
    return lab


def _skin_mask(image: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    return cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))


def _hsv_fire_boxes(image: np.ndarray) -> List[Dict[str, Any]]:
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask_orange = cv2.inRange(hsv, (5, 140, 170), (25, 255, 255))
    mask_yellow = cv2.inRange(hsv, (20, 100, 190), (40, 255, 255))
    mask_core = cv2.inRange(hsv, (0, 40, 220), (40, 180, 255))
    mask = cv2.bitwise_or(mask_orange, mask_yellow)
    mask = cv2.bitwise_or(mask, mask_core)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(_skin_mask(image)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = FIRE_HSV_MAX_AREA_RATIO * h * w
    dets: List[Dict[str, Any]] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < FIRE_HSV_MIN_AREA or area > max_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh < 8 or bw < 6:
            continue
        roi = image[y : y + bh, x : x + bw]
        if roi.size == 0:
            continue
        mean_v = float(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 2].mean())
        if mean_v < 150:
            continue
        if bw > bh * 3.5 and bh < 40:
            continue
        score = min(0.92, 0.55 + area / 12000.0)
        dets.append(
            {
                "model": "fire",
                "label": "fire",
                "score": float(score),
                "box": [float(x), float(y), float(x + bw), float(y + bh)],
            }
        )
    return dets


def detect_fire_smoke(
    image: np.ndarray, want_fire: bool, want_smoke: bool
) -> List[Dict[str, Any]]:
    if not want_fire and not want_smoke:
        return []

    orig_h, orig_w = image.shape[:2]
    infer = image
    scale = 1.0
    if max(orig_h, orig_w) < 640:
        scale = 640 / max(orig_h, orig_w)
        infer = cv2.resize(
            image,
            (int(orig_w * scale), int(orig_h * scale)),
            interpolation=cv2.INTER_LINEAR,
        )

    dets: List[Dict[str, Any]] = []
    try:
        model = get_fire_smoke_model()
        result = model.predict(
            source=infer,
            conf=FIRE_SMOKE_CONF,
            iou=0.45,
            imgsz=640,
            verbose=False,
        )[0]
        names = result.names or {0: "smoke", 1: "fire"}
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls_id in zip(xyxy, confs, clss):
                cls_id = int(cls_id)
                label = _normalize_fire_smoke_label(
                    _resolve_name(names, cls_id), cls_id
                )
                if label == "smoke" and not want_smoke:
                    continue
                if label == "fire" and not want_fire:
                    continue
                if label not in ("fire", "smoke"):
                    continue
                x1, y1, x2, y2 = [float(v) for v in box]
                if scale != 1.0:
                    x1, y1, x2, y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale
                dets.append(
                    {
                        "model": label,
                        "label": label,
                        "score": float(conf),
                        "box": [x1, y1, x2, y2],
                    }
                )
    except Exception as exc:  # noqa: BLE001
        print(f"[fire_smoke] YOLO failed: {exc}")

    if want_fire and not any(d["label"] == "fire" for d in dets):
        dets.extend(_hsv_fire_boxes(image))

    return dets


def detect_fall(image: np.ndarray) -> List[Dict[str, Any]]:
    model = get_fall_model()
    result = model.predict(source=image, conf=CONF, verbose=False)[0]
    dets = _boxes_from_result(result, "fall")
    for d in dets:
        lab = d["label"].lower().replace(" ", "_")
        if "fallen" in lab or lab == "fall":
            d["label"] = "fallen"
            d["model"] = "fall"
        elif "sit" in lab:
            d["label"] = "sitting"
            d["model"] = "fall"
        elif "stand" in lab:
            d["label"] = "standing"
            d["model"] = "fall"
    return dets


def _is_wrong_direction(dx: float, dy: float, correct_direction: str) -> bool:
    """
    correct_direction = allowed traffic flow in the camera frame.
      down  : top -> bottom
      up    : bottom -> top
      right : left -> right
      left  : right -> left
    """
    direction = correct_direction if correct_direction in VALID_DIRECTIONS else "down"
    if direction == "down":
        return dy < -WRONG_WAY_MIN_MOVE
    if direction == "up":
        return dy > WRONG_WAY_MIN_MOVE
    if direction == "right":
        return dx < -WRONG_WAY_MIN_MOVE
    if direction == "left":
        return dx > WRONG_WAY_MIN_MOVE
    return False


def _track_vehicles_wrong_way(
    vehicles: List[Dict[str, Any]], session_id: str, correct_direction: str
) -> bool:
    wrong = False
    with _session_lock:
        prev = _sessions.get(session_id, [])
        matched_prev: List[Dict[str, float]] = []
        for veh in vehicles:
            best: Optional[Dict[str, float]] = None
            best_dist = WRONG_WAY_MATCH_DIST
            for p in prev:
                dist = ((veh["_cx"] - p["cx"]) ** 2 + (veh["_cy"] - p["cy"]) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best = p
            if best is not None:
                dx = veh["_cx"] - best["cx"]
                dy = veh["_cy"] - best["cy"]
                if _is_wrong_direction(dx, dy, correct_direction):
                    wrong = True
                    veh["label"] = f"{veh['label']}:wrong_way"
            matched_prev.append({"cx": veh["_cx"], "cy": veh["_cy"]})
        _sessions[session_id] = matched_prev
    return wrong


def detect_wrong_way(
    image: np.ndarray, session_id: str, correct_direction: str = "down"
) -> Tuple[List[Dict[str, Any]], bool]:
    model = get_coco_model()
    result = model.predict(source=image, conf=CONF, verbose=False)[0]
    current: List[Dict[str, Any]] = []
    names = result.names or {}
    if result.boxes is not None and len(result.boxes) > 0:
        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        for box, conf, cls_id in zip(xyxy, confs, clss):
            if int(cls_id) not in VEHICLE_CLASSES:
                continue
            label = str(names.get(int(cls_id), cls_id))
            x1, y1, x2, y2 = [float(v) for v in box]
            cx, cy = _centroid([x1, y1, x2, y2])
            current.append(
                {
                    "model": "wrong_way",
                    "label": label,
                    "score": float(conf),
                    "box": [x1, y1, x2, y2],
                    "_cx": cx,
                    "_cy": cy,
                }
            )

    wrong = _track_vehicles_wrong_way(current, session_id, correct_direction)
    clean = [
        {
            "model": "wrong_way",
            "label": v["label"],
            "score": v["score"],
            "box": v["box"],
        }
        for v in current
    ]
    return clean, wrong


def run_enabled(
    image: np.ndarray,
    enabled: List[str],
    session_id: str,
    correct_direction: str = "down",
) -> Dict[str, Any]:
    detections: List[Dict[str, Any]] = []
    alerts: List[str] = []
    people_count = 0
    direction = correct_direction if correct_direction in VALID_DIRECTIONS else "down"

    want_people = "people" in enabled
    want_smoke = "smoke" in enabled
    want_fire = "fire" in enabled
    want_fall = "fall" in enabled
    want_wrong = "wrong_way" in enabled

    if want_people and want_wrong:
        model = get_coco_model()
        result = model.predict(source=image, conf=CONF, verbose=False)[0]
        names = result.names or {}
        vehicles_for_track: List[Dict[str, Any]] = []
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls_id in zip(xyxy, confs, clss):
                cls_id = int(cls_id)
                x1, y1, x2, y2 = [float(v) for v in box]
                label = str(names.get(cls_id, cls_id))
                if cls_id == PERSON_CLASS:
                    detections.append(
                        {
                            "model": "people",
                            "label": label,
                            "score": float(conf),
                            "box": [x1, y1, x2, y2],
                        }
                    )
                elif cls_id in VEHICLE_CLASSES:
                    cx, cy = _centroid([x1, y1, x2, y2])
                    vehicles_for_track.append(
                        {
                            "model": "wrong_way",
                            "label": label,
                            "score": float(conf),
                            "box": [x1, y1, x2, y2],
                            "_cx": cx,
                            "_cy": cy,
                        }
                    )
        people_count = sum(1 for d in detections if d["model"] == "people")
        wrong = _track_vehicles_wrong_way(vehicles_for_track, session_id, direction)
        for v in vehicles_for_track:
            detections.append(
                {
                    "model": "wrong_way",
                    "label": v["label"],
                    "score": v["score"],
                    "box": v["box"],
                }
            )
        if wrong:
            alerts.append("wrong_way")
    else:
        if want_people:
            people_dets, people_count = detect_people(image)
            detections.extend(people_dets)
        if want_wrong:
            ww_dets, wrong = detect_wrong_way(image, session_id, direction)
            detections.extend(ww_dets)
            if wrong:
                alerts.append("wrong_way")

    if want_fire or want_smoke:
        fs = detect_fire_smoke(image, want_fire, want_smoke)
        detections.extend(fs)
        if any(d["label"] == "fire" for d in fs):
            alerts.append("fire")
        if any(d["label"] == "smoke" for d in fs):
            alerts.append("smoke")

    if want_fall:
        fall_dets = detect_fall(image)
        detections.extend(fall_dets)
        if any(d["label"] == "fallen" for d in fall_dets):
            alerts.append("fall")

    return {
        "people_count": people_count,
        "detections": detections,
        "alerts": alerts,
        "correct_direction": direction,
    }
