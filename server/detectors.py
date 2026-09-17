"""Detection helpers: people, fire/smoke, fall, wrong-way tracking."""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from models_loader import get_coco_model, get_fall_model, get_fire_smoke_model

PERSON_CLASS = 0  # COCO
CONF = 0.25
PERSON_CONF = 0.12  # CrowdHuman person-only
PERSON_IMGSZ = 640
PERSON_MAX_DET = 300
FIRE_SMOKE_CONF = 0.15
FALL_CONF = 0.12
# Person bbox wider than tall → likely lying / fallen (assist)
FALL_ASPECT_MIN = 1.25
FALL_ASPECT_MAX_H_RATIO = 0.55  # fallen person usually not full frame height
FIRE_HSV_MIN_AREA = 180
FIRE_HSV_MAX_AREA_RATIO = 0.035
SMOKE_HSV_MIN_AREA_RATIO = 0.03

WRONG_WAY_MIN_MOVE = 8.0
WRONG_WAY_MATCH_DIST = 220.0
# Skip only near-full-frame group boxes (webcam close-ups are often 20–50%)
MAX_PERSON_AREA_RATIO = 0.55
VALID_DIRECTIONS = {"down", "up", "right", "left"}

_session_lock = threading.Lock()
# session_id -> { "tracks": [...], "votes": {track_id: [status, ...]} }
_sessions: Dict[str, Dict[str, Any]] = {}
_next_track_id = 1


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


def _iter_person_boxes(image: np.ndarray):
    """Yield individual person detections tuned for crowded scenes."""
    h_img, w_img = image.shape[:2]
    frame_area = float(h_img * w_img)
    model = get_coco_model()
    result = model.predict(
        source=image,
        conf=PERSON_CONF,
        iou=0.5,
        imgsz=PERSON_IMGSZ,
        max_det=PERSON_MAX_DET,
        classes=[PERSON_CLASS],
        verbose=False,
    )[0]
    if result.boxes is None or len(result.boxes) == 0:
        return
    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    for box, conf in zip(xyxy, confs):
        x1, y1, x2, y2 = [float(v) for v in box]
        bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        area = bw * bh
        # Drop huge group boxes that cover many people at once
        if area > MAX_PERSON_AREA_RATIO * frame_area:
            continue
        if bw < 8 or bh < 12:
            continue
        yield x1, y1, x2, y2, float(conf)


def detect_people(image: np.ndarray) -> Tuple[List[Dict[str, Any]], int]:
    detections: List[Dict[str, Any]] = []
    for x1, y1, x2, y2, conf in _iter_person_boxes(image):
        detections.append(
            {
                "model": "people",
                "label": "person",
                "score": conf,
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


def _normalize_fall_label(label: str, cls_id: int) -> str:
    lab = label.lower().strip().replace("-", "_").replace(" ", "_")
    if "fallen" in lab or lab in {"fall", "falling", "liedown", "lie_down", "laying"}:
        return "fallen"
    if "sit" in lab:
        return "sitting"
    if "stand" in lab:
        return "standing"
    # melihuzunoglu: 0=fallen, 1=sitting, 2=standing
    if cls_id == 0:
        return "fallen"
    if cls_id == 1:
        return "sitting"
    if cls_id == 2:
        return "standing"
    return lab


def _aspect_fall_boxes(image: np.ndarray) -> List[Dict[str, Any]]:
    """
    Heuristic: COCO person boxes that are much wider than tall
    often indicate someone lying on the ground.
    """
    h_img, w_img = image.shape[:2]
    model = get_coco_model()
    result = model.predict(source=image, conf=0.3, verbose=False)[0]
    dets: List[Dict[str, Any]] = []
    if result.boxes is None or len(result.boxes) == 0:
        return dets
    xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(int)
    for box, conf, cls_id in zip(xyxy, confs, clss):
        if int(cls_id) != PERSON_CLASS:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        aspect = bw / bh
        if aspect < FALL_ASPECT_MIN:
            continue
        if bh > FALL_ASPECT_MAX_H_RATIO * h_img:
            continue
        # Prefer people lower in the frame (on the ground)
        cy = (y1 + y2) / 2.0
        if cy < 0.35 * h_img:
            continue
        score = min(0.88, 0.4 + (aspect - 1.0) * 0.25 + float(conf) * 0.2)
        dets.append(
            {
                "model": "fall",
                "label": "fallen",
                "score": float(score),
                "box": [x1, y1, x2, y2],
            }
        )
    return dets


def detect_fall(image: np.ndarray) -> List[Dict[str, Any]]:
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
        model = get_fall_model()
        result = model.predict(
            source=infer,
            conf=FALL_CONF,
            iou=0.45,
            imgsz=640,
            verbose=False,
        )[0]
        names = result.names or {0: "fallen", 1: "sitting", 2: "standing"}
        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls_id in zip(xyxy, confs, clss):
                cls_id = int(cls_id)
                label = _normalize_fall_label(_resolve_name(names, cls_id), cls_id)
                if label not in ("fallen", "sitting", "standing"):
                    continue
                x1, y1, x2, y2 = [float(v) for v in box]
                if scale != 1.0:
                    x1, y1, x2, y2 = x1 / scale, y1 / scale, x2 / scale, y2 / scale
                dets.append(
                    {
                        "model": "fall",
                        "label": label,
                        "score": float(conf),
                        "box": [x1, y1, x2, y2],
                    }
                )
    except Exception as exc:  # noqa: BLE001
        print(f"[fall] YOLO failed: {exc}")

    # If specialized model missed a clear lying person, use aspect assist
    if not any(d["label"] == "fallen" for d in dets):
        dets.extend(_aspect_fall_boxes(image))

    return dets


def _box_diag(box: List[float]) -> float:
    x1, y1, x2, y2 = box
    return max(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5, 1.0)


def _classify_axis_move(
    dx: float, dy: float, correct_direction: str
) -> str:
    """
    Return ok | wrong | unknown using the selected axis.
    Allows mild diagonal motion (axis component at least half the other).
    """
    direction = correct_direction if correct_direction in VALID_DIRECTIONS else "down"
    adx, ady = abs(dx), abs(dy)

    if direction in ("left", "right"):
        if adx < WRONG_WAY_MIN_MOVE or adx < 0.45 * ady:
            return "unknown"
        if direction == "right":
            return "ok" if dx > 0 else "wrong"
        return "ok" if dx < 0 else "wrong"

    if ady < WRONG_WAY_MIN_MOVE or ady < 0.45 * adx:
        return "unknown"
    if direction == "down":
        return "ok" if dy > 0 else "wrong"
    return "ok" if dy < 0 else "wrong"


def _track_direction(
    items: List[Dict[str, Any]], session_id: str, correct_direction: str
) -> bool:
    """
    Unique greedy matching + axis-gated classification.
    Sticky status: once OK/Wrong, keep until opposite is seen twice.
    """
    global _next_track_id
    any_wrong = False

    with _session_lock:
        state = _sessions.get(session_id)
        if state is None:
            state = {"tracks": [], "votes": {}, "sticky": {}}
            _sessions[session_id] = state

        prev_tracks: List[Dict[str, Any]] = state["tracks"]
        votes: Dict[int, List[str]] = state["votes"]
        sticky: Dict[int, str] = state.setdefault("sticky", {})

        pairs: List[Tuple[float, int, int]] = []
        for i, item in enumerate(items):
            max_dist = max(WRONG_WAY_MATCH_DIST, 2.8 * _box_diag(item["box"]))
            for j, p in enumerate(prev_tracks):
                dist = ((item["_cx"] - p["cx"]) ** 2 + (item["_cy"] - p["cy"]) ** 2) ** 0.5
                if dist <= max_dist:
                    pairs.append((dist, i, j))
        pairs.sort(key=lambda t: t[0])

        assigned_cur: set[int] = set()
        assigned_prev: set[int] = set()
        match_prev: Dict[int, Dict[str, Any]] = {}

        for _dist, i, j in pairs:
            if i in assigned_cur or j in assigned_prev:
                continue
            assigned_cur.add(i)
            assigned_prev.add(j)
            match_prev[i] = prev_tracks[j]

        new_tracks: List[Dict[str, Any]] = []
        new_votes: Dict[int, List[str]] = {}
        new_sticky: Dict[int, str] = {}

        for i, item in enumerate(items):
            raw = "unknown"
            track_id: Optional[int] = None
            prev = match_prev.get(i)
            if prev is not None:
                track_id = int(prev["id"])
                dx = item["_cx"] - prev["cx"]
                dy = item["_cy"] - prev["cy"]
                raw = _classify_axis_move(dx, dy, correct_direction)
            else:
                track_id = _next_track_id
                _next_track_id += 1

            history = list(votes.get(track_id, []))
            if raw in ("ok", "wrong"):
                history.append(raw)
                history = history[-3:]

            prev_sticky = sticky.get(track_id)

            # One clear move is enough to set status; sticky until opposite appears twice
            status = "unknown"
            if raw in ("ok", "wrong"):
                status = raw
            elif prev_sticky in ("ok", "wrong"):
                status = prev_sticky

            if len(history) >= 2 and history[-1] == history[-2]:
                status = history[-1]

            # Opposite needs two hits to flip sticky
            if prev_sticky in ("ok", "wrong") and raw in ("ok", "wrong") and raw != prev_sticky:
                opp = [h for h in history if h == raw]
                if len(opp) < 2:
                    status = prev_sticky

            if status in ("ok", "wrong"):
                new_sticky[track_id] = status
            elif prev_sticky in ("ok", "wrong"):
                new_sticky[track_id] = prev_sticky

            item["_status"] = status
            if status == "wrong":
                any_wrong = True

            new_tracks.append(
                {
                    "id": track_id,
                    "cx": item["_cx"],
                    "cy": item["_cy"],
                    "box": item["box"],
                }
            )
            new_votes[track_id] = history

        state["tracks"] = new_tracks
        state["votes"] = new_votes
        state["sticky"] = new_sticky

    return any_wrong


def _pack_direction_dets(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for v in items:
        status = v.get("_status", "unknown")
        if status == "wrong":
            label = "Wrong way"
            model = "wrong_way"
        elif status == "ok":
            label = "OK"
            model = "ok_way"
        else:
            label = "Tracking"
            model = "tracking"
        out.append(
            {
                "model": model,
                "label": label,
                "score": v["score"],
                "box": v["box"],
                "status": status,
            }
        )
    return out


def detect_wrong_way(
    image: np.ndarray, session_id: str, correct_direction: str = "down"
) -> Tuple[List[Dict[str, Any]], bool]:
    """Track people walking direction against the chosen correct way."""
    current: List[Dict[str, Any]] = []
    for x1, y1, x2, y2, conf in _iter_person_boxes(image):
        cx, cy = _centroid([x1, y1, x2, y2])
        current.append(
            {
                "score": conf,
                "box": [x1, y1, x2, y2],
                "_cx": cx,
                "_cy": cy,
            }
        )

    wrong = _track_direction(current, session_id, correct_direction)
    return _pack_direction_dets(current), wrong


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

    if want_people or want_wrong:
        people_for_track: List[Dict[str, Any]] = []
        for x1, y1, x2, y2, conf in _iter_person_boxes(image):
            cx, cy = _centroid([x1, y1, x2, y2])
            people_for_track.append(
                {
                    "score": conf,
                    "box": [x1, y1, x2, y2],
                    "_cx": cx,
                    "_cy": cy,
                }
            )
        people_count = len(people_for_track)
        if want_wrong:
            wrong = _track_direction(people_for_track, session_id, direction)
            detections.extend(_pack_direction_dets(people_for_track))
            if wrong:
                alerts.append("wrong_way")
        elif want_people:
            for p in people_for_track:
                detections.append(
                    {
                        "model": "people",
                        "label": "person",
                        "score": p["score"],
                        "box": p["box"],
                    }
                )

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
