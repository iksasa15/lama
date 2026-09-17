"""Lazy-load YOLO weights from Hugging Face Hub / Ultralytics."""

from __future__ import annotations

import os
import threading
from typing import Dict

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

_lock = threading.Lock()
_cache: Dict[str, YOLO] = {}

# Hub repos verified via Hugging Face MCP
# D-Fire fine-tune: class 0=smoke, 1=fire
FIRE_SMOKE_REPO = "rabahdev/fire-smoke-yolov8n"
FIRE_SMOKE_FILE = "best.pt"
FALL_REPO = "melihuzunoglu/human-fall-detection"
FALL_FILE = "best.pt"
# CrowdHuman person-only detector (class 0 = person)
PERSON_REPO = "raghavendra24/crowdhuman-yolov8n"
PERSON_FILE = "crowdhuman_yolov8n_best.pt"


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _download(repo_id: str, filename: str) -> str:
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=_token(),
    )


def get_person_model() -> YOLO:
    """CrowdHuman YOLOv8n — dedicated person detector (crowds / varied clothing)."""
    with _lock:
        if "person" not in _cache:
            path = _download(PERSON_REPO, PERSON_FILE)
            _cache["person"] = YOLO(path)
        return _cache["person"]


# Back-compat alias used by detectors / fall assist
def get_coco_model() -> YOLO:
    return get_person_model()


def get_fire_smoke_model() -> YOLO:
    with _lock:
        if "fire_smoke" not in _cache:
            path = _download(FIRE_SMOKE_REPO, FIRE_SMOKE_FILE)
            _cache["fire_smoke"] = YOLO(path)
        return _cache["fire_smoke"]


def get_fall_model() -> YOLO:
    with _lock:
        if "fall" not in _cache:
            path = _download(FALL_REPO, FALL_FILE)
            _cache["fall"] = YOLO(path)
        return _cache["fall"]


def preload_models() -> None:
    """Warm caches in background so first UI click is faster."""
    try:
        get_person_model()
        print("[preload] person (CrowdHuman) ready")
    except Exception as exc:  # noqa: BLE001
        print(f"[preload] person failed: {exc}")
    try:
        get_fire_smoke_model()
        print("[preload] fire/smoke ready")
    except Exception as exc:  # noqa: BLE001
        print(f"[preload] fire/smoke failed: {exc}")
    try:
        get_fall_model()
        print("[preload] fall ready")
    except Exception as exc:  # noqa: BLE001
        print(f"[preload] fall failed: {exc}")
