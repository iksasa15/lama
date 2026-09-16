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


def _token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def _download(repo_id: str, filename: str) -> str:
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=_token(),
    )


def get_coco_model() -> YOLO:
    """Official YOLOv8n COCO (people + vehicles)."""
    with _lock:
        if "coco" not in _cache:
            _cache["coco"] = YOLO("yolov8n.pt")
        return _cache["coco"]


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
        get_coco_model()
        print("[preload] coco ready")
    except Exception as exc:  # noqa: BLE001
        print(f"[preload] coco failed: {exc}")
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
