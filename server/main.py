"""FastAPI server for multi-model YOLO detection."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from detectors import run_enabled
from models_loader import preload_models


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=preload_models, daemon=True).start()
    yield


app = FastAPI(title="HF Multi-Model Detection", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=2)


def _decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    return img


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/detect")
async def detect(
    image: UploadFile = File(...),
    session_id: str = Form("default"),
    enabled: str = Form("[]"),
    correct_direction: str = Form("down"),
):
    """
    Multipart form:
      - image: JPEG/PNG bytes
      - session_id: string for wrong-way tracking
      - enabled: JSON list e.g. ["people","smoke","fire","fall","wrong_way"]
      - correct_direction: down | up | right | left
    """
    raw = await image.read()
    try:
        enabled_list: List[str] = json.loads(enabled)
    except json.JSONDecodeError:
        enabled_list = []

    if not enabled_list:
        return {"people_count": 0, "detections": [], "alerts": []}

    img = _decode_image(raw)

    loop_result = await _run_in_executor(
        img, enabled_list, session_id, correct_direction
    )
    return loop_result


async def _run_in_executor(
    img: np.ndarray,
    enabled: List[str],
    session_id: str,
    correct_direction: str,
):
    import asyncio

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor,
        lambda: run_enabled(img, enabled, session_id, correct_direction),
    )


@app.get("/")
def root():
    return {
        "service": "HF Multi-Model Detection",
        "endpoints": ["/api/health", "/api/detect"],
    }
