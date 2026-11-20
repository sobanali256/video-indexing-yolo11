"""YOLO11l behind load_model() / infer(); swap this module for another backend later."""
import os
import tempfile
import time
from pathlib import Path

from .aggregate import COCO_IDS

IMGSZ = 640
CONF = 0.4
IOU = 0.7
_INDEX = {cid: i for i, cid in enumerate(COCO_IDS)}


def default_weights():
    return Path(__file__).resolve().parent.parent / "yolo11l.pt"


def summarize(cls_ids, confs):
    """One frame's detections -> (counts[5], max_conf[5]) in OBJECTS order."""
    counts = [0] * len(COCO_IDS)
    best = [0.0] * len(COCO_IDS)
    for c, p in zip(cls_ids, confs):
        i = _INDEX.get(int(c))
        if i is None:
            continue
        counts[i] += 1
        if p > best[i]:
            best[i] = float(p)
    return counts, best


class Model:
    def __init__(self, yolo, device, half=False):
        self.yolo = yolo
        self.device = device
        self.half = half
        # Stage-0 profile: Ultralytics' per-image ms summed over all images, plus predict() wall time
        self.speed = {"preprocess_ms": 0.0, "inference_ms": 0.0, "postprocess_ms": 0.0,
                      "predict_wall_s": 0.0, "predict_calls": 0, "images": 0}


def load_model(weights=None, device=None, offline=True, half=False):
    weights = Path(weights) if weights else default_weights()
    weights = weights.resolve()
    if not weights.is_file():
        raise FileNotFoundError(
            f"weights not found: {weights}. Run `python prepare.py` first (needs internet)."
        )
    if offline:
        os.environ.setdefault("YOLO_OFFLINE", "True")  # Ultralytics checks value.lower() == "true"
    os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(tempfile.gettempdir(), "yolo_config"))
    import torch
    from ultralytics import YOLO, settings

    try:
        settings.update(sync=False)  # no analytics / network calls
    except Exception:
        pass
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    half = bool(half) and str(device).startswith("cuda")  # FP16 only on the GPU
    return Model(YOLO(str(weights), task="detect"), device, half)


def infer(model, frames, classes=COCO_IDS):
    """frames: list of BGR ndarrays. Returns one (counts[5], max_conf[5]) per frame.

    classes=None disables the class filter (counts are still reported for the 5 classes).
    """
    t0 = time.perf_counter()
    results = model.yolo.predict(
        frames, imgsz=IMGSZ, conf=CONF, iou=IOU, half=model.half, verbose=False,
        device=model.device, classes=classes, save=False,
    )
    sp = model.speed
    sp["predict_wall_s"] += time.perf_counter() - t0
    sp["predict_calls"] += 1
    out = []
    for r in results:
        speed = getattr(r, "speed", None) or {}
        sp["preprocess_ms"] += speed.get("preprocess") or 0.0
        sp["inference_ms"] += speed.get("inference") or 0.0
        sp["postprocess_ms"] += speed.get("postprocess") or 0.0
        sp["images"] += 1
        if r.boxes is None or len(r.boxes) == 0:
            out.append(summarize([], []))
        else:
            d = r.boxes.data.cpu()  # one transfer: columns x1 y1 x2 y2 conf cls
            out.append(summarize(d[:, 5].tolist(), d[:, 4].tolist()))
    return out
