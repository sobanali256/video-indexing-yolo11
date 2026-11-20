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


# ---------------------------------------------------------------- direct path (Stage 2a)
# predict() letterboxes every image on the calling thread and re-runs setup_source on every call. The
# direct path lets decode threads letterbox (Ultralytics' own LetterBox, same arguments the predictor
# uses) and runs the predictor's own AutoBackend + NMS with the predictor's own arguments.

def prepare_direct(model):
    """Run predict() once so Ultralytics builds its predictor (fused AutoBackend, fp16 if requested, warmup),
    then keep what the direct path needs. Must be called before make_letterbox / infer_tensor."""
    import numpy as np

    dummy = np.zeros((IMGSZ * 9 // 16, IMGSZ, 3), dtype=np.uint8)
    infer(model, [dummy])
    p = model.yolo.predictor
    model.direct = {
        "backend": p.model,
        "args": p.args,
        "imgsz": tuple(p.imgsz) if isinstance(p.imgsz, (list, tuple)) else (p.imgsz, p.imgsz),
        "stride": p.model.stride,
    }
    a = p.args
    log_args = (a.conf, a.iou, a.classes, a.agnostic_nms, a.max_det, p.model.fp16, model.direct["imgsz"])
    return log_args


def make_letterbox(model):
    """-> prepare(image): the predictor's letterbox for a batch of same-shape images (auto=True minimal
    rectangle, as predict() uses for a .pt model or a dynamic engine). One per thread."""
    from ultralytics.data.augment import LetterBox

    d = model.direct
    lb = LetterBox(d["imgsz"], auto=True, stride=d["stride"])
    return lambda image: lb(image=image)


def infer_tensor(model, images):
    """images: list of letterboxed BGR uint8 arrays of one shape. Same preprocessing as
    BasePredictor.preprocess (BGR->RGB, BCHW, to device, half/float, /255), the predictor's own
    backend, and DetectionPredictor.postprocess's NMS call with the predictor's args."""
    import numpy as np
    import torch
    from ultralytics.utils import ops

    d = model.direct
    backend, a = d["backend"], d["args"]
    t0 = time.perf_counter()
    with torch.inference_mode():
        x = torch.from_numpy(np.stack(images)).to(backend.device)
        x = x[..., [2, 1, 0]].permute(0, 3, 1, 2).contiguous()  # BGR->RGB, BHWC->BCHW
        x = x.half() if backend.fp16 else x.float()
        x /= 255
        preds = backend(x)
        dets = ops.non_max_suppression(
            preds, a.conf, a.iou, a.classes, a.agnostic_nms, max_det=a.max_det,
            nc=len(backend.names), end2end=getattr(backend, "end2end", False), rotated=False,
        )
        out = []
        for det in dets:
            if det is None or len(det) == 0:
                out.append(summarize([], []))
            else:
                c = det.cpu()  # columns x1 y1 x2 y2 conf cls
                out.append(summarize(c[:, 5].tolist(), c[:, 4].tolist()))
    sp = model.speed
    sp["predict_wall_s"] += time.perf_counter() - t0
    sp["predict_calls"] += 1
    sp["images"] += len(images)
    return out
