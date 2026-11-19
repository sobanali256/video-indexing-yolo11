"""Untimed preparation step (needs internet): fetch yolo11l.pt next to run.py and check it loads.

    python prepare.py
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "yolo11l.pt"


def download():
    if WEIGHTS.is_file():
        print(f"{WEIGHTS} already present")
        return
    os.environ.setdefault("YOLO_CONFIG_DIR", str(HERE / ".yolo_config"))
    try:
        from ultralytics.utils.downloads import attempt_download_asset

        attempt_download_asset(str(WEIGHTS))
    except Exception as e:
        print(f"attempt_download_asset failed ({e}); trying YOLO('yolo11l.pt')", file=sys.stderr)
    if not WEIGHTS.is_file():
        os.chdir(HERE)
        from ultralytics import YOLO

        YOLO("yolo11l.pt")  # downloads into the current directory
    if not WEIGHTS.is_file():
        raise SystemExit(f"could not download weights to {WEIGHTS}")


def main():
    download()
    from vidx.model import infer, load_model

    import numpy as np

    model = load_model(WEIGHTS, offline=False)
    out = infer(model, [np.zeros((480, 640, 3), dtype=np.uint8)])
    print(f"model OK on {model.device}; blank-frame result: {out}")


if __name__ == "__main__":
    main()
