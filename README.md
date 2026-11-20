# video-indexing-yolo11

Per-second object indexing of FineVideo with Ultralytics YOLO11l (person, car, dog, laptop, cell phone + people count), optimised for speed.

## Performance (NVIDIA L4, details in `HARDWARE.md`)
| | Brute force (every frame, FP32) | This pipeline (defaults) |
|---|---|---|
| All 50 files (12.98 M frames) | ~34.7 h (projected from a timed file) | **1 h 52 min** (measured) |
| Throughput | 104 frames/s | 1,932 frames/s (**18.6x**) |
| Tune reference (train-00000): F1 / people exact / within ±1 | 1 / 1 / 1 | 0.988 / 0.959 / 0.999 |
| Holdout reference (train-00024, -00049) | 1 / 1 / 1 | 0.990 / 0.968 / 0.999 |

Pass thresholds: F1 >= 0.95, exact >= 0.85, within ±1 >= 0.95. Parameters were chosen on the tune file only; the holdout files were only scored.
How: every frame is decoded and timestamped, but only 3 evenly spaced frames per second go to the model (FP16); decode,
sampling and letterboxing run in 2 threads while the GPU infers batches built across videos.

## Install (Ubuntu 22.04, Python 3.11, CUDA 12)
```
sudo apt-get install -y libgl1 libglib2.0-0     # runtime libs for OpenCV
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare (untimed, needs internet)
Downloads `yolo11l.pt` next to `run.py` and loads it once to check it works.
```
python prepare.py
```

## Run (no internet needed)
```
python run.py --input <folder of .parquet files> --output results.jsonl
```
Defaults: 3 evenly spaced frames per second go to the model (`--sample 3`; every frame is still decoded and timestamped) in FP16 (`--precision fp16`). `--sample 0 --precision fp32` is the brute-force reference.
Optional flags: `--batch 32`, `--resume` (continue an interrupted run via `results.jsonl.done`), `--limit-videos N`, `--dump-frames DIR` (per-frame npz for offline experiments; needs `--sample 0 --decode-threads 0`), `--decode-threads N` (default 2: decode/letterbox in N threads overlapped with inference, batches across videos; 0 = one video at a time through `predict()`; not with `--dump-frames`).
A run writes `<output>.config.json` (sample, precision); `--resume` refuses to continue an output made with other settings. Stage timings (decode / infer / aggregate / total) are printed to stderr.

Durable checkpoints for long runs on a machine whose disk can be wiped (Colab). Off unless a flag is given; the plain command above is unchanged.
```
python run.py --input DIR --output results.jsonl --checkpoint-dir /content/drive/MyDrive/vidx-ckpt
python run.py --input DIR --output results.jsonl --checkpoint-hf-repo USER/REPO   # private HF dataset repo, token in HF_TOKEN
```
`--run-name NAME` (default: output file name) separates runs, `--checkpoint-minutes M` sets the interval (default 10); both flags can be combined. A snapshot is also taken when the run ends, aborts or is interrupted. After a reset, repeat the same command with `--resume`: the newest valid snapshot is restored if it is ahead of the local files. A fresh run (no `--resume`) refuses to start when a checkpoint of that name exists (exit 2). `progress.json` in the checkpoint holds the accumulated compute time (`compute_s`) over all sessions.

## Score a result against a reference
```
python score.py reference.jsonl candidate.jsonl
```

## Tests (CPU, standard library only)
```
python -m unittest discover -s tests -v
```

Colab L4 workflow: see `colab/SETUP.md`.

## Performance
Work in progress.
