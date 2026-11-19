# video-indexing-yolo11

Per-second object indexing of FineVideo with Ultralytics YOLO11l (person, car, dog, laptop, cell phone + people count), optimised for speed.

> Work in progress: the current `run.py` is the correct every-frame baseline. The performance section will be filled in once the speed-ups are validated.

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
Optional flags: `--batch 32`, `--resume` (continue an interrupted run via `results.jsonl.done`), `--limit-videos N`, `--dump-frames DIR` (per-frame npz for offline experiments). Stage timings (decode / infer / aggregate / total) are printed to stderr.

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
