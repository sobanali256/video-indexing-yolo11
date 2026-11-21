# video-indexing-yolo11

Per-second object indexing of FineVideo with Ultralytics **YOLO11l**: for every second of every video, which of
**person, car, dog, laptop, cell phone** appear and how many people are visible. The aim is to be as fast as possible
while staying close to the every-frame brute-force result.

```
python run.py --input <folder of .parquet files> --output results.jsonl
```

## Performance (NVIDIA L4, details in [`HARDWARE.md`](HARDWARE.md))
| | Brute force (every frame, FP32) | This pipeline (defaults) |
|---|---|---|
| All 50 files (1,650 videos, 12.98 M frames) | ~34.7 h (projected from a timed file) | **1 h 52 min** (measured, 0 errors) |
| Throughput | 104 frames/s | **1,932 frames/s (18.6x)** |
| Tune reference (`train-00000`): F1 / people exact / within ±1 | 1 / 1 / 1 | 0.988 / 0.959 / 0.999 |
| Holdout reference (`train-00024`, `train-00049`) | 1 / 1 / 1 | 0.990 / 0.968 / 0.999 |

Pass thresholds: objects F1 >= 0.95, people exact >= 0.85, people within ±1 >= 0.95. Every setting was chosen on the
tune file only; the holdout files were only scored. `results.jsonl.gz` is the output of the full run (tag `results-v1`).

Where the speed comes from (each step measured on the L4 against the reference):

| Step | Change | Frames/s | Gain |
|---|---|---|---|
| Brute force | every frame, FP32, one video at a time | 104 | 1x |
| FP16 | half-precision inference | 155 | 1.5x |
| Sampling | 3 evenly spaced frames per second go to the model | 1,214 | 11.7x |
| Pipelining | decode threads overlap the GPU, batches across videos, direct inference | 1,932 | 18.6x |

## How it works
```
parquet files -> video rows (--resume skips videos already done)
  decode threads (auto, 2-4): decode EVERY frame (PyAV) -> exact timestamp -> group by second
                       -> keep 3 evenly spaced frames per second (all if a second has <= 3)
                       -> rotate if the video has a rotation tag -> Ultralytics LetterBox (640, minimal rectangle)
  main thread (GPU):   batches of 32 frames of one letterboxed shape, mixed across videos
                       -> YOLO11l FP16 (conf 0.4, class-aware NMS IoU 0.7, the predictor's own backend and arguments)
  per second:          object present if detected in >= half the sampled frames; people = median count (rounded down)
                       -> one line per second up to the one holding the last frame (empty seconds included)
```
- Every frame is still decoded, so second boundaries, frame counts and the last second are exact; only the frames sent
  to the model are sampled (`vidx/sampling.py`, shared with the offline simulator).
- The direct inference path gives identical detections to Ultralytics `predict()` (checked on 1,000 frames, FP32 and FP16).
- Model rules kept: official `yolo11l.pt`, imgsz 640, conf 0.4, Ultralytics defaults otherwise (aspect-preserving letterbox,
  class-aware NMS IoU 0.7, BGR frames), rotation applied before inference.

## Install (Ubuntu 22.04, Python 3.11, NVIDIA driver >= 525 for CUDA 12)
```
git clone https://github.com/sobanali256/video-indexing-yolo11.git && cd video-indexing-yolo11
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0          # runtime libs for OpenCV (PyAV wheels bundle FFmpeg)
sudo apt-get install -y python3.11 python3.11-venv   # only if Python 3.11 is not installed (Ubuntu 22.04 ships 3.10)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Already have a Python 3.11 environment? `pip install -r requirements.txt` in it is enough. The install needs ~5 GB of
disk (PyTorch + CUDA wheels); torch 2.6.0 cu126 wheels run on any NVIDIA driver >= 525 (CUDA 12 minor-version compatibility).

## Prepare (untimed, needs internet)
Downloads `yolo11l.pt` next to `run.py` and runs it once on the GPU to check it works (prints the device it used).
```
python prepare.py
```

## Run (no internet needed)
```
python run.py --input <folder of .parquet files> --output results.jsonl
```
Run from any directory (paths are resolved from `run.py`). Expected time: ~1,900 video frames/s on an NVIDIA L4, i.e.
~12 min for 13 h of 30 fps video; progress and timings are logged to stderr every 10 videos.
Output: JSON Lines, one line per video second, e.g.
`{"video": "train-00003-of-01357.parquet:4", "second": 12, "objects": ["person", "laptop"], "people": 2}`.
A video that cannot be decoded gives one `{"video": ..., "error": ...}` line; if it fails partway, the seconds decoded
before the error are kept and one error line is appended. One bad video never stops the run.

| Flag | Default | Meaning |
|---|---|---|
| `--sample K` | 3 | frames per second sent to the model; `0` = every frame (brute force) |
| `--precision fp16\|fp32` | fp16 | model precision on the GPU |
| `--decode-threads N` | auto | decode/letterbox threads overlapped with inference; `auto` = one per 4 CPUs, 2 to 4 (4 on a 32-thread machine); `0` = one video at a time through `predict()`. It changes speed only, not which frames are used |
| `--batch B` | 32 | inference batch size (split in halves on CUDA out-of-memory) |
| `--resume` | off | continue an interrupted run (uses `<output>.done`); refuses other `--sample`/`--precision` (`<output>.config.json`) |
| `--limit-videos N` | all | only the first N videos |
| `--dump-frames DIR` | off | per-frame npz for offline experiments (needs `--sample 0`; with `auto` threads it switches to `--decode-threads 0` by itself) |
| `--stats-json PATH` | off | timing stats (frames, decode/infer time, settings) |

Brute-force reference: `python run.py --input DIR --output ref.jsonl --sample 0 --precision fp32 --decode-threads 0`.

Exit codes: `0` done; `2` bad arguments, missing weights or a resume/checkpoint conflict; `3` three inference batches
failed in a row (nothing is lost: restart and re-run with `--resume`). If the fast inference path cannot start, the run
logs a warning and falls back to `--decode-threads 0`.

### Durable checkpoints (long runs on a machine whose disk can be wiped, e.g. Colab)
Off unless a flag is given; the plain command above is unchanged.
```
python run.py --input DIR --output results.jsonl --checkpoint-dir /content/drive/MyDrive/vidx-ckpt
python run.py --input DIR --output results.jsonl --checkpoint-hf-repo USER/REPO   # private HF dataset repo, token in HF_TOKEN
```
`--run-name NAME` (default: output file name) separates runs, `--checkpoint-minutes M` sets the interval (default 10).
A snapshot is also taken when the run ends, aborts or is interrupted. After a reset, repeat the same command with
`--resume`: the newest valid snapshot is restored if it is ahead of the local files. A fresh run refuses to start when a
checkpoint of that name exists (exit 2). `progress.json` in the checkpoint holds the accumulated compute time.

## Validation and tools
| Command | Purpose |
|---|---|
| `python score.py ref.jsonl cand.jsonl` | the grader's three checks (objects F1, people exact, within ±1) |
| `python score.py --ref tune=ref_tune.jsonl --ref holdout=ref_holdout.jsonl cand.jsonl` | score against several references at once |
| `python bench.py run --label L --input DIR --out OUT --ref tune=... -- <run.py flags>` | timed run + scoring, one row in `bench_log.csv` |
| `python bench.py extrapolate --from-label L --frames-csv frame_counts.csv` | project a full-run time from a timed run |
| `python tools/simulate_sampling.py --dumps DIR --ref ref_tune.jsonl` | score sampling settings offline from every-frame dumps (no GPU) |
| `python tools/count_frames.py --input DIR --out frame_counts.csv` | count frames of all videos without the model (demux only) |
| `python tools/check_schema.py results.jsonl` | output format check |
| `python tools/batch_agreement.py --input DIR --direct --precision fp16` | direct inference path vs `predict()` |
| `python tools/decoder_agreement.py`, `tools/profile_decode.py`, `tools/find_rotated.py` | decoder, decode-speed and rotation checks |

References: every-frame brute force on 3 parquet files spread over the dataset: **tune** = `train-00000` (all tuning),
**holdout** = `train-00024` + `train-00049` (scored only). All measured runs are in `worklog/bench_log.csv`; the full
history of decisions and results is in `worklog/WORKLOG.md` and `worklog/plans/`.

## Tests (CPU, no GPU or model needed)
```
python -m unittest discover -s tests -v
```

## Repository layout
| Path | Contents |
|---|---|
| `run.py` | entry point: arguments, result loop, resume, checkpoints |
| `vidx/` | `source` (parquet rows), `decode` (PyAV, timestamps, rotation), `sampling`, `model` (YOLO11l, direct path), `pipeline` (decode threads + cross-video batches), `aggregate` (per-second rules), `output` (writer + resume), `checkpoint` |
| `prepare.py` | untimed preparation: downloads `yolo11l.pt` |
| `score.py`, `bench.py`, `tools/` | scoring, timing, analysis |
| `colab/` | Colab notebooks: `baseline.ipynb` (references), `speedup.ipynb` (stages, full run), `SETUP.md` |
| `results.jsonl.gz` | results for the first 50 parquet files |
| `HARDWARE.md` | machine and total time of the full run |
| `worklog/` | worklog, plans, bench log |
| `ai_chats/` | AI chat logs |
