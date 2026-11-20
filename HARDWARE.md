# Hardware and full-run time

## Where the full run ran
| | |
|---|---|
| Platform | Google Colab Pro (website), GPU runtime |
| GPU | NVIDIA L4, 23,034 MiB, driver 580.82.07 |
| CPU | Intel Xeon @ 2.20 GHz, 12 vCPUs (1 socket, 2 threads per core) |
| RAM | 52 GiB |
| Disk | Colab local disk (`/content`); parquet files downloaded before the run (not timed) |
| Python | 3.11 (uv venv; the Colab kernel itself is 3.13) |
| Packages | torch 2.6.0+cu126, ultralytics 8.3.100, av 14.2.0 (exact pins in `requirements.txt`) |

## What ran
- Code: tag **`results-v1`** (commit `cfd9662`), `run.py` defaults: 3 sampled frames per second, FP16, 2 decode threads, batch 32.
- Command (the checkpoint/stats flags only add Drive snapshots and a stats file; they do not change results):
  ```
  python run.py --input /content/finevideo/data --output /content/results.jsonl \
    --checkpoint-dir /content/drive/MyDrive/vidx/ckpt --run-name full50 --stats-json /content/drive/MyDrive/vidx/full50.stats.json
  ```
- Input: the first 50 FineVideo parquet files (`train-00000` … `train-00049`), 1,650 videos, 12,981,615 frames, 461,757 output seconds (~128 h of video).

## Total time
| | |
|---|---|
| **Wall time, one uninterrupted session** | **6,721 s = 1 h 52 min** (2026-11-20 15:15:36 → 17:07:37 UTC) |
| Model load + startup | 3.3 s |
| Processing loop | 6,718 s → 1,932 decoded frames/s |
| Frames sent to the model | 1,385,035 (10.7% of all frames) |
| Videos with errors | 0 |

## Brute-force comparison (same L4)
The every-frame brute-force reference was not run on all 50 files. It was timed on `train-00000` (33 videos, 253,918 frames:
2,447 s, 103.9 frames/s, nothing else on the GPU) and projected with the exact frame count of all 50 files
(`tools/count_frames.py`): **124,936 s ≈ 34.7 h**. The full run above is **18.6x** faster.
All measured runs are logged in `worklog/bench_log.csv`.
