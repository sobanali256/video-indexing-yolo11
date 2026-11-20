# Colab L4 setup (colab.research.google.com)

The ready-made notebook `colab/baseline.ipynb` contains every step below as cells.

Open it in the browser: `https://colab.research.google.com/github/sobanali256/video-indexing-yolo11/blob/main/colab/baseline.ipynb`
(or File → Open notebook → GitHub). Save a copy to Drive if you want to keep cell outputs. The code reaches the VM
via `git clone` (the VS Code Colab extension was unreliable, so we use the website).
Shell commands go in cells as `!cmd` or `%%bash`. Every `!` line is a fresh shell, so `source activate` does
not persist; always call the venv tools by full path (`/content/venv/bin/python`, `/content/venv/bin/hf`).
Keep the browser tab open during long runs. A runtime reset wipes `/content`.

Never paste the HF token into a file or a saved cell: use a Colab Secret named `HF_TOKEN`, or the `getpass` prompt.

## 1. Connect and check the machine
**Runtime → Change runtime type → L4 GPU**, then Connect (Colab Pro account).
```
!nvidia-smi; df -h /content; free -h; nproc; lscpu | grep "Model name"; python3 --version
```
Note GPU, CPU model + cores, RAM, disk for HARDWARE.md. Need ~25 GB for all 50 files plus headroom (only 3 files are needed first).

## 2. Python 3.11 venv, code, model
The grader uses Python 3.11; Colab's default differs, so use a uv-managed 3.11 venv.
```
%%bash
set -e
cd /content
pip install -q uv
[ -d /content/venv ] || uv venv --python 3.11 /content/venv
[ -d video-indexing-yolo11 ] || git clone https://github.com/sobanali256/video-indexing-yolo11.git
cd video-indexing-yolo11 && git pull
apt-get install -y -q libgl1 libglib2.0-0
uv pip install --python /content/venv/bin/python -r requirements.txt
/content/venv/bin/python -c "import vidx,torch,av,ultralytics;print(torch.__version__,torch.cuda.is_available(),av.__version__,ultralytics.__version__)"
/content/venv/bin/python prepare.py
```
If a pin fails to resolve, fix it in `requirements.txt` and log it in `worklog/WORKLOG.md`.

## 3. Hugging Face login
```python
from huggingface_hub import login
try:
    from google.colab import userdata
    login(userdata.get("HF_TOKEN"))
except Exception:
    import getpass
    login(getpass.getpass("HF token: "))
```
The token is cached on the VM (`~/.cache/huggingface`), outside the repo; the venv's `hf` uses it too.

## 4. Google Drive (checkpoints, dumps, reference copies)
```python
from google.colab import drive
drive.mount("/content/drive")
```
`DRIVE=/content/drive/MyDrive/vidx` below. If the mount fails: create a **private** dataset repo
on huggingface.co (token needs write access) and replace `--checkpoint-dir …` with `--checkpoint-hf-repo YOURNAME/vidx-ckpt`;
`--dump-frames` can then stay on local disk and the npz files are uploaded to `<run-name>/frames/`.

## 5. Reference sets: tune and holdout
The brute-force reference uses 3 parquet files spread over the dataset:
- **tune** = `train-00000`: all tuning of sampling and other speed-ups happens on this file only.
- **holdout** = `train-00024` + `train-00049`: only scored, never tuned on.

Brute force is **not** run on all 50 files (projected 25+ h). Instead: a clean timing on the tune file, a model-free frame
count of all 50 files, and an extrapolation (step 12). `--limit-videos` is not used.

## 6. Data: only the 3 reference files first
```
!/content/venv/bin/hf download HuggingFaceFV/finevideo --repo-type dataset --include "data/train-00000-*" "data/train-00024-*" "data/train-00049-*" --local-dir /content/finevideo
```
Symlinks keep the original file names, so video ids (`<parquet name>:<row>`) do not change:
```
%%bash
set -e
shopt -s failglob  # an unmatched glob (missing download) aborts instead of linking a literal pattern
mkdir -p /content/subset/tune /content/subset/holdout /content/ref
D=/content/finevideo/data
ln -sf $D/train-00000-*.parquet /content/subset/
ln -sf $D/train-00024-*.parquet /content/subset/
ln -sf $D/train-00049-*.parquet /content/subset/
ln -sf $D/train-00000-*.parquet /content/subset/tune/
ln -sf $D/train-00024-*.parquet /content/subset/holdout/
ln -sf $D/train-00049-*.parquet /content/subset/holdout/
```
After a reset, redo steps 1-4, re-download the 3 reference files (the step 6 download command), recreate the symlinks and, if the 50-file download (step 8) had not finished, start it again; finished downloads are skipped.

## 7. Clean timing run on the whole tune file (no download running, nothing else on the GPU)
The plain grader command (no dumps, no checkpoints) wrapped by `bench.py`; its output is the tune reference:
```
%%bash
cd /content/video-indexing-yolo11
DRIVE=/content/drive/MyDrive/vidx
mkdir -p $DRIVE/reference
/content/venv/bin/python bench.py run --label "brute force baseline" --input /content/subset/tune --out /content/ref/ref_tune.jsonl --log $DRIVE/bench_log.csv -- --sample 0 --precision fp32 --decode-threads 0
cp /content/ref/ref_tune.jsonl $DRIVE/reference/
```
The timing must come from **one uninterrupted run**: after a runtime reset just run it again from scratch. The
`bench_log.csv` row must say `status=ok, clean=yes` (a download or a foreign GPU process during the run makes it `clean=no`).
`bench_log.csv` is copied into the repo later and committed as evidence for the write-up.

## 8. Background download of all 50 files
Into the **same** `--local-dir`, so the 3 existing files are skipped (check the start of `download.log`):
```
!nohup /content/venv/bin/hf download HuggingFaceFV/finevideo --repo-type dataset --include "data/train-000[0-4]*" --local-dir /content/finevideo > /content/download.log 2>&1 &
!head -c 1500 /content/download.log
```

## 9. Tune dump run (may overlap the download; not timed)
Dumps and checkpoints go to Drive, the output stays local. Gives per-frame data for offline tuning and a determinism check:
```
!cd /content/video-indexing-yolo11 && nohup /content/venv/bin/python run.py --input /content/subset/tune --output /content/ref/ref_tune_dump.jsonl --dump-frames /content/drive/MyDrive/vidx/frames --checkpoint-dir /content/drive/MyDrive/vidx/ckpt --run-name ref_tune_dump --sample 0 --precision fp32 --decode-threads 0 > /content/tune_dump.log 2>&1 &
```
After a reset: redo steps 1-4, re-download the 3 reference files (the step 6 download command), recreate the symlinks and, if the 50-file download (step 8) had not finished, start it again, then the same command **with `--resume`**. A fresh run without
`--resume` refuses to start when a checkpoint with that run name exists. Elapsed time across sessions: `progress.json`
in the newest checkpoint generation (`compute_s`, `wall_first_start`, `wall_last_end`, `sessions`).

## 10. Holdout reference (no dumps: holdout is never tuned on)
```
!cd /content/video-indexing-yolo11 && nohup /content/venv/bin/python run.py --input /content/subset/holdout --output /content/ref/ref_holdout.jsonl --checkpoint-dir /content/drive/MyDrive/vidx/ckpt --run-name ref_holdout --sample 0 --precision fp32 --decode-threads 0 > /content/holdout.log 2>&1 &
```
After a reset: redo steps 1-4, re-download the 3 reference files (the step 6 download command), recreate the symlinks and, if the 50-file download (step 8) had not finished, start it again, then use `--resume`.

## 11. Checks, then copy the references to Drive
```
%%bash
cd /content/video-indexing-yolo11
PY=/content/venv/bin/python
DRIVE=/content/drive/MyDrive/vidx
$PY tools/check_schema.py /content/ref/ref_tune.jsonl
$PY tools/check_schema.py /content/ref/ref_holdout.jsonl
$PY score.py /content/ref/ref_tune.jsonl /content/ref/ref_tune.jsonl          # expect 1.0 / 100% / 100%
$PY score.py /content/ref/ref_holdout.jsonl /content/ref/ref_holdout.jsonl
$PY score.py /content/ref/ref_tune.jsonl /content/ref/ref_tune_dump.jsonl     # determinism: about 1.0
$PY tools/find_rotated.py --input /content/subset/tune --max-files 1
$PY tools/decoder_agreement.py --input /content/subset/tune --n 20
$PY tools/batch_agreement.py --input /content/subset/tune --frames 1000
mkdir -p $DRIVE/reference
cp /content/ref/ref_tune.jsonl /content/ref/ref_tune_dump.jsonl /content/ref/ref_holdout.jsonl $DRIVE/reference/
```
Rotation: for a rotated clip found above, compare against `ffmpeg` autorotate (one frame with and without `-noautorotate`).
Damaged input: truncate a video's bytes and corrupt a copy of a parquet file, run on that folder; each must give error lines and the run must continue.
Offline: run once with networking disabled; nothing may download.

## 12. After the 50-file download has finished
```
%%bash
cd /content/video-indexing-yolo11
PY=/content/venv/bin/python
DRIVE=/content/drive/MyDrive/vidx
$PY tools/count_frames.py --input /content/finevideo/data --out $DRIVE/frame_counts.csv --workers 8
$PY tools/count_frames.py --input /content/subset/tune --out $DRIVE/frame_counts_tune.csv --compare-dumps $DRIVE/frames
$PY bench.py extrapolate --from-label "brute force baseline" --frames-csv $DRIVE/frame_counts.csv --log $DRIVE/bench_log.csv
```
`count_frames` only demuxes (no model): it counts video packets, skipping size-0 and discard packets. The `--compare-dumps`
gaps against the dumps of the tune file must be about 0. Compare the resolution mix printed for the tune file with the one
for all 50 files; if they differ a lot, mention it in the notes. The `extrapolated` row in `bench_log.csv` is labelled
as such: measured per-frame cost x total frames, measured on the Colab GPU (the grader uses an RTX 4090).

## 13. Scoring a candidate later
Tune is the only set to tune on; holdout is only scored:
```
!cd /content/video-indexing-yolo11 && /content/venv/bin/python score.py --ref tune=/content/ref/ref_tune.jsonl --ref holdout=/content/ref/ref_holdout.jsonl candidate.jsonl --json scores.json
```
Each reference selects its own parquet files from the candidate; an empty reference is a FAIL; `--only-common` scores only
the videos present in both files and labels the result PARTIAL. `bench.py run … --ref tune=… --ref holdout=…` does the
same after a timed run and logs the scores.
