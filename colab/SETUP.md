# Colab L4 setup (VS Code Colab extension)

The ready-made notebook `colab/baseline.ipynb` contains every step below as cells.

How the extension works: you open a notebook in VS Code and pick a Colab server as its kernel; cells run on
Google's VM, not on your machine. Local files are **not** synced, so the code reaches Colab via `git clone`.
Shell commands go in cells as `!cmd` or `%%bash`. Every `!` line is a fresh shell, so `source activate` does
not persist; always call the venv tools by full path (`/content/venv/bin/python`, `/content/venv/bin/hf`).
Keep VS Code open and the machine awake during long runs. A runtime reset wipes `/content`.

Never paste the HF token into a file or a saved cell: use a Colab Secret named `HF_TOKEN`, or the `getpass` prompt.

## 1. Connect and check the machine
VS Code: install the "Google Colab" extension, open `colab/baseline.ipynb`, **Select Kernel → Colab → New Colab
Server → GPU → L4**, sign in with the Colab Pro account.
```
!nvidia-smi; df -h /content; free -h; nproc; lscpu | grep "Model name"; python3 --version
```
Note GPU, CPU model + cores, RAM, disk for HARDWARE.md. Need ~25 GB data plus headroom.

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
/content/venv/bin/python -c "import torch,av,ultralytics;print(torch.__version__,torch.cuda.is_available(),av.__version__,ultralytics.__version__)"
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

## 4. Data
First two files (foreground), then all 50 in the background:
```
!/content/venv/bin/hf download HuggingFaceFV/finevideo --repo-type dataset --include "data/train-0000[0-1]*" --local-dir /content/finevideo
!df -h /content
!nohup /content/venv/bin/hf download HuggingFaceFV/finevideo --repo-type dataset --include "data/train-000[0-4]*" --local-dir /content/finevideo > /content/download.log 2>&1 &
```
Files land in `/content/finevideo/data/`. After a reset, re-run these; finished files are skipped.

## 5. Durable checkpoint storage
Try Google Drive first:
```python
from google.colab import drive
drive.mount("/content/drive")
```
If the mount fails through the extension: create a **private** dataset repo on huggingface.co (token needs write
access) and replace `--checkpoint-dir …` with `--checkpoint-hf-repo YOURNAME/vidx-ckpt` below; `--dump-frames`
can then stay on local disk (`reference/frames`) and the npz files are uploaded to `<run-name>/frames/`.

## 6. Baseline on the subset (reference + per-frame dumps)
Runs in the background so the notebook stays usable:
```
!cd /content/video-indexing-yolo11 && nohup /content/venv/bin/python run.py --input /content/finevideo/data --output reference/ref_subset.jsonl --limit-videos 64 --dump-frames /content/drive/MyDrive/vidx/frames --checkpoint-dir /content/drive/MyDrive/vidx --run-name ref_subset > /content/baseline.log 2>&1 &
!tail -5 /content/baseline.log
```
`--limit-videos 64` ≈ the first two files. After a reset: redo steps 1–5, then the same command **with `--resume`**.
A fresh run without `--resume` refuses to start when a checkpoint with that run name exists.
Record the final timing line (decode / infer / aggregate / total, frames/s) in the worklog.
Elapsed time across sessions: `progress.json` in the newest checkpoint generation (`compute_s`, `wall_first_start`,
`wall_last_end`, `sessions`).

## 7. Checks
```
%%bash
cd /content/video-indexing-yolo11
PY=/content/venv/bin/python
$PY tools/check_schema.py reference/ref_subset.jsonl
$PY score.py reference/ref_subset.jsonl reference/ref_subset.jsonl     # expect 1.0 / 100% / 100%
$PY tools/find_rotated.py --input /content/finevideo/data --max-files 2
$PY tools/decoder_agreement.py --input /content/finevideo/data --n 20
$PY tools/batch_agreement.py --input /content/finevideo/data --frames 1000
```
Rotation: for a rotated clip found above, compare against `ffmpeg` autorotate (one frame with and without `-noautorotate`).
Damaged input: truncate a video's bytes and corrupt a copy of a parquet file, run on that folder; each must give error lines and the run must continue.
Offline: run once with networking disabled; nothing may download.

## 8. Scoring a candidate later
```
!cd /content/video-indexing-yolo11 && /content/venv/bin/python score.py reference/ref_subset.jsonl candidate.jsonl
```
