# CLAUDE.md — Video Indexing at Speed (YOLO11l take-home)

## Start of every session
1. Read `worklog/WORKLOG.md` (committed to the repo, so never put secrets in it). It holds pending/completed work, open/closed issues and where the last session stopped.
2. Pick up from its **Next up** section. If unsure what to do next, ask the user.
3. Update `worklog/WORKLOG.md` before ending a session or after any milestone (dated entries, absolute dates).

## Task (from the brief)
`python run.py --input <folder of .parquet files> --output results.jsonl`
For every second of every video, report which of **person, car, dog, laptop, cell phone** appear and how many people are visible. Make it as **fast as possible while still passing the accuracy checks**. Frame-by-frame brute force is the reference; our job is to get near-identical results in far less time.

**Deadline:** Sunday 22 Nov 2026, 23:55 PKT — last commit before then is graded.

### Data
- FineVideo (`HuggingFaceFV/finevideo`), first 50 Parquet files (~25 GB, ~1,600 videos, ~125 h).
- `hf download HuggingFaceFV/finevideo --repo-type dataset --include "data/train-000[0-4]*" --local-dir finevideo`
- One row = one video. `mp4` column = video bytes. `json` column = metadata (not needed).

### Model rules (must not be violated)
- Ultralytics **YOLO11l**, official `yolo11l.pt`, **imgsz 640**, **conf >= 0.4**.
- Ultralytics defaults otherwise: aspect-preserving letterbox, **class-aware NMS IoU 0.7**, **BGR** frames.
- If a video has a **rotation tag**, rotate frames before inference.
- COCO ids: person=0, car=2, dog=16, laptop=63, cell phone=67.

### Output (JSON Lines, line order irrelevant)
```
{"video": "train-00003-of-01357.parquet:4", "second": 12, "objects": ["person", "laptop"], "people": 2}
{"video": "train-00003-of-01357.parquet:5", "error": "could not decode video"}
```
- `video` = parquet basename + `:` + 0-based row index.
- Second N = frames with timestamp in [N, N+1), timestamps counted from the **first decoded frame**.
- Emit every second up to and including the one holding the last frame; empty seconds → `"objects": [], "people": 0`.
- Object names exactly as above. Damaged video → one `error` line; if it fails partway, keep lines already written and append one error line. Never crash the whole run.

### How it is graded (reference = model on every frame)
- Object present in a second if detected in **>= half** of that second's frames.
- People = **median** of per-frame person counts, **rounded down** at .5.
- Pass needs all: objects **F1 >= 0.95**; people **exact on >= 85%** of seconds; people **within ±1 on >= 95%**. Fail → speed doesn't count.

### Grading machine
RTX 4090 24 GB, 32 CPU threads, 64 GB RAM, Ubuntu 22.04, **Python 3.11**, CUDA 12 driver. Fresh VM, `pip install -r requirements.txt`, optional apt-get lines from README, optional **untimed preparation command (has internet)** — use it to fetch `yolo11l.pt` and build anything GPU-specific (e.g. TensorRT engine). Timed run has **no internet**. Hard stop at 6 h. Test set ≈13 h of unseen, possibly damaged/unusual FineVideo files.

### Deliverables (public repo)
`run.py` (+ modules), `requirements.txt` (exact pins), `README.md` (install/prep/run steps), `results.jsonl.gz` (all 50 files), `HARDWARE.md` (where/what ran the full run + total time), `WRITEUP.md`, `ai_chats/` (full, unedited Claude Code `.jsonl` from `~/.claude/projects/`, tokens replaced with `[REMOVED]`; web chat share links in `ai_chats/links.txt`).

## HARD RULES
- **Never write, draft, rewrite or check answers in `WRITEUP.md`.** The brief requires the user's own words, without AI. Claude may only create the empty question headings if asked.
- **Never commit secrets.** No HF tokens in code, notebooks, or outputs. Scan before every push (`git grep -nE "hf_[A-Za-z0-9]{20,}"`).
- **Never commit** data, weights, or raw `results.jsonl` (only `results.jsonl.gz`). `worklog/WORKLOG.md` IS committed: keep secrets and tokens out of it.
- Correctness first: every speed change must be validated against a brute-force reference on a sample before it's accepted.
- **Tune only on `train-00000`.** `train-00024` and `train-00049` are held out: scored only, never used to choose parameters.
- When anything is ambiguous, **ask the user**.

## Workflow (agreed with user)
1. **Plan mode first** for every non-trivial change — planning with **Opus**.
2. **Independent verification** of the plan by a separate subagent (fresh context) before execution.
3. **Execution with Sonnet** (user switches with `/model`; `opusplan` is not available).
4. **After every major change, run `/code-review`** and fix findings before moving on.
5. Log the outcome in `worklog/WORKLOG.md`. Commit only when the user asks.

## Inference backend (TensorRT deferred)
- **Now:** correct baseline with plain `yolo11l.pt` on PyTorch, then batching / pipelining / sampling / FP16.
- **Design for a swap:** keep the model behind one small interface (e.g. `load_model()` + `infer(batch)`) so another backend can be dropped in without touching the rest of the pipeline.
- **TensorRT is a stretch goal**, only if benchmarks show GPU inference is the bottleneck. If added:
  - Never commit `.engine` files (already gitignored).
  - Build the engine in a separate `prepare.py` run once by the grader in the untimed prep step on their RTX 4090 (downloads `yolo11l.pt`, exports engine). Document the command in README.
  - `run.py` must fall back to the `.pt` model (FP16) if the engine is missing or fails to load, and log which backend it used. A TensorRT problem may cost speed, never break the run.
  - Use dynamic input shapes (or otherwise preserve Ultralytics' default rect/minimum-rectangle letterboxing) — fixed 640x640 padding could change detections vs. the reference.
  - Keep TensorRT only if it passes `score.py` against the every-frame reference.

## Environment
- Local: Windows 11, Python 3.14 (dev/editing only, no GPU work here).
- Compute: **Google Colab Pro, L4 GPU (24 GB)**, via the VS Code Colab extension. Code reaches Colab by `git clone`/`pull` of this repo; data lives on the Colab disk (check free space before downloading 25 GB). HF token via Colab Secrets, never in files. Colab sessions can disconnect → full runs must be **resumable** (skip videos already in the output).
- Target runtime is Python 3.11 + CUDA 12; keep `requirements.txt` compatible with it.
