# Speed-up plan: sampling, FP16, parallel pipeline, scheduling (TensorRT last)

## Context
The brute-force baseline is complete and validated (tune 33 videos / 9,144 s, holdout 66 videos / 17,586 s, deterministic, decoders agree, batching safe). On the L4 it runs at 103.9 frames/s, about 34.7 h for all 50 files and ~3.6 h for the 13 h test set. We now need to be much faster without failing the accuracy checks.

User decisions:
- **Margin:** accept a setting only if it reaches F1 ≥ 0.98, people exact ≥ 0.92 and within ±1 ≥ 0.98 on **both** tune and holdout.
- **Scope:** staged work. Every stage leaves a validated, runnable version. TensorRT comes last, only if time remains.

Deadline: Sun 22 Nov 23:55 PKT. The full 50-file run (for `results.jsonl.gz`) must also fit before then.

## Research findings (what limits us)
1. **The model is ~98% of the time** (holdout split: infer 9,617 s, decode 180 s). The `infer` timer in `run.py:process_frames` also includes Ultralytics' CPU work: per-image `LetterBox` in a Python list comprehension, `np.stack`, BGR→RGB flip (`predictor.py` v8.3.100), plus `setup_source` on every `predict()` call. So part of "infer" is single-threaded CPU work, not GPU work. Stage 0 measures this.
2. **FP32 today.** `vidx/model.py:infer` passes `half=False`. The L4 and the 4090 both have FP16 tensor cores; FP16 typically gives ~2× on YOLO-L.
3. **Every frame goes through the model,** but the grading per second is a vote: object present if seen in ≥ half the frames, people = median (rounded down). A few well-chosen frames per second usually give the same vote. This is the largest win available (~30 fps video → 3–6 inferred frames/s ≈ 5–10× less model work).
4. **Once sampling cuts model work, decoding becomes the bottleneck.** We still have to decode every frame: H.264 can't skip frames, and we need every timestamp to know the seconds and frames per second. On the 4090 (2–3× an L4) this matters even more. Remedies:
   - convert only the sampled frames to BGR (`to_ndarray` / sws_scale is a large share of decode cost);
   - decode in **parallel worker processes** (the grader has 32 threads).
5. **Letterbox compatibility:** Ultralytics uses the minimal-rectangle letterbox (`auto=True`) when all images in a call share a shape and the model is `.pt` **or a dynamic engine**. So a dynamic TensorRT engine keeps reference-identical letterboxing, and a static 640×640 one would not (CLAUDE.md rule confirmed).
6. **Scheduling:**
   - One video at a time leaves the GPU idle during decode and gives small last batches.
   - Cross-video batching (frames grouped by letterboxed shape) keeps batches full.
   - Longest-videos-first ordering avoids a long single-video tail at the end of the run.

## Process
- Before execution, a separate subagent with fresh context verifies this plan independently (workflow step 2).
- Execution is done with Sonnet, with `/code-review` after each stage.
- Stage 0's numbers decide which later steps are worth doing.

## Stage 0: profile (≈30 min Colab, small code change)
- In `vidx/model.py:infer`, collect Ultralytics' per-result `r.speed` (preprocess / inference / postprocess ms) into the `Timers`. Write them to `--stats-json` and add them to `Timers.summary` (`run.py`).
- Add a decode micro-benchmark to a new `tools/profile_decode.py`: decode only, decode + full BGR conversion, decode + conversion of 1 in 6 frames, each with `thread_type` AUTO at 1/2/4 threads.
- Run on 5 tune videos with `bench.py run --label profile`, sampling GPU utilisation with `nvidia-smi dmon`.
- **Output:** the real split of CPU preprocessing / GPU / NMS / decode. This decides whether Stage 2 is needed on the L4 and how many workers to start.

## Stage 1: sampling + FP16 (biggest gain, smallest change)
### 1a. Offline sampling simulator (no GPU)
New `tools/simulate_sampling.py`:
- reads the tune per-frame dumps (`$DRIVE/frames/*.npz` from step 9: `t`, `counts` for every frame);
- re-aggregates using only the selected frames, through `vidx.aggregate.aggregate`;
- scores against `ref_tune.jsonl` using `score.py`'s functions (import, don't duplicate);
- reports, per strategy: F1 / exact / within±1 and the fraction of frames inferred.

Strategies:
- **Fixed k per second:** k = 1…8, evenly spaced (`idx_j = floor((j+0.5)·n/k)`). Also a variant that anchors the first and last frame.
- **Adaptive:** k0 frames first, then add frames up to k1 for a second only if the k0 frames disagree (an object's vote isn't unanimous, or the person counts differ).
- **Rule:** choose the cheapest strategy that meets the margin on tune with extra headroom. Use adaptive only if it is clearly cheaper than fixed k at the same accuracy.

### 1b. Implement sampling in the pipeline
- New `vidx/sampling.py`, pure functions: `pick(n, k)` and adaptive-refine helpers (unit-tested).
- Change `vidx/decode.py`:
  - decode every frame and stamp it (unchanged `Timestamper`);
  - group raw `av.VideoFrame`s by second (`int(t)`), buffering only the current second (≤ ~60 YUV frames);
  - when the second closes, select indices, then convert and rotate **only** those frames;
  - yield them with the second's frame count `n`.
- Adaptive (only if chosen): buffer converted candidate frames for a chunk of ~16 seconds. Pass 1 infers every second's k0 frames in one batch; pass 2 infers the extra candidates only for seconds that need them. Batches stay large and memory stays bounded.
- `aggregate` stays unchanged: it votes over the sampled frames, and second numbering is still the same (every frame is timestamped).
- New flags `--sample K` (and `--refine K1` if adaptive), defaulting to the tuned values, with `--sample 0` meaning every frame. The grader command stays `run.py --input --output`.

### 1c. FP16
- `infer(..., half=True)`, behind a `--fp32` escape flag.
- Measure FP16 on its own on 5 tune videos at every frame (`score.py --only-common` against `ref_tune`), so a sampling error and an FP16 error are never confused.

### Validation gate (Stage 1)
- `bench.py run --label "stage1 ..." --ref tune=ref_tune.jsonl` must meet the margin and records the speed.
- Then a single holdout run, `--ref holdout=ref_holdout.jsonl`, scored only and not tuned on. It must meet the margin too.
- Also: `check_schema`, unit tests, and per-video worst cases (no video type, such as high fps or rotated, much worse than the rest).

### Sampling edge cases
- If a second has n ≤ k frames, use all of them (unit-tested).
- Prefer odd k, so the sampled median is a real frame count and there are no vote ties.

## Stage 2: parallel pipeline + scheduling
New `vidx/pipeline.py`, used when `--workers N > 0`. `--workers 0` keeps the current single-process path for tests and as a fallback.

**Done in two steps; the Stage 0 numbers decide what is worth doing:**
- **2a:** decode workers + a plain `multiprocessing` queue (letterboxed numpy arrays pickled through the queue) + cross-video batching + direct-tensor inference.
- **2b (only if profiling shows the queue/IPC is the bottleneck):** replace the queue payload with the shared-memory ring buffer described below.

**Main process (GPU owner)**
- Reads parquet rows with the existing `vidx.source.iter_rows`.
- Orders the videos longest-first by mp4 byte size (check the correlation with `frame_counts.csv`).
- Feeds (video_id, bytes) to a bounded task queue.

**N decode worker processes**
- Start method is `spawn`, created before CUDA init.
- Default N = `max(1, os.cpu_count() - 4)`, with FFmpeg `thread_count` 2 each, so threads don't oversubscribe the cores. Test explicitly on Colab's low core count (where N may be 1–4) as well as with a forced larger N.
- Each worker: decode, timestamp, sample, convert, rotate, then **letterbox in the worker** using Ultralytics' own `LetterBox(640, auto=True, stride=32)`, the identical function the reference uses.
- 2a: puts letterboxed uint8 frames on a bounded `multiprocessing` queue (backpressure via maxsize). 2b only: writes them into slots of a shared-memory ring buffer instead (e.g. 512 × 640×640×3 ≈ 630 MB).
- Sends small metadata messages: (video_id, second, slot, letterboxed shape); per-video end message with per-second frame counts and any error text.

**GPU loop (main process)**
- Bucket frames by letterboxed shape **across videos**. Flush a bucket at batch size B, or when it is old or the queue runs dry.
- Copy to pinned memory, transfer uint8 to the GPU, BGR→RGB, `.half()/255`, forward, then `ultralytics.utils.ops.non_max_suppression` with the exact predictor arguments (conf 0.4, iou 0.7, classes, agnostic False, max_det 300 — confirm the cfg defaults), counting classes only (no box rescale needed).
- A second thread prepares the next batch while the current one runs (double buffering).
- A new `infer_tensor()` in `vidx/model.py` keeps the `load_model()` / `infer()` interface for the backend swap.

**Completion and robustness**
- When a video's end message has arrived and all its frames are inferred: `aggregate` → existing `OutputWriter.write_video` → sidecar → checkpoint. Resume and checkpoint code is unchanged.
- A worker crash (FFmpeg segfault on a damaged file) or a per-video stall timeout: lines for the frames received so far plus one error line (same semantics as now), then the worker is respawned. One bad video never stops the run.
- Logging every 10 videos: queue depth and GPU idle %, showing whether workers or the GPU is the bottleneck.

### Validation gate (Stage 2)
- **Direct path vs `predict()`:** identical counts on 1,000 frames (extend `tools/batch_agreement.py` with `--direct`).
- **Stage 2 vs Stage 1 on tune:** results identical (same sampled frames). Then holdout, plus the margin check.
- **Damaged input:** truncated mp4 and a corrupt parquet → error lines, and the run continues.
- **Kill and resume:** kill mid-run, then `--resume`.
- **Offline run** (no network).

## Stage 3 (only if time remains and still GPU-bound): TensorRT
Per the CLAUDE.md rules:
- `prepare.py` exports a **dynamic**, FP16 engine on the grader's GPU in the untimed prep step.
- `load_model()` falls back to `.pt` FP16 and logs which backend it used.
- Keep it only if it passes the margin on tune and holdout.

## Timeline (PKT) and fallback
| When | Work |
|---|---|
| Fri 20 evening | Stage 0 + 1a simulator → choose k |
| **Sat 21 morning** | Start the 50-file re-download in the background (the runtime reset wiped them). Stage 1b/1c implemented, validated on tune and holdout. **This is a shippable version.** **Fully offline run test** (network disabled); if Ultralytics attempts network calls, fix the offline settings. |
| Sat 21 afternoon/evening | Stage 2a (+2b only if Stage 0 says so) + validation gate |
| **Sat 21 night, before the user sleeps** | Launch the full 50-file resumable run (checkpoints on Drive) with the best version validated by then: Stage 2 if it passed its gate, else Stage 1. At ~6–10× it is ≈ 3.5–6 h on the L4. **Tag that commit** (e.g. `results-run`); the run uses default flags only, so the submitted `run.py` defaults are exactly the configuration that produced the results. |
| Sun 22 | Check the run finished (`--resume` if not), `results.jsonl.gz` + check_schema, HARDWARE.md (L4 + total time), README, ai_chats, secret scan, push. **WRITEUP.md: the user writes it, no AI; budget 2–3 h.** WORKLOG and bench_log stay current throughout so the user has the facts at hand. Stage 3 only if all of that is done. |

**Rule after the tag:** any later code change must not alter the defaults or the results. If it would, either re-run or don't merge it.

## Files
- **New:** `vidx/sampling.py`, `vidx/pipeline.py`, `tools/simulate_sampling.py`, `tools/profile_decode.py`, tests `tests/test_sampling.py`, `tests/test_pipeline.py` (fake decode and infer, like `test_run.py`).
- **Changed:**
  - `vidx/decode.py` (per-second grouping, selective conversion);
  - `vidx/model.py` (half, `infer_tensor`, speed stats);
  - `run.py` (flags `--sample/--refine/--workers/--fp32`, pipeline dispatch, timers);
  - `tools/batch_agreement.py` (`--direct`);
  - `colab/baseline.ipynb` + `SETUP.md` (new stage cells; copy logs and stats.json to Drive; wait for background runs before checks and copies);
  - README (flags, performance section).
- **Reused:** `vidx.aggregate.aggregate/median_floor`, `score.py` scoring, `bench.py run/extrapolate`, `vidx.source.iter_rows/list_inputs`, `vidx.output.OutputWriter`, `vidx.checkpoint`, `run.py:run_inference` (OOM split).

## Verification (every stage)
1. `python -m unittest discover -s tests` locally (CPU, fakes).
2. Colab: `bench.py run --ref tune=$DRIVE/reference/ref_tune.jsonl` → margin met and speed logged in `bench_log.csv`.
3. Once per stage: holdout run, `--ref holdout=$DRIVE/reference/ref_holdout.jsonl` → margin met (never used to pick parameters).
4. `tools/check_schema.py` on the output; damaged-input and resume checks for Stage 2.
5. `bench.py extrapolate` → projected 50-file and test-set time; log the results in WORKLOG.
6. `/code-review` after each stage, per the workflow.

## Amendments after independent review (2026-11-20, approved by the user)
These override the sections above where they conflict.

**User decisions**
- **FP16 is allowed** if it passes the margin. It is measured separately from sampling, mentioned in the write-up, and `--fp32` stays as an escape flag.
- **Stage 2a is threads-first, in one process:** a decode/letterbox prefetch thread (PyAV releases the GIL) + direct-tensor inference + cross-video batching. Worker processes (the old 2a) are used only if Stage 0 or 2a shows decode saturating a core. The shared-memory step (2b) applies only to processes.
- **Start the 50-file download tonight (Fri),** not Saturday morning.

**Blockers fixed in the plan**
- **B1. Partial decode failure.** The per-second buffering loop is wrapped in try/except. On a decode error the buffered second is sampled and yielded, then the error is re-raised. Unit test: a fake decoder that fails partway through a second.
- **B2. Frame counts.** `Timers` and the stats JSON get `decoded_frames` and `inferred_frames`. `bench.py` (fps_loop, `extrapolate`'s per-frame cost) uses decoded frames.

**Important changes**
- **I1. One function for grouping and selection.** It lives in `vidx/sampling.py` and both the simulator and the pipeline call it. A second closes when `int(t)` changes. Non-monotonic timestamps are handled the same way in both, with a unit test. The simulator uses the exact `t_num/t_den` from the dumps, not the float `t`. First check that `$DRIVE/frames` holds 33 npz files.
- **I3. Direct path.** Run `predict()` once on a dummy batch with the same arguments, then use `model.yolo.predictor.model` (the fused AutoBackend with the right precision). NMS arguments come from `predictor.args`, never hard-coded, and keep `classes=COCO_IDS`. Warm up each letterbox shape. `cudnn.benchmark` only if validated.
- **I4. Stage 2 gate.**
  - In FP16, ≥ 0.999 of frames must have identical counts vs Stage 1, plus the margin.
  - Bit-identical is required only for FP32 `--direct` vs `predict()`.
  - `batch_agreement --direct` also compares FP16 direct vs FP16 `predict()`.
- **I5. Holdout discipline.**
  - Accept on tune at F1 ≥ 0.985, exact ≥ 0.93, within ±1 ≥ 0.985.
  - Fallback ladder, fixed before holdout is ever looked at: chosen k → k+2 → every frame (FP16) → every frame (FP32).
  - If holdout fails, step down the ladder. Never re-tune after seeing holdout.
- **I6. Scheduling.** Longest-first only within a look-ahead window of ~64 videos; no full scan of the mp4 column. If processes are ever used, workers receive `(path, row)` and read the parquet themselves.
- **I7. Workers (processes only).** Set `YOLO_OFFLINE` / `YOLO_CONFIG_DIR` before spawn; `cv2.setNumThreads(1)`; daemon workers; watch RAM per worker (ultralytics import ≈ 0.5 GB).
- **I8. Inference failures across videos.**
  - A failed batch marks every video in it as pending.
  - "3 in a row" means 3 consecutive failed batches.
  - The pending / write-later / exit-3 semantics stay as they are.
  - A video is complete only when its end marker has arrived and none of its frames are still in buckets.

**Nits taken**
- N1: Stage 0 also records `predict()` wall time minus the sum of `r.speed` (per-call overhead).
- N2: the anchored first/last variant is dropped.
- N4: `--refine` (adaptive) only if fixed k fails the margin.
