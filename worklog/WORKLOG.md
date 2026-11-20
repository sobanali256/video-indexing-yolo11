# WORKLOG (committed to the repo: never put secrets or tokens here)

Deadline: **Sun 22 Nov 2026, 23:55 PKT**

## Next up
- **(2026-11-20) Speed-up plan APPROVED:** `worklog/plans/2026-11-20-speedup-plan.md`. Margin: F1 >= 0.98, exact >= 0.92, within1 >= 0.98 on tune AND holdout. Stages: 0 profile -> 1 sampling (offline simulator on tune dumps) + FP16 -> 2a decode workers + mp queue + cross-video batching + direct-tensor infer (2b shared memory only if profiling says so) -> 3 TensorRT if time. Schedule: Sat morning re-download 50 files + offline-run test; **Sat night launch full 50-file run** with best validated stage, tag that commit (defaults = results config); Sun deliverables + user writes WRITEUP.md (2-3 h). Independent review done (2 blockers, 9 important; all folded into the plan's 'Amendments' section). User decisions: FP16 allowed if it passes the margin (note in write-up); Stage 2a threads-first in one process; start the 50-file download tonight. Next action: check `$DRIVE/frames` has 33 npz, start the download, then Stage 0 + 1a executed with Sonnet.
0. **Where we stopped (2026-11-20):** Colab (website, not the VS Code extension — it was buggy) session reached step 11 "Checks"; the cell had been running > 30 min. Collect from the user: the `bench_log.csv` "brute force baseline" row, machine check output, and whatever the checks cell printed. Then investigate the slow checks (open issue #8).
1. (done) code review of 2026-11-20 changes; commits 4009eeb, 17a55b9 (MPLBACKEND fix, Colab website), f3e3b60 (single `--include` for the 3 reference files).
2. Colab L4, run order in `colab/baseline.ipynb` / `colab/SETUP.md` (Drive = `/content/drive/MyDrive/vidx`):
   1. machine check, py3.11 venv + install + `prepare.py`, HF login, Drive mount
   2. download only the 3 reference files (00000, 00024, 00049); symlinks into `/content/subset/{,tune,holdout}`
   3. clean timing run on the tune file with `bench.py run --label "brute force baseline"` (no download, nothing else on the GPU; one uninterrupted run); output = `ref_tune.jsonl`
   4. start the 50-file background download into the same `--local-dir`
   5. tune dump run (`--dump-frames`, checkpoints on Drive) + determinism check vs `ref_tune.jsonl`
   6. holdout reference run (no dumps)
   7. checks (schema, self-scores, rotation, decoder/batch agreement), copy references to `$DRIVE/reference/`
   8. after the download: `tools/count_frames.py` on all 50 files and `--compare-dumps` on the tune file, then `bench.py extrapolate`
3. Copy `bench_log.csv` into the repo as evidence for the write-up (user writes WRITEUP.md).
4. Then speed-up planning (sampling, decode speed, batching, FP16, pipelining, maybe TensorRT): tune on `train-00000` only, validate each change on holdout with `score.py --ref tune=... --ref holdout=...`.

## Pending
- [ ] HF access + token (Colab Secret)
- [ ] Colab (website): confirm Drive mount works; then reset the runtime mid-run and resume from the checkpoint
- [ ] Colab L4 environment verified (GPU, disk, Python/CUDA versions)
- [ ] Download 50 parquet files to Colab
- [ ] Verify `requirements.txt` pins in a fresh Python 3.11 venv (PROVISIONAL: torch 2.6.0/cu126, torchvision 0.21.0, ultralytics 8.3.100, av 14.2.0, pyarrow 19.0.1, numpy 1.26.4, opencv-python 4.10.0.84)
- [ ] Run `prepare.py` on Colab (check download path logic works; weights end up at repo root `yolo11l.pt`)
- [ ] Baseline run on the subset with `--dump-frames`; record stage timings and fps; extrapolate full-run time (state honestly in the write-up that brute force ran on a subset only)
- [ ] `score.py ref ref` -> F1 1.0 / 100% / 100%
- [ ] `tools/check_schema.py` on the subset output
- [ ] Rotation: `tools/find_rotated.py`, then compare a rotated clip against `ffmpeg` autorotate; confirm the sign (display-matrix angles are counter-clockwise) and that PyAV >= 14 exposes `frame.rotation` (else the side-data / legacy-tag fallbacks are used, and they are untested)
- [ ] Decoder agreement (`tools/decoder_agreement.py`, ~20 videos); below ~0.99 -> ask the user which decoder is the reference
- [ ] Batch vs single-frame agreement >= 0.999 over 1k frames, classes filter vs none identical (`tools/batch_agreement.py`)
- [ ] Damaged input: truncated mp4 bytes and corrupted parquet each give error lines and the run continues
- [ ] Offline run (network disabled): nothing downloads, no crash
- [ ] Test `source.py` (never run locally: pip downloads were too slow; helper script idea: small synthetic parquet incl. a corrupt file and a null mp4) and then against real FineVideo parquet (row-group layout, `as_buffer()` on the mp4 column)
- [ ] Speed optimisations (iterative, each validated)
- [ ] Resumable full run on all 50 files -> results.jsonl.gz
- [ ] HARDWARE.md (Colab L4 specs + full-run time)
- [ ] README.md performance section
- [ ] WRITEUP.md: **user writes this, no AI**
- [ ] ai_chats/: copy Claude Code .jsonl, redact tokens, add web links
- [ ] Final secret scan + push before deadline

## Completed
- 2026-11-20: Stage 2a GPU checks prepared:
  - `tools/batch_agreement.py --direct --precision fp32|fp16`: predict() vs the direct path on the same frames and batches.
  - speedup.ipynb section 4: 4.1 agreement, 4.2 tune timing with threads 0/1/2 (labels include the GPU name), 4.3 holdout.
  - A setup guard on the Stage 1 and Stage 2a cells stops with "run section 0 first" after a reset.
  - The T4 came from the reconnect; the L4 is still available, so timings run on the L4.
- 2026-11-20: **Stage 2a code (off by default, `--decode-threads N`; not yet run on a GPU):**
  - `vidx/pipeline.py`: N decode threads (decode + sample + rotate + letterbox) feed a bounded queue. The main thread buckets frames by letterboxed shape across videos and runs a bucket when it is full or the queue runs dry. A video is emitted once its END has arrived and no frame of it waits in a bucket.
  - `vidx/model.py`: `prepare_direct` (one predict() call, then the predictor's own AutoBackend + args), `make_letterbox` (Ultralytics LetterBox, auto=True, the predictor's stride/imgsz), `infer_tensor` (BGR→RGB, BCHW, half/float, /255, the predictor's NMS args).
  - `run.py`: `iter_videos` / `sequential_results` / `pipeline_results`, one shared result loop (pending / 3-failed-batches / exit 3 kept).
  - **Code review #5** (7 findings, all fixed):
    - a letterbox setup error in a thread would hang the run → `make_prepare` moved inside try/finally, and the run raises once all threads have failed;
    - a stale "ok" (inferred before a failed batch) cleared pending → "ok" carries `last_batch`, and only a batch newer than every failed one clears pending;
    - an error from the shared iterator was swallowed (exit 0, videos missing) → stored and re-raised on the main thread;
    - speedup.ipynb profile cells now pass `--sample 0`;
    - resume without a checkpoint could mix settings → `<output>.config.json` (in commit c711611);
    - pipeline decode time excludes time blocked on the queue (sum over threads);
    - README documents `--decode-threads` as EXPERIMENTAL.
  - Tests: 179 pass. Stage 1 was committed separately (c711611) and its snapshot tested alone (167 pass).
  - **Next:** GPU check of the direct path vs predict() (FP32 bit-identical expected, FP16 >= 0.999), then a timing run with `--decode-threads 1/2` on tune, then the margin check on tune + holdout.
- 2026-11-20: **Stage 1 gate PASSED (k=3 + FP16)**. Colab gave a **Tesla T4** after the reset (not the L4), so these speeds are not comparable with the L4 baseline.
  - k=3 FP32 tune (pipeline vs simulator check): F1 0.9879 / exact 0.9586 / within1 0.9989 vs simulator 0.9880 / 0.9584 / 0.9988. Off by about 1–2 seconds out of 9,144, not bit-identical. Consistent with batch composition / GPU (T4 vs the L4 reference) flipping conf ~0.4 detections (batch_agreement already showed < 0.1% such frames). Not a sampling bug: the inferred frames (27,431 = 0.108) match the simulator's fraction exactly. clean=no (download running).
  - **k=3 FP16 tune: F1 0.9877, exact 0.9586, within1 0.9989** → accepted (bar 0.985 / 0.93 / 0.985). T4: 544.9 decoded fps; decode 201 s, infer 261 s of 466 s; per image inf 7.04 ms, pre 1.52, post 0.80.
  - **k=3 FP16 holdout: F1 0.9901, exact 0.9683, within1 0.9990** → accepted. T4: 564.5 fps (486,831 frames, 862 s).
  - Speed on T4: brute force on the T4 is ~20 ms/frame FP32 (from the k3 fp32 row: inf 17.2 + pre 1.8 + post 1.1) ≈ 48 fps, so k=3 FP16 ≈ **11x on the same GPU**. Decode is now ~43% of the loop on the T4 runtime (few CPU cores), so Stage 2a overlap matters.
  - **Defaults set:** `DEFAULT_SAMPLE = 3`, `DEFAULT_PRECISION = "fp16"` (README updated). Brute-force reference runs in baseline.ipynb / SETUP.md now pass `--sample 0 --precision fp32` explicitly.
- 2026-11-20: **Code review #4 (Stage 0/1, commit 4626e7b)**, 6 findings, all fixed (164 tests pass):
  - (important) Unbounded per-second buffer on broken timestamps → `MAX_GROUP = 240` frames per group in `vidx.sampling` (shared by simulator and pipeline; no effect on normal videos up to 240 fps).
  - (important) The fingerprint config broke resume of older checkpoints → config is left out for the baseline (sample 0, fp32); the error message names --sample / --precision.
  - (nit) `inferred_frames` is counted per successful batch.
  - (nit) ProfileNoteTest moved above the main guard.
  - (nit) `--sample < 0` and `--dump-frames` with `--sample > 0` are rejected (exit 2).
  - (nit) The simulator docstring states that partial-failure videos are not simulated.
- 2026-11-20: **Stage 0 profile (L4, 5 tune videos, 43,431 frames, clean):**
  - Every frame, FP32: 99.3 fps. Per image: pre 1.58 ms (CPU letterbox), inference 7.32, post 0.67; decode 0.375 ms/frame.
  - Every frame, **FP16: 155.3 fps** (inference 3.76 ms, 1.95x faster). FP16 alone vs reference: **F1 0.9995, exact 0.9981, within1 1.0000** (negligible loss).
  - Decode only (`profile_decode`, 12 CPUs, FFmpeg auto threads): 5,304 fps; + BGR every frame 3,138 fps; + BGR for 5 frames/s 4,883 fps. Single thread 1,376 fps, so auto threading matters.
  - Conclusion: even with sampling, the GPU stays the bottleneck on the L4. Model cost per inferred image ≈ 5.9 ms FP16 (of which ~1.5 ms is CPU letterbox) vs decode ≈ 0.2 ms per decoded frame.
  - Sequential estimate per decoded frame: k=5 ≈ 1.27 ms (~4.6 h for 50 files); k=3 ≈ 0.85 ms (~3.1 h).
  - Stage 2a (decode + letterbox in a prefetch thread, direct tensors) should give roughly another 1.5x.
  - Bench note: `--only-common` gives PARTIAL, which bench logs as `fail_accuracy`; expected for profile runs.
  - **Decision (by plan rule: cheapest accepted setting):** k=3 + FP16. Ladder fixed: k=3 → k=5 → every frame FP16 → every frame FP32.
- 2026-11-20: **Sampling simulator on tune** (33 videos, 253,918 frames, 9,144 s; k=0 sanity check 1.0/1.0/1.0). Columns: inferred fraction / F1 / exact / within1:
  - k=1 0.036 / 0.9775 / 0.9301 / 0.9913 (not accepted)
  - k=2 0.072 / 0.9657 / 0.9190 / 0.9922 (not accepted)
  - **k=3 0.108 (9.3x) / 0.9880 / 0.9584 / 0.9988 (accepted)**
  - k=4 0.144 / 0.9825 / 0.9475 / 0.9968 (not accepted)
  - **k=5 0.180 (5.6x) / 0.9911 / 0.9705 / 0.9992 (accepted)**
  - k=6 0.216 / 0.9881 / 0.9630 / 0.9983 (accepted)
  - k=7 0.252 (4.0x) / 0.9929 / 0.9738 / 0.9993 (accepted)
  - k=9 0.324 / 0.9947 / 0.9776 / 0.9999 (accepted)
  - adaptive 3:7 0.158 / 0.9929 / 0.9751 / 0.9996 (accepted)
  - **adaptive 3:9 0.158 (6.3x) / 0.9947 / 0.9777 / 0.9999 (accepted)**
  - adaptive 5:9 0.254 / 0.9943 / 0.9795 / 0.9998 (accepted)

  Odd k clearly beats even k (vote ties / median of an even count). Every strategy passes the grader thresholds, even k=1. The weak videos are the same across strategies: people exact in videos 13, 5, 30; F1 in 15 and 20. k=3 has the thinnest F1 headroom (0.988 vs the 0.985 acceptance). Choice between k=3 and k=5 waits for the Stage 0 profile: if decode dominates after sampling, k=5 costs little extra time. Adaptive (N4) stays dropped unless fixed k fails.
- 2026-11-20: **Stage 0 + Stage 1 code (not yet run on Colab):**
  - `vidx/sampling.py`: `pick(n, k)` (evenly spaced centres; all frames if n <= k or k <= 0), `iter_groups` (a group closes when int(t) changes; on a decode error the buffered group is yielded, then the error re-raised = blocker B1), `sample_groups`.
  - `vidx/decode.py`: `decode_video(data, sample=0, counter=None)` decodes and stamps every frame, converts only the sampled ones, counts decoded frames.
  - `vidx/model.py`: `load_model(half=)` (FP16 only on CUDA); `infer` sums Ultralytics `r.speed` and predict() wall time into `model.speed`.
  - `run.py`: `--sample K` (default `DEFAULT_SAMPLE=0`) and `--precision fp16|fp32` (default `DEFAULT_PRECISION="fp32"`); defaults change only after the tune+holdout gate. `Timers.frames` = decoded frames, plus `inferred_frames` (blocker B2). Stats JSON adds inferred_frames, sample, precision, model_speed. The checkpoint fingerprint includes sample + precision.
  - `bench.py`: these go into the notes column (no new CSV columns).
  - New tools: `tools/simulate_sampling.py` (offline, tune dumps + score.py, accept 0.985/0.93/0.985, k=0 sanity check must be 1.0), `tools/profile_decode.py`.
  - Tests: 160 pass (new test_sampling, test_simulate_sampling, ProfileNoteTest).
- 2026-11-20: **Brute-force baseline projection (Colab L4):** `bench.py extrapolate` -> projected **124,936 s (34.70 h)** for **12,981,615 frames** (all 50 files). Derived: ~9.6 ms/frame (~104 fps end-to-end); dataset averages ~28.8 fps of video (12.98 M frames / ~125 h). Unseen ~13 h test set ≈ 1.35 M frames ≈ ~3.6 h brute force on L4 (under the 6 h cap, but this is the number to beat). Still to collect: the tune row's decode_s / infer_s / aggregate_s split (decides whether decode or GPU is the bottleneck) and the checks output.
- 2026-11-20: **Tune measured row** (bench_log.csv, clean=yes, L4): 33 videos, 253,918 frames, wall 2447.3 s, startup 2.7 s, loop 2443.7 s, 103.9 fps. CSV has no decode/infer split: it is in `/content/ref/ref_tune.jsonl.stats.json` (not collected yet).
- 2026-11-20: **Step 11 checks:** tune schema OK (33 videos, 9,144 seconds, 0 errors); tune self-score 1.0/1.0/1.0; decoder agreement (PyAV vs cv2, 20 videos) micro F1 1.0000; batch vs single identical counts 1.0000 (max conf gap 0.00173); classes filter vs none identical. **Holdout reference is EMPTY** (check_schema: 0 videos), see issue #10. Rotation (`find_rotated`) and determinism (tune vs tune_dump) outputs were not in the paste.
- 2026-11-20: Reference-set change (plan "Tune/holdout reference sets, clean timing, frame count and extrapolation"). Decision: the brute-force reference is now 3 parquet files spread over the dataset; **tune = train-00000** (all tuning), **holdout = train-00024 + train-00049** (scored only, never tuned on; rule added to CLAUDE.md). Brute force is not run on all 50 files (projected 25+ h): a clean timing on the tune file + a model-free frame count of all 50 files + extrapolation instead. Code: `run.py --stats-json PATH` (opt-in, written in `finally`: videos, errors, frames, decode_s, infer_s, aggregate_s, loop_s, startup_s, exit_code, argv; grader command unchanged); `score.py` multi-reference (`--ref NAME=PATH`, per-reference file set from second + error lines, candidate filtered to it, empty reference = FAIL, `--only-common` = PARTIAL, `--json`; positional form kept); new `bench.py` (`run` = timed subprocess + scoring + CSV row, clean detection via /proc download scan and nvidia-smi foreign GPU pids every 30 s; `extrapolate` = fixed + frames x per-frame from the newest clean ok row); new `tools/count_frames.py` (PyAV demux packet count, CSV, totals, `--compare-dumps`); `colab/baseline.ipynb` + `colab/SETUP.md` rewritten for the new run order. Tests: all pass (new `test_score_multi`, `test_bench`, `test_count_frames`, `StatsJsonTest` in `test_run`). Not run on Colab yet; `/code-review` pending.
- 2026-11-19: Durable checkpoints for long Colab runs (new `vidx/checkpoint.py`, imported only when `--checkpoint-dir` / `--checkpoint-hf-repo` is given; plain `run.py --input --output` unchanged). Generation folders `gen-NNNN/` (gz output, done.txt, progress.json, manifest.json last with sizes + sha256 + run fingerprint), newest 2 kept, snapshot every `--checkpoint-minutes` (10) at a video boundary plus a final one in `finally` (SIGTERM mapped to exit); every snapshot failure only warns. `--resume` restores the newest valid generation if ahead of the local files; a fresh run over an existing checkpoint exits 2. Sidecar is now `video_id<TAB>line_count` (plain ids still read) and `_recover` redoes a video whose lines are incomplete. `--dump-frames` writes `<name>.tmp.npz` then `os.replace`. `huggingface_hub==0.29.3` pinned (provisional). Tests: `tests/test_checkpoint.py`, extended `test_output.py` / `test_run.py` (smoke test uses fake `iter_rows` + decode because `.venv` has no pyarrow). Not yet run on Colab; `/code-review` pending.
- 2026-11-19: Housekeeping: worklog moved to `worklog/WORKLOG.md`, removed from `.gitignore`, CLAUDE.md paths and wording updated (worklog is now committed).
- 2026-11-19: Wrote the baseline pipeline per plan: `run.py`, `prepare.py`, `score.py`, `vidx/{source,decode,model,aggregate,output}.py`, `tools/{find_rotated,decoder_agreement,batch_agreement,check_schema}.py`, `tests/` (CPU, stdlib unittest), `requirements.txt` (provisional pins), `colab/SETUP.md`, README install/prepare/run.
- 2026-11-19: Local checks: `python -m unittest discover -s tests -v` passes (42 tests), every .py compiles.
- 2026-11-19: Added TensorRT-deferred backend design to CLAUDE.md; first commit made (brief PDF stays out of repo).
- 2026-11-19: Read brief. Git repo initialised in D:\video-indexing-yolo11, tracking origin/main (remote had LICENSE only). Created .gitignore, CLAUDE.md, README.md stub, ai_chats/links.txt, this WORKLOG.

## Open issues
- #2 (2026-11-19) Dev machine is Python 3.14; grader is 3.11. Pins are provisional until verified in a 3.11 venv on Colab.
- #5 (2026-11-19) `ultralytics` depends on `opencv-python` (non-headless), which may be installed next to `opencv-python-headless` and needs `libgl1`. README/SETUP list `apt-get install libgl1 libglib2.0-0`. Revisit if the grader VM lacks apt access.
- #6 (2026-11-19) Assumptions made while coding (review these):
  - Parquet that cannot be opened -> one error line `<file>:0`. A file that opens but has an unreadable row group -> one error line per remaining row of that group (row counts come from the footer).
  - Error text format: `could not decode video: <ExceptionType: message>` (truncated to 200 chars); zero decoded frames without an exception -> `no frames decoded`.
  - `--limit-videos N` counts the first N videos in sorted file/row order, including ones skipped by `--resume`.
  - The inference batch size is the whole `predict()` list; relies on Ultralytics treating a list of ndarrays as one batch.
  - `prepare.py` downloads with `ultralytics.utils.downloads.attempt_download_asset(<abs path>)`, falling back to `YOLO("yolo11l.pt")` in the repo dir. Untested.
  - Offline settings: `YOLO_OFFLINE=1`, `settings.update(sync=False)`, `YOLO_CONFIG_DIR` in the temp dir (set with `setdefault`). Untested; confirm with the offline check.
  - `io.BytesIO(memoryview(buffer))` copies the mp4 bytes once per video (cheap relative to decode); `as_buffer()` itself does not copy.
  - Stream-level rotation side data via `stream.side_data["DISPLAYMATRIX"]` is a guess about the PyAV API; it is wrapped in try/except and only a fallback after `frame.rotation`.
  - `--resume` fsync is not done (flush only); a VM crash could lose the last video, which is simply redone because the sidecar is written after the output.
- #4 (2026-11-19) RESOLVED: worklog is now committed in `worklog/`.
- #7 (2026-11-19) Checkpoint assumptions (review, untested on Colab / the real HF Hub):
  - `huggingface_hub==0.29.3` is believed to have `HfApi.upload_folder`, `delete_folder`, `list_repo_files`, `hf_hub_download(local_dir=)`; only a fake API object was tested. The repo must exist already (not auto-created).
  - `--checkpoint-dir` stores under `DIR/<run-name>/gen-NNNN` (the plan said straight into DIR) so `--run-name` and the fresh-run guard work for folders too.
  - HF `--dump-frames` files go to `<run-name>/frames/` in the same commit as the generation (never pruned), so old generations can be deleted without losing frames. Dump files are not restored automatically.
  - Restore compares done-video counts: local wins ties; local counts only if output and sidecar both exist. A restore error (network, corrupt store) exits 2 rather than silently starting over.
  - A checkpoint with a different fingerprint (input path, `--limit-videos`, output name, run name; superseded input-size rule: see Code review #2) is refused (exit 2) even if older than local files.
  - The final snapshot always makes a new generation, even if nothing changed. Time lost after a crashed session's last snapshot is not in `compute_s`.
  - A half-written dump is only protected from partial reads, not cleaned up at start; `*.tmp.npz` files are ignored by the HF uploader.
  - SIGTERM handler is `sys.exit(143)`, installed only with a checkpoint flag. Windows `.venv` runs Python 3.14, so the tests have not run on 3.11.

- #8 (2026-11-20) Colab step 11 "Checks" ran > 30 min. Likely cause (unverified): `tools/decoder_agreement.py --n 20` runs YOLO on **every frame of 20 whole videos twice** (PyAV and cv2), i.e. roughly a brute-force run on ~40 video-passes; `batch_agreement` (1000 frames × 3 passes) is small. Also a `%%bash` cell shows output only when it ends, so progress is invisible. To do: confirm with `ps`/`nvidia-smi`, cap decoder_agreement to N seconds or frames per video, run each check in its own cell / with progress output.
- #9 (2026-11-20) Colab facts learned: Colab kernel Python is 3.13 (venv is uv-managed 3.11.18); notebook kernel exports `MPLBACKEND=module://matplotlib_inline...` which breaks Ultralytics in subprocesses (fixed in `vidx/__init__.py`); repeated `hf download --include` flags appeared to keep only the last pattern (fixed: one `--include` with 3 patterns; cause not verified with `ls`). requirements.txt pins installed cleanly on py3.11 (54 packages, torch 2.6.0+cu126) — pins verified for install; runtime not yet verified.

- #10 (2026-11-20) `ref_holdout.jsonl` had 0 videos when step 11 ran. Step 10 starts the holdout run with `nohup ... &`, so the likely cause (unverified) is that it was still running or had crashed when the checks started. If it was still running, it shared the GPU with decoder_agreement, which may also explain #8. To do: `tail /content/holdout.log`, `nvidia-smi`, `wc -l /content/ref/ref_holdout.jsonl`; once it is complete, rerun the holdout schema and self-score checks. Notebook fix to consider: a wait/assert cell before step 11.
  - Update 2026-11-20: a later cell found `/content/holdout.log`, `/content/ref/*`, `/content/subset/holdout` and `ref_tune.jsonl.stats.json` all missing, and no GPU processes. So the **Colab runtime had been reset** (/content wiped), which means the holdout log is lost. When step 11 ran, the holdout file existed but was empty (check_schema opens it without a guard). That fits a run that had just started or failed early. Evidence still available: the `$DRIVE/ckpt/ref_holdout` checkpoint and `$DRIVE/reference`. Also lost: the tune decode/infer split. Notebook gap: stats.json and the logs are not copied to Drive.
  - Update 2026-11-20 (Drive listing): the holdout run **completed**. Final checkpoint `ckpt/ref_holdout/gen-0014` (00:48 UTC): 66 videos, 486,831 frames, 0 errors, decode 180 s, infer 9,617 s, aggregate 0.6 s. But `$DRIVE/reference/ref_holdout.jsonl` was copied at 00:24, **before the run ended**, so it is a partial file. Fix: restore from `gen-0014/results.jsonl.gz`, then check schema and self-score (expect 66 videos). The holdout run started 22:05, so it shared the GPU with the tune dump run and the checks. Its timing (~51 fps) is therefore not clean, but the split is still telling: **inference is ~98% of the time, decode ~2%**, so the GPU is the bottleneck.
  - **RESOLVED 2026-11-20:** holdout restored from `gen-0014`: 17,586 lines (the partial Drive copy had 11,568), schema OK with 66 videos and 0 errors, self-score 1.0/1.0/1.0. Determinism: tune vs tune_dump scores 1.0/1.0/1.0 over 9,144 seconds. Drive copy `$DRIVE/reference/ref_holdout.jsonl` overwritten with the restored file (17,586 lines, confirmed). **Baseline complete** (rotation check output still not seen).

## Deferred
- #3 (2026-11-19) **Deferred, design noted.** TensorRT: L4 dev vs RTX 4090 grader, engine must be built in grader prep step. Rules recorded in CLAUDE.md "Inference backend". Baseline first with PyTorch .pt; model behind load_model()/infer(batch).

## Closed issues
- 2026-11-19 **Code review #1** (10 findings, all closed; 55 tests pass):
  - R1 Inference failures were handled like decode failures. Fix: `run_inference` splits a batch in halves on CUDA OOM; other model errors raise `InferenceError`, the video is not written or marked done, and 3 in a row abort the run (exit 3, retry with `--resume`).
  - R2 `YOLO_OFFLINE` was set to "1". Fix: now "True" (Ultralytics compares `value.lower() == "true"`).
  - R3 A `--dump-frames` failure discarded the video's lines. Fix: the dump has its own try/except and only logs a warning.
  - R4 `opencv-python-headless` clashed with the `opencv-python` that ultralytics pulls in. Fix: pinned `opencv-python==4.10.0.84` (numpy<2 compatible); SETUP.md wording updated. Still PROVISIONAL until verified (also supersedes #5's duplicate-wheel concern).
  - R5 Leading frames without pts then a pts reset time to 0. Fix: `first_pts` is anchored so the first pts frame continues the estimated timeline.
  - R6 No-rate fallback used absolute index/30. Fix: `last_t + step` (1/rate, else median of previous positive gaps, else 1/30; first frame 0).
  - R7 `--resume` with an output but no `.done` sidecar silently wiped the output. Fix: `ResumeError` with a clear message (main exits 2).
  - R8 All frame details were kept in memory. Fix: `process_frames(..., keep_details=)`; full records only with `--dump-frames`.
  - R9 Two GPU transfers per result in `infer`. Fix: one `boxes.data.cpu()`, conf = column 4, cls = column 5.
  - R10 `tools/batch_agreement.py` re-implemented batching. Fix: it calls `run.process_frames` with batch 1 vs `--batch`, plus a `classes=None` wrapper.
- 2026-11-19 **Code review #2 (checkpoints)** (10 findings closed; 95 tests pass):
  - C1 Fingerprint covered every parquet, so a changed download set blocked resume. Fix: fingerprint = resolved input path + `--limit-videos` + output name + run name; the manifest records name+size of each parquet that had a video processed, and resume refuses only if one of those is missing or changed size (new files are fine).
  - C2 One `InferenceError` video was silently dropped with exit 0. Fix: failed ids wait in a pending list; the next successful video first writes them as `inference failed: ...` error lines (marked done); 3 in a row abort with exit 3 without writing them (`--resume` retries); leftovers at the normal end are written as error lines.
  - C3 A store listing failure counted as "no checkpoint". Fix: `Checkpointer.startup()` lists every store once and raises `CheckpointError` (exit 2) if any fails, for fresh and resume runs alike.
  - C4 `restore()` died on one bad generation. Fix: fetch/validate errors per generation are logged and the next older generation / other store is tried.
  - C5 A missing Drive mount would be recreated as a local folder. Fix: `DirStore` requires the checkpoint dir's parent to exist ("checkpoint dir parent missing — is Google Drive mounted?"), at listing and at publish; only the dir and `<run-name>` are ever created.
  - C6 One generation number for all stores, and a prune error counted as a failed publish. Fix: number computed per store inside its own try/except; prune failure only warns, the publish still counts and dump frames are marked sent.
  - C7 A sidecar entry with no output lines (legacy, or count 0) counted as done. Fix: done needs >= 1 complete line (exactly the recorded count if known); `write_video` refuses an empty list, so a count of 0 is never written.
  - C8 `huggingface_hub==0.29.3` has no `hf` CLI. Fix: pinned `huggingface_hub==0.34.4` (still PROVISIONAL, unverified on Colab; `HfApi` methods used by `HFStore` are only tested against a fake).
  - C9 `prune()` kept the newest 2 numbers even if invalid. Fix: `prune_store` keeps the newest 2 valid generations (manifest + sizes; HF: manifest present in the listing) and deletes invalid/partial and older ones.
  - C10 (**partial**) Re-listing stores every snapshot and re-reading files for sha256. Fix: generation numbers are listed once at startup and kept in memory per store; size and sha256 are computed while writing. Deliberately NOT done: background upload thread.
- #1 (2026-11-19) The worklog used to be private; it is now committed in `worklog/`, so chat logs and the repo agree.
- #4 (2026-11-19) Decision made: commit the worklog in `worklog/`.

## Session log
- 2026-11-19: Session "Alethia": repo setup.
- 2026-11-19: Execution session (Sonnet): housekeeping + local code per the approved plan (steps 1-2). Colab steps not started.
- 2026-11-20: Code review #3 (bench/score/count_frames), fixes:
  - bench: a scoring exception no longer prevents the CSV row (row logged with blank scores and the error, truncated, in notes; exit 1).
  - bench: GPU-busy detection now scans `/proc/<pid>/fd` for open `/dev/nvidia*` fds outside bench's own process tree (reason `other GPU process pid=.. cmd=..`); no /proc -> clean "no", unknown. The nvidia-smi compute-apps check is gone (nvidia-smi is only used for the GPU name). The process tree is computed once per check and passed to both scanners.
  - bench: a failed or PARTIAL `--ref` set gives status `fail_accuracy` and a non-zero exit (`failed` stays for a non-zero run.py exit); `extrapolate` only uses `status=ok` rows.
  - colab notebook + SETUP.md: reset instructions now also say to re-download the 3 reference files, recreate the symlinks and restart the 50-file download if unfinished; the symlink cell uses `shopt -s failglob` plus an existence check so an unmatched glob fails loudly.
  - tools/count_frames.py: per-file error handling (an error row per failing file, also for metadata access), BrokenProcessPool retries the unfinished files serially, the CSV is always written, error count printed.
  - run.py: the SIGTERM->SystemExit handler is installed when `--stats-json` or a checkpoint flag is set (never for the plain grader command).
  - score.py: the two-file positional mode no longer filters the candidate (extra candidate seconds are FP again); filtering stays in `--ref` mode with a loud WARNING for ignored candidate videos and an explicit message when the candidate shares no parquet file with a reference. `--only-common` with no overlapping videos reports "no overlapping videos between candidate and reference" (FAIL, not empty_reference).
  - shared helpers: `vidx.source.list_inputs` (used by run.py and count_frames) and public `vidx.source.short_error` (was `_short`).
  - tests added/adjusted in test_bench, test_score_multi, test_count_frames, test_run.
