# WORKLOG (committed to the repo: never put secrets or tokens here)

Deadline: **Sun 22 Nov 2026, 23:55 PKT**

## Next up
- **(2026-11-20) Baseline complete.** Next: plan-mode (Opus) speed-up plan. Order: frame sampling first (inference is ~98% of time), then FP16 / batch size / decode-infer overlap, TensorRT only if still GPU-bound. Also fix the notebook gaps: copy logs + stats.json to Drive; wait for the holdout run to finish before the checks / copy step. Rotation check still to confirm.
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
