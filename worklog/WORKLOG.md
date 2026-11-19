# WORKLOG (committed to the repo: never put secrets or tokens here)

Deadline: **Sun 22 Nov 2026, 23:55 PKT**

## Next up
1. User: review the code written on 2026-11-19, then `/code-review` it and fix findings.
2. User: push the setup commit `c63d7cd` (it has not reached GitHub; push from a terminal where the credential prompt works).
3. Colab L4 (see `colab/SETUP.md`): py3.11 venv, `pip install -r requirements.txt` (pins are provisional), `python prepare.py`, HF login via Colab Secret, download 2 parquet files, then the full 50 in the background.
4. Colab: baseline on the subset with `--dump-frames` -> `reference/ref_subset.jsonl`; record fps and stage timings; run the checks listed under Pending.
5. Then speed-up planning (sampling, decode speed, batching, FP16, pipelining, maybe TensorRT), each validated with `score.py` against the reference.

## Pending
- [ ] HF access + token (Colab Secret)
- [ ] Colab: confirm Drive mount via VS Code extension, else create private HF dataset repo and use `--checkpoint-hf-repo`; then reset the runtime mid-subset-run and resume from the checkpoint
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
