"""Index videos per second with YOLO11l.

    python run.py --input <folder of .parquet files> --output results.jsonl

Heavy dependencies (torch, ultralytics, av, cv2, pyarrow) are only imported
where they are needed, so process_frames() can be tested with fakes.
"""
import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import NamedTuple

from vidx.aggregate import aggregate, error_line, make_lines

_PROCESS_START = time.perf_counter()  # approximates process start: startup_s = imports + CUDA init + model load
log = logging.getLogger("run")
MAX_INFERENCE_FAILURES = 3
# Defaults of the grader command (`run.py --input --output`). They change only after a setting passed the
# accuracy margin on tune AND holdout; the commit that produced results.jsonl.gz is tagged with them.
# Stage 1 gate passed 2026-11-20 (k=3 fp16): tune F1 0.9877 / exact 0.9586 / within1 0.9989,
# holdout F1 0.9901 / exact 0.9683 / within1 0.9990 (accept bar 0.985 / 0.93 / 0.985).
DEFAULT_SAMPLE = 3  # frames per second sent to the model; 0 = every frame (brute force)
DEFAULT_PRECISION = "fp16"
# Stage 2a gate passed 2026-11-20 (L4): direct path == predict() (1.0000, fp32 and fp16); tune 1,900 fps with
# 2 threads (1,214 with 0) at identical accuracy; holdout F1 0.9902 / exact 0.9680 / within1 0.9990.
# 2 threads: no cost on the L4 (GPU-bound) and decode headroom for a faster GPU.
DEFAULT_DECODE_THREADS = 2  # 0 = sequential decode -> predict() per video; N > 0 = vidx.pipeline (Stage 2a)


class Record(NamedTuple):
    t: object  # Fraction
    counts: list
    conf: list = None  # the fields below are only filled with keep_details (--dump-frames)
    pts: object = None
    rotation: int = 0
    shape: tuple = ()


class InferenceError(Exception):
    """The model failed on a batch (not a decode problem): the video must be retried, not recorded."""


class Timers:
    """frames = frames decoded (the unit of speed and extrapolation); inferred_frames = frames sent to
    the model. Without sampling both are equal. Fakes that bypass the decoder count inferred frames as
    decoded (see process_frames)."""

    def __init__(self):
        self.decode = self.infer = self.aggregate = 0.0
        self.videos = self.errors = 0
        self.decoded_frames = self.inferred_frames = 0

    @property
    def frames(self):
        return self.decoded_frames

    def summary(self, total):
        fps = self.frames / total if total > 0 else 0.0
        return (
            f"videos {self.videos} (errors {self.errors}), frames {self.frames} "
            f"(inferred {self.inferred_frames}) | "
            f"decode {self.decode:.1f}s, infer {self.infer:.1f}s, aggregate {self.aggregate:.1f}s, "
            f"total {total:.1f}s | {fps:.1f} frames/s overall"
        )


def _is_oom(e):
    return type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower()


def _empty_cuda_cache():
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


def run_inference(infer_fn, images):
    """infer_fn(images), splitting the batch in halves on CUDA OOM. Other failures -> InferenceError."""
    try:
        return infer_fn(images)
    except InferenceError:
        raise
    except Exception as e:
        if not _is_oom(e) or len(images) == 1:
            raise InferenceError(f"{type(e).__name__}: {e}"[:200]) from e
        log.warning("CUDA out of memory on a batch of %d: retrying in halves", len(images))
        _empty_cuda_cache()
    mid = len(images) // 2
    return run_inference(infer_fn, images[:mid]) + run_inference(infer_fn, images[mid:])


def process_frames(frames, infer_fn, batch_size, timers, keep_details=False):
    """Run frames (iterator of vidx.decode.Frame) through infer_fn in batches.

    infer_fn(list_of_images) -> list of (counts, conf). A batch is flushed when
    the frame shape changes so the letterbox stays per-frame minimal.
    Returns (records, error). On a decode failure the records of all frames
    decoded so far are still returned together with the error text.
    An inference failure raises InferenceError (see run_inference).
    keep_details=False stores only (t, counts) per frame to save memory.
    """
    records = []
    pending = []
    error = None

    def flush():
        nonlocal pending
        if not pending:
            return
        batch, pending = pending, []
        t0 = time.perf_counter()
        results = run_inference(infer_fn, [f.image for f in batch])
        timers.infer += time.perf_counter() - t0
        timers.inferred_frames += len(batch)
        for f, (counts, conf) in zip(batch, results):
            if keep_details:
                records.append(Record(f.t, counts, conf, f.pts, f.rotation, tuple(f.image.shape)))
            else:
                records.append(Record(f.t, counts))

    it = iter(frames)
    while True:
        t0 = time.perf_counter()
        try:
            frame = next(it)
        except StopIteration:
            break
        except Exception as e:  # decode failure: keep the frames we have
            error = f"{type(e).__name__}: {e}"[:200]
            break
        finally:
            timers.decode += time.perf_counter() - t0
        if pending and frame.image.shape != pending[0].image.shape:
            flush()
        pending.append(frame)
        if len(pending) >= batch_size:
            flush()
    flush()  # frames decoded before a decode error are still valid
    if not getattr(frames, "counts_decoded", False):  # plain iterators (tests, tools): decoded == inferred
        timers.decoded_frames += len(records)
    return records, error


class CountedFrames:
    """Wraps a decode generator that increments timers.decoded_frames itself (see process_frames)."""

    counts_decoded = True

    def __init__(self, gen):
        self.gen = gen

    def __iter__(self):
        return self.gen


def video_lines(video_id, records, error):
    """Records (+ optional error) -> output lines. Partial last second is kept."""
    lines = []
    if records:
        rows = aggregate((r.t, r.counts) for r in records)
        lines += make_lines(video_id, rows)
    if error or not records:
        lines.append(error_line(video_id, f"could not decode video: {error}" if error else "no frames decoded"))
    return lines


def npz_paths(directory, video_id):
    """(temp, final) npz paths. savez appends ".npz" unless the name already ends with it."""
    stem = Path(directory) / video_id.replace(":", "__")
    return Path(str(stem) + ".tmp.npz"), Path(str(stem) + ".npz")


def dump_frames(directory, video_id, records):
    import numpy as np

    Path(directory).mkdir(parents=True, exist_ok=True)
    tmp, final = npz_paths(directory, video_id)
    n = len(records)
    shape = np.array([r.shape for r in records], dtype=np.int32).reshape(n, -1)
    np.savez_compressed(
        tmp,
        t=np.array([float(r.t) for r in records], dtype=np.float64),
        t_num=np.array([r.t.numerator for r in records], dtype=np.int64),
        t_den=np.array([r.t.denominator for r in records], dtype=np.int64),
        pts=np.array([-1 if r.pts is None else r.pts for r in records], dtype=np.int64),
        shape=shape,
        rotation=np.array([r.rotation for r in records], dtype=np.int32),
        counts=np.array([r.counts for r in records], dtype=np.int32).reshape(n, 5),
        max_conf=np.array([r.conf for r in records], dtype=np.float32).reshape(n, 5),
    )
    os.replace(tmp, final)  # a half-written dump never has the final name


def process_video(video_id, data, infer_fn, args, timers):
    """Decode + infer + aggregate one video -> output lines.

    Raises only InferenceError (the caller must not record that video).
    """
    from vidx.decode import decode_video

    try:
        records, error = process_frames(
            CountedFrames(decode_video(data, sample=args.sample, counter=timers)),
            infer_fn, args.batch, timers, keep_details=bool(args.dump_frames),
        )
        t0 = time.perf_counter()
        lines = video_lines(video_id, records, error)
        timers.aggregate += time.perf_counter() - t0
    except InferenceError:
        raise
    except Exception as e:
        return [error_line(video_id, f"could not decode video: {type(e).__name__}: {e}"[:200])]
    if args.dump_frames and records:
        try:
            dump_frames(args.dump_frames, video_id, records)
        except Exception as e:
            log.warning("%s: dump-frames failed: %s: %s", video_id, type(e).__name__, e)
    return lines


def iter_videos(files, args, writer, ckpt, iter_rows):
    """(video_id, data, err) of every video still to do, in sorted file/row order. --limit-videos counts
    the first N videos including ones skipped because they are already done."""
    seen = 0
    for path in files:
        if args.limit_videos is not None and seen >= args.limit_videos:
            return
        for video_id, data, err in iter_rows(path):
            if args.limit_videos is not None and seen >= args.limit_videos:
                return
            seen += 1
            if writer.is_done(video_id):
                if ckpt:
                    ckpt.note_input(path.name)
                continue
            yield video_id, data, err


def sequential_results(videos, infer_fn, args, timers):
    """One video at a time (decode, then predict() in batches). Yields (kind, video_id, payload, batch_no):
    ("ok", id, lines, None), ("source_error", id, lines, None) or ("fail", id, message, unique_no).
    batch_no None on "ok" means "inferred after every earlier failure" (strict video order)."""
    failures = 0
    for video_id, data, err in videos:
        if err:
            yield "source_error", video_id, [error_line(video_id, err)], None
            continue
        try:
            lines = process_video(video_id, data, infer_fn, args, timers)
        except InferenceError as e:
            failures += 1
            yield "fail", video_id, str(e), failures
            continue
        yield "ok", video_id, lines, None


def pipeline_results(videos, model, args, timers):
    """Stage 2a: decode threads + cross-video batches + direct tensor inference (vidx.pipeline).
    Call only after vidx.model.prepare_direct(model) succeeded."""
    from vidx.decode import decode_video
    from vidx.model import infer_tensor, make_letterbox
    from vidx.pipeline import Pipeline

    def decode_fn(data, counter):
        return decode_video(data, sample=args.sample, counter=counter)

    pipe = Pipeline(videos, decode_fn, lambda: make_letterbox(model), lambda imgs: infer_tensor(model, imgs),
                    args.batch, args.decode_threads, timers, run_inference, video_lines)
    yield from pipe.run()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Per-second object indexing with YOLO11l")
    ap.add_argument("--input", required=True, help="folder of .parquet files (or one file)")
    ap.add_argument("--output", required=True, help="results.jsonl")
    ap.add_argument("--batch", type=int, default=32, help="inference batch size (default 32)")
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE, metavar="K",
                    help=f"frames per second sent to the model, evenly spaced; 0 = every frame (default {DEFAULT_SAMPLE})")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default=DEFAULT_PRECISION,
                    help=f"model precision on the GPU (default {DEFAULT_PRECISION})")
    ap.add_argument("--decode-threads", type=int, default=DEFAULT_DECODE_THREADS, metavar="N",
                    help="N > 0: decode/letterbox in N threads overlapped with inference, batches across videos; "
                         f"0 = one video at a time (default {DEFAULT_DECODE_THREADS})")
    ap.add_argument("--resume", action="store_true", help="continue a previous run (uses <output>.done)")
    ap.add_argument("--limit-videos", type=int, default=None, help="only the first N videos")
    ap.add_argument("--dump-frames", default=None, metavar="DIR", help="save per-frame npz per video")
    ap.add_argument("--checkpoint-dir", default=None, metavar="DIR",
                    help="also snapshot progress into this folder (e.g. a Google Drive mount)")
    ap.add_argument("--checkpoint-hf-repo", default=None, metavar="USER/REPO",
                    help="also snapshot progress into this private HF dataset repo")
    ap.add_argument("--run-name", default=None, help="checkpoint name (default: output file name)")
    ap.add_argument("--stats-json", default=None, metavar="PATH",
                    help="write timing stats (videos, frames, decode_s, infer_s, loop_s, startup_s, ...) here; off by default")
    ap.add_argument("--checkpoint-minutes", type=float, default=10, help="snapshot interval (default 10)")
    return ap.parse_args(argv)


def setup_checkpoint(args, files):
    """Only called when a checkpoint flag is set; the only place vidx.checkpoint is imported."""
    from vidx import checkpoint

    run_name = args.run_name or Path(args.output).name
    stores = []
    if args.checkpoint_dir:
        stores.append(checkpoint.DirStore(args.checkpoint_dir, run_name))
    if args.checkpoint_hf_repo:
        stores.append(checkpoint.HFStore(args.checkpoint_hf_repo, run_name))
    fp = checkpoint.make_fingerprint(Path(args.input).resolve(), args.limit_videos, args.output, run_name,
                                     config={"sample": args.sample, "precision": args.precision})
    ckpt = checkpoint.Checkpointer(
        stores, args.output, fp, args.checkpoint_minutes, args.dump_frames,
        inputs={f.name: f.stat().st_size for f in files},
    )
    if args.resume:
        log.info("checkpoint: %s", ckpt.restore())
    elif ckpt.has_existing():
        raise checkpoint.CheckpointError(
            f"a checkpoint named {run_name!r} already exists: use --resume, or a different --run-name"
        )
    return ckpt


def write_stats(path, stats, timers, exit_code, argv):
    """Write the --stats-json file. Never raises: a stats problem must not change the run's outcome."""
    try:
        d = {
            "videos": timers.videos if timers else 0,
            "errors": timers.errors if timers else 0,
            "frames": timers.frames if timers else 0,  # decoded frames
            "inferred_frames": timers.inferred_frames if timers else 0,
            "decode_s": timers.decode if timers else 0.0,
            "infer_s": timers.infer if timers else 0.0,
            "aggregate_s": timers.aggregate if timers else 0.0,
            "loop_s": stats.get("loop_s"),  # loop start -> loop end; None if the loop never started
            "startup_s": stats.get("startup_s"),
            "sample": stats.get("sample"),
            "precision": stats.get("precision"),
            "decode_threads": stats.get("decode_threads"),
            "model_speed": stats.get("model_speed"),
            "exit_code": exit_code,
            "argv": list(argv),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
    except Exception as e:
        print(f"stats-json: could not write {path}: {e}", file=sys.stderr)


def main(argv=None):
    args = parse_args(argv)
    if not args.stats_json:
        return _main(args, {})
    stats = {}
    exit_code = 1  # stays 1 if an exception escapes
    try:
        exit_code = _main(args, stats)
        return exit_code
    except SystemExit as e:
        exit_code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        raise
    finally:
        write_stats(args.stats_json, stats, stats.get("timers"), exit_code,
                    sys.argv[1:] if argv is None else argv)


def _main(args, stats):
    if args.sample < 0:
        print("--sample must be >= 0 (0 = every frame)", file=sys.stderr)
        return 2
    if args.decode_threads < 0:
        print("--decode-threads must be >= 0", file=sys.stderr)
        return 2
    if args.dump_frames and args.decode_threads > 0:
        print("--dump-frames is only supported with --decode-threads 0", file=sys.stderr)
        return 2
    if args.dump_frames and args.sample > 0:
        print("--dump-frames needs every frame (--sample 0): the sampling simulator treats dumps as every-frame",
              file=sys.stderr)
        return 2
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s %(message)s")
    from vidx.model import infer, load_model
    from vidx.output import OutputWriter, ResumeError
    from vidx.source import iter_rows, list_inputs

    files = list_inputs(args.input)
    if not files:
        print(f"no .parquet files found in {args.input}", file=sys.stderr)
        return 2
    ckpt = None
    if args.checkpoint_dir or args.checkpoint_hf_repo:
        try:
            ckpt = setup_checkpoint(args, files)
        except Exception as e:
            print(f"checkpoint: {e}", file=sys.stderr)
            return 2
    if args.stats_json or args.checkpoint_dir or args.checkpoint_hf_repo:
        # run `finally` (stats / checkpoint) on SIGTERM; never for the plain grader command
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    try:
        model = load_model(half=args.precision == "fp16")
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2
    log.info("model loaded on %s (%s, sample %s)", model.device, "fp16" if model.half else "fp32", args.sample)
    stats["sample"] = args.sample
    stats["precision"] = "fp16" if model.half else "fp32"
    stats["model_speed"] = model.speed  # filled in place by infer()

    def infer_fn(images):
        return infer(model, images)

    timers = Timers()
    stats["timers"] = timers
    try:
        from vidx.output import check_config

        # legacy (no config file) = the old defaults; with a checkpoint the fingerprint already checked it
        check_config(args.output, args.resume, {"sample": args.sample, "precision": args.precision},
                     legacy_config=None if ckpt else {"sample": 0, "precision": "fp32"})
        writer = OutputWriter(args.output, resume=args.resume)
    except ResumeError as e:
        print(str(e), file=sys.stderr)
        return 2
    if args.resume:
        log.info("resuming: %d videos already done", len(writer.done))
    start = time.perf_counter()
    stats["startup_s"] = start - _PROCESS_START
    pending = []  # (video_id, message) of videos whose inference failed: written only once we know the model still works
    failed_batches = set()  # failed batch ids since the last success ("3 in a row" counts batches)

    def record(video_id, lines):
        timers.videos += 1
        if any("error" in ln for ln in lines):
            timers.errors += 1
        writer.write_video(video_id, lines)
        if ckpt:
            ckpt.note_input(video_id.rpartition(":")[0])  # parquet file name

    def write_pending():
        for video_id, message in pending:
            record(video_id, [error_line(video_id, f"inference failed: {message}")])
        pending.clear()
        failed_batches.clear()

    videos = iter_videos(files, args, writer, ckpt, iter_rows)
    threads = args.decode_threads
    if threads > 0:
        try:
            from vidx.model import prepare_direct

            log.info("direct path: conf, iou, classes, agnostic, max_det, fp16, imgsz = %s", prepare_direct(model))
        except Exception as e:  # the faster path may cost speed, never the run: fall back to predict() per video
            log.warning("direct path unavailable (%s: %s): falling back to --decode-threads 0", type(e).__name__, e)
            threads = 0
    stats["decode_threads"] = threads
    if threads > 0:
        results = pipeline_results(videos, model, args, timers)
    else:
        results = sequential_results(videos, infer_fn, args, timers)

    try:
        for kind, video_id, payload, batch_no in results:
            if kind == "fail":
                pending.append((video_id, payload))
                failed_batches.add(batch_no)
                log.error("%s: inference failed, not recorded yet (%d failed batches in a row): %s",
                          video_id, len(failed_batches), payload)
                if len(failed_batches) >= MAX_INFERENCE_FAILURES:  # pending ones stay unwritten: --resume retries them
                    print(f"aborting: {len(failed_batches)} batches in a row failed in inference "
                          f"(last: {payload}). Restart the runtime and re-run with --resume.",
                          file=sys.stderr)
                    return 3
                continue
            # the model works again (a batch newer than every failed one succeeded), so those earlier failures
            # were video-specific. A pipeline result inferred before a failure proves nothing.
            if kind == "ok" and (batch_no is None or not failed_batches or batch_no > max(failed_batches)):
                write_pending()
            record(video_id, payload)
            if timers.videos % 10 == 0:
                log.info("%s | %s", video_id, timers.summary(time.perf_counter() - start))
            if ckpt:
                ckpt.maybe_snapshot(timers)
        write_pending()  # normal end: nothing followed them, record them as errors
    finally:
        results.close()  # stops pipeline threads on an early return / exception
        stats["loop_s"] = time.perf_counter() - start
        writer.close()
        if ckpt:
            ckpt.snapshot(timers)  # also on exit 3, an exception, KeyboardInterrupt or SIGTERM; never raises
        print(timers.summary(time.perf_counter() - start), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
