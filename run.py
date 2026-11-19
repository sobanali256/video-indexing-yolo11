"""Index videos per second with YOLO11l.

    python run.py --input <folder of .parquet files> --output results.jsonl

Heavy dependencies (torch, ultralytics, av, cv2, pyarrow) are only imported
where they are needed, so process_frames() can be tested with fakes.
"""
import argparse
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import NamedTuple

from vidx.aggregate import aggregate, error_line, make_lines

log = logging.getLogger("run")
MAX_INFERENCE_FAILURES = 3


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
    def __init__(self):
        self.decode = self.infer = self.aggregate = 0.0
        self.frames = self.videos = self.errors = 0

    def summary(self, total):
        fps = self.frames / total if total > 0 else 0.0
        return (
            f"videos {self.videos} (errors {self.errors}), frames {self.frames} | "
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
    timers.frames += len(records)
    return records, error


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
            decode_video(data), infer_fn, args.batch, timers, keep_details=bool(args.dump_frames)
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


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Per-second object indexing with YOLO11l")
    ap.add_argument("--input", required=True, help="folder of .parquet files (or one file)")
    ap.add_argument("--output", required=True, help="results.jsonl")
    ap.add_argument("--batch", type=int, default=32, help="inference batch size (default 32)")
    ap.add_argument("--resume", action="store_true", help="continue a previous run (uses <output>.done)")
    ap.add_argument("--limit-videos", type=int, default=None, help="only the first N videos")
    ap.add_argument("--dump-frames", default=None, metavar="DIR", help="save per-frame npz per video")
    ap.add_argument("--checkpoint-dir", default=None, metavar="DIR",
                    help="also snapshot progress into this folder (e.g. a Google Drive mount)")
    ap.add_argument("--checkpoint-hf-repo", default=None, metavar="USER/REPO",
                    help="also snapshot progress into this private HF dataset repo")
    ap.add_argument("--run-name", default=None, help="checkpoint name (default: output file name)")
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
    fp = checkpoint.make_fingerprint(Path(args.input).resolve(), args.limit_videos, args.output, run_name)
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


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(name)s %(message)s")
    from vidx.model import infer, load_model
    from vidx.output import OutputWriter, ResumeError
    from vidx.source import iter_rows

    src = Path(args.input)
    files = [src] if src.is_file() else sorted(src.glob("*.parquet"))
    if not files:
        print(f"no .parquet files found in {src}", file=sys.stderr)
        return 2
    ckpt = None
    if args.checkpoint_dir or args.checkpoint_hf_repo:
        try:
            ckpt = setup_checkpoint(args, files)
        except Exception as e:
            print(f"checkpoint: {e}", file=sys.stderr)
            return 2
        signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))  # run `finally` on SIGTERM
    try:
        model = load_model()
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2
    log.info("model loaded on %s", model.device)

    def infer_fn(images):
        return infer(model, images)

    timers = Timers()
    try:
        writer = OutputWriter(args.output, resume=args.resume)
    except ResumeError as e:
        print(str(e), file=sys.stderr)
        return 2
    if args.resume:
        log.info("resuming: %d videos already done", len(writer.done))
    start = time.perf_counter()
    seen = 0
    pending = []  # (video_id, message) of videos whose inference failed: written only once we know the model still works

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

    try:
        for path in files:
            if args.limit_videos is not None and seen >= args.limit_videos:
                break
            for video_id, data, err in iter_rows(path):
                if args.limit_videos is not None and seen >= args.limit_videos:
                    break
                seen += 1
                if writer.is_done(video_id):
                    if ckpt:
                        ckpt.note_input(path.name)
                    continue
                if err:
                    lines = [error_line(video_id, err)]
                else:
                    try:
                        lines = process_video(video_id, data, infer_fn, args, timers)
                    except InferenceError as e:
                        pending.append((video_id, str(e)))
                        log.error("%s: inference failed, not recorded yet (%d in a row): %s",
                                  video_id, len(pending), e)
                        if len(pending) >= MAX_INFERENCE_FAILURES:  # pending ones stay unwritten: --resume retries them
                            print(f"aborting: {len(pending)} videos in a row failed in inference "
                                  f"(last: {e}). Restart the runtime and re-run with --resume.",
                                  file=sys.stderr)
                            return 3
                        continue
                    write_pending()  # the model works again, so those earlier failures were video-specific
                record(video_id, lines)
                if timers.videos % 10 == 0:
                    log.info("%s | %s", video_id, timers.summary(time.perf_counter() - start))
                if ckpt:
                    ckpt.maybe_snapshot(timers)
        write_pending()  # normal end: nothing followed them, record them as errors
    finally:
        writer.close()
        if ckpt:
            ckpt.snapshot(timers)  # also on exit 3, an exception, KeyboardInterrupt or SIGTERM; never raises
        print(timers.summary(time.perf_counter() - start), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
