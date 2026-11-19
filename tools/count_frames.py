"""Model-free frame count: demux every video (no decoding) and count its video packets.

    python tools/count_frames.py --input /content/finevideo/data --out frame_counts.csv [--workers 8]
        [--compare-dumps DIR]

Counts packets on the stream chosen by vidx.decode.pick_video_stream, skipping packets with size 0 (flush)
or is_discard. CSV columns: file,row,video_id,frames,duration_s,width,height,codec,error.
--compare-dumps compares each count with len(t) of the video's dump npz (written by run.py --dump-frames).
PyAV is imported only when a video is really demuxed.
"""
import argparse
import csv
import io
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vidx.source import list_inputs, short_error  # noqa: E402

COLUMNS = ["file", "row", "video_id", "frames", "duration_s", "width", "height", "codec", "error"]


def count_packets(packets):
    """Pure counting. packets: iterable of objects with .size and .is_discard.
    Returns (frames, error); on an exception while iterating the count so far is kept."""
    n = 0
    try:
        for p in packets:
            if p.size == 0 or getattr(p, "is_discard", False):
                continue
            n += 1
    except Exception as e:
        return n, short_error(e)
    return n, None


def demux_video(data):
    """Real PyAV demux of one mp4 buffer -> dict(frames, duration_s, width, height, codec, error)."""
    import av

    from vidx.decode import pick_video_stream

    container = av.open(io.BytesIO(memoryview(data)))
    try:
        stream = pick_video_stream(container)
        if stream is None:
            raise ValueError("no video stream")
        frames, error = count_packets(container.demux(stream))
        duration = None
        if stream.duration is not None and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        elif container.duration:
            duration = container.duration / 1_000_000
        ctx = stream.codec_context
        return {"frames": frames, "duration_s": duration, "width": ctx.width, "height": ctx.height,
                "codec": ctx.name, "error": error}
    finally:
        container.close()


def summarize_video(video_id, data, err, demux_fn=demux_video):
    """One CSV row (dict) for a video; any failure goes into the error column."""
    file, _, row = video_id.rpartition(":")
    out = {"file": file, "row": row, "video_id": video_id, "frames": None, "duration_s": None,
           "width": None, "height": None, "codec": None, "error": None}
    if err:
        out["error"] = err
        return out
    try:
        out.update(demux_fn(data))
    except Exception as e:
        out["error"] = short_error(e)
    return out


def process_file(path, demux_fn=demux_video, rows_fn=None):
    if rows_fn is None:
        from vidx.source import iter_rows as rows_fn
    return [summarize_video(vid, data, err, demux_fn) for vid, data, err in rows_fn(Path(path))]


def error_row(path, e):
    """One CSV row recording that a whole file could not be processed."""
    name = Path(path).name
    return {"file": name, "row": "", "video_id": f"{name}:0", "frames": None, "duration_s": None,
            "width": None, "height": None, "codec": None, "error": f"file failed: {short_error(e)}"}


def safe_process_file(path, process_fn=process_file):
    """process_file, but any exception (metadata access, rows iteration) becomes one error row."""
    try:
        return process_fn(path)
    except Exception as e:
        return [error_row(path, e)]


def _worker(path):
    return safe_process_file(path)


def process_all(files, workers, worker_fn=_worker):
    """Rows of all files, in file order. A crashed pool re-runs the unfinished files serially."""
    rows = []
    if workers <= 1:
        for f in files:
            rows += worker_fn(f)
        return rows
    done = {}
    try:
        with ProcessPoolExecutor(workers) as ex:
            futures = [(f, ex.submit(worker_fn, f)) for f in files]
            for f, fut in futures:
                try:
                    done[f] = fut.result()
                except BrokenProcessPool:
                    raise
                except Exception as e:
                    done[f] = [error_row(f, e)]
    except BrokenProcessPool as e:
        print(f"worker pool broke ({e}); retrying {len(files) - len(done)} files serially", file=sys.stderr)
    for f in files:
        if f not in done:
            try:
                done[f] = worker_fn(f)
            except BaseException as e:  # noqa: BLE001 - never lose the CSV
                done[f] = [error_row(f, e)]
        rows += done[f]
    return rows


def write_csv(rows, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in COLUMNS})


def totals(rows):
    """videos, frames, hours and the share of frames per resolution."""
    frames = sum(r["frames"] or 0 for r in rows)
    hours = sum(r["duration_s"] or 0 for r in rows) / 3600
    by_res = defaultdict(int)
    for r in rows:
        if r["frames"] and r["width"] and r["height"]:
            by_res[f"{r['width']}x{r['height']}"] += r["frames"]
    share = {k: v / frames for k, v in sorted(by_res.items(), key=lambda kv: -kv[1])} if frames else {}
    return {"videos": len(rows), "errors": sum(1 for r in rows if r["error"]), "frames": frames,
            "hours": hours, "resolution_share": share}


def _npz_len(path):
    import numpy as np

    with np.load(path) as z:
        return len(z["t"])


def compare_dumps(rows, dump_dir, npz_len=_npz_len):
    """Compare packet counts with len(t) of each dump npz. Returns a summary dict."""
    from run import npz_paths

    compared = missing = 0
    sum_count = sum_dump = 0
    worst = (0, None)
    for r in rows:
        if not r["frames"]:
            continue
        path = npz_paths(dump_dir, r["video_id"])[1]
        if not path.exists():
            missing += 1
            continue
        n = npz_len(path)
        compared += 1
        sum_count += r["frames"]
        sum_dump += n
        gap = abs(r["frames"] - n)
        if gap > worst[0]:
            worst = (gap, r["video_id"])
    pct = 100.0 * (sum_count - sum_dump) / sum_dump if sum_dump else 0.0
    return {"compared": compared, "missing_dumps": missing, "largest_gap": worst[0],
            "largest_gap_video": worst[1], "total_gap_pct": pct, "sum_count": sum_count, "sum_dump": sum_dump}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True, help="folder of .parquet files (or one file)")
    ap.add_argument("--out", required=True, help="CSV to write")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--compare-dumps", default=None, metavar="DIR")
    args = ap.parse_args(argv)
    files = list_inputs(args.input)
    if not files:
        print(f"no .parquet files in {args.input}", file=sys.stderr)
        return 2
    workers = max(1, min(args.workers, len(files)))
    rows = process_all(files, workers)
    write_csv(rows, args.out)
    t = totals(rows)
    print(f"videos {t['videos']} (errors {t['errors']}), frames {t['frames']}, hours {t['hours']:.2f}")
    for res, share in list(t["resolution_share"].items())[:10]:
        print(f"  {res}: {100 * share:.1f}% of frames")
    if args.compare_dumps:
        c = compare_dumps(rows, args.compare_dumps)
        print(f"compare-dumps: {c['compared']} compared, {c['missing_dumps']} without dump, "
              f"largest gap {c['largest_gap']} frames ({c['largest_gap_video']}), "
              f"total gap {c['total_gap_pct']:+.4f}% (packets {c['sum_count']} vs dumps {c['sum_dump']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
