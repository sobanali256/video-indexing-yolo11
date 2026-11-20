"""Stage-0 decode profile (CPU only, no model): how fast can one process decode, and what does BGR
conversion cost? Decides whether decoding needs threads or worker processes once sampling cuts model work.

    python tools/profile_decode.py --input /content/subset/tune --videos 5

Modes per FFmpeg thread count (0 = FFmpeg's automatic choice):
  decode   decode + timestamp only (lower bound: work that sampling cannot skip)
  bgr_all  decode + BGR conversion of every frame (what brute force does)
  sample5  decode + BGR conversion of 5 frames per second (vidx.decode with sample=5)
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vidx.decode import Timestamper, _open, pick_video_stream  # noqa: E402
from vidx.sampling import sample_groups  # noqa: E402
from vidx.source import iter_rows, list_inputs  # noqa: E402


def run_mode(data, mode, threads):
    """-> (decoded frames, converted frames, seconds)."""
    t0 = time.perf_counter()
    container = _open(data)
    decoded = converted = 0
    try:
        stream = pick_video_stream(container)
        if stream is None:
            return 0, 0, 0.0
        stream.thread_type = "AUTO"
        if threads:
            stream.thread_count = threads
        stamper = Timestamper(stream.time_base, stream.guessed_rate or stream.average_rate)

        def stamped():
            nonlocal decoded
            for frame in container.decode(stream):
                decoded += 1
                yield frame, stamper.stamp(frame.pts)

        frames = stamped()
        if mode == "sample5":
            frames = sample_groups(frames, lambda x: x[1], 5)
        for frame, _ in frames:
            if mode != "decode":
                frame.to_ndarray(format="bgr24")
                converted += 1
    finally:
        container.close()
    return decoded, converted, time.perf_counter() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--videos", type=int, default=5)
    ap.add_argument("--threads", default="0,1,2,4", help="FFmpeg thread counts to try (0 = auto)")
    ap.add_argument("--modes", default="decode,bgr_all,sample5")
    args = ap.parse_args(argv)

    datas = []
    for path in list_inputs(args.input):
        for vid, data, err in iter_rows(path):
            if not err and data is not None:
                datas.append((vid, bytes(memoryview(data))))
            if len(datas) >= args.videos:
                break
        if len(datas) >= args.videos:
            break
    print(f"{len(datas)} videos, cpu_count={os.cpu_count()}")
    print(f"{'mode':<9}{'threads':>8}{'frames':>9}{'seconds':>9}{'fps':>9}{'ms/frame':>10}")
    for threads in (int(x) for x in args.threads.split(",")):
        for mode in args.modes.split(","):
            dec = sec = 0
            for _, data in datas:
                d, _, s = run_mode(data, mode, threads)
                dec += d
                sec += s
            fps = dec / sec if sec else 0.0
            print(f"{mode:<9}{threads:>8}{dec:>9}{sec:>9.1f}{fps:>9.0f}{1000 / fps if fps else 0:>10.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
