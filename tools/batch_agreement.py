"""Batch vs single-frame inference, and classes-filter vs no-filter (needs the model, i.e. a GPU box).

    python tools/batch_agreement.py --input finevideo --frames 1000

Targets: batch vs single >= 0.999 identical frames; filter vs no filter identical.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run  # noqa: E402
from vidx.decode import decode_video  # noqa: E402
from vidx.model import infer, load_model  # noqa: E402
from vidx.source import iter_rows  # noqa: E402


def collect(input_dir, n):
    """Up to n Frames: every 15th frame of the first videos, so several videos are used."""
    frames = []
    for path in sorted(Path(input_dir).glob("*.parquet")):
        for _, data, err in iter_rows(path):
            if err:
                continue
            try:
                for i, f in enumerate(decode_video(data)):
                    if i % 15 == 0:
                        frames.append(f)
                        if len(frames) >= n:
                            return frames
            except Exception as e:
                print("decode problem, skipping rest of video:", e, file=sys.stderr)
    return frames


def same(a, b):
    return sum(1 for x, y in zip(a, b) if x.counts == y.counts) / max(len(a), 1)


def max_conf_gap(a, b):
    return max((abs(p - q) for x, y in zip(a, b) for p, q in zip(x.conf, y.conf)), default=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--frames", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    model = load_model()
    frames = collect(args.input, args.frames)
    print(f"{len(frames)} frames collected")

    def pipeline(infer_fn, batch_size):
        records, error = run.process_frames(frames, infer_fn, batch_size, run.Timers(), keep_details=True)
        if error:
            sys.exit(f"pipeline error: {error}")
        return records

    def filtered(images):
        return infer(model, images)

    def unfiltered(images):
        return infer(model, images, classes=None)

    single = pipeline(filtered, 1)
    batched = pipeline(filtered, args.batch)
    nofilter = pipeline(unfiltered, 1)

    print(f"batch vs single:        identical counts {same(single, batched):.4f} (need >= 0.999), "
          f"max conf gap {max_conf_gap(single, batched):.5f}")
    print(f"classes filter vs none: identical counts {same(single, nofilter):.4f} (need 1.0), "
          f"max conf gap {max_conf_gap(single, nofilter):.5f}")


if __name__ == "__main__":
    main()
