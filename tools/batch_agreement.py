"""Batch vs single-frame inference, and classes-filter vs no-filter (needs the model, i.e. a GPU box).

    python tools/batch_agreement.py --input finevideo --frames 1000
    python tools/batch_agreement.py --input finevideo --frames 1000 --direct --precision fp16

Targets: batch vs single >= 0.999 identical frames; filter vs no filter identical.
--direct (Stage 2a): predict() vs the direct path (thread-side LetterBox + infer_tensor) on the same frames in
the same batches. FP32 should be identical (1.0000); FP16 >= 0.999.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import run  # noqa: E402
from vidx.decode import decode_video  # noqa: E402
from vidx.model import infer, infer_tensor, load_model, make_letterbox, prepare_direct  # noqa: E402
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
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp32")
    ap.add_argument("--direct", action="store_true", help="compare predict() with the Stage 2a direct path")
    args = ap.parse_args()
    model = load_model(half=args.precision == "fp16")
    frames = collect(args.input, args.frames)
    print(f"{len(frames)} frames collected, precision {'fp16' if model.half else 'fp32'}")
    if args.direct:
        direct_check(model, frames, args.batch)
        return

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


def direct_check(model, frames, batch_size):
    """Same batches (consecutive frames of one shape, at most batch_size) through predict() and the direct path."""
    print("direct path args (conf, iou, classes, agnostic, max_det, fp16, imgsz):", prepare_direct(model))
    letterbox = make_letterbox(model)
    batches, cur = [], []
    for f in frames:
        if cur and (f.image.shape != cur[0].image.shape or len(cur) >= batch_size):
            batches.append(cur)
            cur = []
        cur.append(f)
    if cur:
        batches.append(cur)
    via_predict, via_direct = [], []
    for b in batches:
        images = [f.image for f in b]
        via_predict += infer(model, images)
        via_direct += infer_tensor(model, [letterbox(im) for im in images])
    same_counts = sum(1 for (a, _), (b, _) in zip(via_predict, via_direct) if a == b) / max(len(frames), 1)
    gap = max((abs(p - q) for (_, ca), (_, cb) in zip(via_predict, via_direct) for p, q in zip(ca, cb)), default=0.0)
    need = "1.0" if not model.half else ">= 0.999"
    print(f"predict() vs direct: identical counts {same_counts:.4f} (need {need}), max conf gap {gap:.5f}, "
          f"{len(batches)} batches")


if __name__ == "__main__":
    main()
