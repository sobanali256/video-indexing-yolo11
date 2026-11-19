"""Score PyAV decoding against cv2.VideoCapture on N videos (needs the model, i.e. a GPU box).

    python tools/decoder_agreement.py --input finevideo --n 20

Both decoders go through the same model and aggregation; the PyAV result is the
reference. Agreement below ~0.99 means: ask the user which decoder is the reference.
"""
import argparse
import os
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import score  # noqa: E402
from run import Timers, process_frames, video_lines  # noqa: E402
from vidx.decode import Frame, decode_video  # noqa: E402
from vidx.model import infer, load_model  # noqa: E402
from vidx.source import iter_rows  # noqa: E402


def cv2_frames(data):
    import cv2

    fd, tmp = tempfile.mkstemp(suffix=".mp4")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(memoryview(data))
        cap = cv2.VideoCapture(tmp)  # applies rotation metadata on its own
        first = None
        while True:
            ok, image = cap.read()
            if not ok:
                break
            ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            t = Fraction(ms / 1000.0).limit_denominator(1_000_000)
            first = t if first is None else first
            yield Frame(image, max(t - first, Fraction(0)), None, 0)
        cap.release()
    finally:
        os.remove(tmp)


def lines_for(video_id, frames, infer_fn, batch):
    records, error = process_frames(frames, infer_fn, batch, Timers())
    return video_lines(video_id, records, error), len(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()
    model = load_model()

    def infer_fn(images):
        return infer(model, images)

    ref_lines, cand_lines = [], []
    done = 0
    for path in sorted(Path(args.input).glob("*.parquet")):
        for video_id, data, err in iter_rows(path):
            if err:
                continue
            a, na = lines_for(video_id, decode_video(data), infer_fn, args.batch)
            b, nb = lines_for(video_id, cv2_frames(data), infer_fn, args.batch)
            print(f"{video_id}: pyav {na} frames, cv2 {nb} frames")
            ref_lines += a
            cand_lines += b
            done += 1
            if done >= args.n:
                break
        if done >= args.n:
            break
    m = score.compute(score.parse_lines(ref_lines), score.parse_lines(cand_lines))
    score.report(m)
    print(f"decoder agreement (micro F1): {m['micro_f1']:.4f}  (< 0.99: ask the user which decoder is the reference)")


if __name__ == "__main__":
    main()
