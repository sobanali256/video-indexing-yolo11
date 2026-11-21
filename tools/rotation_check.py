"""Rotation check against ffmpeg (ground truth: the ffmpeg CLI applies display rotation by default).

Takes one real video from a parquet file, writes copies that carry rotation metadata (90 / 180 / 270), and compares
the first frame our decoder yields (vidx.decode.decode_video, rotation applied) with the first frame ffmpeg decodes
from the same file. A correct rotation gives the same shape and a small pixel difference; a wrong one gives another
shape or a large difference. Needs the ffmpeg CLI (`apt-get install ffmpeg`; present on Colab).

    python tools/rotation_check.py --input /content/subset/tune --row 0 --work /content/rotcheck
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vidx.decode import decode_video, probe_rotation  # noqa: E402
from vidx.source import iter_rows, list_inputs  # noqa: E402

MAX_MEAN_DIFF = 8.0  # mean |difference| per pixel channel (0-255) still counted as "same image" (colour conversion)


def ffmpeg(*args):
    return subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-y", *args], capture_output=True, text=True)


def make_rotated(src, dst, degrees):
    """Copy the stream and add rotation metadata. Newer ffmpeg: -display_rotation (counter-clockwise) on the input;
    older ffmpeg (4.x, e.g. Ubuntu 22.04): the mp4 'rotate' tag (clockwise). Returns how it was written or None."""
    r = ffmpeg("-display_rotation:v:0", str(degrees), "-i", str(src), "-map", "0:v:0", "-c", "copy", str(dst))
    if r.returncode == 0 and dst.exists():
        return f"-display_rotation {degrees}"
    r = ffmpeg("-i", str(src), "-map", "0:v:0", "-c", "copy", "-metadata:s:v:0", f"rotate={degrees}", str(dst))
    if r.returncode == 0 and dst.exists():
        return f"rotate={degrees}"
    print(f"ffmpeg could not write {dst}: {r.stderr.strip()[:300]}", file=sys.stderr)
    return None


def ffmpeg_first_frame(path, png):
    r = ffmpeg("-i", str(path), "-frames:v", "1", str(png))  # autorotate is on by default
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[:300])
    import cv2

    img = cv2.imread(str(png))  # BGR, like our frames
    if img is None:
        raise RuntimeError(f"could not read {png}")
    return img


def compare(ours, truth):
    """-> (same_shape, mean_abs_diff or None, best np.rot90 k of ours vs truth)."""
    import numpy as np

    best = None
    for k in range(4):
        cand = np.rot90(ours, k)
        if cand.shape == truth.shape:
            d = float(np.abs(cand.astype(np.int16) - truth.astype(np.int16)).mean())
            if best is None or d < best[1]:
                best = (k, d)
    same = ours.shape == truth.shape
    diff = float(np.abs(ours.astype(np.int16) - truth.astype(np.int16)).mean()) if same else None
    return same, diff, (best[0] if best else None)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="parquet folder or file")
    ap.add_argument("--row", type=int, default=0, help="which video (global row index in the first file)")
    ap.add_argument("--work", default="rotcheck", help="scratch folder for the mp4/png files")
    args = ap.parse_args(argv)

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    data = None
    for vid, buf, err in iter_rows(list_inputs(args.input)[0]):
        if vid.endswith(f":{args.row}"):
            if err:
                sys.exit(f"{vid}: {err}")
            data = bytes(memoryview(buf))
            break
    if data is None:
        sys.exit(f"row {args.row} not found")
    src = work / "src.mp4"
    src.write_bytes(data)

    ok = True
    print(f"{'case':<10}{'written as':<24}{'probe(stream,frame,tag)':<26}{'ours':<16}{'ffmpeg':<16}"
          f"{'mean diff':>10}  best k  result")
    for degrees in (0, 90, 180, 270):
        if degrees == 0:
            path, how = src, "original"
        else:
            path = work / f"rot{degrees}.mp4"
            how = make_rotated(src, path, degrees)
            if how is None:
                ok = False
                continue
        file_bytes = path.read_bytes()
        probe = probe_rotation(file_bytes)
        ours = next(iter(decode_video(file_bytes))).image
        truth = ffmpeg_first_frame(path, work / f"ffmpeg_{degrees}.png")
        same, diff, best_k = compare(ours, truth)
        good = same and diff is not None and diff <= MAX_MEAN_DIFF and best_k == 0
        if degrees and probe == (0, 0, None):
            good = False  # the metadata did not reach the file: the case tests nothing
        ok &= good
        print(f"{degrees:<10}{how:<24}{str(probe):<26}{str(ours.shape):<16}{str(truth.shape):<16}"
              f"{'' if diff is None else f'{diff:.2f}':>10}  {best_k!s:<7} {'OK' if good else 'FAIL'}")
    print("rotation check:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
