"""Scan parquet files for videos carrying rotation metadata.

    python tools/find_rotated.py --input finevideo --max-files 2
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vidx.decode import probe_rotation  # noqa: E402
from vidx.source import iter_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--max-files", type=int, default=2)
    args = ap.parse_args()
    files = sorted(Path(args.input).glob("*.parquet"))[: args.max_files]
    scanned = found = 0
    for path in files:
        for video_id, data, err in iter_rows(path):
            if err:
                continue
            scanned += 1
            try:
                stream_rot, frame_rot, tag = probe_rotation(data)
            except Exception as e:
                print(f"{video_id}: probe failed: {e}", file=sys.stderr)
                continue
            if stream_rot or frame_rot or tag not in (None, "", "0"):
                found += 1
                print(f"{video_id}: stream_rotation_ccw={stream_rot} frame_rotation_ccw={frame_rot} rotate_tag={tag}")
    print(f"scanned {scanned} videos, {found} with rotation metadata")


if __name__ == "__main__":
    main()
