"""Damaged-input test set, and a checker for the run's output on it.

    python tools/make_damaged.py make  --input /content/subset/tune --out /content/damaged
    python run.py --input /content/damaged --output /content/damaged.jsonl
    python tools/make_damaged.py check --out /content/damaged --results /content/damaged.jsonl

`make` takes two real videos (A, B) from the first parquet file and writes:
  damaged.parquet   rows: 0 valid A | 1 A truncated to 40% | 2 random bytes | 3 null | 4 empty bytes |
                          5 A with a zeroed block in the middle | 6 valid B
  corrupt.parquet   random bytes (not a parquet file)
  truncated.parquet damaged.parquet cut in half (footer missing)
`check` verifies the brief's rules: every video has output; at most one error line per video; a video that cannot
be decoded has exactly one error line; valid videos have seconds and no error; seconds are contiguous from 0
(the schema is checked by tools/check_schema.py).
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vidx.source import iter_rows, list_inputs  # noqa: E402

# video -> what must hold: "seconds" (seconds, no error), "error" (exactly one error line, no seconds),
# "any" (seconds and/or one error line: decoders may recover part of a damaged stream)
EXPECT = {
    "damaged.parquet:0": "seconds", "damaged.parquet:1": "any", "damaged.parquet:2": "error",
    "damaged.parquet:3": "error", "damaged.parquet:4": "error", "damaged.parquet:5": "any",
    "damaged.parquet:6": "seconds", "corrupt.parquet:0": "error", "truncated.parquet:0": "error",
}


def make(args):
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    videos = []
    for _, buf, err in iter_rows(list_inputs(args.input)[0]):
        if not err:
            videos.append(bytes(memoryview(buf)))
        if len(videos) == 2:
            break
    a, b = videos
    zeroed = bytearray(a)
    mid, span = len(a) // 2, max(1, len(a) // 20)
    zeroed[mid:mid + span] = bytes(span)
    rows = [a, a[: int(len(a) * 0.4)], os.urandom(200_000), None, b"", bytes(zeroed), b]
    table = pa.table({"mp4": pa.array(rows, type=pa.binary())})
    pq.write_table(table, out / "damaged.parquet", row_group_size=2)
    (out / "corrupt.parquet").write_bytes(os.urandom(100_000))
    whole = (out / "damaged.parquet").read_bytes()
    (out / "truncated.parquet").write_bytes(whole[: len(whole) // 2])
    print(f"wrote {out}: damaged.parquet ({len(rows)} rows), corrupt.parquet, truncated.parquet")
    return 0


def check(args):
    seconds, errors = defaultdict(list), defaultdict(list)
    with open(args.results, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            (errors if "error" in d else seconds)[d["video"]].append(d.get("error", d.get("second")))
    ok = True
    print(f"{'video':<22}{'expect':<9}{'seconds':>8}  errors  result")
    for vid, rule in EXPECT.items():
        n_sec, errs = len(seconds.get(vid, [])), errors.get(vid, [])
        contiguous = sorted(seconds.get(vid, [])) == list(range(n_sec))
        if rule == "seconds":
            good = n_sec > 0 and not errs
        elif rule == "error":
            good = n_sec == 0 and len(errs) == 1
        else:
            good = (n_sec > 0 or errs) and len(errs) <= 1
        good = bool(good) and contiguous
        ok &= good
        print(f"{vid:<22}{rule:<9}{n_sec:>8}  {len(errs):<6}  {'OK' if good else 'FAIL'}"
              f"{'  ' + errs[0][:70] if errs else ''}")
    extra = (set(seconds) | set(errors)) - set(EXPECT)
    if extra:
        ok = False
        print("unexpected videos in the output:", sorted(extra))
    print("damaged-input check:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--input", required=True, help="parquet folder or file with real videos")
    m.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("--out", required=True, help="(unused; kept for symmetry)")
    c.add_argument("--results", required=True)
    args = ap.parse_args(argv)
    return make(args) if args.cmd == "make" else check(args)


if __name__ == "__main__":
    sys.exit(main())
