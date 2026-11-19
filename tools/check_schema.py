"""Check a results file against the output format. Exit 1 if anything is wrong.

    python tools/check_schema.py results.jsonl
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vidx.aggregate import OBJECTS  # noqa: E402


def check(path):
    problems = []
    seconds = defaultdict(list)
    errors = defaultdict(int)
    with open(path, encoding="utf-8") as f:
        for n, raw in enumerate(f, 1):
            if not raw.strip():
                continue
            try:
                d = json.loads(raw)
            except ValueError:
                problems.append(f"line {n}: not valid JSON")
                continue
            if not isinstance(d, dict):
                problems.append(f"line {n}: not a JSON object")
                continue
            video = d.get("video")
            if not isinstance(video, str) or ":" not in video:
                problems.append(f"line {n}: bad video id {video!r}")
                continue
            if "error" in d:
                if set(d) != {"video", "error"} or not isinstance(d["error"], str):
                    problems.append(f"line {n}: malformed error line")
                errors[video] += 1
                continue
            if set(d) != {"video", "second", "objects", "people"}:
                problems.append(f"line {n}: wrong keys {sorted(d)}")
                continue
            sec, objs, ppl = d["second"], d["objects"], d["people"]
            if not isinstance(sec, int) or isinstance(sec, bool) or sec < 0:
                problems.append(f"line {n}: bad second {sec!r}")
            if not isinstance(ppl, int) or isinstance(ppl, bool) or ppl < 0:
                problems.append(f"line {n}: bad people {ppl!r}")
            if not isinstance(objs, list) or any(o not in OBJECTS for o in objs):
                problems.append(f"line {n}: bad objects {objs!r}")
            elif objs != [o for o in OBJECTS if o in objs]:
                problems.append(f"line {n}: objects not unique/in canonical order {objs!r}")
            seconds[video].append(sec)
    for video, secs in seconds.items():
        if sorted(secs) != list(range(len(secs))):
            problems.append(f"{video}: seconds not contiguous from 0 (got {len(secs)} lines, max {max(secs)})")
    for video, n in errors.items():
        if n > 1:
            problems.append(f"{video}: {n} error lines")
    print(f"{len(seconds)} videos with seconds, {len(errors)} with error lines")
    return problems


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    problems = check(sys.argv[1])
    for p in problems[:50]:
        print("PROBLEM:", p)
    if len(problems) > 50:
        print(f"... and {len(problems) - 50} more")
    print("schema OK" if not problems else f"{len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
