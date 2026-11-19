"""Compare a candidate results file against a reference (grader's three thresholds).

    python score.py reference.jsonl candidate.jsonl

Standard library only. Exit code 0 if all thresholds pass, 1 otherwise.
"""
import argparse
import json
import sys
from collections import defaultdict

F1_MIN = 0.95
EXACT_MIN = 0.85
WITHIN1_MIN = 0.95


def parse_lines(lines):
    """Iterable of dicts -> (seconds, errors).
    seconds: {(video, second): (frozenset(objects), people)}; errors: {video: message}."""
    seconds, errors = {}, {}
    for d in lines:
        video = d.get("video")
        if "error" in d:
            errors[video] = d["error"]
        elif "second" in d:
            seconds[(video, d["second"])] = (frozenset(d.get("objects", [])), d.get("people", 0))
    return seconds, errors


def load_results(path):
    bad = 0
    lines = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                lines.append(json.loads(raw))
            except ValueError:
                bad += 1
    if bad:
        print(f"warning: {bad} unparseable lines in {path}", file=sys.stderr)
    return parse_lines(lines)


def f1(tp, fp, fn):
    denom = 2 * tp + fp + fn
    return 1.0 if denom == 0 else 2 * tp / denom


def compute(ref, cand):
    """ref, cand: (seconds, errors) as returned by parse_lines. Returns a metrics dict."""
    ref_s, ref_e = ref
    cand_s, cand_e = cand
    per_video = defaultdict(lambda: [0, 0, 0])  # tp, fp, fn
    exact = within1 = total = 0
    people_by_video = defaultdict(lambda: [0, 0])  # exact hits, seconds

    for key, (r_obj, r_ppl) in ref_s.items():
        c_obj, c_ppl = cand_s.get(key, (frozenset(), 0))  # missing second == empty
        tp, fp, fn = len(r_obj & c_obj), len(c_obj - r_obj), len(r_obj - c_obj)
        v = per_video[key[0]]
        v[0] += tp
        v[1] += fp
        v[2] += fn
        total += 1
        people_by_video[key[0]][1] += 1
        if c_ppl == r_ppl:
            exact += 1
            people_by_video[key[0]][0] += 1
        if abs(c_ppl - r_ppl) <= 1:
            within1 += 1
    for key, (c_obj, _) in cand_s.items():
        if key not in ref_s:  # extra second: every claimed object is a false positive
            per_video[key[0]][1] += len(c_obj)

    tp = sum(v[0] for v in per_video.values())
    fp = sum(v[1] for v in per_video.values())
    fn = sum(v[2] for v in per_video.values())
    video_f1 = {vid: f1(*v) for vid, v in per_video.items()}
    return {
        "micro_f1": f1(tp, fp, fn),
        "macro_f1": sum(video_f1.values()) / len(video_f1) if video_f1 else 1.0,
        "people_exact": exact / total if total else 1.0,
        "people_within1": within1 / total if total else 1.0,
        "ref_seconds": total,
        "cand_seconds": len(cand_s),
        "missing_seconds": sum(1 for k in ref_s if k not in cand_s),
        "extra_seconds": sum(1 for k in cand_s if k not in ref_s),
        "ref_errors": len(ref_e),
        "cand_errors": len(cand_e),
        "error_mismatch": len(set(ref_e) ^ set(cand_e)),
        "video_f1": video_f1,
        "video_people_exact": {k: (h / n if n else 1.0) for k, (h, n) in people_by_video.items()},
    }


def passes(m):
    return {
        "objects micro F1": (m["micro_f1"], F1_MIN),
        "people exact": (m["people_exact"], EXACT_MIN),
        "people within +-1": (m["people_within1"], WITHIN1_MIN),
    }


def report(m, worst=10, out=None):
    out = out or sys.stdout
    ok = True
    for name, (value, threshold) in passes(m).items():
        good = value >= threshold
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {name}: {value:.4f} (need >= {threshold})", file=out)
    print(f"macro F1 per video: {m['macro_f1']:.4f}", file=out)
    print(
        f"seconds: ref {m['ref_seconds']}, cand {m['cand_seconds']}, "
        f"missing {m['missing_seconds']}, extra {m['extra_seconds']}",
        file=out,
    )
    print(
        f"error lines: ref {m['ref_errors']}, cand {m['cand_errors']}, "
        f"mismatched videos {m['error_mismatch']}",
        file=out,
    )
    if m["video_f1"]:
        print(f"worst {worst} videos (F1, people exact):", file=out)
        ranked = sorted(m["video_f1"].items(), key=lambda kv: (kv[1], m["video_people_exact"].get(kv[0], 1.0)))
        for vid, score in ranked[:worst]:
            print(f"  {vid}: F1 {score:.3f}, people exact {m['video_people_exact'].get(vid, 1.0):.3f}", file=out)
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("reference")
    ap.add_argument("candidate")
    ap.add_argument("--worst", type=int, default=10)
    args = ap.parse_args(argv)
    m = compute(load_results(args.reference), load_results(args.candidate))
    return 0 if report(m, args.worst) else 1


if __name__ == "__main__":
    sys.exit(main())
