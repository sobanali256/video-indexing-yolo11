"""Compare a candidate results file against one or more references (grader's three thresholds).

    python score.py reference.jsonl candidate.jsonl
    python score.py --ref tune=ref_tune.jsonl --ref holdout=ref_holdout.jsonl candidate.jsonl [--json out.json]

With --ref, each reference defines a set of parquet files (from its second lines and its error lines);
the candidate is filtered to those files and scored against that reference alone (a WARNING is printed
for ignored candidate videos). The two-file form does not filter: extra candidate seconds are false
positives. An empty reference is a FAIL. --only-common scores only the videos present in both files and labels the result PARTIAL.
Standard library only. Exit code 0 only if every set is a full PASS, 1 otherwise.
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


def file_of(video):
    """Parquet file name of a video id "<file>:<row>"."""
    return str(video).rpartition(":")[0]


def file_set(parsed):
    """Parquet files that appear in a parsed results file, in second lines or in error lines."""
    seconds, errors = parsed
    return {file_of(v) for v, _ in seconds} | {file_of(v) for v in errors}


def video_set(parsed):
    seconds, errors = parsed
    return {v for v, _ in seconds} | set(errors)


def restrict(parsed, keep):
    """Keep only entries whose video satisfies keep(video). Returns (filtered, number of videos dropped)."""
    seconds, errors = parsed
    dropped = {v for v in video_set(parsed) if not keep(v)}
    return (
        {k: v for k, v in seconds.items() if keep(k[0])},
        {v: m for v, m in errors.items() if keep(v)},
    ), len(dropped)


def score_set(name, ref, cand, only_common=False, filter_candidate=True):
    """Score parsed candidate against parsed reference -> result dict (status PASS / FAIL / PARTIAL).

    filter_candidate=True (--ref mode): the candidate is restricted to the reference's parquet files.
    False (positional mode): the whole candidate is scored, so extra candidate seconds count as FP."""
    files = file_set(ref)
    ignored = 0
    cand_f = cand
    if filter_candidate:
        cand_f, ignored = restrict(cand, lambda v: file_of(v) in files)
    no_file_overlap = bool(files) and not (files & file_set(cand))
    ref_f = ref
    no_overlap = False
    if only_common:
        common = video_set(ref) & video_set(cand_f)
        no_overlap = not common
        ref_f, _ = restrict(ref, common.__contains__)
        cand_f, _ = restrict(cand_f, common.__contains__)
    m = compute(ref_f, cand_f)
    empty = not ref[0]  # the reference itself has no seconds
    unscorable = empty or no_overlap or m["ref_seconds"] == 0
    ok = (not unscorable) and all(v >= t for v, t in passes(m).values())
    if not ok:
        status = "FAIL"
    else:
        status = "PARTIAL" if only_common else "PASS"
    return {"name": name, "metrics": m, "status": status, "empty_reference": empty, "no_overlap": no_overlap,
            "no_file_overlap": no_file_overlap, "files": sorted(files), "ignored_candidate_videos": ignored,
            "only_common": only_common, "filtered": filter_candidate}


def score_files(refs, cand_path, only_common=False, worst=10, out=None, filter_candidate=True):
    """refs: [(name, path)]. Prints one block per set and a summary. Returns (results, all_pass)."""
    out = out or sys.stdout
    cand = load_results(cand_path)
    results = []
    for name, path in refs:
        r = score_set(name, load_results(path), cand, only_common, filter_candidate)
        results.append(r)
        print(f"== {name}: {path} ({len(r['files'])} files) ==", file=out)
        if r["empty_reference"]:
            print("FAIL  reference has no seconds: nothing to score", file=out)
        elif r["no_overlap"]:
            print("FAIL  no overlapping videos between candidate and reference", file=out)
        elif r["metrics"]["ref_seconds"] == 0:
            print("FAIL  no reference seconds to score among the common videos", file=out)
        else:
            report(r["metrics"], worst, out)
        if filter_candidate:
            if r["no_file_overlap"]:
                print(f"WARNING: candidate has NO parquet file in common with reference {name!r}: "
                      f"wrong candidate file or wrong reference?", file=out)
            if r["ignored_candidate_videos"]:
                print(f"WARNING: {r['ignored_candidate_videos']} candidate videos ignored "
                      f"(files not in this reference)", file=out)
            else:
                print("candidate videos ignored (files not in this reference): 0", file=out)
        if only_common:
            print("PARTIAL: only videos present in both files were scored", file=out)
    print("== summary ==", file=out)
    for r in results:
        m = r["metrics"]
        print(f"{r['status']}  {r['name']}: F1 {m['micro_f1']:.4f}, exact {m['people_exact']:.4f}, "
              f"within1 {m['people_within1']:.4f}, seconds {m['ref_seconds']}", file=out)
    all_pass = bool(results) and all(r["status"] == "PASS" for r in results)
    print(f"overall: {'PASS' if all_pass else 'NOT PASS'}", file=out)
    return results, all_pass


def result_json(r):
    m = r["metrics"]
    keys = ("micro_f1", "macro_f1", "people_exact", "people_within1", "ref_seconds", "cand_seconds",
            "missing_seconds", "extra_seconds", "ref_errors", "cand_errors", "error_mismatch")
    d = {k: m[k] for k in keys}
    d.update(status=r["status"], no_overlap=r["no_overlap"], no_file_overlap=r["no_file_overlap"],
             ignored_candidate_videos=r["ignored_candidate_videos"],
             files=r["files"], only_common=r["only_common"])
    return d


def parse_ref(text):
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"--ref must be NAME=PATH, got {text!r}")
    return name, path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="*", help="REFERENCE CANDIDATE, or just CANDIDATE when --ref is used")
    ap.add_argument("--ref", action="append", type=parse_ref, default=[], metavar="NAME=PATH")
    ap.add_argument("--only-common", action="store_true", help="score only videos present in both files (PARTIAL)")
    ap.add_argument("--json", default=None, metavar="PATH", help="write metrics per set as JSON")
    ap.add_argument("--worst", type=int, default=10)
    args = ap.parse_args(argv)
    if args.ref:
        if len(args.files) != 1:
            ap.error("with --ref give exactly one positional argument: the candidate")
        refs, cand = args.ref, args.files[0]
    else:
        if len(args.files) != 2:
            ap.error("give REFERENCE CANDIDATE (or --ref NAME=PATH ... CANDIDATE)")
        refs, cand = [("reference", args.files[0])], args.files[1]
    results, ok = score_files(refs, cand, args.only_common, args.worst, filter_candidate=bool(args.ref))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"sets": {r["name"]: result_json(r) for r in results}, "overall_pass": ok}, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
