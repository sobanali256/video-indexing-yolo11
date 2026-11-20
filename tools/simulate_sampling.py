"""Offline sampling simulator: no GPU, no video decoding.

Re-aggregates the per-frame dumps of an every-frame run (run.py --dump-frames) using only the frames a
sampling strategy would send to the model, and scores the result against the reference with score.py.
Grouping and selection are vidx.sampling's, i.e. exactly what the pipeline does; timestamps are the
exact Fractions (t_num / t_den), so every frame lands in the same second as in the pipeline.

    python tools/simulate_sampling.py --dumps $DRIVE/frames --ref $DRIVE/reference/ref_tune.jsonl
    python tools/simulate_sampling.py ... --k 1,3,5,7 --adaptive 3:7,5:9 --json sim.json

Only for the TUNE set (the holdout has no dumps, by design). FP16 effects are not simulated.
"""
import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import score  # noqa: E402
from vidx.aggregate import aggregate, make_lines  # noqa: E402
from vidx.sampling import iter_groups, pick  # noqa: E402

# Tune acceptance (plan amendment I5): above the grader's 0.95 / 0.85 / 0.95 with margin.
ACCEPT = {"micro_f1": 0.985, "people_exact": 0.93, "people_within1": 0.985}


def video_id_of(npz_path):
    """train-x.parquet__4.npz -> train-x.parquet:4 (inverse of run.npz_paths)."""
    name = Path(npz_path).name
    stem = name[: -len(".npz")]
    file_part, _, row = stem.rpartition("__")
    return f"{file_part}:{row}"


def load_dump(path):
    """-> [(Fraction t, counts tuple)] in decode order."""
    import numpy as np

    with np.load(path) as d:
        t_num, t_den, counts = d["t_num"], d["t_den"], d["counts"]
        return [(Fraction(int(a), int(b)), tuple(int(c) for c in row)) for a, b, row in zip(t_num, t_den, counts)]


def disagree(counts_list):
    """True if the sampled frames of a group do not all agree (presence of any class, or person count)."""
    first = counts_list[0]
    pres = [c > 0 for c in first]
    for c in counts_list[1:]:
        if c[0] != first[0] or [x > 0 for x in c] != pres:
            return True
    return False


def select(frames, k, adaptive=None):
    """Frames the pipeline would infer. adaptive=(k0, k1): k0 per group, plus pick(n, k1) when the
    k0 frames disagree (union). Returns (selected frames in order, number inferred)."""
    out = []
    for group in iter_groups(frames, lambda f: f[0]):
        n = len(group)
        if adaptive:
            k0, k1 = adaptive
            idx = pick(n, k0)
            if len(idx) > 1 and disagree([group[i][1] for i in idx]):
                idx = sorted(set(idx) | set(pick(n, k1)))
        else:
            idx = pick(n, k)
        out.extend(group[i] for i in idx)
    return out


def simulate(videos, ref, strategy):
    """videos: {video_id: frames}; strategy: ("k", k) or ("adaptive", (k0, k1)). Returns a result dict."""
    lines, inferred, total = [], 0, 0
    for vid, frames in videos.items():
        kind, val = strategy
        sel = select(frames, val, None) if kind == "k" else select(frames, 0, val)
        inferred += len(sel)
        total += len(frames)
        lines += make_lines(vid, aggregate(sel))
    seconds, _ = score.parse_lines(lines)
    cand = (seconds, dict(ref[1]))  # decode errors are not simulated: copy the reference's error lines
    r = score.score_set("tune", ref, cand, only_common=False, filter_candidate=True)
    m = r["metrics"]
    ok = all(m[key] >= v for key, v in ACCEPT.items())
    return {"strategy": strategy, "inferred": inferred, "frames": total, "fraction": inferred / max(total, 1),
            "f1": m["micro_f1"], "exact": m["people_exact"], "within1": m["people_within1"],
            "grader_pass": r["status"] == "PASS", "accept": ok and r["status"] == "PASS",
            "worst": sorted(m["video_people_exact"].items(), key=lambda kv: kv[1])[:3],
            "worst_f1": sorted(m["video_f1"].items(), key=lambda kv: kv[1])[:3]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dumps", required=True, help="folder of <video>.npz from run.py --dump-frames")
    ap.add_argument("--ref", required=True, help="every-frame reference jsonl of the same videos")
    ap.add_argument("--k", default="1,2,3,4,5,6,7,8,9", help="fixed k values, comma separated")
    ap.add_argument("--adaptive", default="", help="k0:k1 pairs, comma separated, e.g. 3:7,5:9")
    ap.add_argument("--json", default=None, help="also write all results here")
    args = ap.parse_args(argv)

    ref_s, ref_e = score.load_results(args.ref)
    paths = sorted(p for p in Path(args.dumps).glob("*.npz") if not p.name.endswith(".tmp.npz"))
    videos = {video_id_of(p): load_dump(p) for p in paths}
    ref_videos = {v for v, _ in ref_s} | set(ref_e)
    missing = sorted(ref_videos - set(videos))
    extra = sorted(set(videos) - ref_videos)
    for v in extra:
        del videos[v]
    if missing:
        print(f"WARNING: {len(missing)} reference videos have no dump; they are left out of the scoring: "
              f"{missing[:5]}{' ...' if len(missing) > 5 else ''}", file=sys.stderr)
    if extra:
        print(f"WARNING: {len(extra)} dumps are not in the reference and are ignored", file=sys.stderr)
    keep = set(videos)
    ref, _ = score.restrict((ref_s, ref_e), keep.__contains__)
    if not videos or not ref[0]:
        sys.exit("nothing to simulate: no dumps matching the reference")

    # sanity: every-frame re-aggregation must reproduce the reference exactly
    base = simulate(videos, ref, ("k", 0))
    print(f"{len(videos)} videos, {base['frames']} frames, {len(ref[0])} seconds")
    print(f"check k=0 (every frame) vs reference: F1 {base['f1']:.4f} exact {base['exact']:.4f} "
          f"within1 {base['within1']:.4f}  (must be 1.0000)")

    strategies = [("k", int(k)) for k in args.k.split(",") if k.strip()]
    for pair in filter(None, (p.strip() for p in args.adaptive.split(","))):
        k0, k1 = (int(x) for x in pair.split(":"))
        strategies.append(("adaptive", (k0, k1)))
    print(f"accept on tune: F1 >= {ACCEPT['micro_f1']}, exact >= {ACCEPT['people_exact']}, "
          f"within1 >= {ACCEPT['people_within1']}")
    print(f"{'strategy':<14}{'inferred':>10}{'speedup':>9}{'F1':>8}{'exact':>8}{'within1':>9}  grader  accept")
    results = [base]
    for st in strategies:
        r = simulate(videos, ref, st)
        results.append(r)
        name = f"k={st[1]}" if st[0] == "k" else f"adapt {st[1][0]}:{st[1][1]}"
        print(f"{name:<14}{r['fraction']:>10.3f}{1 / max(r['fraction'], 1e-9):>8.1f}x{r['f1']:>8.4f}"
              f"{r['exact']:>8.4f}{r['within1']:>9.4f}  {'PASS' if r['grader_pass'] else 'FAIL':<6}  "
              f"{'yes' if r['accept'] else 'no'}")
    print("\nworst videos per strategy (people exact | F1):")
    for r in results[1:]:
        st = r["strategy"]
        name = f"k={st[1]}" if st[0] == "k" else f"adapt {st[1][0]}:{st[1][1]}"
        w = ", ".join(f"{v.rpartition(':')[2]}={x:.3f}" for v, x in r["worst"])
        wf = ", ".join(f"{v.rpartition(':')[2]}={x:.3f}" for v, x in r["worst_f1"])
        print(f"  {name:<14} exact: {w} | F1: {wf}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump([{**r, "strategy": list(r["strategy"])} for r in results], f, indent=2, default=str)
    return 0 if base["f1"] == base["exact"] == base["within1"] == 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
