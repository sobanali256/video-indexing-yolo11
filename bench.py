"""Timed runs of run.py with a CSV evidence log, and extrapolation of the brute-force time.

    python bench.py run --label "brute force baseline" --input /content/subset/tune \\
        --out /content/ref/ref_tune.jsonl --log bench_log.csv [--ref tune=PATH --ref holdout=PATH] \\
        [--notes TEXT] [--only-common] [-- <extra run.py args>]
    python bench.py extrapolate --from-label "brute force baseline" --frames-csv frame_counts.csv --log bench_log.csv

`run` starts run.py (same Python), times the whole process with a monotonic clock, scores the output against
the given references and appends one row to the CSV. A run is "clean" only if, at start and every 30 s until
the end, no download process (hf download / huggingface-cli / snapshot_download) and no foreign GPU process
(a process outside bench's tree holding an open /dev/nvidia* fd) was running. On systems without /proc the run is never marked clean ("unknown (no /proc)").
Standard library only.
"""
import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import score

COLUMNS = [
    "timestamp_utc", "label", "kind", "status", "clean", "clean_reason", "gpu", "input", "videos", "frames",
    "wall_s", "startup_s", "loop_s", "fps_loop",
    "f1_tune", "exact_tune", "within1_tune", "pass_tune",
    "f1_holdout", "exact_holdout", "within1_holdout", "pass_holdout",
    "total_frames", "projected_s", "projected_h", "notes", "command",
]
DOWNLOAD_MARKERS = ("hf download", "huggingface-cli", "snapshot_download")
CLEAN_INTERVAL_S = 30.0


# ---------------------------------------------------------------- clean detection

def _read(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return None


def proc_tree(roots, proc_root="/proc"):
    """PIDs of roots and all their descendants (from /proc/<pid>/stat). None if /proc is unavailable."""
    if not os.path.isdir(proc_root):
        return None
    parent = {}
    for name in os.listdir(proc_root):
        if not name.isdigit():
            continue
        raw = _read(os.path.join(proc_root, name, "stat"))
        if not raw:
            continue
        try:  # "pid (comm) state ppid ...": comm may contain spaces/parens
            parent[int(name)] = int(raw.decode("utf-8", "replace").rpartition(")")[2].split()[1])
        except (ValueError, IndexError):
            continue
    tree = set(roots)
    grew = True
    while grew:
        grew = False
        for pid, ppid in parent.items():
            if ppid in tree and pid not in tree:
                tree.add(pid)
                grew = True
    return tree


def _cmdline(proc_root, name):
    raw = _read(os.path.join(proc_root, name, "cmdline"))
    if not raw:
        return ""
    return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()


def scan_downloads(roots, tree=None, proc_root="/proc"):
    """Descriptions of running download processes outside the roots' tree. None if /proc is unavailable.
    tree: precomputed proc_tree(roots) (computed here when omitted)."""
    if tree is None:
        tree = proc_tree(roots, proc_root)
    if tree is None or not os.path.isdir(proc_root):
        return None
    found = []
    for name in os.listdir(proc_root):
        if not name.isdigit() or int(name) in tree:
            continue
        cmd = _cmdline(proc_root, name)
        if any(m in cmd for m in DOWNLOAD_MARKERS):
            found.append(f"{name}: {cmd[:80]}")
    return found


def scan_gpu_users(roots, tree=None, proc_root="/proc"):
    """Processes outside the roots' tree holding an open /dev/nvidia* fd ("pid=N cmd=..."). None if /proc is
    unavailable. (nvidia-smi PIDs are unreliable inside Colab's container, so /proc/<pid>/fd is scanned.)"""
    if tree is None:
        tree = proc_tree(roots, proc_root)
    if tree is None or not os.path.isdir(proc_root):
        return None
    found = []
    for name in os.listdir(proc_root):
        if not name.isdigit() or int(name) in tree:
            continue
        fd_dir = os.path.join(proc_root, name, "fd")
        try:
            fds = os.listdir(fd_dir)
        except OSError:  # no permission or the process just exited
            continue
        for fd in fds:
            try:
                target = os.readlink(os.path.join(fd_dir, fd))
            except OSError:
                continue
            if target.startswith("/dev/nvidia"):
                found.append(f"pid={name} cmd={_cmdline(proc_root, name)[:80]}")
                break
    return found


def gpu_name():
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        if res.returncode == 0:
            return res.stdout.strip().splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError, IndexError):
        pass
    return ""


class CleanMonitor:
    """Checks at start and every `interval` seconds until stop() (plus once at the end).

    proc_scan(roots, tree) -> list of download descriptions, or None if there is no /proc.
    gpu_scan(roots, tree) -> list of "pid=N cmd=..." of foreign GPU users, or None if there is no /proc.
    tree_fn(roots) -> PIDs of roots and descendants (computed once per check), or None.
    """

    def __init__(self, roots, proc_scan=None, gpu_scan=None, interval=CLEAN_INTERVAL_S, tree_fn=None):
        self.tree_fn = tree_fn or proc_tree
        self.roots = set(roots)
        self.proc_scan = proc_scan or scan_downloads
        self.gpu_scan = gpu_scan or scan_gpu_users
        self.interval = interval
        self.reasons = []
        self.unknown = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def add_root(self, pid):
        self.roots.add(pid)

    def check_once(self):
        roots = set(self.roots)
        try:
            tree = self.tree_fn(roots)
        except Exception:
            tree = None
        try:
            downloads = self.proc_scan(roots, tree)
        except Exception as e:  # a failing detector must never break the run
            downloads = None
            self._note(f"scan failed: {type(e).__name__}")
        if downloads is None:
            self.unknown = True
        else:
            for d in downloads:
                self._note(f"download running ({d})")
        try:
            gpu = self.gpu_scan(roots, tree)
        except Exception as e:
            gpu = None
            self._note(f"gpu scan failed: {type(e).__name__}")
        if gpu is None:
            self.unknown = True
        else:
            for g in gpu:
                self._note(f"other GPU process {g}")

    def _note(self, reason):
        if reason not in self.reasons and len(self.reasons) < 5:
            self.reasons.append(reason)

    def _loop(self):
        while not self._stop.wait(self.interval):
            self.check_once()

    def start(self):
        self.check_once()
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        self.check_once()

    def result(self):
        """(clean, reason): clean is "yes" or "no"."""
        if self.reasons:
            return "no", "; ".join(self.reasons)
        if self.unknown:
            return "no", "unknown (no /proc)"
        return "yes", "no download or foreign GPU process seen"


# ---------------------------------------------------------------- CSV

def now_utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_row(log_path, row):
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = not p.exists() or p.stat().st_size == 0
    with open(p, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore", restval="")
        if new:
            w.writeheader()
        w.writerow({k: ("" if v is None else v) for k, v in row.items()})


def read_rows(log_path):
    if not Path(log_path).exists():
        return []
    with open(log_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(x, digits=3):
    return "" if x is None else f"{x:.{digits}f}"


def load_stats(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


# ---------------------------------------------------------------- run

def run_bench(args, extra, proc_scan=None, gpu_scan=None, gpu_name_fn=None, tree_fn=None):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stats_path = args.stats_json or (str(out) + ".stats.json")
    try:
        os.remove(stats_path)  # never read a stale file
    except OSError:
        pass
    cmd = [sys.executable, args.run_script, "--input", args.input, "--output", str(out),
           "--stats-json", stats_path, *extra]
    mon = CleanMonitor({os.getpid()}, proc_scan, gpu_scan, args.clean_interval, tree_fn)
    mon.start()
    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(cmd)
        mon.add_root(proc.pid)
        rc = proc.wait()
    finally:
        wall = time.monotonic() - t0
        mon.stop()
    clean, reason = mon.result()
    stats = load_stats(stats_path)
    frames, loop_s = stats.get("frames"), stats.get("loop_s")
    fps = frames / loop_s if frames and loop_s else None
    row = {
        "timestamp_utc": now_utc(), "label": args.label, "kind": "measured",
        "status": "ok" if rc == 0 else "failed", "clean": clean, "clean_reason": reason,
        "gpu": (gpu_name_fn or gpu_name)(), "input": args.input,
        "videos": stats.get("videos"), "frames": frames,
        "wall_s": fmt(wall), "startup_s": fmt(stats.get("startup_s")), "loop_s": fmt(loop_s),
        "fps_loop": fmt(fps, 2), "notes": args.notes or "", "command": shlex.join(cmd),
    }
    exit_code = rc
    if rc != 0:
        row["notes"] = (row["notes"] + f" run.py exit code {rc}").strip()
    elif args.ref:
        try:
            results, all_pass = score.score_files(args.ref, str(out), args.only_common, out=sys.stderr)
            for r in results:
                if r["name"] in ("tune", "holdout"):
                    m = r["metrics"]
                    row[f"f1_{r['name']}"] = fmt(m["micro_f1"], 4)
                    row[f"exact_{r['name']}"] = fmt(m["people_exact"], 4)
                    row[f"within1_{r['name']}"] = fmt(m["people_within1"], 4)
                    row[f"pass_{r['name']}"] = r["status"].lower()
            if not all_pass:  # any FAIL or PARTIAL set: the timing must not feed extrapolation
                row["status"] = "fail_accuracy"
                exit_code = 1
        except Exception as e:  # scoring must never prevent the timing row from being logged
            row["status"] = "fail_accuracy"
            exit_code = 1
            msg = f"scoring failed: {type(e).__name__}: {e}"[:200]
            row["notes"] = (row["notes"] + " " + msg).strip()
            print(f"bench: {msg}", file=sys.stderr)
    append_row(args.log, row)
    print(f"bench: {row['status']}, clean={clean} ({reason}), wall {row['wall_s']}s, loop {row['loop_s']}s, "
          f"{row['frames']} frames -> {args.log}", file=sys.stderr)
    return exit_code


# ---------------------------------------------------------------- extrapolate

def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def newest_clean_ok(rows, label):
    best = None
    for r in rows:
        if (r.get("label") == label and r.get("kind") == "measured" and r.get("status") == "ok"
                and r.get("clean") == "yes" and _f(r.get("frames")) and _f(r.get("loop_s"))
                and _f(r.get("wall_s")) is not None):
            if best is None or r["timestamp_utc"] >= best["timestamp_utc"]:
                best = r
    return best


def total_frames_from_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return int(sum(_f(r.get("frames")) or 0 for r in csv.DictReader(f)))


def project(src, total_frames):
    """(per_frame_s, fixed_s, projected_s) from a clean measured row."""
    per_frame = float(src["loop_s"]) / float(src["frames"])
    fixed = float(src["wall_s"]) - float(src["loop_s"])
    return per_frame, fixed, fixed + total_frames * per_frame


def extrapolate(args):
    src = newest_clean_ok(read_rows(args.log), args.from_label)
    if src is None:
        print(f"bench: no clean, ok, measured row with label {args.from_label!r} in {args.log}", file=sys.stderr)
        return 2
    total = args.total_frames if args.total_frames else total_frames_from_csv(args.frames_csv)
    per_frame, fixed, projected = project(src, total)
    gpu = src.get("gpu", "")
    notes = (f"extrapolated: measured per-frame cost x total frames of 50 files; measured on {gpu or 'unknown GPU'} "
             f"(grader uses RTX 4090); per_frame_s={per_frame:.6f} fixed_s={fixed:.1f}; "
             f"source row {src['timestamp_utc']}")
    if args.notes:
        notes += "; " + args.notes
    row = {
        "timestamp_utc": now_utc(), "label": args.from_label, "kind": "extrapolated", "status": "ok",
        "clean": src.get("clean"), "clean_reason": src.get("clean_reason"), "gpu": gpu, "input": src.get("input"),
        "total_frames": total, "projected_s": fmt(projected, 1), "projected_h": fmt(projected / 3600, 2),
        "notes": notes, "command": shlex.join([sys.executable, *sys.argv]),
    }
    append_row(args.log, row)
    print(f"bench: projected {projected:.0f}s ({projected / 3600:.2f} h) for {total} frames", file=sys.stderr)
    return 0


# ---------------------------------------------------------------- CLI

def parse_args(argv):
    argv = list(argv)
    extra = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="time one run.py run and log it")
    r.add_argument("--label", required=True)
    r.add_argument("--input", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--ref", action="append", type=score.parse_ref, default=[], metavar="NAME=PATH")
    r.add_argument("--log", default="bench_log.csv", help="CSV file to append to")
    r.add_argument("--notes", default="")
    r.add_argument("--only-common", action="store_true")
    r.add_argument("--run-script", default=str(Path(__file__).resolve().parent / "run.py"), help=argparse.SUPPRESS)
    r.add_argument("--stats-json", default=None, help=argparse.SUPPRESS)
    r.add_argument("--clean-interval", type=float, default=CLEAN_INTERVAL_S, help=argparse.SUPPRESS)
    e = sub.add_parser("extrapolate", help="project the full brute-force time from a clean measured row")
    e.add_argument("--from-label", required=True)
    e.add_argument("--frames-csv", default=None, help="CSV written by tools/count_frames.py")
    e.add_argument("--total-frames", type=int, default=None)
    e.add_argument("--log", default="bench_log.csv")
    e.add_argument("--notes", default="")
    args = ap.parse_args(argv)
    if args.cmd == "extrapolate" and not (args.frames_csv or args.total_frames):
        ap.error("give --frames-csv or --total-frames")
    return args, extra


def main(argv=None, proc_scan=None, gpu_scan=None, gpu_name_fn=None, tree_fn=None):
    args, extra = parse_args(sys.argv[1:] if argv is None else argv)
    if args.cmd == "run":
        return run_bench(args, extra, proc_scan, gpu_scan, gpu_name_fn, tree_fn)
    return extrapolate(args)


if __name__ == "__main__":
    sys.exit(main())
