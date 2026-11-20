import contextlib
import csv
import io
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import bench  # noqa: E402

FAKE_RUN = r'''
import json, sys
a = sys.argv[1:]
def val(flag):
    return a[a.index(flag) + 1]
out, stats = val("--output"), val("--stats-json")
with open(out, "w") as f:
    f.write(json.dumps({"video": "a.parquet:0", "second": 0, "objects": ["car"], "people": 1}) + "\n")
    f.write(json.dumps({"video": "a.parquet:0", "second": 1, "objects": [], "people": 0}) + "\n")
if "--no-stats" not in a:
    json.dump({"videos": 1, "errors": 0, "frames": 200, "loop_s": 20.0, "startup_s": 5.0, "exit_code": 0}, open(stats, "w"))
sys.exit(7 if "--fail" in a else 0)
'''


def quiet(fn, *a, **kw):
    with contextlib.redirect_stderr(io.StringIO()), contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **kw)


class BenchRunTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.script = self.root / "fake_run.py"
        self.script.write_text(FAKE_RUN, encoding="utf-8")
        self.log = self.root / "log" / "bench_log.csv"
        ref = self.root / "ref.jsonl"
        ref.write_text(
            json.dumps({"video": "a.parquet:0", "second": 0, "objects": ["car"], "people": 1}) + "\n"
            + json.dumps({"video": "a.parquet:0", "second": 1, "objects": [], "people": 0}) + "\n",
            encoding="utf-8")
        self.ref = str(ref)

    def bench(self, *extra, proc_scan=lambda roots, tree: [], gpu_scan=lambda roots, tree: [], tail=()):
        argv = ["run", "--label", "L", "--input", str(self.root), "--out", str(self.root / "out.jsonl"),
                "--log", str(self.log), "--run-script", str(self.script), "--clean-interval", "0.05",
                *extra, "--", *tail]
        return quiet(bench.main, argv, proc_scan=proc_scan, gpu_scan=gpu_scan, gpu_name_fn=lambda: "FakeGPU",
                          tree_fn=lambda roots: set(roots))

    def rows(self):
        with open(self.log, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def test_row_written_with_scores_and_blanks(self):
        self.assertEqual(self.bench("--ref", f"tune={self.ref}"), 0)
        (r,) = self.rows()
        self.assertEqual(list(r), bench.COLUMNS)
        self.assertEqual((r["label"], r["kind"], r["status"], r["clean"], r["gpu"]), ("L", "measured", "ok", "yes", "FakeGPU"))
        self.assertEqual((r["videos"], r["frames"], r["loop_s"], r["startup_s"]), ("1", "200", "20.000", "5.000"))
        self.assertEqual(r["fps_loop"], "10.00")
        self.assertGreater(float(r["wall_s"]), 0)
        self.assertEqual((r["f1_tune"], r["pass_tune"]), ("1.0000", "pass"))
        self.assertEqual((r["f1_holdout"], r["pass_holdout"], r["total_frames"], r["projected_s"]), ("", "", "", ""))
        self.assertIn("--stats-json", r["command"])

    def test_scoring_exception_still_logs_row(self):
        orig = bench.score.score_files

        def boom(*a, **k):
            raise ValueError("bad ref " + "x" * 500)

        bench.score.score_files = boom
        try:
            rc = self.bench("--ref", f"tune={self.ref}")
        finally:
            bench.score.score_files = orig
        self.assertEqual(rc, 1)
        (r,) = self.rows()
        self.assertEqual((r["f1_tune"], r["pass_tune"]), ("", ""))
        self.assertIn("scoring failed: ValueError", r["notes"])
        self.assertLess(len(r["notes"]), 300)
        self.assertEqual(r["frames"], "200")

    def test_accuracy_failure_status_and_exit_code(self):
        bad = self.root / "bad_ref.jsonl"
        bad.write_text(json.dumps({"video": "a.parquet:0", "second": 0, "objects": ["dog"], "people": 3}) + chr(10),
                       encoding="utf-8")
        self.assertEqual(self.bench("--ref", f"tune={bad}"), 1)
        (r,) = self.rows()
        self.assertEqual((r["status"], r["pass_tune"]), ("fail_accuracy", "fail"))
        self.assertIsNone(bench.newest_clean_ok(self.rows(), "L"))  # never feeds extrapolation

    def test_partial_is_fail_accuracy(self):
        self.assertEqual(self.bench("--ref", f"tune={self.ref}", "--only-common"), 1)
        (r,) = self.rows()
        self.assertEqual((r["status"], r["pass_tune"]), ("fail_accuracy", "partial"))

    def test_missing_stats_leaves_blanks(self):
        self.assertEqual(self.bench(tail=["--no-stats"]), 0)
        (r,) = self.rows()
        self.assertEqual((r["status"], r["frames"], r["loop_s"], r["fps_loop"]), ("ok", "", "", ""))

    def test_failed_run(self):
        self.assertEqual(self.bench("--ref", f"tune={self.ref}", tail=["--fail"]), 7)
        (r,) = self.rows()
        self.assertEqual(r["status"], "failed")
        self.assertEqual(r["pass_tune"], "")
        self.assertIn("exit code 7", r["notes"])

    def test_rows_are_appended_with_one_header(self):
        self.bench()
        self.bench()
        with open(self.log, encoding="utf-8") as f:
            self.assertEqual(sum(1 for line in f if line.startswith("timestamp_utc")), 1)
        self.assertEqual(len(self.rows()), 2)

    def test_download_marks_run_unclean(self):
        self.bench(proc_scan=lambda roots, tree: ["99: hf download HuggingFaceFV/finevideo"])
        (r,) = self.rows()
        self.assertEqual(r["clean"], "no")
        self.assertIn("download running", r["clean_reason"])

    def test_foreign_gpu_process_marks_run_unclean(self):
        self.bench(gpu_scan=lambda roots, tree: ["pid=4242 cmd=python x.py"])
        (r,) = self.rows()
        self.assertEqual(r["clean"], "no")
        self.assertIn("other GPU process pid=4242", r["clean_reason"])

    def test_no_proc_is_unknown_not_clean(self):
        self.bench(proc_scan=lambda roots, tree: None)
        (r,) = self.rows()
        self.assertEqual((r["clean"], r["clean_reason"]), ("no", "unknown (no /proc)"))


class CleanDetectorTest(unittest.TestCase):

    def test_scan_gpu_users_with_fake_readlink(self):
        links = {"/p/10/fd/0": "/dev/null", "/p/20/fd/3": "/dev/nvidiactl", "/p/30/fd/4": "/dev/nvidia0",
                 "/p/40/fd/1": "/tmp/x"}
        dirs = {"/p/10/fd": ["0"], "/p/20/fd": ["3"], "/p/30/fd": ["4"], "/p/40/fd": ["1"], "/p/50/fd": None}
        cmds = {"20": "python other.py", "30": "bench child"}
        real = (bench.os.path.isdir, bench.os.listdir, bench.os.readlink, bench._read)

        def listdir(p):
            p = p.replace("\\", "/")
            if p == "/p":
                return ["10", "20", "30", "40", "50", "self"]
            v = dirs[p]
            if v is None:
                raise PermissionError(p)
            return v

        def readlink(p):
            return links[p.replace("\\", "/")]

        bench.os.path.isdir = lambda p: True
        bench.os.listdir = listdir
        bench.os.readlink = readlink
        bench._read = lambda p: cmds.get(p.replace("\\", "/").split("/")[2], "").replace(" ", "\0").encode()
        try:
            found = bench.scan_gpu_users({1}, tree={30}, proc_root="/p")
        finally:
            bench.os.path.isdir, bench.os.listdir, bench.os.readlink, bench._read = real
        self.assertEqual(found, ["pid=20 cmd=python other.py"])  # 30 is in the tree, 50 unreadable

    def test_scan_gpu_users_no_proc(self):
        self.assertIsNone(bench.scan_gpu_users({1}, proc_root="/definitely/not/here"))

    def test_unknown_gpu_scan_is_not_clean(self):
        m = bench.CleanMonitor({1}, lambda r, t: [], lambda r, t: None, 0.01, tree_fn=lambda r: None)
        m.start()
        m.stop()
        self.assertEqual(m.result(), ("no", "unknown (no /proc)"))

    def test_tree_computed_once_per_check_and_shared(self):
        seen = []
        m = bench.CleanMonitor({1}, lambda r, t: seen.append(("d", t)) or [], lambda r, t: seen.append(("g", t)) or [],
                               60, tree_fn=lambda r: {1, 2})
        m.check_once()
        self.assertEqual(seen, [("d", {1, 2}), ("g", {1, 2})])

    def make_proc(self, root, procs):
        for pid, ppid, cmd in procs:
            d = Path(root) / str(pid)
            d.mkdir()
            (d / "cmdline").write_bytes(cmd.replace(" ", "\0").encode())
            (d / "stat").write_text(f"{pid} (some cmd) S {ppid} 1 1 0\n")

    def test_scan_downloads_in_fake_proc(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.make_proc(tmp, [
                (10, 1, "bash"),
                (20, 1, "/content/venv/bin/hf download HuggingFaceFV/finevideo"),
                (30, 10, "python bench.py run"),
                (31, 30, "python run.py hf download-looking-arg"),  # inside bench's own tree: ignored
            ])
            found = bench.scan_downloads({30}, proc_root=tmp)
            self.assertEqual(len(found), 1)
            self.assertTrue(found[0].startswith("20:"))
            self.assertEqual(bench.proc_tree({30}, tmp), {30, 31})

    def test_no_proc_dir(self):
        self.assertIsNone(bench.scan_downloads({1}, proc_root="/definitely/not/here"))

    def test_clean_when_nothing_found(self):
        m = bench.CleanMonitor({1}, lambda r, t: [], lambda r, t: [], 0.01)
        m.start()
        m.stop()
        self.assertEqual(m.result()[0], "yes")

    def test_periodic_check_catches_late_download(self):
        calls = {"n": 0}

        def scan(roots, tree):
            calls["n"] += 1
            return [] if calls["n"] < 3 else ["5: snapshot_download"]

        m = bench.CleanMonitor({1}, scan, lambda r, t: [], 0.01)
        m.start()
        time.sleep(0.3)  # the thread checks every 0.01 s
        m.stop()
        self.assertEqual(m.result()[0], "no")

    def test_failing_detector_does_not_raise_and_is_unclean(self):
        def boom(roots, tree):
            raise OSError("x")

        m = bench.CleanMonitor({1}, boom, lambda r, t: [], 0.01)
        m.start()
        m.stop()
        self.assertEqual(m.result()[0], "no")


class ExtrapolateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.log = Path(self.tmp.name) / "log.csv"
        self.frames_csv = Path(self.tmp.name) / "fc.csv"
        self.frames_csv.write_text("file,row,video_id,frames,error\nf,0,f:0,600,\nf,1,f:1,400,\nf,2,f:2,,bad\n")

    def add(self, **kw):
        row = {"timestamp_utc": "2026-11-20T10:00:00Z", "label": "bf", "kind": "measured", "status": "ok",
               "clean": "yes", "gpu": "L4", "input": "tune", "frames": "1000", "wall_s": "120", "loop_s": "100"}
        row.update(kw)
        bench.append_row(self.log, row)

    def extrapolate(self, *extra):
        return quiet(bench.main, ["extrapolate", "--from-label", "bf", "--frames-csv", str(self.frames_csv),
                                  "--log", str(self.log), *extra])

    def test_math_and_row(self):
        self.add()
        self.assertEqual(self.extrapolate(), 0)
        rows = bench.read_rows(self.log)
        r = rows[-1]
        self.assertEqual((r["kind"], r["total_frames"]), ("extrapolated", "1000"))
        # per_frame 0.1 s, fixed 20 s, 1000 frames -> 20 + 100
        self.assertEqual(r["projected_s"], "120.0")
        self.assertEqual(r["projected_h"], "0.03")
        self.assertIn("extrapolated: measured per-frame cost", r["notes"])
        self.assertIn("grader uses RTX 4090", r["notes"])
        self.assertIn("L4", r["notes"])
        self.assertIn("2026-11-20T10:00:00Z", r["notes"])

    def test_project_function(self):
        per_frame, fixed, projected = bench.project({"loop_s": "100", "frames": "1000", "wall_s": "120"}, 50000)
        self.assertAlmostEqual(per_frame, 0.1)
        self.assertAlmostEqual(fixed, 20)
        self.assertAlmostEqual(projected, 5020)

    def test_picks_newest_clean_ok_measured_row(self):
        self.add(timestamp_utc="2026-11-20T09:00:00Z", loop_s="200", wall_s="220")  # older
        self.add(timestamp_utc="2026-11-20T11:00:00Z", loop_s="50", wall_s="60")  # newest clean ok: used
        self.add(timestamp_utc="2026-11-20T12:00:00Z", clean="no", loop_s="1", wall_s="2")
        self.add(timestamp_utc="2026-11-20T13:00:00Z", status="failed", loop_s="1", wall_s="2")
        self.add(timestamp_utc="2026-11-20T14:00:00Z", kind="extrapolated", loop_s="1", wall_s="2")
        self.add(timestamp_utc="2026-11-20T15:00:00Z", label="other", loop_s="1", wall_s="2")
        self.assertEqual(self.extrapolate(), 0)
        r = bench.read_rows(self.log)[-1]
        # per_frame 0.05, fixed 10, total 1000 -> 60
        self.assertEqual(r["projected_s"], "60.0")
        self.assertIn("2026-11-20T11:00:00Z", r["notes"])

    def test_no_eligible_row(self):
        self.add(clean="no")
        self.assertEqual(self.extrapolate(), 2)
        self.assertEqual(len(bench.read_rows(self.log)), 1)


if __name__ == "__main__":
    unittest.main()


class ProfileNoteTest(unittest.TestCase):
    def test_empty_and_full(self):
        self.assertEqual(bench.profile_note({}), "")
        note = bench.profile_note({
            "inferred_frames": 50, "sample": 5, "precision": "fp16", "decode_s": 2.0, "infer_s": 10.0,
            "model_speed": {"preprocess_ms": 100.0, "inference_ms": 300.0, "postprocess_ms": 50.0,
                            "predict_wall_s": 0.5, "predict_calls": 2, "images": 50},
        })
        self.assertIn("inferred=50", note)
        self.assertIn("sample=5", note)
        self.assertIn("precision=fp16", note)
        self.assertIn("pre=2.00 inf=6.00 post=1.00 other=1.00", note)  # (500 - 450) ms / 50 images
