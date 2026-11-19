import csv
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import count_frames as cf  # noqa: E402
import run  # noqa: E402


def pkt(size=10, discard=False):
    return SimpleNamespace(size=size, is_discard=discard)


class CountPacketsTest(unittest.TestCase):
    def test_skips_empty_and_discard(self):
        self.assertEqual(cf.count_packets([pkt(), pkt(0), pkt(5, True), pkt(7)]), (2, None))

    def test_exception_keeps_partial_count(self):
        def gen():
            yield pkt()
            yield pkt()
            raise RuntimeError("truncated")

        n, error = cf.count_packets(gen())
        self.assertEqual(n, 2)
        self.assertIn("truncated", error)

    def test_packet_without_is_discard_attribute(self):
        self.assertEqual(cf.count_packets([SimpleNamespace(size=3)]), (1, None))


def fake_demux(data):
    n, w, h = {b"a": (10, 640, 360), b"b": (30, 1280, 720), b"c": (60, 1280, 720)}[bytes(data)]
    return {"frames": n, "duration_s": n / 10, "width": w, "height": h, "codec": "h264", "error": None}


def fake_rows(path):
    yield "f.parquet:0", b"a", None
    yield "f.parquet:1", b"b", None
    yield "f.parquet:2", None, "null mp4"
    yield "f.parquet:3", b"zzz", None  # demux raises KeyError
    yield "f.parquet:4", b"c", None


class FileProcessingTest(unittest.TestCase):
    def setUp(self):
        self.rows = cf.process_file("f.parquet", demux_fn=fake_demux, rows_fn=fake_rows)

    def test_rows_and_errors(self):
        self.assertEqual([r["frames"] for r in self.rows], [10, 30, None, None, 60])
        self.assertEqual(self.rows[0]["file"], "f.parquet")
        self.assertEqual(self.rows[0]["row"], "0")
        self.assertEqual(self.rows[2]["error"], "null mp4")
        self.assertIn("KeyError", self.rows[3]["error"])

    def test_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "sub", "c.csv")
            cf.write_csv(self.rows, out)
            with open(out, newline="", encoding="utf-8") as fh:
                got = list(csv.DictReader(fh))
        self.assertEqual(list(got[0]), cf.COLUMNS)
        self.assertEqual(got[1]["video_id"], "f.parquet:1")
        self.assertEqual(got[1]["frames"], "30")
        self.assertEqual(got[2]["frames"], "")

    def test_totals(self):
        t = cf.totals(self.rows)
        self.assertEqual((t["videos"], t["errors"], t["frames"]), (5, 2, 100))
        self.assertAlmostEqual(t["hours"], 10.0 / 3600)  # (1 + 3 + 6) s
        self.assertAlmostEqual(t["resolution_share"]["1280x720"], 0.9)
        self.assertAlmostEqual(t["resolution_share"]["640x360"], 0.1)

    def test_compare_dumps(self):
        lengths = {"f.parquet:0": 10, "f.parquet:1": 28}  # row 4 has no dump
        by_name = {run.npz_paths(".", vid)[1].name: n for vid, n in lengths.items()}
        with tempfile.TemporaryDirectory() as tmp:
            for vid in lengths:
                run.npz_paths(tmp, vid)[1].write_bytes(b"x")
            c = cf.compare_dumps(self.rows, tmp, npz_len=lambda p: by_name[p.name])
        self.assertEqual((c["compared"], c["missing_dumps"]), (2, 1))
        self.assertEqual((c["largest_gap"], c["largest_gap_video"]), (2, "f.parquet:1"))
        self.assertAlmostEqual(c["total_gap_pct"], 100 * 2 / 38)


class FailingFileTest(unittest.TestCase):
    def test_worker_error_row_and_csv_still_written(self):
        def fake_process(path):
            if "bad" in str(path):
                raise RuntimeError("metadata exploded")
            return cf.process_file(path, demux_fn=fake_demux, rows_fn=fake_rows)

        rows = []
        for p in ("good.parquet", "bad.parquet"):
            rows += cf.safe_process_file(p, fake_process)
        self.assertEqual(len(rows), 6)
        last = rows[-1]
        self.assertEqual((last["file"], last["video_id"]), ("bad.parquet", "bad.parquet:0"))
        self.assertIn("metadata exploded", last["error"])
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "c.csv")
            cf.write_csv(rows, out)
            with open(out, newline="", encoding="utf-8") as fh:
                got = list(csv.DictReader(fh))
        self.assertEqual(len(got), 6)
        self.assertEqual(cf.totals(rows)["errors"], 3)

    def test_main_writes_csv_when_a_file_fails(self):
        import io
        from contextlib import redirect_stdout

        with tempfile.TemporaryDirectory() as tmp:
            (os.path.join(tmp, "in") and os.mkdir(os.path.join(tmp, "in")))
            for n in ("a.parquet", "b.parquet"):
                open(os.path.join(tmp, "in", n), "wb").close()  # not real parquet: iter_rows yields an error row
            out = os.path.join(tmp, "o.csv")
            saved = cf.process_file

            def boom(path, **kw):
                raise OSError("disk gone")

            cf.process_file = boom
            try:
                buf = io.StringIO()
                with redirect_stdout(buf):
                    rc = cf.main(["--input", os.path.join(tmp, "in"), "--out", out, "--workers", "1"])
            finally:
                cf.process_file = saved
            with open(out, newline="", encoding="utf-8") as fh:
                got = list(csv.DictReader(fh))
        self.assertEqual(rc, 0)
        self.assertEqual(len(got), 2)
        self.assertIn("errors 2", buf.getvalue())

    def test_broken_pool_retries_serially(self):
        from concurrent.futures.process import BrokenProcessPool

        class DeadPool:
            def __init__(self, n):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def submit(self, fn, f):
                from concurrent.futures import Future

                fut = Future()
                fut.set_exception(BrokenProcessPool("died"))
                return fut

        saved = cf.ProcessPoolExecutor
        cf.ProcessPoolExecutor = DeadPool
        try:
            import contextlib
            import io

            with contextlib.redirect_stderr(io.StringIO()):
                rows = cf.process_all(["x.parquet", "y.parquet"], 2, worker_fn=lambda f: [{"file": f, "error": None}])
        finally:
            cf.ProcessPoolExecutor = saved
        self.assertEqual([r["file"] for r in rows], ["x.parquet", "y.parquet"])


if __name__ == "__main__":
    unittest.main()
