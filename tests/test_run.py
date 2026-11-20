import json
import os
import sys
import unittest
from pathlib import Path
from fractions import Fraction as F
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run  # noqa: E402
from vidx.decode import Frame  # noqa: E402
from vidx.model import summarize  # noqa: E402


def frame(t, shape=(4, 6, 3), person=0):
    return Frame(SimpleNamespace(shape=shape, person=person), t, None, 0)


class FakeInfer:
    """Counts persons from the fake image; remembers each batch it was given."""

    def __init__(self, fail_on_call=None, oom_above=None):
        self.batches = []
        self.ok_batches = []
        self.fail_on_call = fail_on_call
        self.oom_above = oom_above  # raise a fake CUDA OOM for batches larger than this

    def __call__(self, images):
        self.batches.append([im.shape for im in images])
        if self.fail_on_call == len(self.batches):
            raise RuntimeError("some other model failure")
        if self.oom_above is not None and len(images) > self.oom_above:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        self.ok_batches.append(list(images))
        return [([im.person, 0, 0, 0, 0], [0.9, 0, 0, 0, 0]) for im in images]


class ProcessFramesTest(unittest.TestCase):
    def test_batching(self):
        fi = FakeInfer()
        frames = [frame(F(i, 10)) for i in range(7)]
        records, error = run.process_frames(iter(frames), fi, 3, run.Timers())
        self.assertIsNone(error)
        self.assertEqual(len(records), 7)
        self.assertEqual([len(b) for b in fi.batches], [3, 3, 1])

    def test_flush_on_shape_change(self):
        fi = FakeInfer()
        shapes = [(4, 6, 3)] * 2 + [(6, 4, 3)] * 2 + [(4, 6, 3)]
        frames = [frame(F(i, 10), shape=s) for i, s in enumerate(shapes)]
        records, _ = run.process_frames(iter(frames), fi, 32, run.Timers(), keep_details=True)
        self.assertEqual([len(b) for b in fi.batches], [2, 2, 1])
        self.assertEqual([r.shape for r in records], shapes)

    def test_decode_failure_keeps_partial_second(self):
        def gen():
            yield frame(F(0), person=1)
            yield frame(F(1), person=1)
            yield frame(F(21, 10), person=2)  # part of second 2
            raise ValueError("corrupt packet")

        records, error = run.process_frames(gen(), FakeInfer(), 32, run.Timers())
        self.assertIn("corrupt packet", error)
        self.assertEqual(len(records), 3)
        lines = run.video_lines("a.parquet:7", records, error)
        self.assertEqual([l.get("second") for l in lines[:-1]], [0, 1, 2])
        self.assertEqual(lines[2]["people"], 2)  # partial second 2 emitted
        self.assertEqual(lines[2]["objects"], ["person"])
        self.assertEqual(set(lines[-1]), {"video", "error"})

    def test_failure_before_any_frame(self):
        def gen():
            raise ValueError("moov atom not found")
            yield

        records, error = run.process_frames(gen(), FakeInfer(), 32, run.Timers())
        lines = run.video_lines("a.parquet:7", records, error)
        self.assertEqual(len(lines), 1)
        self.assertIn("error", lines[0])

    def test_zero_frames_without_exception_is_error(self):
        records, error = run.process_frames(iter([]), FakeInfer(), 32, run.Timers())
        lines = run.video_lines("v:0", records, error)
        self.assertEqual(lines, [{"video": "v:0", "error": "no frames decoded"}])

    def test_inference_failure_raises_inference_error(self):
        frames = [frame(F(i, 10)) for i in range(6)]
        with self.assertRaises(run.InferenceError):
            run.process_frames(iter(frames), FakeInfer(fail_on_call=2), 3, run.Timers())

    def test_oom_batch_is_split_and_retried(self):
        fi = FakeInfer(oom_above=2)
        frames = [frame(F(i, 10), person=i) for i in range(8)]
        records, error = run.process_frames(iter(frames), fi, 8, run.Timers())
        self.assertIsNone(error)
        self.assertEqual([r.counts[0] for r in records], list(range(8)))  # order kept
        self.assertEqual(max(len(b) for b in fi.ok_batches), 2)
        self.assertEqual(sum(len(b) for b in fi.ok_batches), 8)

    def test_oom_at_batch_size_one_is_inference_error(self):
        with self.assertRaises(run.InferenceError):
            run.process_frames(iter([frame(F(0))]), FakeInfer(oom_above=0), 8, run.Timers())

    def test_keep_details(self):
        frames = [frame(F(0)), frame(F(1, 2))]
        lite, _ = run.process_frames(iter(frames), FakeInfer(), 8, run.Timers())
        full, _ = run.process_frames(iter(frames), FakeInfer(), 8, run.Timers(), keep_details=True)
        self.assertIsNone(lite[0].conf)
        self.assertEqual(full[0].shape, (4, 6, 3))
        self.assertEqual(full[0].conf[0], 0.9)

    def test_clean_video_has_no_error_line(self):
        records, error = run.process_frames(iter([frame(F(0)), frame(F(1, 2))]), FakeInfer(), 8, run.Timers())
        lines = run.video_lines("v:1", records, error)
        self.assertEqual(lines, [{"video": "v:1", "second": 0, "objects": [], "people": 0}])


def make_args(**kw):
    return SimpleNamespace(**{"batch": 4, "dump_frames": None, "sample": 0, **kw})


def fake_decode(n=3, fail_after=None):
    def decode(data, sample=0, counter=None):
        for i in range(n):
            if fail_after == i:
                raise ValueError("corrupt packet")
            if counter is not None:
                counter.decoded_frames += 1
            yield frame(F(i, 2), person=1)
    return decode


class ProcessVideoTest(unittest.TestCase):
    def run_video(self, infer_fn, args=None, decode=None):
        import vidx.decode

        orig = vidx.decode.decode_video
        vidx.decode.decode_video = decode or fake_decode()
        try:
            return run.process_video("v:0", b"", infer_fn, args or make_args(), run.Timers())
        finally:
            vidx.decode.decode_video = orig

    def test_ok(self):
        lines = self.run_video(FakeInfer())
        self.assertEqual([l["second"] for l in lines], [0, 1])

    def test_inference_error_propagates_not_a_line(self):
        with self.assertRaises(run.InferenceError):
            self.run_video(FakeInfer(fail_on_call=1))

    def test_decode_error_keeps_lines_and_one_error(self):
        lines = self.run_video(FakeInfer(), decode=fake_decode(5, fail_after=3))
        self.assertEqual(len([l for l in lines if "error" in l]), 1)
        self.assertEqual(len([l for l in lines if "second" in l]), 2)

    def test_dump_failure_keeps_lines(self):
        orig = run.dump_frames

        def boom(*a):
            raise OSError("disk full")

        run.dump_frames = boom
        try:
            lines = self.run_video(FakeInfer(), args=make_args(dump_frames="somewhere"))
        finally:
            run.dump_frames = orig
        self.assertEqual([l["second"] for l in lines], [0, 1])


class MainAbortTest(unittest.TestCase):
    def test_three_inference_failures_in_a_row_abort_without_writing(self):
        import tempfile
        from pathlib import Path

        import vidx.decode
        import vidx.model
        import vidx.source

        saved = (vidx.decode.decode_video, vidx.model.load_model, vidx.model.infer, vidx.source.iter_rows)

        def bad_infer(model, images):
            raise RuntimeError("driver broke")

        vidx.decode.decode_video = fake_decode()
        vidx.model.load_model = lambda **kw: SimpleNamespace(device="cpu", half=False, speed={})
        vidx.model.infer = bad_infer
        vidx.source.iter_rows = lambda path: ((f"p:{i}", b"", None) for i in range(10))
        try:
            with tempfile.TemporaryDirectory() as tmp:
                (Path(tmp) / "a.parquet").touch()
                out = Path(tmp) / "r.jsonl"
                rc = run.main(["--input", tmp, "--output", str(out)])
                self.assertEqual(rc, 3)
                self.assertEqual(out.read_text(), "")
                self.assertEqual(Path(str(out) + ".done").read_text(), "")
        finally:
            (vidx.decode.decode_video, vidx.model.load_model, vidx.model.infer, vidx.source.iter_rows) = saved


def fake_decode_by_data(data, sample=0, counter=None):
    """Deterministic fake video: data is the video id as bytes; person count depends on it."""
    n = 2 + len(data) % 3
    for i in range(n):
        if counter is not None:
            counter.decoded_frames += 1
        yield frame(F(i, 2), person=(len(data) + i) % 4)


class FakeRun:
    """Patches the heavy modules so run.main() works on fake videos; restores them afterwards."""

    def __init__(self, n_videos=8, crash_after=None, bad_infer=False, bad_calls=()):
        import tempfile

        self.n, self.crash_after, self.bad_infer = n_videos, crash_after, bad_infer
        self.bad_calls, self.calls = set(bad_calls), 0  # infer call numbers that fail (one call per video)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "in").mkdir()
        (self.root / "in" / "a.parquet").write_bytes(b"x" * 10)

    def rows(self, path):
        for i in range(self.n):
            if self.crash_after is not None and i >= self.crash_after:
                raise KeyboardInterrupt  # stands in for a kill; `finally` still runs
            yield f"a.parquet:{i}", f"video{i * 7}".encode(), None

    def __enter__(self):
        import vidx.decode
        import vidx.model
        import vidx.source

        self.mods = (vidx.decode, vidx.model, vidx.source)
        self.saved = (vidx.decode.decode_video, vidx.model.load_model, vidx.model.infer, vidx.source.iter_rows)
        vidx.decode.decode_video = fake_decode_by_data
        vidx.model.load_model = lambda **kw: SimpleNamespace(device="cpu", half=False, speed={})
        vidx.model.infer = self.infer
        vidx.source.iter_rows = self.rows
        return self

    def infer(self, model, images):
        self.calls += 1
        if self.bad_infer or self.calls - 1 in self.bad_calls:
            raise RuntimeError("driver broke")
        return FakeInfer()(images)

    def __exit__(self, *exc):
        d, m, s = self.mods
        d.decode_video, m.load_model, m.infer, s.iter_rows = self.saved
        self.tmp.cleanup()

    def main(self, out, *extra):
        """These tests exercise the sequential path (one infer call per video); the threaded pipeline (the
        default) is covered by tests/test_pipeline.py, which passes --decode-threads explicitly."""
        if "--decode-threads" not in extra:
            extra = (*extra, "--decode-threads", "0")
        return run.main(["--input", str(self.root / "in"), "--output", str(self.root / out), *extra])

    def text(self, out):
        return (self.root / out).read_text(encoding="utf-8")


class PendingInferenceFailureTest(unittest.TestCase):
    def error_videos(self, fr, out="r.jsonl"):
        rows = [json.loads(x) for x in fr.text(out).splitlines()]
        return {r["video"]: r["error"] for r in rows if "error" in r}

    def test_bad_sample_options_rejected(self):
        with FakeRun(n_videos=2) as fr:
            self.assertEqual(fr.main("r.jsonl", "--sample", "-1"), 2)
            self.assertEqual(fr.main("r2.jsonl", "--sample", "3", "--dump-frames", str(fr.root / "d")), 2)

    def test_single_failure_followed_by_success_is_written_as_error_line_and_done(self):
        with FakeRun(n_videos=4, bad_calls={1}) as fr:
            self.assertEqual(fr.main("r.jsonl"), 0)
            errors = self.error_videos(fr)
            self.assertEqual(list(errors), ["a.parquet:1"])
            self.assertTrue(errors["a.parquet:1"].startswith("inference failed: "))
            done = (fr.root / "r.jsonl.done").read_text().splitlines()
            self.assertEqual(len(done), 4)
            self.assertIn("a.parquet:1	1", done)

    def test_failure_at_the_end_of_the_run_is_written_as_error_line(self):
        with FakeRun(n_videos=3, bad_calls={1, 2}) as fr:
            self.assertEqual(fr.main("r.jsonl"), 0)
            self.assertEqual(sorted(self.error_videos(fr)), ["a.parquet:1", "a.parquet:2"])
            self.assertEqual(len((fr.root / "r.jsonl.done").read_text().splitlines()), 3)

    def test_success_resets_the_counter(self):
        with FakeRun(n_videos=7, bad_calls={0, 1, 3, 4}) as fr:  # never 3 in a row
            self.assertEqual(fr.main("r.jsonl"), 0)
            self.assertEqual(len(self.error_videos(fr)), 4)

    def test_abort_leaves_pending_unwritten_so_resume_retries_them(self):
        with FakeRun(n_videos=6, bad_calls={1, 2, 3}) as fr:
            self.assertEqual(fr.main("r.jsonl"), 3)
            self.assertEqual(self.error_videos(fr), {})
            self.assertEqual((fr.root / "r.jsonl.done").read_text().splitlines(), ["a.parquet:0	1"])
            fr.bad_calls = set()
            self.assertEqual(fr.main("r.jsonl", "--resume"), 0)
            self.assertEqual(self.error_videos(fr), {})
            self.assertEqual(len((fr.root / "r.jsonl.done").read_text().splitlines()), 6)


class CheckpointMainTest(unittest.TestCase):
    def test_plain_run_does_not_import_checkpoint_module_or_write_extra_files(self):
        sys.modules.pop("vidx.checkpoint", None)
        with FakeRun() as fr:
            self.assertEqual(fr.main("r.jsonl"), 0)
            self.assertNotIn("vidx.checkpoint", sys.modules)
            self.assertEqual(sorted(p.name for p in fr.root.iterdir()), ["in", "r.jsonl", "r.jsonl.config.json", "r.jsonl.done"])

    def test_fresh_run_against_existing_checkpoint_exits_2(self):
        with FakeRun() as fr:
            ck = str(fr.root / "drive")
            self.assertEqual(fr.main("r.jsonl", "--checkpoint-dir", ck), 0)
            before = fr.text("r.jsonl")
            self.assertEqual(fr.main("r.jsonl", "--checkpoint-dir", ck), 2)
            self.assertEqual(fr.text("r.jsonl"), before)  # not overwritten
            self.assertEqual(fr.main("r.jsonl", "--checkpoint-dir", ck, "--run-name", "other"), 0)

    def test_resume_from_checkpoint_after_losing_local_files_matches_uninterrupted_run(self):
        # Parquet fixture: .venv has no pyarrow, so iter_rows and decode are faked (see FakeRun).
        with FakeRun(n_videos=8) as ref:
            self.assertEqual(ref.main("ref.jsonl"), 0)
            expected = sorted(ref.text("ref.jsonl").splitlines())
        with FakeRun(n_videos=8, crash_after=5) as fr:
            ck = str(fr.root / "drive")
            with self.assertRaises(KeyboardInterrupt):
                fr.main("r.jsonl", "--checkpoint-dir", ck, "--checkpoint-minutes", "0")
            (fr.root / "r.jsonl").unlink()  # runtime reset: local disk is gone
            (fr.root / "r.jsonl.done").unlink()
            fr.crash_after = None
            self.assertEqual(fr.main("r.jsonl", "--checkpoint-dir", ck, "--resume"), 0)
            self.assertEqual(sorted(fr.text("r.jsonl").splitlines()), expected)
            done = (fr.root / "r.jsonl.done").read_text().splitlines()
            self.assertEqual(len(done), 8)

    def test_resume_survives_new_parquet_files_but_refuses_a_changed_processed_one(self):
        with FakeRun(n_videos=4) as fr:
            ck = str(fr.root / "drive")
            self.assertEqual(fr.main("r.jsonl", "--checkpoint-dir", ck), 0)
            (fr.root / "in" / "b.parquet").write_bytes(b"y")  # downloaded later: fine
            self.assertEqual(fr.main("r.jsonl", "--checkpoint-dir", ck, "--resume"), 0)
            (fr.root / "in" / "a.parquet").write_bytes(b"x" * 11)  # processed file changed size: refused
            self.assertEqual(fr.main("r.jsonl", "--checkpoint-dir", ck, "--resume"), 2)
            (fr.root / "in" / "a.parquet").unlink()  # processed file gone: refused
            self.assertEqual(fr.main("r.jsonl", "--checkpoint-dir", ck, "--resume"), 2)

    def test_checkpoint_dir_with_missing_parent_exits_2_and_creates_nothing(self):
        with FakeRun() as fr:
            ck = fr.root / "nodrive" / "MyDrive" / "ckpt"
            self.assertEqual(fr.main("r.jsonl", "--checkpoint-dir", str(ck)), 2)
            self.assertEqual(fr.main("r.jsonl", "--checkpoint-dir", str(ck), "--resume"), 2)
            self.assertFalse((fr.root / "nodrive").exists())

    def test_snapshot_oserror_keeps_run_and_exit_code(self):
        import vidx.checkpoint as cp

        orig = cp.DirStore.publish

        def boom(self, staging, n):
            raise OSError("Drive disconnected")

        cp.DirStore.publish = boom
        try:
            with FakeRun() as fr:
                ck = str(fr.root / "drive")
                self.assertEqual(fr.main("r.jsonl", "--checkpoint-dir", ck, "--checkpoint-minutes", "0"), 0)
                self.assertEqual(len((fr.root / "r.jsonl.done").read_text().splitlines()), 8)
            with FakeRun(bad_infer=True) as fr:
                ck = str(fr.root / "drive")
                self.assertEqual(fr.main("r.jsonl", "--checkpoint-dir", ck), 3)  # abort code preserved
        finally:
            cp.DirStore.publish = orig

    def test_npz_temp_name_ends_with_npz_and_differs_from_final(self):
        tmp, final = run.npz_paths("d", "a.parquet:3")
        self.assertEqual((tmp.name, final.name), ("a.parquet__3.tmp.npz", "a.parquet__3.npz"))
        try:
            import numpy as np
        except ImportError:
            return
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            rec = run.Record(F(0), [1, 0, 0, 0, 0], [0.9] * 5, 0, 0, (4, 6, 3))
            run.dump_frames(d, "a.parquet:3", [rec])
            self.assertEqual(sorted(p.name for p in Path(d).iterdir()), ["a.parquet__3.npz"])
            self.assertEqual(np.load(Path(d) / final.name)["counts"].tolist(), [[1, 0, 0, 0, 0]])


class StatsJsonTest(unittest.TestCase):
    KEYS = {"videos", "errors", "frames", "inferred_frames", "decode_s", "infer_s", "aggregate_s", "loop_s", "startup_s",
            "sample", "precision", "model_speed", "decode_threads",
            "exit_code", "argv"}

    def stats(self, fr, name="s.json"):
        return json.loads((fr.root / name).read_text(encoding="utf-8"))

    def test_keys_present_after_normal_run(self):
        with FakeRun(n_videos=3) as fr:
            path = str(fr.root / "s.json")
            self.assertEqual(fr.main("r.jsonl", "--stats-json", path), 0)
            d = self.stats(fr)
        self.assertEqual(set(d), self.KEYS)
        self.assertEqual((d["videos"], d["errors"], d["exit_code"]), (3, 0, 0))
        self.assertGreater(d["frames"], 0)
        self.assertGreaterEqual(d["loop_s"], 0)
        self.assertGreaterEqual(d["startup_s"], 0)
        self.assertIn("--stats-json", d["argv"])

    def test_written_when_run_aborts(self):
        with FakeRun(n_videos=6, bad_infer=True) as fr:
            path = str(fr.root / "s.json")
            self.assertEqual(fr.main("r.jsonl", "--stats-json", path), 3)
            d = self.stats(fr)
        self.assertEqual(set(d), self.KEYS)
        self.assertEqual(d["exit_code"], 3)
        self.assertIsNotNone(d["loop_s"])

    def test_written_when_run_is_interrupted(self):
        with FakeRun(n_videos=6, crash_after=2) as fr:
            path = str(fr.root / "s.json")
            with self.assertRaises(KeyboardInterrupt):
                fr.main("r.jsonl", "--stats-json", path)
            d = self.stats(fr)
        self.assertEqual(set(d), self.KEYS)
        self.assertEqual(d["videos"], 2)
        self.assertEqual(d["exit_code"], 1)

    def test_written_on_early_exit(self):
        with FakeRun() as fr:
            for p in (fr.root / "in").glob("*.parquet"):
                p.unlink()
            path = str(fr.root / "s.json")
            self.assertEqual(fr.main("r.jsonl", "--stats-json", path), 2)
            d = self.stats(fr)
        self.assertEqual((d["exit_code"], d["videos"], d["loop_s"]), (2, 0, None))

    def test_sigterm_handler_installed_with_stats_json_only(self):
        import signal

        old = signal.getsignal(signal.SIGTERM)
        try:
            with FakeRun(n_videos=1) as fr:
                self.assertEqual(fr.main("r.jsonl"), 0)
                self.assertIs(signal.getsignal(signal.SIGTERM), old)  # plain command: untouched
                self.assertEqual(fr.main("r2.jsonl", "--stats-json", str(fr.root / "s.json")), 0)
                handler = signal.getsignal(signal.SIGTERM)
                self.assertIsNot(handler, old)
                with self.assertRaises(SystemExit) as cm:
                    handler(signal.SIGTERM, None)
                self.assertEqual(cm.exception.code, 128 + signal.SIGTERM)
        finally:
            signal.signal(signal.SIGTERM, old)

    def test_off_by_default(self):
        with FakeRun() as fr:
            self.assertEqual(fr.main("r.jsonl"), 0)
            self.assertEqual(sorted(p.name for p in fr.root.iterdir()), ["in", "r.jsonl", "r.jsonl.config.json", "r.jsonl.done"])


class SummarizeTest(unittest.TestCase):
    def test_summarize(self):
        counts, best = summarize([0, 0, 2, 67, 5], [0.5, 0.9, 0.6, 0.41, 0.99])
        self.assertEqual(counts, [2, 1, 0, 0, 1])  # class 5 (bus) ignored
        self.assertEqual(best, [0.9, 0.6, 0.0, 0.0, 0.41])


class RunConfigResumeTest(unittest.TestCase):
    """<output>.config.json: --resume never mixes sampling / precision settings in one output."""

    def test_resume_with_other_settings_refused(self):
        with FakeRun(n_videos=4) as fr:
            self.assertEqual(fr.main("r.jsonl", "--limit-videos", "2", "--sample", "3"), 0)
            self.assertEqual(fr.main("r.jsonl", "--resume", "--sample", "5"), 2)
            self.assertEqual(fr.main("r.jsonl", "--resume", "--sample", "3"), 0)

    def test_legacy_output_counts_as_brute_force(self):
        with FakeRun(n_videos=4) as fr:
            self.assertEqual(fr.main("r.jsonl", "--limit-videos", "2", "--sample", "0", "--precision", "fp32"), 0)
            (fr.root / "r.jsonl.config.json").unlink()  # as written before the config file existed
            self.assertEqual(fr.main("r.jsonl", "--resume"), 2)  # new defaults (k=3 fp16) differ
            self.assertEqual(fr.main("r.jsonl", "--resume", "--sample", "0", "--precision", "fp32"), 0)


if __name__ == "__main__":
    unittest.main()
