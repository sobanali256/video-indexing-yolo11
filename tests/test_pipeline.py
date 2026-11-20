"""vidx.pipeline (Stage 2a) with fake decode / letterbox / inference, and run.main --decode-threads end to end."""
import json
import os
import sys
import unittest
from fractions import Fraction as F
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import run  # noqa: E402
from vidx.pipeline import Pipeline  # noqa: E402
from tests.test_run import FakeInfer, FakeRun, fake_decode_by_data, frame  # noqa: E402


def make_decode(videos):
    """videos: {data: (n_frames, fail_after or None, shape)}. Fake decode keyed by the data bytes."""

    def decode(data, counter):
        n, fail_after, shape = videos[data]
        for i in range(n):
            if fail_after is not None and i == fail_after:
                raise ValueError("corrupt packet")
            counter.decoded_frames += 1
            yield frame(F(i, 2), shape=shape, person=(len(data) + i) % 3)

    return decode


def identity_prepare():
    return lambda image: image


def collect(pipe):
    return {vid: (kind, payload, batch) for kind, vid, payload, batch in pipe.run()}


class PipelineTest(unittest.TestCase):
    def make(self, rows, videos, infer=None, batch=4, threads=2, idle_wait=0.01):
        timers = run.Timers()
        pipe = Pipeline(iter(rows), make_decode(videos), identity_prepare, infer or FakeInfer(), batch, threads,
                        timers, run.run_inference, run.video_lines, idle_wait=idle_wait)
        return pipe, timers

    def expected(self, vid, data, n, fail_after=None):
        """What the sequential path produces for the same fake frames."""
        frames = [frame(F(i, 2), person=(len(data) + i) % 3) for i in range(n if fail_after is None else fail_after)]
        records = [SimpleNamespace(t=f.t, counts=[f.image.person, 0, 0, 0, 0]) for f in frames]
        err = "ValueError: corrupt packet" if fail_after is not None else None
        return run.video_lines(vid, records, err)

    def test_same_lines_as_sequential_and_counts(self):
        videos = {b"a": (7, None, (4, 6, 3)), b"bb": (5, None, (4, 6, 3)), b"ccc": (9, None, (6, 4, 3))}
        rows = [("p:0", b"a", None), ("p:1", b"bb", None), ("p:2", b"ccc", None)]
        pipe, timers = self.make(rows, videos)
        out = collect(pipe)
        self.assertEqual(set(out), {"p:0", "p:1", "p:2"})
        for vid, data, _ in rows:
            kind, lines, _ = out[vid]
            self.assertEqual(kind, "ok")
            self.assertEqual(lines, self.expected(vid, data, videos[data][0]))
        self.assertEqual((timers.decoded_frames, timers.inferred_frames), (21, 21))

    def test_batches_mix_videos_and_respect_shape(self):
        fi = FakeInfer()
        videos = {b"a": (3, None, (4, 6, 3)), b"bb": (3, None, (4, 6, 3)), b"ccc": (2, None, (6, 4, 3))}
        rows = [("p:0", b"a", None), ("p:1", b"bb", None), ("p:2", b"ccc", None)]
        pipe, _ = self.make(rows, videos, infer=fi, batch=4, threads=1, idle_wait=5)
        collect(pipe)
        self.assertIn(4, [len(b) for b in fi.batches])  # 3 frames of a + 1 of bb share a batch
        for b in fi.batches:
            self.assertEqual(len(set(b)), 1)  # never two shapes in one batch

    def test_decode_failure_keeps_frames_and_adds_error(self):
        videos = {b"a": (6, 3, (4, 6, 3)), b"bb": (4, None, (4, 6, 3))}
        rows = [("p:0", b"a", None), ("p:1", b"bb", None)]
        out = collect(self.make(rows, videos)[0])
        kind, lines, _ = out["p:0"]
        self.assertEqual(kind, "ok")
        self.assertEqual(lines, self.expected("p:0", b"a", 6, fail_after=3))
        self.assertIn("corrupt packet", lines[-1]["error"])

    def test_source_error_and_empty_video(self):
        videos = {b"e": (0, None, (4, 6, 3))}
        rows = [("p:0", None, "could not read row: X"), ("p:1", b"e", None)]
        out = collect(self.make(rows, videos)[0])
        self.assertEqual(out["p:0"][0], "source_error")
        self.assertEqual(out["p:0"][1], [{"video": "p:0", "error": "could not read row: X"}])
        self.assertEqual(out["p:1"][1], [{"video": "p:1", "error": "no frames decoded"}])

    def test_inference_failure_marks_videos_in_the_batch(self):
        fi = FakeInfer(fail_on_call=1)
        videos = {b"a": (2, None, (4, 6, 3)), b"bb": (2, None, (4, 6, 3)), b"ccc": (4, None, (4, 6, 3))}
        rows = [("p:0", b"a", None), ("p:1", b"bb", None), ("p:2", b"ccc", None)]
        out = collect(self.make(rows, videos, infer=fi, batch=4, threads=1, idle_wait=5)[0])
        self.assertEqual(out["p:0"][0], "fail")
        self.assertEqual(out["p:1"][0], "fail")
        self.assertEqual(out["p:0"][2], out["p:1"][2])  # same failed batch
        self.assertEqual(out["p:2"][0], "ok")

    def test_early_close_stops_threads(self):
        videos = {bytes([i]) * (i + 1): (50, None, (4, 6, 3)) for i in range(20)}
        rows = [(f"p:{i}", bytes([i]) * (i + 1), None) for i in range(20)]
        pipe, _ = self.make(rows, videos, batch=2, threads=3)
        gen = pipe.run()
        next(gen)
        gen.close()
        self.assertTrue(pipe.stop.is_set())


class PipelineFailureTest(unittest.TestCase):
    def make(self, rows, make_prepare=identity_prepare):
        videos = {b"a": (3, None, (4, 6, 3)), b"bb": (3, None, (4, 6, 3))}
        return Pipeline(iter(rows) if isinstance(rows, list) else rows, make_decode(videos), make_prepare,
                        FakeInfer(), 4, 2, run.Timers(), run.run_inference, run.video_lines, idle_wait=0.01)

    def test_letterbox_setup_failure_ends_instead_of_hanging(self):
        def broken():
            raise KeyError("direct")

        pipe = self.make([("p:0", b"a", None)], make_prepare=broken)
        with self.assertRaises(RuntimeError):
            list(pipe.run())

    def test_iterator_error_is_raised_not_swallowed(self):
        def rows():
            yield ("p:0", b"a", None)
            raise OSError("parquet read failed")

        pipe = self.make(rows())
        with self.assertRaises(OSError):
            list(pipe.run())

    def test_ok_carries_last_batch(self):
        out = list(self.make([("p:0", b"a", None), ("p:1", b"bb", None)]).run())
        self.assertTrue(all(kind == "ok" and batch >= 1 for kind, _, _, batch in out))


class RunMainPipelineTest(unittest.TestCase):
    """run.main with --decode-threads: same output as the sequential path on the same fake videos."""

    def patch_direct(self, fr):
        import vidx.model

        saved = (vidx.model.prepare_direct, vidx.model.make_letterbox, vidx.model.infer_tensor)
        vidx.model.prepare_direct = lambda model: ()
        vidx.model.make_letterbox = lambda model: (lambda image: image)
        vidx.model.infer_tensor = lambda model, images: fr.infer(model, images)
        return saved

    def restore(self, saved):
        import vidx.model

        vidx.model.prepare_direct, vidx.model.make_letterbox, vidx.model.infer_tensor = saved

    def lines(self, fr, out):
        return sorted(fr.text(out).splitlines())

    def test_same_output_as_sequential(self):
        with FakeRun(n_videos=9) as fr:
            saved = self.patch_direct(fr)
            try:
                self.assertEqual(fr.main("seq.jsonl"), 0)
                self.assertEqual(fr.main("pipe.jsonl", "--decode-threads", "2", "--batch", "3"), 0)
            finally:
                self.restore(saved)
            self.assertEqual(self.lines(fr, "seq.jsonl"), self.lines(fr, "pipe.jsonl"))
            done = (fr.root / "pipe.jsonl.done").read_text(encoding="utf-8").split()
            self.assertEqual(len([d for d in done if d.startswith("a.parquet:")]), 9)

    def test_resume_skips_done_videos(self):
        with FakeRun(n_videos=6) as fr:
            saved = self.patch_direct(fr)
            try:
                self.assertEqual(fr.main("r.jsonl", "--decode-threads", "2", "--limit-videos", "3"), 0)
                self.assertEqual(fr.main("r.jsonl", "--decode-threads", "2", "--resume"), 0)
            finally:
                self.restore(saved)
            vids = [json.loads(x)["video"] for x in fr.text("r.jsonl").splitlines()]
            self.assertEqual(sorted(set(vids)), [f"a.parquet:{i}" for i in range(6)])

    def test_dump_frames_rejected_with_threads(self):
        with FakeRun(n_videos=2) as fr:
            self.assertEqual(fr.main("r.jsonl", "--decode-threads", "2", "--dump-frames", str(fr.root / "d")), 2)


if __name__ == "__main__":
    unittest.main()
