import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from fractions import Fraction as F
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import numpy  # noqa: F401
except ImportError:  # pragma: no cover
    numpy = None

import run  # noqa: E402
from tools import simulate_sampling as sim  # noqa: E402
from vidx.aggregate import aggregate, make_lines  # noqa: E402


def records(n_frames, fps, person_fn):
    return [run.Record(F(i, fps), [person_fn(i), 0, 0, 0, 0], [0.9, 0, 0, 0, 0], i, 0, (4, 6, 3))
            for i in range(n_frames)]


@unittest.skipIf(numpy is None, "numpy not installed")
class SimulateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dumps = self.root / "frames"
        vids = {
            "f.parquet:0": records(90, 30, lambda i: 1),  # steady: any k is exact
            "f.parquet:1": records(60, 30, lambda i: 2 if i % 2 else 1),  # flickers every frame
        }
        lines = []
        for vid, recs in vids.items():
            run.dump_frames(self.dumps, vid, recs)
            lines += make_lines(vid, aggregate((r.t, r.counts) for r in recs))
        lines.append({"video": "f.parquet:2", "error": "could not decode video: x"})
        self.ref = self.root / "ref.jsonl"
        self.ref.write_text("".join(json.dumps(l) + "\n" for l in lines), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_video_id_roundtrip(self):
        _, final = run.npz_paths(self.dumps, "train-00000-of-01357.parquet:12")
        self.assertEqual(sim.video_id_of(final), "train-00000-of-01357.parquet:12")

    def test_every_frame_reproduces_reference_and_k_reduces_frames(self):
        out = io.StringIO()
        with redirect_stdout(out):
            rc = sim.main(["--dumps", str(self.dumps), "--ref", str(self.ref), "--k", "1,3",
                           "--adaptive", "1:5", "--json", str(self.root / "r.json")])
        self.assertEqual(rc, 0, out.getvalue())
        res = json.loads((self.root / "r.json").read_text(encoding="utf-8"))
        base, k1, k3, ad = res
        self.assertEqual((base["f1"], base["exact"], base["within1"]), (1.0, 1.0, 1.0))
        self.assertEqual(base["inferred"], 150)
        self.assertEqual(k1["inferred"], 5)  # one frame per second, 5 seconds
        self.assertEqual(k3["inferred"], 15)
        self.assertEqual(ad["inferred"], 5)  # k0=1: nothing to disagree with -> no refinement

    def test_disagree(self):
        self.assertFalse(sim.disagree([(1, 0, 0, 0, 0), (1, 0, 0, 0, 0)]))
        self.assertTrue(sim.disagree([(1, 0, 0, 0, 0), (2, 0, 0, 0, 0)]))
        self.assertTrue(sim.disagree([(1, 0, 0, 0, 0), (1, 1, 0, 0, 0)]))
        self.assertFalse(sim.disagree([(1, 2, 0, 0, 0), (1, 1, 0, 0, 0)]))  # same presence, same people


if __name__ == "__main__":
    unittest.main()
