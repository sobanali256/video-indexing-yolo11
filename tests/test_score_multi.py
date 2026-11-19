import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import score  # noqa: E402


def sec(video, s, objects=(), people=0):
    return {"video": video, "second": s, "objects": list(objects), "people": people}


def err(video):
    return {"video": video, "error": "bad"}


class Files:
    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def write(self, name, lines):
        p = self.root / name
        p.write_text("".join(json.dumps(x) + "\n" for x in lines), encoding="utf-8")
        return str(p)


def run_main(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        rc = score.main(list(argv))
    return rc, out.getvalue()


class MultiReferenceTest(unittest.TestCase):
    def setUp(self):
        self.f = Files()
        self.addCleanup(self.f.tmp.cleanup)
        self.tune = self.f.write("tune.jsonl", [sec("a.parquet:0", 0, ["car"], 1), sec("a.parquet:1", 0, [], 0)])
        # holdout file b appears in seconds, file c ONLY in an error line
        self.hold = self.f.write("hold.jsonl", [sec("b.parquet:0", 0, ["dog"], 2), err("c.parquet:3")])

    def test_filtering_across_references_with_error_only_file(self):
        cand = self.f.write("cand.jsonl", [
            sec("a.parquet:0", 0, ["car"], 1), sec("a.parquet:1", 0, [], 0),
            sec("b.parquet:0", 0, ["dog"], 2), err("c.parquet:3"),
            sec("z.parquet:0", 0, ["person"], 5),  # in no reference
        ])
        results, ok = score.score_files([("tune", self.tune), ("holdout", self.hold)], cand, out=io.StringIO())
        self.assertTrue(ok)
        by = {r["name"]: r for r in results}
        self.assertEqual(by["tune"]["files"], ["a.parquet"])
        self.assertEqual(by["holdout"]["files"], ["b.parquet", "c.parquet"])
        self.assertEqual(by["tune"]["ignored_candidate_videos"], 3)  # b:0, c:3 error, z:0
        self.assertEqual(by["holdout"]["ignored_candidate_videos"], 3)  # a:0, a:1, z:0
        self.assertEqual(by["tune"]["metrics"]["extra_seconds"], 0)
        self.assertEqual(by["holdout"]["metrics"]["cand_errors"], 1)

    def test_one_failing_set_fails_overall(self):
        cand = self.f.write("cand.jsonl", [sec("a.parquet:0", 0, ["car"], 1), sec("a.parquet:1", 0, [], 0),
                                           sec("b.parquet:0", 0, [], 0)])
        results, ok = score.score_files([("tune", self.tune), ("holdout", self.hold)], cand, out=io.StringIO())
        self.assertFalse(ok)
        self.assertEqual([r["status"] for r in results], ["PASS", "FAIL"])

    def test_empty_reference_is_fail(self):
        empty = self.f.write("empty.jsonl", [])
        cand = self.f.write("cand.jsonl", [sec("a.parquet:0", 0, ["car"], 1)])
        rc, text = run_main("--ref", f"tune={empty}", cand)
        self.assertEqual(rc, 1)
        self.assertIn("FAIL  tune", text)
        self.assertIn("reference has no seconds", text)

    def test_only_common_is_partial(self):
        ref = self.f.write("ref.jsonl", [sec("a.parquet:0", 0, ["car"], 1), sec("a.parquet:1", 0, ["dog"], 1)])
        cand = self.f.write("cand.jsonl", [sec("a.parquet:0", 0, ["car"], 1)])  # a:1 not processed yet
        results, ok = score.score_files([("tune", ref)], cand, only_common=True, out=io.StringIO())
        self.assertEqual(results[0]["status"], "PARTIAL")
        self.assertFalse(ok)  # never an overall PASS
        self.assertEqual(results[0]["metrics"]["ref_seconds"], 1)
        results, ok = score.score_files([("tune", ref)], cand, out=io.StringIO())
        self.assertEqual(results[0]["status"], "FAIL")  # without --only-common the missing video counts

    def test_positional_form_still_works(self):
        rc, text = run_main(self.tune, self.tune)
        self.assertEqual(rc, 0)
        self.assertIn("PASS  objects micro F1: 1.0000", text)
        rc, _ = run_main(self.tune, self.hold)
        self.assertEqual(rc, 1)

    def test_positional_mode_does_not_filter_candidate(self):
        # candidate has an extra video (z) not in the reference: positional mode counts its objects as FP
        cand = self.f.write("cand_extra.jsonl", [sec("a.parquet:0", 0, ["car"], 1), sec("a.parquet:1", 0, [], 0),
                                                 sec("z.parquet:0", 0, ["person", "dog", "car"], 1)])
        results, ok = score.score_files([("reference", self.tune)], cand, out=io.StringIO(), filter_candidate=False)
        self.assertFalse(ok)
        self.assertEqual(results[0]["metrics"]["extra_seconds"], 1)
        self.assertEqual(results[0]["ignored_candidate_videos"], 0)
        rc, _ = run_main(self.tune, cand)
        self.assertEqual(rc, 1)
        rc, text = run_main("--ref", f"tune={self.tune}", cand)  # --ref mode filters, with a loud warning
        self.assertEqual(rc, 0)
        self.assertIn("WARNING: 1 candidate videos ignored", text)

    def test_ref_mode_warns_when_no_file_overlap(self):
        cand = self.f.write("cand_other.jsonl", [sec("z.parquet:0", 0, ["car"], 1)])
        rc, text = run_main("--ref", f"tune={self.tune}", cand)
        self.assertEqual(rc, 1)
        self.assertIn("NO parquet file in common", text)

    def test_only_common_without_overlap_is_fail_not_empty_reference(self):
        cand = self.f.write("cand_other.jsonl", [sec("a.parquet:7", 0, ["car"], 1)])
        results, ok = score.score_files([("tune", self.tune)], cand, only_common=True, out=io.StringIO())
        r = results[0]
        self.assertEqual(r["status"], "FAIL")
        self.assertFalse(r["empty_reference"])
        self.assertTrue(r["no_overlap"])
        rc, text = run_main("--ref", f"tune={self.tune}", "--only-common", cand)
        self.assertEqual(rc, 1)
        self.assertIn("no overlapping videos between candidate and reference", text)
        self.assertNotIn("reference has no seconds", text)

    def test_bad_arguments(self):
        with self.assertRaises(SystemExit):
            run_main(self.tune)  # one positional without --ref
        with self.assertRaises(SystemExit):
            run_main("--ref", "noequals", self.tune)

    def test_json_output(self):
        cand = self.f.write("cand.jsonl", [sec("a.parquet:0", 0, ["car"], 1), sec("a.parquet:1", 0, [], 0),
                                           sec("b.parquet:0", 0, ["dog"], 2), err("c.parquet:3")])
        out = str(self.f.root / "m.json")
        rc, _ = run_main("--ref", f"tune={self.tune}", "--ref", f"holdout={self.hold}", "--json", out, cand)
        self.assertEqual(rc, 0)
        with open(out, encoding="utf-8") as fh:
            d = json.load(fh)
        self.assertTrue(d["overall_pass"])
        self.assertEqual(set(d["sets"]), {"tune", "holdout"})
        self.assertEqual(d["sets"]["tune"]["status"], "PASS")
        self.assertEqual(d["sets"]["holdout"]["micro_f1"], 1.0)
        self.assertEqual(d["sets"]["holdout"]["files"], ["b.parquet", "c.parquet"])


if __name__ == "__main__":
    unittest.main()
