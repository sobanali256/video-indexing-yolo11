import contextlib
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import score  # noqa: E402


def sec(video, s, objects=(), people=0):
    return {"video": video, "second": s, "objects": list(objects), "people": people}


def metrics(ref, cand):
    return score.compute(score.parse_lines(ref), score.parse_lines(cand))


class ScoreTest(unittest.TestCase):
    def test_identical(self):
        ref = [sec("v:0", 0, ["person"], 2), sec("v:0", 1, ["car", "dog"], 0), sec("v:1", 0, [], 0)]
        m = metrics(ref, list(ref))
        self.assertEqual(m["micro_f1"], 1.0)
        self.assertEqual(m["macro_f1"], 1.0)
        self.assertEqual(m["people_exact"], 1.0)
        self.assertEqual(m["people_within1"], 1.0)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(score.report(m))

    def test_both_empty_is_perfect(self):
        ref = [sec("v:0", 0), sec("v:0", 1)]
        m = metrics(ref, list(ref))
        self.assertEqual(m["micro_f1"], 1.0)
        self.assertEqual(m["video_f1"]["v:0"], 1.0)

    def test_missing_seconds_count_as_empty(self):
        ref = [sec("v:0", 0, ["car"], 1), sec("v:0", 1, [], 0)]
        cand = [sec("v:0", 1, [], 0)]
        m = metrics(ref, cand)
        self.assertEqual(m["missing_seconds"], 1)
        self.assertEqual(m["micro_f1"], 0.0)  # tp 0, fn 1
        self.assertEqual(m["people_exact"], 0.5)  # 1 vs predicted 0 is wrong; empty second matches
        self.assertEqual(m["people_within1"], 1.0)

    def test_extra_seconds_are_false_positives(self):
        ref = [sec("v:0", 0, ["car"], 0)]
        cand = [sec("v:0", 0, ["car"], 0), sec("v:0", 1, ["dog", "car"], 3)]
        m = metrics(ref, cand)
        self.assertEqual(m["extra_seconds"], 1)
        self.assertAlmostEqual(m["micro_f1"], 2 * 1 / (2 * 1 + 2 + 0))
        self.assertEqual(m["people_exact"], 1.0)  # people only judged on reference seconds

    def test_people_tolerance(self):
        ref = [sec("v:0", i, [], 2) for i in range(4)]
        cand = [sec("v:0", 0, [], 2), sec("v:0", 1, [], 3), sec("v:0", 2, [], 1), sec("v:0", 3, [], 5)]
        m = metrics(ref, cand)
        self.assertEqual(m["people_exact"], 0.25)
        self.assertEqual(m["people_within1"], 0.75)

    def test_thresholds_fail(self):
        ref = [sec("v:0", i, ["car"], 0) for i in range(10)]
        cand = [sec("v:0", i, [] if i < 2 else ["car"], 0) for i in range(10)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertFalse(score.report(metrics(ref, cand)))

    def test_error_lines_counted_separately(self):
        ref = [sec("v:0", 0, ["car"], 0), {"video": "v:1", "error": "bad"}]
        cand = [sec("v:0", 0, ["car"], 0), {"video": "v:2", "error": "bad"}]
        m = metrics(ref, cand)
        self.assertEqual((m["ref_errors"], m["cand_errors"], m["error_mismatch"]), (1, 1, 2))
        self.assertEqual(m["micro_f1"], 1.0)

    def test_empty_files(self):
        m = metrics([], [])
        self.assertEqual((m["micro_f1"], m["people_exact"], m["people_within1"]), (1.0, 1.0, 1.0))


if __name__ == "__main__":
    unittest.main()
