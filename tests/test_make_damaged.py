import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import make_damaged as md  # noqa: E402


def sec(v, n):
    return [{"video": v, "second": i, "objects": [], "people": 0} for i in range(n)]


def err(v):
    return [{"video": v, "error": "could not decode video: x"}]


GOOD = (sec("damaged.parquet:0", 5) + sec("damaged.parquet:1", 2) + err("damaged.parquet:1") + err("damaged.parquet:2")
        + err("damaged.parquet:3") + err("damaged.parquet:4") + sec("damaged.parquet:5", 5) + sec("damaged.parquet:6", 4)
        + err("corrupt.parquet:0") + err("truncated.parquet:0"))


class CheckTest(unittest.TestCase):
    def run_check(self, lines):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "r.jsonl")
            with open(p, "w", encoding="utf-8") as f:
                f.write("".join(json.dumps(l) + "\n" for l in lines))
            with redirect_stdout(io.StringIO()):
                return md.main(["check", "--out", d, "--results", p])

    def test_good_output_passes(self):
        self.assertEqual(self.run_check(GOOD), 0)

    def test_missing_video_fails(self):
        self.assertEqual(self.run_check([l for l in GOOD if l["video"] != "damaged.parquet:6"]), 1)

    def test_two_error_lines_fail(self):
        self.assertEqual(self.run_check(GOOD + err("damaged.parquet:2")), 1)

    def test_error_on_valid_video_fails(self):
        self.assertEqual(self.run_check(GOOD + err("damaged.parquet:0")), 1)

    def test_gap_in_seconds_fails(self):
        bad = [l for l in GOOD if not (l["video"] == "damaged.parquet:0" and l.get("second") == 2)]
        self.assertEqual(self.run_check(bad), 1)


if __name__ == "__main__":
    unittest.main()
