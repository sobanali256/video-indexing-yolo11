import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidx.output import OutputWriter, ResumeError, sidecar_path  # noqa: E402


def lines_for(video, n):
    return [{"video": video, "second": s, "objects": [], "people": 0} for s in range(n)]


class OutputTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = Path(self.tmp.name) / "results.jsonl"

    def tearDown(self):
        self.tmp.cleanup()

    def read(self):
        return [json.loads(x) for x in self.out.read_text(encoding="utf-8").splitlines()]

    def test_write_and_sidecar(self):
        w = OutputWriter(self.out)
        w.write_video("a:0", lines_for("a:0", 2))
        w.write_video("a:1", [{"video": "a:1", "error": "bad"}])
        w.close()
        self.assertEqual(len(self.read()), 3)
        self.assertEqual(sidecar_path(self.out).read_text().split("\n"), ["a:0\t2", "a:1\t1", ""])

    def test_fresh_run_resets_old_files(self):
        w = OutputWriter(self.out)
        w.write_video("a:0", lines_for("a:0", 2))
        w.close()
        w = OutputWriter(self.out)  # no resume: start over
        w.close()
        self.assertEqual(self.out.read_text(), "")
        self.assertEqual(sidecar_path(self.out).read_text(), "")

    def test_resume_drops_half_written_video(self):
        w = OutputWriter(self.out)
        w.write_video("a:0", lines_for("a:0", 2))
        w.close()
        # crash: lines of a:1 reached the output (one truncated), sidecar never got a:1
        with open(self.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(lines_for("a:1", 1)[0]) + "\n")
            f.write('{"video": "a:1", "sec')
        w = OutputWriter(self.out, resume=True)
        self.assertTrue(w.is_done("a:0"))
        self.assertFalse(w.is_done("a:1"))
        w.write_video("a:1", lines_for("a:1", 3))
        w.close()
        rows = self.read()
        self.assertEqual([r["video"] for r in rows], ["a:0", "a:0", "a:1", "a:1", "a:1"])

    def test_resume_redoes_video_whose_last_line_was_cut_off(self):
        w = OutputWriter(self.out)
        w.write_video("a:0", lines_for("a:0", 2))
        w.write_video("a:1", lines_for("a:1", 3))
        w.close()
        text = self.out.read_text(encoding="utf-8")
        self.out.write_text(text[:-10], encoding="utf-8")  # sidecar says a:1 is done, its last line is cut
        w = OutputWriter(self.out, resume=True)
        self.assertEqual(w.done, {"a:0"})
        w.close()
        self.assertEqual([r["video"] for r in self.read()], ["a:0", "a:0"])
        self.assertEqual(sidecar_path(self.out).read_text(), "a:0\t2\n")

    def test_resume_accepts_plain_sidecar_lines(self):
        self.out.write_text("".join(json.dumps(x) + "\n" for x in lines_for("a:0", 2)), encoding="utf-8")
        sidecar_path(self.out).write_text("a:0\n", encoding="utf-8")  # format of earlier runs
        w = OutputWriter(self.out, resume=True)
        self.assertEqual(w.done, {"a:0"})
        w.close()
        self.assertEqual(len(self.read()), 2)
        self.assertEqual(sidecar_path(self.out).read_text(), "a:0\t2\n")  # upgraded with the count

    def test_resume_redoes_sidecar_entry_without_any_output_line(self):
        self.out.write_text("".join(json.dumps(x) + "\n" for x in lines_for("a:0", 1)), encoding="utf-8")
        sidecar_path(self.out).write_text("a:0\t1\na:1\na:2\t0\na:3\t2\n", encoding="utf-8")
        w = OutputWriter(self.out, resume=True)  # legacy a:1, count-0 a:2 and a:3 have no lines in the output
        self.assertEqual(w.done, {"a:0"})
        w.close()
        self.assertEqual(sidecar_path(self.out).read_text(), "a:0\t1\n")

    def test_write_video_refuses_empty_lines(self):
        w = OutputWriter(self.out)
        with self.assertRaises(ValueError):
            w.write_video("a:0", [])
        w.close()
        self.assertEqual(sidecar_path(self.out).read_text(), "")

    def test_resume_ignores_truncated_sidecar_entry(self):
        w = OutputWriter(self.out)
        w.write_video("a:1", lines_for("a:1", 1))
        w.close()
        with open(sidecar_path(self.out), "a", encoding="utf-8") as f:
            f.write("a:1")  # "a:1x" truncated mid-write, no newline
        with open(self.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(lines_for("a:12", 1)[0]) + "\n")
        w = OutputWriter(self.out, resume=True)
        self.assertEqual(w.done, {"a:1"})
        w.close()
        self.assertEqual([r["video"] for r in self.read()], ["a:1"])

    def test_resume_with_output_but_no_sidecar_refuses(self):
        w = OutputWriter(self.out)
        w.write_video("a:0", lines_for("a:0", 2))
        w.close()
        sidecar_path(self.out).unlink()
        before = self.out.read_text()
        with self.assertRaises(ResumeError) as cm:
            OutputWriter(self.out, resume=True)
        self.assertIn("sidecar", str(cm.exception))
        self.assertEqual(self.out.read_text(), before)  # not wiped
        OutputWriter(self.out).close()  # without --resume it starts fresh

    def test_resume_with_empty_output_and_no_sidecar_is_fine(self):
        self.out.write_text("")
        w = OutputWriter(self.out, resume=True)
        self.assertEqual(w.done, set())
        w.close()

    def test_resume_without_existing_files(self):
        w = OutputWriter(self.out, resume=True)
        self.assertEqual(w.done, set())
        w.close()


if __name__ == "__main__":
    unittest.main()
