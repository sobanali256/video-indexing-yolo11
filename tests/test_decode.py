import os
import sys
import unittest
from fractions import Fraction as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidx.decode import Timestamper, normalize_rotation  # noqa: E402


class TimestamperTest(unittest.TestCase):
    def test_exact_from_first_pts(self):
        ts = Timestamper(F(1, 90000), F(30))
        self.assertEqual(ts.stamp(9000), F(0))
        self.assertEqual(ts.stamp(12003), F(3003, 90000))
        self.assertEqual(ts.stamp(9000 + 90000), F(1))

    def test_negative_pts_clamped(self):
        ts = Timestamper(F(1, 1000), F(25))
        self.assertEqual(ts.stamp(100), F(0))
        self.assertEqual(ts.stamp(60), F(0))  # earlier than the first frame
        self.assertEqual(ts.clamped, 1)
        self.assertEqual(ts.stamp(140), F(40, 1000))

    def test_negative_first_pts_ok(self):
        ts = Timestamper(F(1, 1000), None)
        self.assertEqual(ts.stamp(-500), F(0))
        self.assertEqual(ts.stamp(-460), F(40, 1000))

    def test_missing_pts_uses_rate(self):
        ts = Timestamper(F(1, 1000), F(25))
        self.assertEqual(ts.stamp(0), F(0))
        self.assertEqual(ts.stamp(None), F(1, 25))
        self.assertEqual(ts.stamp(None), F(2, 25))
        self.assertEqual(ts.fallbacks, 2)

    def test_missing_pts_without_rate_and_no_history_uses_1_30(self):
        ts = Timestamper(F(1, 1000), None)
        self.assertEqual(ts.stamp(None), F(0))
        self.assertEqual(ts.stamp(None), F(1, 30))
        self.assertEqual(ts.stamp(None), F(2, 30))

    def test_missing_pts_without_rate_uses_median_gap(self):
        ts = Timestamper(F(1, 1000), None)
        for pts in (1000, 1040, 1080, 1130):  # gaps 40, 40, 50 ms
            ts.stamp(pts)
        self.assertEqual(ts.stamp(None), F(130 + 40, 1000))
        self.assertEqual(ts.stamp(None), F(130 + 80, 1000))

    def test_pts_after_leading_gap_continues_timeline(self):
        ts = Timestamper(F(1, 1000), F(10))
        self.assertEqual(ts.stamp(None), F(0))
        self.assertEqual(ts.stamp(None), F(1, 10))
        self.assertEqual(ts.stamp(5000), F(2, 10))  # not reset to 0
        self.assertEqual(ts.stamp(5100), F(3, 10))
        self.assertEqual(ts.stamp(4000), F(0))  # earlier than the anchor: clamped
        self.assertEqual(ts.clamped, 1)

    def test_pts_after_leading_gap_without_rate(self):
        ts = Timestamper(F(1, 90000), None)
        ts.stamp(None)  # 0
        self.assertEqual(ts.stamp(7), F(1, 30))
        self.assertEqual(ts.stamp(7 + 3003), F(1, 30) + F(3003, 90000))

    def test_no_time_base(self):
        ts = Timestamper(None, F(10))
        self.assertEqual(ts.stamp(5), F(0))
        self.assertEqual(ts.stamp(6), F(1, 10))


class RotationTest(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize_rotation(0), 0)
        self.assertEqual(normalize_rotation(90), 90)
        self.assertEqual(normalize_rotation(-90), 270)
        self.assertEqual(normalize_rotation(180), 180)
        self.assertEqual(normalize_rotation(-180), 180)
        self.assertEqual(normalize_rotation(450), 90)
        self.assertEqual(normalize_rotation(89.9), 90)
        self.assertEqual(normalize_rotation(None), 0)
        self.assertEqual(normalize_rotation("x"), 0)


if __name__ == "__main__":
    unittest.main()
