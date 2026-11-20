import os
import sys
import unittest
from fractions import Fraction as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidx.sampling import MAX_GROUP, pick, sample_groups  # noqa: E402


class PickTest(unittest.TestCase):
    def test_off_or_small_group_keeps_all(self):
        self.assertEqual(pick(30, 0), list(range(30)))
        self.assertEqual(pick(3, 5), [0, 1, 2])  # n < k
        self.assertEqual(pick(5, 5), [0, 1, 2, 3, 4])  # n == k
        self.assertEqual(pick(0, 5), [])

    def test_evenly_spaced_centres(self):
        self.assertEqual(pick(30, 3), [5, 15, 25])
        self.assertEqual(pick(30, 5), [3, 9, 15, 21, 27])
        self.assertEqual(pick(10, 1), [5])

    def test_strictly_increasing_in_range(self):
        for n in range(1, 130):
            for k in range(1, 12):
                idx = pick(n, k)
                self.assertEqual(len(idx), min(n, k))
                self.assertEqual(idx, sorted(set(idx)))
                self.assertTrue(all(0 <= i < n for i in idx))


def ts(*pairs):
    """[(t, tag)] items."""
    return [(F(t), tag) for t, tag in pairs]


class SampleGroupsTest(unittest.TestCase):
    def run_sg(self, items, k):
        return [tag for _, tag in sample_groups(iter(items), lambda x: x[0], k)]

    def test_one_frame_per_second_minimum(self):
        items = [(F(i, 10), i) for i in range(30)]  # 3 seconds of 10 frames
        self.assertEqual(self.run_sg(items, 1), [5, 15, 25])

    def test_k_zero_keeps_everything(self):
        items = [(F(i, 10), i) for i in range(25)]
        self.assertEqual(self.run_sg(items, 0), list(range(25)))

    def test_short_last_second_kept(self):
        items = [(F(i, 10), i) for i in range(12)]  # second 1 has 2 frames, k=5 -> both
        self.assertEqual(self.run_sg(items, 5), [1, 3, 5, 7, 9, 10, 11])

    def test_every_second_with_frames_keeps_a_frame(self):
        items = ts((0, "a"), (F(5, 2), "b"), (7, "c"))  # seconds 0, 2, 7 (gaps)
        self.assertEqual(self.run_sg(items, 3), ["a", "b", "c"])

    def test_backwards_timestamp_starts_new_group(self):
        items = [(F(0), 0), (F(1, 2), 1), (F(3, 2), 2), (F(9, 10), 3), (F(2), 4)]
        # groups: [0,1] (s0), [2] (s1), [3] (s0 again), [4] (s2); k=1 picks one from each group
        self.assertEqual(self.run_sg(items, 1), [1, 2, 3, 4])

    def test_failure_mid_second_yields_buffered_group_then_raises(self):
        def gen():
            for i in range(14):  # second 0 complete, second 1 has 4 frames
                yield (F(i, 10), i)
            raise ValueError("corrupt packet")

        got = []
        with self.assertRaises(ValueError):
            for _, tag in sample_groups(gen(), lambda x: x[0], 3):
                got.append(tag)
        self.assertEqual(got, [1, 5, 8, 10, 12, 13])  # partial second 1 sampled before the error

    def test_stuck_timestamps_are_capped(self):
        items = [(F(0), i) for i in range(MAX_GROUP * 2 + 10)]  # pts never advance
        got = self.run_sg(items, 1)
        self.assertEqual(len(got), 3)  # groups of MAX_GROUP, MAX_GROUP, 10: memory stays bounded

    def test_failure_before_any_frame(self):
        def gen():
            raise ValueError("moov atom not found")
            yield

        with self.assertRaises(ValueError):
            list(sample_groups(gen(), lambda x: x[0], 3))


if __name__ == "__main__":
    unittest.main()
