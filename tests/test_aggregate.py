import os
import sys
import unittest
from fractions import Fraction as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vidx.aggregate import OBJECTS, aggregate, error_line, make_lines, median_floor  # noqa: E402


def c(person=0, car=0, dog=0, laptop=0, phone=0):
    return [person, car, dog, laptop, phone]


class MedianTest(unittest.TestCase):
    def test_floor_at_half(self):
        self.assertEqual(median_floor([1, 2]), 1)  # 1.5 -> 1
        self.assertEqual(median_floor([0, 1]), 0)
        self.assertEqual(median_floor([2, 3]), 2)

    def test_odd_even(self):
        self.assertEqual(median_floor([3, 1, 2]), 2)
        self.assertEqual(median_floor([1, 1, 3, 3]), 2)
        self.assertEqual(median_floor([5]), 5)
        self.assertEqual(median_floor([]), 0)


class AggregateTest(unittest.TestCase):
    def test_half_rule_even(self):
        # 2 of 4 frames -> present (hits*2 >= n)
        frames = [(F(i, 10), c(car=1 if i < 2 else 0)) for i in range(4)]
        self.assertEqual(aggregate(frames), [(0, ["car"], 0)])

    def test_half_rule_below(self):
        frames = [(F(i, 10), c(car=1 if i < 1 else 0)) for i in range(4)]
        self.assertEqual(aggregate(frames), [(0, [], 0)])

    def test_half_rule_odd(self):
        # n=3: 1 hit is not enough, 2 hits is
        one = [(F(i, 10), c(dog=1 if i == 0 else 0)) for i in range(3)]
        two = [(F(i, 10), c(dog=1 if i < 2 else 0)) for i in range(3)]
        self.assertEqual(aggregate(one), [(0, [], 0)])
        self.assertEqual(aggregate(two), [(0, ["dog"], 0)])

    def test_single_frame_second(self):
        self.assertEqual(aggregate([(F(0), c(laptop=2, person=1))]), [(0, ["person", "laptop"], 1)])

    def test_object_order_fixed(self):
        row = aggregate([(F(0), c(phone=1, laptop=1, dog=1, car=1, person=1))])
        self.assertEqual(row[0][1], OBJECTS)
        self.assertEqual(OBJECTS, ["person", "car", "dog", "laptop", "cell phone"])

    def test_people_median_floor(self):
        frames = [(F(0), c(person=1)), (F(1, 4), c(person=2))]  # median 1.5 -> 1
        self.assertEqual(aggregate(frames)[0][2], 1)

    def test_person_object_but_median_zero(self):
        frames = [(F(0), c(person=1)), (F(1, 4), c(person=0)), (F(1, 2), c(person=0)), (F(3, 4), c(person=1))]
        self.assertEqual(aggregate(frames), [(0, ["person"], 0)])  # 2/4 frames, median 0.5 -> 0

    def test_empty_seconds_filled(self):
        rows = aggregate([(F(0), c(car=1)), (F(3, 1), c(person=2))])
        self.assertEqual(rows, [(0, ["car"], 0), (1, [], 0), (2, [], 0), (3, ["person"], 2)])

    def test_second_boundary_exact(self):
        rows = aggregate([(F(999, 1000), c()), (F(1), c(person=1))])
        self.assertEqual([r[0] for r in rows], [0, 1])

    def test_no_frames(self):
        self.assertEqual(aggregate([]), [])

    def test_partial_last_second(self):
        # video failed after 0.2 s of second 2: partial second is aggregated from what exists
        frames = [(F(0), c()), (F(1), c()), (F(2), c(person=1)), (F(21, 10), c(person=1))]
        self.assertEqual(aggregate(frames)[-1], (2, ["person"], 1))

    def test_lines(self):
        self.assertEqual(
            make_lines("a.parquet:3", [(0, ["car"], 1)]),
            [{"video": "a.parquet:3", "second": 0, "objects": ["car"], "people": 1}],
        )
        self.assertEqual(error_line("a.parquet:3", "x"), {"video": "a.parquet:3", "error": "x"})


if __name__ == "__main__":
    unittest.main()
