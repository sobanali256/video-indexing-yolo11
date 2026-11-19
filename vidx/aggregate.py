"""Per-frame detections -> per-second output lines. Standard library only."""
from fractions import Fraction

OBJECTS = ["person", "car", "dog", "laptop", "cell phone"]  # fixed output order
COCO_IDS = [0, 2, 16, 63, 67]  # same order as OBJECTS
PERSON = 0  # index of "person" in OBJECTS


def median_floor(values):
    """Median of ints, rounded down (so 1.5 -> 1)."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) // 2


def aggregate(frames):
    """frames: iterable of (t, counts) with t an exact Fraction (>= 0) and counts
    a 5-sequence of per-class detection counts for that frame.

    Returns [(second, objects, people)] for every second from 0 to the last
    second that holds a frame. Seconds without frames give ([], 0).
    """
    by_second = {}
    for t, counts in frames:
        by_second.setdefault(int(t // 1), []).append(counts)
    if not by_second:
        return []
    rows = []
    for second in range(max(by_second) + 1):
        group = by_second.get(second)
        if not group:
            rows.append((second, [], 0))
            continue
        n = len(group)
        objects = []
        for i, name in enumerate(OBJECTS):
            hits = sum(1 for c in group if c[i] > 0)
            if hits * 2 >= n:
                objects.append(name)
        people = median_floor([c[PERSON] for c in group])
        rows.append((second, objects, people))
    return rows


def make_lines(video_id, rows):
    return [{"video": video_id, "second": s, "objects": o, "people": p} for s, o, p in rows]


def error_line(video_id, message):
    return {"video": video_id, "error": str(message)}
