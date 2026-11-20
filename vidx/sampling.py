"""Per-second frame sampling. Standard library only.

Used by both the decoder (vidx.decode) and the offline simulator (tools/simulate_sampling.py),
so the simulator's scores predict the pipeline exactly: same grouping, same selection.

A "group" is a run of consecutive frames whose int(t) is the same second. A group closes as soon
as a frame of a different second arrives (also when timestamps go backwards), so a second can, in
rare broken files, produce two groups; both the simulator and the pipeline then sample each group.
Every group yields at least one frame, so every second that holds a frame keeps an output line.
"""


def pick(n, k):
    """Indices of the frames to keep from a group of n frames: k evenly spaced ones.

    k <= 0 (sampling off) or n <= k -> all n frames. Otherwise the centres of k equal slices:
    floor((j + 0.5) * n / k) for j in 0..k-1 (strictly increasing because n > k).
    """
    if k <= 0 or n <= k:
        return list(range(n))
    return [(2 * j + 1) * n // (2 * k) for j in range(k)]


def second_of(t):
    """Second index of a timestamp (Fraction or int, >= 0); same rule as vidx.aggregate."""
    return int(t // 1)


def iter_groups(items, t_of):
    """Yield lists of consecutive items that share a second (see module docstring).

    If the iterable raises (decode failure), the group collected so far is yielded first and then
    the exception is re-raised, so frames decoded before the error are never lost.
    """
    group = []
    current = None
    it = iter(items)
    while True:
        try:
            item = next(it)
        except StopIteration:
            break
        except Exception:
            if group:
                yield group
            raise
        s = second_of(t_of(item))
        if group and s != current:
            yield group
            group = []
        current = s
        group.append(item)
    if group:
        yield group


def sample_groups(items, t_of, k):
    """Yield the selected items (pick(n, k) of each group), group by group, in input order."""
    for group in iter_groups(items, t_of):
        yield from (group[i] for i in pick(len(group), k))
