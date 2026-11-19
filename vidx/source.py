"""Parquet rows -> (video_id, mp4 buffer or None, error or None)."""
import logging

log = logging.getLogger("vidx.source")


def iter_rows(path):
    """Yield (video_id, data, error) for every row of one parquet file.

    - File cannot be opened at all: one error row with id "<file>:0".
    - File opens but a row cannot be read: one error row for that row, then carry on.
    data is a pyarrow Buffer (bytes-like) when error is None.
    """
    import pyarrow.parquet as pq

    name = path.name
    try:
        pf = pq.ParquetFile(path)
        n_groups = pf.metadata.num_row_groups
        names = pf.schema_arrow.names
    except Exception as e:
        yield f"{name}:0", None, f"could not open parquet: {_short(e)}"
        return

    index = 0  # global row counter across row groups
    for g in range(n_groups):
        rows_in_group = pf.metadata.row_group(g).num_rows
        if "mp4" not in names:
            for _ in range(rows_in_group):
                yield f"{name}:{index}", None, "parquet has no mp4 column"
                index += 1
            continue
        produced = 0
        try:
            for batch in pf.iter_batches(batch_size=1, row_groups=[g], columns=["mp4"]):
                scalar = batch.column(0)[0]
                if not scalar.is_valid:
                    yield f"{name}:{index}", None, "null mp4"
                else:
                    yield f"{name}:{index}", scalar.as_buffer(), None
                index += 1
                produced += 1
        except Exception as e:
            log.warning("%s row group %d unreadable after %d rows: %s", name, g, produced, e)
            for _ in range(rows_in_group - produced):
                yield f"{name}:{index}", None, f"could not read row: {_short(e)}"
                index += 1


def _short(e, limit=200):
    return f"{type(e).__name__}: {e}"[:limit]
