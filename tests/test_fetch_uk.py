from datetime import datetime, timedelta

from carbonwatch.fetch_uk import date_chunks


def test_chunks_cover_range_without_overlap():
    start, end = datetime(2024, 1, 1), datetime(2024, 2, 15)
    chunks = date_chunks(start, end)
    assert chunks[0][0] == start
    assert chunks[-1][1] == end
    for (_, hi), (lo, _) in zip(chunks, chunks[1:]):
        assert hi == lo
    assert all(hi - lo <= timedelta(days=13) for lo, hi in chunks)


def test_single_short_range():
    start, end = datetime(2024, 1, 1), datetime(2024, 1, 2)
    assert date_chunks(start, end) == [(start, end)]
