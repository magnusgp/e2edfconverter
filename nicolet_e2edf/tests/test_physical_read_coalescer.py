from __future__ import annotations

import io
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np
import pytest

import nicolet_e2edf.nicolet.data as data_module
from nicolet_e2edf.nicolet.data import (
    _MAX_COALESCED_READ_BYTES,
    _coalesced_spans,
    _PhysicalReadRequest,
    _read_coalesced_requests,
    read_nervus_data,
)
from nicolet_e2edf.nicolet.types import MainIndexEntry, NervusHeader, SegmentInfo, StaticPacket


def _segment(
    counts: list[int],
    scales: list[float],
    offsets: list[float] | None = None,
) -> SegmentInfo:
    channel_count = len(counts)
    return SegmentInfo(
        dateOLE=0.0,
        date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        duration=1.0,
        chName=[f"C{idx + 1}" for idx in range(channel_count)],
        refName=["REF"] * channel_count,
        samplingRate=np.asarray(counts, dtype=float),
        scale=np.asarray(scales, dtype=float),
        sampleCount=np.asarray(counts, dtype=int),
        eegOffset=None if offsets is None else np.asarray(offsets, dtype=float),
    )


def _header(
    path: Path,
    entries: list[tuple[int, int, int]],
    segments: list[SegmentInfo],
    static_indices: list[int] | None = None,
) -> NervusHeader:
    channel_count = len(segments[0].sampleCount)
    indices = static_indices or list(range(1, channel_count + 1))
    header = NervusHeader(filename=path, format="nicolet-e")
    header.StaticPackets = [
        StaticPacket(tag=str(channel), index=indices[channel], IDStr=str(channel))
        for channel in range(channel_count)
    ]
    header.MainIndex = [
        MainIndexEntry(sectionIdx=section, offset=offset, blockL=count * 2, sectionL=count * 2)
        for section, offset, count in entries
    ]
    header.Segments = segments
    header.matchingChannels = list(range(1, channel_count + 1))
    return header


def _request(start: int, samples: int) -> _PhysicalReadRequest:
    return _PhysicalReadRequest(start, samples, 0, 0, 1.0, 0.0)


def test_span_planner_merges_overlap_duplicates_and_adjacency_but_not_gaps() -> None:
    requests = [
        _request(0, 2),
        _request(0, 2),
        _request(2, 3),
        _request(10, 1),
        _request(14, 1),
    ]
    assert _coalesced_spans(requests, max_span_bytes=8) == [
        (0, 8),
        (10, 12),
        (14, 16),
    ]


def test_span_planner_enforces_exact_eight_mib_cap() -> None:
    request = _request(7, (_MAX_COALESCED_READ_BYTES + 6) // 2)
    spans = _coalesced_spans([request])
    assert spans == [
        (7, 7 + _MAX_COALESCED_READ_BYTES),
        (7 + _MAX_COALESCED_READ_BYTES, 7 + _MAX_COALESCED_READ_BYTES + 6),
    ]
    assert max(end - start for start, end in spans) == _MAX_COALESCED_READ_BYTES


def test_boundary_scatter_preserves_unaligned_int16_and_duplicate_requests() -> None:
    payload = bytes(range(20))
    requests = [
        _PhysicalReadRequest(0, 6, 0, 0, 1.0, 0.0),
        _PhysicalReadRequest(1, 5, 1, 0, -2.0, 3.0),
        _PhysicalReadRequest(0, 6, 2, 0, 0.5, -1.0),
    ]
    output = np.zeros((3, 6), dtype=np.float32)
    _read_coalesced_requests(io.BytesIO(payload), requests, output, max_span_bytes=7)
    raw_zero = np.frombuffer(payload[0:12], dtype="<i2")
    raw_one = np.frombuffer(payload[1:11], dtype="<i2")
    np.testing.assert_array_equal(output[0], raw_zero)
    np.testing.assert_array_equal(output[1, :5], raw_one * -2.0 + 3.0)
    np.testing.assert_array_equal(output[2], raw_zero * 0.5 - 1.0)


def test_multichannel_interleaving_and_contiguous_sections_equal_old_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = np.asarray([11, 12, 21, 22, 13, 14, 23, 24], dtype="<i2")
    recording = tmp_path / "interleaved.e"
    recording.write_bytes(raw.tobytes())
    header = _header(
        recording,
        entries=[(1, 0, 2), (2, 4, 2), (1, 8, 2), (2, 12, 2)],
        segments=[_segment([4, 4], [2.0, -0.5], [1.0, 10.0])],
    )
    old = read_nervus_data(recording, header)

    real_open = Path.open
    reads: list[tuple[int, int]] = []

    class TrackingFile:
        def __init__(self, wrapped):
            self.wrapped = wrapped

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.wrapped.close()

        def seek(self, offset: int, whence: int = 0):
            return self.wrapped.seek(offset, whence)

        def read(self, size: int = -1):
            reads.append((self.wrapped.tell(), size))
            return self.wrapped.read(size)

    def tracking_open(self: Path, *args, **kwargs):
        return TrackingFile(real_open(self, *args, **kwargs))

    monkeypatch.setattr(Path, "open", tracking_open)
    coalesced = read_nervus_data(recording, header, coalesce_reads=True)
    np.testing.assert_array_equal(coalesced, old)
    np.testing.assert_array_equal(coalesced[0], np.asarray([23, 25, 27, 29], dtype=np.float32))
    np.testing.assert_array_equal(coalesced[1], np.asarray([-0.5, -1, -1.5, -2], dtype=np.float32))
    assert reads == [(0, 16)]


def test_overlapping_channel_requests_are_read_once_and_scaled_independently(
    tmp_path: Path,
) -> None:
    raw = np.asarray([-4, 5, 300], dtype="<i2")
    recording = tmp_path / "duplicate.e"
    recording.write_bytes(raw.tobytes())
    header = _header(
        recording,
        entries=[(1, 0, 3)],
        segments=[_segment([3, 3], [1.0, 2.0], [0.0, -7.0])],
        static_indices=[1, 1],
    )
    old = read_nervus_data(recording, header)
    coalesced = read_nervus_data(recording, header, coalesce_reads=True)
    np.testing.assert_array_equal(coalesced, old)
    np.testing.assert_array_equal(coalesced[0], raw.astype(np.float32))
    np.testing.assert_array_equal(coalesced[1], raw * 2.0 - 7.0)


def test_segment_window_across_contiguous_sections_is_exact(tmp_path: Path) -> None:
    raw = np.asarray([100, -3, 7, 9, 11, 13, -15, 17, 19, 21, 23, 25], dtype="<i2")
    recording = tmp_path / "segments.e"
    recording.write_bytes(raw.tobytes())
    header = _header(
        recording,
        entries=[(1, 0, 3), (1, 6, 4), (1, 14, 5)],
        segments=[
            _segment([5], [0.25], [3.0]),
            _segment([7], [-2.0], [-7.0]),
        ],
    )
    old = read_nervus_data(recording, header, begsample=3, endsample=10)
    coalesced = read_nervus_data(
        recording,
        header,
        begsample=3,
        endsample=10,
        coalesce_reads=True,
    )
    np.testing.assert_array_equal(coalesced, old)
    expected = np.concatenate((raw[2:5] * 0.25 + 3.0, raw[5:10] * -2.0 - 7.0))
    np.testing.assert_array_equal(coalesced[0], expected.astype(np.float32))


def test_truncation_matches_old_partial_read_and_zero_fill(tmp_path: Path) -> None:
    raw = np.asarray([1, -2, 3, -4, 5], dtype="<i2")
    recording = tmp_path / "truncated.e"
    recording.write_bytes(raw.tobytes())
    header = _header(recording, [(1, 0, 8)], [_segment([8], [1.5], [2.0])])
    old = read_nervus_data(recording, header)
    coalesced = read_nervus_data(recording, header, coalesce_reads=True)
    np.testing.assert_array_equal(coalesced, old)
    np.testing.assert_array_equal(coalesced[0, :5], raw * 1.5 + 2.0)
    np.testing.assert_array_equal(coalesced[0, 5:], np.zeros(3, dtype=np.float32))


def test_short_read_halts_later_logical_section_even_if_its_offset_is_readable(
    tmp_path: Path,
) -> None:
    recording = tmp_path / "logical-short.e"
    recording.write_bytes(np.asarray([71, 72], dtype="<i2").tobytes())
    header = _header(
        recording,
        entries=[(1, 20, 2), (1, 0, 2)],
        segments=[_segment([4], [1.0])],
    )
    old = read_nervus_data(recording, header)
    coalesced = read_nervus_data(recording, header, coalesce_reads=True)
    np.testing.assert_array_equal(coalesced, old)
    np.testing.assert_array_equal(coalesced, np.zeros((1, 4), dtype=np.float32))


def test_mixed_rate_rejection_is_unchanged(tmp_path: Path) -> None:
    recording = tmp_path / "mixed.e"
    recording.write_bytes(np.asarray([1, 2, 3, 4, 5], dtype="<i2").tobytes())
    header = _header(
        recording,
        entries=[(1, 0, 2), (2, 4, 3)],
        segments=[_segment([2, 3], [1.0, 1.0])],
    )
    for coalesce in (False, True):
        with pytest.raises(NotImplementedError, match="Mixed sampling rates"):
            read_nervus_data(recording, header, coalesce_reads=coalesce)


def test_legacy_dispatch_ignores_modern_coalescing(monkeypatch: pytest.MonkeyPatch) -> None:
    import nicolet_e2edf.nicolet.legacy_eeg as legacy_eeg

    sentinel = np.asarray([[123.0]], dtype=np.float32)
    header = NervusHeader(filename=Path("unused.eeg"), format="nervus-eeg")
    calls: list[tuple] = []

    def fake_legacy(path, passed_header, **kwargs):
        calls.append((path, passed_header, kwargs))
        return sentinel

    monkeypatch.setattr(legacy_eeg, "read_legacy_data", fake_legacy)
    result = read_nervus_data("unused.eeg", header, coalesce_reads=True)
    assert result is sentinel
    assert len(calls) == 1


def test_environment_opt_in_and_explicit_false_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = np.asarray([1, 2, 3], dtype="<i2")
    recording = tmp_path / "env.e"
    recording.write_bytes(raw.tobytes())
    header = _header(recording, [(1, 0, 3)], [_segment([3], [1.0])])
    calls = 0
    real_coalesced = data_module._read_nervus_data_coalesced

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_coalesced(*args, **kwargs)

    monkeypatch.setattr(data_module, "_read_nervus_data_coalesced", counted)
    monkeypatch.setenv("NICOLET_E2EDF_COALESCE_READS", "1")
    env_result = read_nervus_data(recording, header)
    default_result = read_nervus_data(recording, header, coalesce_reads=False)
    np.testing.assert_array_equal(env_result, default_result)
    assert calls == 1


def test_deterministic_random_differential_windows_and_truncation(tmp_path: Path) -> None:
    rng = np.random.default_rng(20260803)
    for case_idx in range(75):
        channel_count = int(rng.integers(1, 5))
        sample_count = int(rng.integers(2, 65))
        section_specs: list[dict[str, object]] = []
        logical_sections: list[list[dict[str, object]]] = []
        scales = rng.choice(np.asarray([-2.0, -0.5, 0.25, 1.0, 3.0]), channel_count)
        offsets = rng.choice(np.asarray([-11.0, 0.0, 7.0]), channel_count)

        for channel in range(channel_count):
            raw = rng.integers(-32768, 32768, size=sample_count, dtype=np.int16).astype("<i2")
            possible = np.arange(1, sample_count)
            split_count = min(int(rng.integers(1, 6)), sample_count)
            cuts = (
                sorted(rng.choice(possible, size=split_count - 1, replace=False).tolist())
                if split_count > 1
                else []
            )
            starts = [0, *cuts]
            ends = [*cuts, sample_count]
            channel_specs = []
            for logical_idx, (start, end) in enumerate(zip(starts, ends, strict=True)):
                spec = {
                    "section": channel + 1,
                    "logical_idx": logical_idx,
                    "raw": raw[start:end],
                }
                section_specs.append(spec)
                channel_specs.append(spec)
            logical_sections.append(channel_specs)

        physical_order = list(section_specs)
        rng.shuffle(physical_order)
        payload = bytearray()
        for spec in physical_order:
            payload.extend(rng.bytes(int(rng.choice(np.asarray([0, 2, 4])))))
            spec["offset"] = len(payload)
            payload.extend(np.asarray(spec["raw"], dtype="<i2").tobytes())

        if case_idx % 4 == 0 and payload:
            payload = payload[: int(rng.integers(0, len(payload) + 1))]
        recording = tmp_path / f"random-{case_idx}.e"
        recording.write_bytes(payload)
        entries: list[tuple[int, int, int]] = []
        for channel_specs in logical_sections:
            for spec in channel_specs:
                entries.append(
                    (
                        cast(int, spec["section"]),
                        cast(int, spec["offset"]),
                        int(np.asarray(spec["raw"]).size),
                    )
                )
        header = _header(
            recording,
            entries,
            [_segment([sample_count] * channel_count, scales.tolist(), offsets.tolist())],
        )

        windows = [(1, sample_count)]
        windows.extend(
            (
                start := int(rng.integers(1, sample_count + 1)),
                int(rng.integers(start, sample_count + 1)),
            )
            for _ in range(3)
        )
        for begsample, endsample in windows:
            old = read_nervus_data(recording, header, begsample=begsample, endsample=endsample)
            coalesced = read_nervus_data(
                recording,
                header,
                begsample=begsample,
                endsample=endsample,
                coalesce_reads=True,
            )
            np.testing.assert_array_equal(
                coalesced,
                old,
                err_msg=f"case={case_idx} window={begsample}:{endsample}",
            )
