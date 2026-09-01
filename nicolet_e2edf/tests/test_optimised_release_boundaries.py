from __future__ import annotations

import io
import struct
from pathlib import Path

import numpy as np
import pytest

import nicolet_e2edf.nicolet.cli as cli_module
import nicolet_e2edf.nicolet.header as header_module
from nicolet_e2edf.nicolet.data import _PhysicalReadRequest, _read_coalesced_requests
from nicolet_e2edf.nicolet.edf_writer import write_edf
from nicolet_e2edf.nicolet.header import _read_main_index, _read_qi_index2, read_nervus_header

_QI_INDEX2_OFFSET = 188_664
_QI_INDEX2_RECORD = struct.Struct("<HHIIIIIIIQQI")
_MAIN_INDEX_RECORD = struct.Struct("<QQQ")


def test_qi_index2_bulk_decodes_complete_records() -> None:
    values = (
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        (14 << 32) | 13,
        15,
    )
    handle = io.BytesIO(b"\x00" * _QI_INDEX2_OFFSET + _QI_INDEX2_RECORD.pack(*values))

    assert _read_qi_index2(handle, {"LQi": 1}) == [
        {
            "index": (3, 4),
            "misc1": 5,
            "indexIdx": 6,
            "misc2": [7, 8, 9],
            "sectionIdx": 10,
            "misc3": 11,
            "offset": 12,
            "blockL": 13,
            "sectionL": 14,
            "dataL": 15,
        }
    ]


def test_qi_index2_rejects_truncated_and_impossible_counts_before_large_read() -> None:
    truncated = io.BytesIO(b"\x00" * _QI_INDEX2_OFFSET + b"\x00" * (_QI_INDEX2_RECORD.size - 1))
    with pytest.raises(EOFError, match="Unexpected end of file"):
        _read_qi_index2(truncated, {"LQi": 1})

    class NoBulkRead(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            if size is not None and size > _QI_INDEX2_RECORD.size:
                raise AssertionError("unsafe bulk read was attempted")
            return super().read(size)

    guarded = NoBulkRead(b"\x00" * _QI_INDEX2_OFFSET)
    with pytest.raises(EOFError, match="cannot fit within the remaining stream"):
        _read_qi_index2(guarded, {"LQi": 2})


def test_main_index_bulk_decodes_complete_chain() -> None:
    first = _MAIN_INDEX_RECORD.pack(7, 101, (33 << 32) | 22)
    second = _MAIN_INDEX_RECORD.pack(8, 202, (55 << 32) | 44)
    next_pointer = 8 + len(first) + 8
    payload = struct.pack("<Q", 1) + first + struct.pack("<Q", next_pointer)
    payload += struct.pack("<Q", 1) + second + struct.pack("<Q", 0)

    entries = _read_main_index(io.BytesIO(payload), 0, 2)
    assert [(e.sectionIdx, e.offset, e.blockL, e.sectionL) for e in entries] == [
        (7, 101, 22, 33),
        (8, 202, 44, 55),
    ]


@pytest.mark.parametrize(
    ("payload", "nr_entries", "message"),
    [
        (struct.pack("<Q", 0), 1, "zero records"),
        (struct.pack("<Q", 2), 1, "only 1 expected"),
        (struct.pack("<Q", 1) + b"\x00" * (_MAIN_INDEX_RECORD.size - 1), 1, "end of file"),
    ],
)
def test_main_index_rejects_malformed_or_truncated_block_counts(
    payload: bytes, nr_entries: int, message: str
) -> None:
    with pytest.raises((EOFError, ValueError), match=message):
        _read_main_index(io.BytesIO(payload), 0, nr_entries)


def test_main_index_rejects_block_that_cannot_fit_stream_before_bulk_read() -> None:
    class NoBulkRead(io.BytesIO):
        def read(self, size: int | None = -1) -> bytes:
            if size is not None and size > _MAIN_INDEX_RECORD.size:
                raise AssertionError("unsafe bulk read was attempted")
            return super().read(size)

    payload = struct.pack("<Q", 2) + bytes(_MAIN_INDEX_RECORD.size)
    with pytest.raises(EOFError, match="cannot fit within the remaining stream"):
        _read_main_index(NoBulkRead(payload), 0, 2)


def test_main_index_accepts_more_than_one_million_records_when_stream_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record_count = 1_000_001
    payload = (
        struct.pack("<Q", record_count)
        + bytes(record_count * _MAIN_INDEX_RECORD.size)
        + struct.pack("<Q", 0)
    )
    monkeypatch.setattr(header_module, "MainIndexEntry", lambda **_kwargs: None)

    entries = _read_main_index(io.BytesIO(payload), 0, record_count)

    assert len(entries) == record_count


def _minimal_modern_header(path: Path) -> None:
    payload = bytearray(172_232)
    struct.pack_into("<I", payload, 24, 1)
    struct.pack_into("<I", payload, 172, 0)
    struct.pack_into("<IIIIQ", payload, 172_208, 0, 0, 1, 0, 0)
    path.write_bytes(payload)


def test_public_header_default_includes_qi_index2_and_explicit_skip_omits_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "minimal.e"
    _minimal_modern_header(source)
    calls = 0

    def fake_qi_index2(handle, qi_index):
        nonlocal calls
        calls += 1
        return [{"sentinel": True}]

    monkeypatch.setattr(header_module, "_read_qi_index2", fake_qi_index2)
    monkeypatch.setattr(header_module, "_read_tsinfo_packets", lambda *args: [])
    _, public_default = read_nervus_header(source)
    _, conversion_skip = read_nervus_header(source, include_qi_index2=False)

    assert calls == 1
    assert public_default.QIIndex2 == [{"sentinel": True}]
    assert conversion_skip.QIIndex2 == []


def test_conversion_path_explicitly_skips_qi_index2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class HeaderReached(Exception):
        pass

    seen: dict[str, object] = {}

    def fake_header(path, **kwargs):
        seen.update(kwargs)
        raise HeaderReached

    monkeypatch.setattr(cli_module, "read_nervus_header", fake_header)
    with pytest.raises(HeaderReached):
        cli_module.convert_file(tmp_path / "input.e", tmp_path, [])
    assert seen == {"include_qi_index2": False}


@pytest.mark.parametrize(
    ("data", "sfreq", "golden_hex"),
    [
        (
            np.array([[0.0, -1.0], [1.0, 0.0], [2.0, 1.0], [0.5, 0.5]]),
            4,
            "00800000ff7f00c000800000ff7fff3f",
        ),
        (
            np.array([[0.0], [1.0], [2.0], [0.5], [1.5]]),
            4,
            "00800000ff7f00c0ff3f000000000000",
        ),
        (
            np.array([[np.nan], [-np.inf], [np.inf], [2.0]]),
            4,
            "008000800080ff7f",
        ),
    ],
    ids=["full-record-rounding", "partial-record-padding", "non-finite-boundary"],
)
def test_vectorised_writer_data_bytes_match_golden(
    tmp_path: Path, data: np.ndarray, sfreq: int, golden_hex: str
) -> None:
    output = tmp_path / "golden.edf"
    write_edf(output, data, sfreq, [f"C{index + 1}" for index in range(data.shape[1])])
    header_bytes = 256 + data.shape[1] * 256
    assert output.read_bytes()[header_bytes:] == bytes.fromhex(golden_hex)


def test_coalesced_cap_short_read_zeroes_the_unavailable_suffix() -> None:
    payload = np.arange(12, dtype="<i2").tobytes()

    class ShortSecondSpan(io.BytesIO):
        calls = 0

        def read(self, size: int | None = -1) -> bytes:
            self.calls += 1
            if self.calls == 2:
                size = 3
            return super().read(size)

    output = np.full((1, 12), 999.0, dtype=np.float32)
    _read_coalesced_requests(
        ShortSecondSpan(payload),
        [_PhysicalReadRequest(0, 12, 0, 0, 1.0, 0.0)],
        output,
        max_span_bytes=8,
    )
    np.testing.assert_array_equal(output[0, :5], np.arange(5, dtype=np.float32))
    np.testing.assert_array_equal(output[0, 5:], np.zeros(7, dtype=np.float32))
