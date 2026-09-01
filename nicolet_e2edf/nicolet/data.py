# This file includes logic adapted from FieldTrip's read_nervus_data.m.
# FieldTrip is released under the GPL-3.0 licence. Copyright (C) the FieldTrip project.

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .types import NervusHeader

_MAX_COALESCED_READ_BYTES = 8 * 1024 * 1024
_COALESCE_READS_ENV = "NICOLET_E2EDF_COALESCE_READS"


@dataclass(slots=True, frozen=True)
class _PhysicalReadRequest:
    physical_start: int
    sample_count: int
    output_channel: int
    output_start: int
    scale: float
    offset: float

    @property
    def physical_end(self) -> int:
        return self.physical_start + self.sample_count * 2


def _normalise_channel_selection(header: NervusHeader, channels: Iterable[int] | None) -> list[int]:
    """Return a 1-based sorted list of channels to read."""

    if channels is None:
        if header.matchingChannels:
            return sorted(header.matchingChannels)
        return list(range(1, len(header.TSInfo) + 1))
    channel_list = sorted(set(int(ch) for ch in channels))
    if any(ch < 1 for ch in channel_list):
        raise ValueError("Channel indices must be 1-based and positive")
    return channel_list


def _lookup_static_index(header: NervusHeader, channel_zero_based: int) -> int:
    target_tag = str(channel_zero_based)
    for packet in header.StaticPackets:
        if packet.tag.strip() == target_tag:
            return packet.index
    raise KeyError(f"Static packet for channel {channel_zero_based + 1} not found")


def _collect_sections(header: NervusHeader, section_idx: int):
    sections = [entry for entry in header.MainIndex if entry.sectionIdx == section_idx]
    lengths = [entry.sectionL // 2 for entry in sections]
    cumulative = np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))
    return sections, np.array(lengths, dtype=np.int64), cumulative


def _accumulate_segment_lengths(header: NervusHeader, channel_zero_based: int) -> np.ndarray:
    lengths = [int(segment.sampleCount[channel_zero_based]) for segment in header.Segments]
    return np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))


def _read_section_chunk(handle, entry, start_offset_samples: int, count: int) -> np.ndarray:
    handle.seek(entry.offset + start_offset_samples * 2, 0)
    return np.fromfile(handle, dtype="<i2", count=count)


def _read_channel_window(
    handle,
    sections,
    cumulative_lengths,
    start_sample: int,
    count: int,
) -> np.ndarray:
    """Read a window of samples for a single channel, handling multi-section storage."""

    if count <= 0:
        return np.empty(0, dtype=np.int16)
    if not sections:
        return np.empty(0, dtype=np.int16)
    output = np.empty(count, dtype=np.int16)
    written = 0
    target_start = start_sample
    target_end = start_sample + count
    start_idx = int(np.searchsorted(cumulative_lengths, target_start, side="right") - 1)
    if start_idx < 0:
        start_idx = 0
    if start_idx >= len(sections):
        start_idx = len(sections) - 1
    for idx in range(start_idx, len(sections)):
        entry = sections[idx]
        section_start = cumulative_lengths[idx]
        section_end = cumulative_lengths[idx + 1]
        if target_end <= section_start:
            break
        if target_start >= section_end:
            continue
        read_start = max(target_start, section_start)
        read_end = min(target_end, section_end)
        samples_to_read = read_end - read_start
        if samples_to_read <= 0:
            continue
        offset_within_section = read_start - section_start
        chunk = _read_section_chunk(handle, entry, offset_within_section, samples_to_read)
        actual = chunk.size
        if actual == 0:
            break
        output[written : written + actual] = chunk
        written += actual
        if actual < samples_to_read:
            break
    return output[:written]


def _coalesced_spans(
    requests: Iterable[_PhysicalReadRequest],
    *,
    max_span_bytes: int = _MAX_COALESCED_READ_BYTES,
) -> list[tuple[int, int]]:
    """Return gap-free physical spans no larger than ``max_span_bytes``."""
    if max_span_bytes <= 0:
        raise ValueError("max_span_bytes must be positive")
    ranges = sorted(
        (request.physical_start, request.physical_end)
        for request in requests
        if request.sample_count > 0
    )
    if not ranges:
        return []
    components: list[tuple[int, int]] = []
    component_start, component_end = ranges[0]
    for start, end in ranges[1:]:
        if start <= component_end:
            component_end = max(component_end, end)
        else:
            components.append((component_start, component_end))
            component_start, component_end = start, end
    components.append((component_start, component_end))
    spans: list[tuple[int, int]] = []
    for component_start, component_end in components:
        start = component_start
        while start < component_end:
            end = min(start + max_span_bytes, component_end)
            spans.append((start, end))
            start = end
    return spans


def _append_window_requests(
    requests: list[_PhysicalReadRequest],
    *,
    entries,
    cumulative_lengths: np.ndarray,
    start_sample: int,
    count: int,
    output_channel: int,
    output_start: int,
    scale: float,
    offset: float,
) -> None:
    """Map a logical channel window onto exact physical section ranges."""
    if count <= 0 or not entries:
        return
    target_end = start_sample + count
    start_idx = int(np.searchsorted(cumulative_lengths, start_sample, side="right") - 1)
    start_idx = min(max(start_idx, 0), len(entries) - 1)
    for idx in range(start_idx, len(entries)):
        entry = entries[idx]
        section_start = int(cumulative_lengths[idx])
        section_end = int(cumulative_lengths[idx + 1])
        if target_end <= section_start:
            break
        if start_sample >= section_end:
            continue
        read_start = max(start_sample, section_start)
        read_end = min(target_end, section_end)
        samples_to_read = read_end - read_start
        if samples_to_read <= 0:
            continue
        requests.append(
            _PhysicalReadRequest(
                physical_start=int(entry.offset) + (read_start - section_start) * 2,
                sample_count=samples_to_read,
                output_channel=output_channel,
                output_start=output_start + (read_start - start_sample),
                scale=scale,
                offset=offset,
            )
        )


def _write_request_samples(
    data: np.ndarray,
    request: _PhysicalReadRequest,
    sample_offset: int,
    raw: np.ndarray,
) -> None:
    if raw.size:
        start = request.output_start + sample_offset
        data[request.output_channel, start : start + raw.size] = (
            raw * request.scale + request.offset
        )


def _read_coalesced_requests(
    handle,
    requests: list[_PhysicalReadRequest],
    data: np.ndarray,
    *,
    max_span_bytes: int = _MAX_COALESCED_READ_BYTES,
) -> None:
    """Read and scatter bounded, gap-free physical spans into logical outputs."""
    spans = _coalesced_spans(requests, max_span_bytes=max_span_bytes)
    if not spans:
        return
    prefix_bytes = [0] * len(requests)
    request_order = sorted(
        range(len(requests)),
        key=lambda idx: (requests[idx].physical_start, requests[idx].physical_end, idx),
    )
    next_request = 0
    active: list[int] = []
    pending_boundary: list[tuple[int, bytes]] = []
    for span_index, (span_start, span_end) in enumerate(spans):
        active = [idx for idx in active if requests[idx].physical_end > span_start]
        while (
            next_request < len(request_order)
            and requests[request_order[next_request]].physical_start < span_end
        ):
            active.append(request_order[next_request])
            next_request += 1
        handle.seek(span_start, 0)
        chunk = handle.read(span_end - span_start)
        actual_end = span_start + len(chunk)
        if pending_boundary:
            if chunk:
                for request_idx, first_byte in pending_boundary:
                    request = requests[request_idx]
                    sample_offset = (span_start - 1 - request.physical_start) // 2
                    raw = np.frombuffer(first_byte + chunk[:1], dtype="<i2", count=1)
                    _write_request_samples(data, request, sample_offset, raw)
            pending_boundary = []
        for request_idx in active:
            request = requests[request_idx]
            covered_start = max(span_start, request.physical_start)
            covered_end = min(actual_end, request.physical_end)
            expected = request.physical_start + prefix_bytes[request_idx]
            if covered_start <= expected and covered_end > expected:
                prefix_bytes[request_idx] = covered_end - request.physical_start
            first_sample = max(0, (span_start - request.physical_start + 1) // 2)
            last_sample = min(
                request.sample_count,
                (actual_end - request.physical_start) // 2,
            )
            if last_sample > first_sample:
                byte_start = request.physical_start + first_sample * 2 - span_start
                byte_end = request.physical_start + last_sample * 2 - span_start
                raw = np.frombuffer(chunk[byte_start:byte_end], dtype="<i2")
                _write_request_samples(data, request, first_sample, raw)
        next_start = spans[span_index + 1][0] if span_index + 1 < len(spans) else None
        if chunk and actual_end == span_end and next_start == span_end:
            position = span_end - 1
            for request_idx in active:
                request = requests[request_idx]
                relative = position - request.physical_start
                if 0 <= relative < request.sample_count * 2 and relative % 2 == 0:
                    pending_boundary.append((request_idx, chunk[-1:]))
    halted_channels: set[int] = set()
    for request_idx, request in enumerate(requests):
        start = request.output_start
        end = start + request.sample_count
        if request.output_channel in halted_channels:
            data[request.output_channel, start:end] = 0.0
            continue
        available = min(request.sample_count, prefix_bytes[request_idx] // 2)
        if available < request.sample_count:
            data[request.output_channel, start + available : end] = 0.0
            halted_channels.add(request.output_channel)


def _read_nervus_data_coalesced(
    path: str | Path,
    header: NervusHeader,
    zero_based_channels: list[int],
    cumulative_segment_lengths: np.ndarray,
    beg_zero: int,
    end_exclusive: int,
    data: np.ndarray,
) -> np.ndarray:
    requests: list[_PhysicalReadRequest] = []
    sections_cache: dict[int, tuple] = {}
    offsets_cache: dict[int, np.ndarray] = {}
    for ch_idx, channel_zb in enumerate(zero_based_channels):
        offsets = offsets_cache.get(channel_zb)
        if offsets is None:
            offsets = _accumulate_segment_lengths(header, channel_zb)
            offsets_cache[channel_zb] = offsets
        if not np.array_equal(offsets, cumulative_segment_lengths):
            raise NotImplementedError(
                "Mixed sampling rates across requested channels are not yet supported."
            )
        sections = sections_cache.get(channel_zb)
        if sections is None:
            section_idx = _lookup_static_index(header, channel_zb)
            sections_cache[channel_zb] = _collect_sections(header, section_idx)
        entries, _section_lengths, cumulative_lengths = sections_cache[channel_zb]
        for seg_idx, segment in enumerate(header.Segments):
            segment_start = int(cumulative_segment_lengths[seg_idx])
            segment_end = int(cumulative_segment_lengths[seg_idx + 1])
            overlap_start = max(beg_zero, segment_start)
            overlap_end = min(end_exclusive, segment_end)
            if overlap_start >= overlap_end:
                continue
            relative_start = overlap_start - beg_zero
            samples_to_copy = overlap_end - overlap_start
            window_start = int(offsets[seg_idx]) + (overlap_start - segment_start)
            scale = float(segment.scale[channel_zb]) if segment.scale is not None else 1.0
            if np.isclose(scale, 0.0):
                scale = 1.0
            offset = 0.0
            if segment.eegOffset is not None and channel_zb < len(segment.eegOffset):
                offset = float(segment.eegOffset[channel_zb])
                if not np.isfinite(offset) or abs(offset) > 1e9:
                    offset = 0.0
            _append_window_requests(
                requests,
                entries=entries,
                cumulative_lengths=cumulative_lengths,
                start_sample=int(window_start),
                count=int(samples_to_copy),
                output_channel=ch_idx,
                output_start=int(relative_start),
                scale=scale,
                offset=offset,
            )
    with Path(path).open("rb") as handle:
        _read_coalesced_requests(handle, requests, data)
    return data


def _coalescing_enabled(coalesce_reads: bool | None) -> bool:
    if coalesce_reads is not None:
        return bool(coalesce_reads)
    return os.environ.get(_COALESCE_READS_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def read_nervus_data(
    path: str | Path,
    header: NervusHeader,
    channels: Iterable[int] | None = None,
    begsample: int | None = None,
    endsample: int | None = None,
    *,
    coalesce_reads: bool | None = None,
) -> np.ndarray:
    """Read waveform data from a Nicolet/Nervus recording.

    Modern reads opt in with ``coalesce_reads=True`` or
    ``NICOLET_E2EDF_COALESCE_READS=1``. The default reader is unchanged.
    """

    if header.format == "nervus-eeg":
        from .legacy_eeg import read_legacy_data

        return read_legacy_data(
            path, header, channels=channels, begsample=begsample, endsample=endsample
        )
    if header.format == "nervus-legacy-e":
        from .legacy import read_legacy_data

        return read_legacy_data(
            path, header, channels=channels, begsample=begsample, endsample=endsample
        )

    if not header.Segments:
        raise ValueError("Header does not contain segment information")
    channel_selection = _normalise_channel_selection(header, channels)
    zero_based_channels = [ch - 1 for ch in channel_selection]
    base_channel = zero_based_channels[0]
    cumulative_segment_lengths = _accumulate_segment_lengths(header, base_channel)
    total_samples = int(cumulative_segment_lengths[-1])
    beg = 1 if begsample is None else int(begsample)
    end = total_samples if endsample is None else int(endsample)
    if beg < 1 or end < beg:
        raise ValueError("Invalid sample range specified")
    beg_zero = beg - 1
    end_exclusive = min(end, total_samples)
    samples_requested = end_exclusive - beg_zero
    data = np.zeros((len(zero_based_channels), samples_requested), dtype=np.float32)
    if _coalescing_enabled(coalesce_reads):
        return _read_nervus_data_coalesced(
            path,
            header,
            zero_based_channels,
            cumulative_segment_lengths,
            beg_zero,
            end_exclusive,
            data,
        )

    sections_cache: dict[int, tuple] = {}
    offsets_cache: dict[int, np.ndarray] = {}
    with Path(path).open("rb") as handle:
        # Iterate per channel so we can reuse section metadata and honour scaling.
        for ch_idx, channel_zb in enumerate(zero_based_channels):
            offsets = offsets_cache.get(channel_zb)
            if offsets is None:
                offsets = _accumulate_segment_lengths(header, channel_zb)
                offsets_cache[channel_zb] = offsets
            if not np.array_equal(offsets, cumulative_segment_lengths):
                raise NotImplementedError(
                    "Mixed sampling rates across requested channels are not yet supported."
                )
            # Collect all MainIndex sections that belong to this channel's data stream.
            sections = sections_cache.get(channel_zb)
            if sections is None:
                section_idx = _lookup_static_index(header, channel_zb)
                sections_cache[channel_zb] = _collect_sections(header, section_idx)
            entries, _section_lengths, cumulative_lengths = sections_cache[channel_zb]
            for seg_idx, segment in enumerate(header.Segments):
                # Translate the global sample window into the portion stored inside this segment.
                segment_start = cumulative_segment_lengths[seg_idx]
                segment_end = cumulative_segment_lengths[seg_idx + 1]
                overlap_start = max(beg_zero, segment_start)
                overlap_end = min(end_exclusive, segment_end)
                if overlap_start >= overlap_end:
                    continue
                relative_start = overlap_start - beg_zero
                samples_to_copy = overlap_end - overlap_start
                channel_offset = offsets[seg_idx]
                window_start = channel_offset + (overlap_start - segment_start)
                raw = _read_channel_window(
                    handle,
                    entries,
                    cumulative_lengths,
                    int(window_start),
                    int(samples_to_copy),
                )
                if raw.size == 0:
                    break
                scale = float(segment.scale[channel_zb]) if segment.scale is not None else 1.0
                if np.isclose(scale, 0.0):
                    scale = 1.0
                offset = 0.0
                if segment.eegOffset is not None and channel_zb < len(segment.eegOffset):
                    offset = float(segment.eegOffset[channel_zb])
                    if not np.isfinite(offset) or abs(offset) > 1e9:
                        offset = 0.0
                count = min(raw.size, samples_to_copy)
                data[ch_idx, relative_start : relative_start + count] = raw[:count] * scale + offset
                if raw.size < samples_to_copy:
                    break
    return data
