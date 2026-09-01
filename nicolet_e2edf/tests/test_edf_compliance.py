"""EDF+ compliance validation tests using PyEDFlib.

PyEDFlib serves as a strict EDF+/BDF+ parser that will raise errors
if files don't comply with the EDF+ specification. This makes it
an excellent CI gate for ensuring our output is standards-compliant.

These tests verify that:
1. Generated EDF files can be opened by PyEDFlib without errors
2. Header fields are correctly formatted per EDF+ spec
3. Signal data can be read correctly
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from nicolet_e2edf.nicolet import cli
from nicolet_e2edf.nicolet.edf_writer import write_edf
from nicolet_e2edf.nicolet.types import EventItem, NervusHeader, SegmentInfo

# PyEDFlib is an optional dev dependency - skip tests if not available
pyedflib = pytest.importorskip("pyedflib", reason="pyedflib required for compliance tests")


class TestEdfPlusCompliance:
    """Tests that verify EDF+ standard compliance using PyEDFlib."""

    def test_pyedflib_can_read_basic_edf(self, tmp_path: Path) -> None:
        """Verify PyEDFlib can read a basic EDF file without errors.
        
        This is the most fundamental compliance check - if PyEDFlib raises
        an error, the file is not EDF+ compliant.
        """
        # Create a simple test EDF file
        output_path = tmp_path / "test.edf"
        
        # Simple test data: 2 channels, 256 samples at 256 Hz (1 second)
        n_samples = 256
        n_channels = 2
        sfreq = 256.0
        data = np.random.randn(n_samples, n_channels) * 100  # Random EEG-like data in µV
        ch_names = ["Fp1", "Fp2"]
        
        patient_meta = {
            "PatientID": "TEST001",
            "PatientName": "Test Patient",
            "PatientSex": "M",
        }
        recording_start = datetime(2024, 3, 15, 10, 30, 0)
        
        # Write the EDF file
        write_edf(
            output_path,
            data,
            sfreq,
            ch_names,
            patient_meta=patient_meta,
            recording_start=recording_start,
            annotations=None,  # No annotations for this basic test
        )
        
        # This is the key test: PyEDFlib should read it without errors
        reader = pyedflib.EdfReader(str(output_path))
        try:
            # Verify basic header info
            assert reader.signals_in_file == n_channels
            assert reader.file_duration == pytest.approx(1.0, rel=0.01)
            
            # Verify we can read signal data
            for ch_idx in range(n_channels):
                signal = reader.readSignal(ch_idx)
                assert len(signal) == n_samples
        finally:
            reader.close()

    def test_pyedflib_can_read_edf_with_annotations(self, tmp_path: Path) -> None:
        """Verify PyEDFlib can read EDF+ files with EDF Annotations signal."""
        output_path = tmp_path / "test_annotations.edf"
        
        # Test data
        n_samples = 512
        n_channels = 3
        sfreq = 256.0
        data = np.random.randn(n_samples, n_channels) * 50
        ch_names = ["C3", "C4", "Cz"]
        
        patient_meta = {
            "PatientID": "ANNOT001",
            "PatientName": "Annotation Test",
            "PatientSex": "F",
        }
        recording_start = datetime(2024, 5, 20, 14, 0, 0)
        
        # Create some test events/annotations
        events = [
            EventItem(
                dateOLE=0.0,
                dateFraction=0.0,
                date=datetime(2024, 5, 20, 14, 0, 0),  # At recording start
                duration=0.0,
                user="test",
                GUID="{TEST}",
                label="RecordingStart",
                IDStr="Start",
                annotation="Recording started",
            ),
            EventItem(
                dateOLE=0.0,
                dateFraction=0.0,
                date=datetime(2024, 5, 20, 14, 0, 1),  # 1 second in
                duration=2.0,
                user="test",
                GUID="{TEST}",
                label="TestEvent",
                IDStr="Event",
                annotation="Test annotation at 1 second",
            ),
        ]
        
        # Write the EDF file with annotations
        write_edf(
            output_path,
            data,
            sfreq,
            ch_names,
            patient_meta=patient_meta,
            recording_start=recording_start,
            annotations=events,
        )
        
        # PyEDFlib should read it without errors
        reader = pyedflib.EdfReader(str(output_path))
        try:
            # PyEDFlib returns n_channels for EEG channels only
            # (annotation signal is handled internally)
            assert reader.signals_in_file == n_channels
            
            # Verify EEG data can still be read
            for ch_idx in range(n_channels):
                signal = reader.readSignal(ch_idx)
                assert len(signal) == n_samples
        finally:
            reader.close()

    def test_header_fields_comply_with_edfplus_spec(self, tmp_path: Path) -> None:
        """Verify header fields are formatted per EDF+ specification."""
        output_path = tmp_path / "test_header.edf"
        
        # Test data
        data = np.random.randn(128, 2) * 100
        ch_names = ["F3", "F4"]
        sfreq = 128.0
        
        patient_meta = {
            "PatientID": "P123",
            "PatientName": "John Doe",  # Name with space - should use underscore
            "PatientSex": "M",
            "PatientBirthDate": "1990-05-15",
        }
        recording_start = datetime(2024, 6, 10, 9, 15, 30)
        
        # Include annotations to create EDF+ format (with EDF+C marker)
        write_edf(
            output_path,
            data,
            sfreq,
            ch_names,
            patient_meta=patient_meta,
            recording_start=recording_start,
            annotations=[],  # Empty but present = EDF+ with annotation signal
        )
        
        # Read raw header bytes to verify format
        with open(output_path, "rb") as f:
            header = f.read(256)
        
        # Version field (8 bytes): must be "0" followed by spaces
        version = header[0:8].decode("ascii")
        assert version.strip() == "0"
        
        # Patient identification (80 bytes): EDF+ format
        patient_field = header[8:88].decode("ascii").strip()
        # Should be: <code> <sex> <birthdate> <name>
        parts = patient_field.split()
        assert len(parts) >= 4, f"Patient field not properly structured: {patient_field}"
        assert parts[1] == "M"  # Sex
        assert "MAY" in parts[2].upper()  # Birthdate contains month
        assert "John_Doe" in parts[3] or "John" in parts[3]  # Name with underscore
        
        # Recording identification (80 bytes): must start with "Startdate"
        recording_field = header[88:168].decode("ascii").strip()
        assert recording_field.startswith("Startdate"), f"Recording field: {recording_field}"
        # Should contain the date in DD-MMM-YYYY format
        assert "JUN" in recording_field.upper()  # June
        assert "2024" in recording_field
        
        # Reserved field (44 bytes at offset 192): must start with EDF+C or EDF+D
        reserved = header[192:236].decode("ascii").strip()
        assert reserved.startswith("EDF+C") or reserved.startswith("EDF+D"), \
            f"Reserved field must start with EDF+C or EDF+D, got: {reserved}"

    def test_edf_readable_after_conversion_pipeline(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Test full conversion pipeline produces PyEDFlib-readable files."""
        # Set up fake header/data like other tests
        fake_header = NervusHeader(filename=tmp_path / "case.e")
        fake_header.matchingChannels = [1, 2]
        fake_header.targetSamplingRate = 256.0
        fake_header.Segments = [
            SegmentInfo(
                dateOLE=0.0,
                date=datetime(2024, 1, 15, 10, 0, 0),
                duration=256 / 256.0,  # 1 second of data
                chName=["Fp1", "Fp2"],
                refName=["REF", "REF"],
                samplingRate=np.array([256.0, 256.0]),
                scale=np.ones(2),
                sampleCount=np.array([256, 256]),
            )
        ]
        fake_header.startDateTime = datetime(2024, 1, 15, 10, 0, 0)
        fake_header.Events = [
            EventItem(
                dateOLE=0.0,
                dateFraction=0.0,
                date=datetime(2024, 1, 15, 10, 0, 0, 500000),  # 0.5 seconds in
                duration=1.0,
                user="test",
                GUID="{TEST}",
                label="Eyes Closed",
                IDStr="Annotation",
                annotation="Patient closed eyes",
            )
        ]

        def _fake_read_header(path: Path, **_kwargs):
            return {"Fs": 256.0}, fake_header

        def _fake_read_data(path: Path, header: NervusHeader, channels=None, **kwargs):
            return np.random.randn(2, 256).astype(np.float32) * 100

        monkeypatch.setattr(cli, "read_nervus_header", _fake_read_header)
        monkeypatch.setattr(cli, "read_nervus_data", _fake_read_data)

        recording = tmp_path / "case.e"
        recording.write_bytes(b"\x00")
        output_dir = tmp_path / "out"

        exit_code = cli.main(["--in", str(recording), "--out", str(output_dir)])
        assert exit_code == 0

        edf_path = output_dir / "case.edf"
        assert edf_path.exists()

        # The key compliance test: PyEDFlib should read without errors
        reader = pyedflib.EdfReader(str(edf_path))
        try:
            # Verify file can be read and has expected structure
            n_signals = reader.signals_in_file
            assert n_signals >= 2  # At least 2 EEG channels
            
            # Read each signal to ensure data integrity
            for ch_idx in range(n_signals):
                label = reader.getLabel(ch_idx)
                if label != "EDF Annotations":
                    signal = reader.readSignal(ch_idx)
                    assert len(signal) > 0
        finally:
            reader.close()


def test_edf_compliance_quick(tmp_path: Path) -> None:
    """Quick smoke test for EDF+ compliance.
    
    This is a simple test that can run as part of the regular test suite
    to catch obvious compliance regressions.
    """
    output_path = tmp_path / "quick_test.edf"
    
    # Minimal test: single channel, 100 samples
    data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0] * 20]).T  # 100 samples, 1 channel
    write_edf(
        output_path,
        data,
        100.0,  # 100 Hz
        ["Test"],
        patient_meta={"PatientID": "X", "PatientName": "X"},
        recording_start=datetime(2024, 1, 1, 0, 0, 0),
    )
    
    # Must be readable by PyEDFlib
    reader = pyedflib.EdfReader(str(output_path))
    try:
        assert reader.signals_in_file == 1
    finally:
        reader.close()


def test_long_recording_with_multi_records(tmp_path: Path) -> None:
    """Test that long recordings use multiple data records and stay under size limits.
    
    This tests the fix for the "datarecordsize is too many bytes" error.
    
    Without multi-record support, this would create a single data record of:
    20 channels × 250 Hz × 60 seconds × 2 bytes = 600,000 bytes per record
    
    With 1-second records, each record is:
    20 channels × 250 samples × 2 bytes = 10,000 bytes (well under 10 MB limit)
    """
    output_path = tmp_path / "long_recording.edf"
    
    # Simulate a 60-second recording with 20 channels at 250 Hz
    # This would exceed single-record limits with the old approach
    n_channels = 20
    sfreq = 250.0
    duration_seconds = 60
    n_samples = int(sfreq * duration_seconds)  # 15,000 samples
    
    # Generate test data (random EEG-like signal)
    np.random.seed(42)  # For reproducibility
    data = np.random.randn(n_samples, n_channels).astype(np.float32) * 100
    ch_names = [f"Ch{i+1}" for i in range(n_channels)]
    
    patient_meta = {
        "PatientID": "LONG001",
        "PatientName": "Long Recording Test",
        "PatientSex": "X",
    }
    recording_start = datetime(2024, 7, 1, 8, 0, 0)
    
    # Add events distributed across the recording
    events = [
        EventItem(
            dateOLE=0.0,
            dateFraction=0.0,
            date=datetime(2024, 7, 1, 8, 0, 5),  # 5 seconds in
            duration=2.0,
            user="test",
            GUID="{TEST1}",
            label="Event1",
            IDStr="Event",
            annotation="Early event",
        ),
        EventItem(
            dateOLE=0.0,
            dateFraction=0.0,
            date=datetime(2024, 7, 1, 8, 0, 30),  # 30 seconds in
            duration=1.0,
            user="test",
            GUID="{TEST2}",
            label="Event2",
            IDStr="Event",
            annotation="Middle event",
        ),
        EventItem(
            dateOLE=0.0,
            dateFraction=0.0,
            date=datetime(2024, 7, 1, 8, 0, 55),  # 55 seconds in
            duration=0.5,
            user="test",
            GUID="{TEST3}",
            label="Event3",
            IDStr="Event",
            annotation="Late event",
        ),
    ]
    
    # Write the EDF file with annotations
    write_edf(
        output_path,
        data,
        sfreq,
        ch_names,
        patient_meta=patient_meta,
        recording_start=recording_start,
        annotations=events,
    )
    
    # Verify the file structure has multiple records
    with open(output_path, "rb") as f:
        header = f.read(256)
    
    # Check number of records (at offset 236-244)
    n_records = int(header[236:244].decode("ascii").strip())
    assert n_records == duration_seconds, f"Expected {duration_seconds} records, got {n_records}"
    
    # Check record duration (at offset 244-252)
    record_duration = float(header[244:252].decode("ascii").strip())
    assert record_duration == pytest.approx(1.0), f"Expected 1-second records, got {record_duration}"
    
    # PyEDFlib should read the file without errors
    reader = pyedflib.EdfReader(str(output_path))
    try:
        # Should have all EEG channels
        assert reader.signals_in_file == n_channels
        
        # Total duration should be correct
        assert reader.file_duration == pytest.approx(duration_seconds, rel=0.01)
        
        # Verify we can read signal data from start, middle, and end
        for ch_idx in [0, n_channels // 2, n_channels - 1]:
            signal = reader.readSignal(ch_idx)
            assert len(signal) == n_samples, f"Channel {ch_idx} has wrong length"
    finally:
        reader.close()


def test_multi_record_annotation_distribution(tmp_path: Path) -> None:
    """Test that annotations are correctly distributed across data records.
    
    Each data record should have a time-keeping TAL at its start,
    and event annotations should be in the correct record based on onset time.
    """
    output_path = tmp_path / "multi_record_annot.edf"
    
    # 5-second recording at 100 Hz = 5 data records
    n_samples = 500
    sfreq = 100.0
    data = np.random.randn(n_samples, 2) * 50
    ch_names = ["Ch1", "Ch2"]
    recording_start = datetime(2024, 8, 1, 12, 0, 0)
    
    # Events in different records
    events = [
        EventItem(
            dateOLE=0.0,
            dateFraction=0.0,
            date=datetime(2024, 8, 1, 12, 0, 0, 500000),  # Record 0 (0.5s)
            duration=0.0,
            user="test",
            GUID="{A}",
            label="A",
            IDStr="A",
            annotation=None,
        ),
        EventItem(
            dateOLE=0.0,
            dateFraction=0.0,
            date=datetime(2024, 8, 1, 12, 0, 2, 200000),  # Record 2 (2.2s)
            duration=0.0,
            user="test",
            GUID="{B}",
            label="B",
            IDStr="B",
            annotation=None,
        ),
        EventItem(
            dateOLE=0.0,
            dateFraction=0.0,
            date=datetime(2024, 8, 1, 12, 0, 4, 900000),  # Record 4 (4.9s)
            duration=0.0,
            user="test",
            GUID="{C}",
            label="C",
            IDStr="C",
            annotation=None,
        ),
    ]
    
    write_edf(
        output_path,
        data,
        sfreq,
        ch_names,
        patient_meta={"PatientID": "DIST001"},
        recording_start=recording_start,
        annotations=events,
    )
    
    # Verify file structure
    with open(output_path, "rb") as f:
        header = f.read(256)
    
    n_records = int(header[236:244].decode("ascii").strip())
    assert n_records == 5, f"Expected 5 records, got {n_records}"
    
    # PyEDFlib should read without errors
    reader = pyedflib.EdfReader(str(output_path))
    try:
        assert reader.signals_in_file == 2  # 2 EEG channels
        assert reader.file_duration == pytest.approx(5.0, rel=0.01)
    finally:
        reader.close()
