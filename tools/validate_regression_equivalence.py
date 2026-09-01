#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from nicolet_e2edf.nicolet.cli import convert_file


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _scrub_sidecar(sidecar: dict) -> dict:
    return {key: value for key, value in sidecar.items() if key != "edf_file"}


def _validate_one_case(case_dir: Path, temp_root: Path) -> dict[str, object]:
    source_path = case_dir / "source.e"
    baseline_edf = case_dir / "reconverted_current.edf"
    baseline_json = case_dir / "reconverted_current.json"
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source file: {source_path}")
    if not baseline_edf.exists() or not baseline_json.exists():
        raise FileNotFoundError(f"Missing baseline outputs in {case_dir}")

    out_dir = temp_root / case_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        generated_edf = convert_file(
            input_path=source_path,
            output_dir=out_dir,
            patient_rules=[],
            json_sidecar=True,
        )
        generated_json = generated_edf.with_suffix(".json")

        baseline_hash = _sha256(baseline_edf)
        generated_hash = _sha256(generated_edf)
        baseline_sidecar = _load_json(baseline_json)
        generated_sidecar = _load_json(generated_json)

        return {
            "case_id": case_dir.name,
            "edf_equal": baseline_hash == generated_hash,
            "channels_equal": baseline_sidecar.get("channels") == generated_sidecar.get("channels"),
            "events_equal": baseline_sidecar.get("events") == generated_sidecar.get("events"),
            "annotations_equal": baseline_sidecar.get("annotations") == generated_sidecar.get("annotations"),
            "sidecar_equal_except_edf_file": _scrub_sidecar(baseline_sidecar) == _scrub_sidecar(generated_sidecar),
            "baseline_edf_sha256": baseline_hash,
            "generated_edf_sha256": generated_hash,
            "baseline_channel_count": baseline_sidecar.get("channel_count"),
            "generated_channel_count": generated_sidecar.get("channel_count"),
            "baseline_event_count": baseline_sidecar.get("event_count"),
            "generated_event_count": generated_sidecar.get("event_count"),
            "baseline_annotation_count": baseline_sidecar.get("annotation_count"),
            "generated_annotation_count": generated_sidecar.get("annotation_count"),
        }
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate optimized converter equivalence on regression cases.")
    parser.add_argument(
        "--cases-root",
        required=True,
        help="Directory containing regression benchmark case folders.",
    )
    parser.add_argument(
        "--results-json",
        required=True,
        help="Where to write the validation summary JSON.",
    )
    parser.add_argument(
        "--results-csv",
        required=True,
        help="Where to write the per-case validation CSV.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel conversions to run.",
    )
    parser.add_argument(
        "--temp-root",
        default=None,
        help="Optional scratch directory root. Defaults to a TemporaryDirectory under the system temp area.",
    )
    args = parser.parse_args()

    cases_root = Path(args.cases_root)
    case_dirs = sorted(path for path in cases_root.iterdir() if path.is_dir())
    results_json = Path(args.results_json)
    results_csv = Path(args.results_csv)
    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    temp_root = None
    cleanup_temp_root = False
    if args.temp_root:
        temp_root = Path(args.temp_root)
        if temp_root.exists():
            shutil.rmtree(temp_root)
        temp_root.mkdir(parents=True, exist_ok=True)
        cleanup_temp_root = True
    else:
        from tempfile import TemporaryDirectory

        temp_manager = TemporaryDirectory(prefix="e2edf-regression-")
        temp_root = Path(temp_manager.name)
        cleanup_temp_root = True
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            future_map = {
                executor.submit(_validate_one_case, case_dir, temp_root): case_dir.name
                for case_dir in case_dirs
            }
            for future in as_completed(future_map):
                case_id = future_map[future]
                try:
                    rows.append(future.result())
                except Exception as exc:
                    rows.append(
                        {
                            "case_id": case_id,
                            "error": f"{type(exc).__name__}: {exc}",
                            "edf_equal": False,
                            "channels_equal": False,
                            "events_equal": False,
                            "annotations_equal": False,
                            "sidecar_equal_except_edf_file": False,
                        }
                    )
    finally:
        if cleanup_temp_root and temp_root is not None:
            shutil.rmtree(temp_root, ignore_errors=True)

    rows.sort(key=lambda row: str(row.get("case_id", "")))
    summary = {
        "n_cases": len(rows),
        "n_errors": sum(1 for row in rows if row.get("error")),
        "edf_equal_all": all(bool(row.get("edf_equal")) for row in rows),
        "channels_equal_all": all(bool(row.get("channels_equal")) for row in rows),
        "events_equal_all": all(bool(row.get("events_equal")) for row in rows),
        "annotations_equal_all": all(bool(row.get("annotations_equal")) for row in rows),
        "sidecar_equal_except_edf_file_all": all(bool(row.get("sidecar_equal_except_edf_file")) for row in rows),
        "mismatch_case_ids": [
            str(row.get("case_id"))
            for row in rows
            if not (
                row.get("edf_equal")
                and row.get("channels_equal")
                and row.get("events_equal")
                and row.get("annotations_equal")
                and row.get("sidecar_equal_except_edf_file")
            )
        ],
        "cases": rows,
    }

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with results_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    results_json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
