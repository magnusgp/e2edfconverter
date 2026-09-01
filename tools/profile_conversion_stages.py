#!/usr/bin/env python3
from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import shutil
import time
from pathlib import Path

from nicolet_e2edf.nicolet.cli import convert_file


def _profile_one_file(input_path: Path, temp_root: Path, top_n: int) -> dict:
    case_root = temp_root / input_path.stem
    out_dir = case_root / "out"
    case_root.mkdir(parents=True, exist_ok=True)

    stage_marks: list[tuple[str, float]] = []

    def status_cb(stage: str) -> None:
        stage_marks.append((stage, time.perf_counter()))

    profile = cProfile.Profile()
    started = time.perf_counter()
    try:
        profile.enable()
        output_path = convert_file(
            input_path=input_path,
            output_dir=out_dir,
            patient_rules=[],
            json_sidecar=True,
            status_cb=status_cb,
        )
        profile.disable()
    finally:
        ended = time.perf_counter()

    # Convert callback marks into stage durations.
    stage_durations: list[dict[str, float | str]] = []
    ordered = stage_marks + [("done", ended)]
    for (stage_name, t0), (_, t1) in zip(ordered, ordered[1:]):
        stage_durations.append(
            {
                "stage": stage_name,
                "seconds": round(t1 - t0, 6),
            }
        )

    s = io.StringIO()
    stats = pstats.Stats(profile, stream=s).sort_stats("cumulative")
    stats.print_stats(top_n)
    top_cumulative = s.getvalue()

    result = {
        "input_path": str(input_path),
        "input_size_bytes": input_path.stat().st_size,
        "output_path": str(output_path),
        "output_size_bytes": output_path.stat().st_size,
        "wall_seconds": round(ended - started, 6),
        "stage_durations": stage_durations,
        "cprofile_top_cumulative": top_cumulative,
    }

    shutil.rmtree(case_root, ignore_errors=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile e2edf conversion stages for one or more .e files.")
    parser.add_argument("inputs", nargs="+", help="Input .e/.eeg files to profile.")
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory where JSON and text profiling results should be written.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of cProfile cumulative entries to keep in the text output.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    temp_root = results_dir / ".tmp_outputs"
    results_dir.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    try:
        for raw_input in args.inputs:
            input_path = Path(raw_input).resolve()
            result = _profile_one_file(input_path, temp_root, args.top_n)
            manifest.append(result)

            stem = input_path.stem
            json_path = results_dir / f"{stem}.profile.json"
            txt_path = results_dir / f"{stem}.cprofile.txt"
            json_path.write_text(json.dumps(result, indent=2) + "\n")
            txt_path.write_text(result["cprofile_top_cumulative"])
            print(f"profiled: {input_path}")
            print(f"  json: {json_path}")
            print(f"  text: {txt_path}")
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)

    summary_path = results_dir / "summary.json"
    summary_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
