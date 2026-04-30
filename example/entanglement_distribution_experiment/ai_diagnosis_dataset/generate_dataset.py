"""CLI for AI-diagnosis sequence-dataset generation."""

from __future__ import annotations

import argparse
import os
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from statistics import mean

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config import IMPAIRMENT_LABELS, default_output_root, get_preset
from scenarios import build_scenario
from simulator import simulate_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a monitoring-episode impairment-diagnosis dataset from SeQUeNCe simulations.")
    parser.add_argument("--mode", choices=["sample", "full"], default="sample", help="Dataset size preset.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional explicit output directory.")
    parser.add_argument("--samples-per-label", type=int, default=None, help="Override the number of episodes generated per impairment label.")
    parser.add_argument("--episode-duration-s", type=float, default=None, help="Override total episode duration in seconds.")
    parser.add_argument("--window-duration-s", type=float, default=None, help="Override monitoring-window duration in seconds.")
    parser.add_argument("--source-frequency-hz", type=float, default=None, help="Override the SPDC source frequency in Hz.")
    parser.add_argument("--visibility-stride-windows", type=int, default=None, help="Override how often active visibility/X-basis diagnostics are measured.")
    parser.add_argument("--temperature-episode-span-s", type=float, default=None, help="Physical monitoring time spanned by peak_shift_temperature windows, without changing acquisition runtime.")
    parser.add_argument("--base-seed", type=int, default=None, help="Override the base RNG seed used to derive scenario seeds.")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel worker processes. Defaults to CPU count minus one.")
    parser.add_argument(
        "--parallel-backend",
        choices=["process", "thread"],
        default="process",
        help="Parallel execution backend. Use 'thread' on Windows if process workers exhaust paging-file memory while importing SciPy/QuTiP.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Ignore existing per-sample checkpoints in the output directory and regenerate all samples.",
    )
    parser.set_defaults(resume=True)
    parser.add_argument("--skip-raw", action="store_true", help="Skip writing raw per-sequence JSON artifacts to reduce I/O and disk usage.")
    parser.add_argument("--validation-off", action="store_true", help="Disable validation retries and accept the first generated scenario for each sample.")
    return parser.parse_args()


def _apply_overrides(args: argparse.Namespace, preset):
    overrides = {}
    if args.samples_per_label is not None:
        overrides["samples_per_label"] = args.samples_per_label
    if args.episode_duration_s is not None:
        overrides["episode_duration_s"] = args.episode_duration_s
    if args.window_duration_s is not None:
        overrides["window_duration_s"] = args.window_duration_s
    if args.source_frequency_hz is not None:
        overrides["source_frequency_hz"] = args.source_frequency_hz
    if args.visibility_stride_windows is not None:
        overrides["visibility_measurement_stride_windows"] = args.visibility_stride_windows
    if args.temperature_episode_span_s is not None:
        overrides["temperature_episode_span_s"] = args.temperature_episode_span_s
    if args.base_seed is not None:
        overrides["base_seed"] = args.base_seed
    return replace(preset, **overrides) if overrides else preset


def _summarize_validation(observable_rows: list[dict], sequence_rows: list[dict]) -> dict:
    by_sample: dict[str, list[dict]] = {}
    for row in observable_rows:
        by_sample.setdefault(row["sample_id"], []).append(row)

    summaries: list[dict] = []
    for item in sequence_rows:
        sample_id = item["sample_id"]
        rows = sorted(by_sample[sample_id], key=lambda r: r["window_index"])
        onset = int(item["fault_onset_window"])
        pre = rows[:onset] or rows
        post = rows[onset:] or rows

        def avg(chunk: list[dict], key: str) -> float | None:
            values = [float(r[key]) for r in chunk if r.get(key) not in ("", None)]
            return mean(values) if values else None

        def extrema(chunk: list[dict], key: str, fn) -> float | None:
            values = [float(r[key]) for r in chunk if r.get(key) not in ("", None)]
            return fn(values) if values else None

        summary = {
            "sample_id": sample_id,
            "label": item["label"],
            "trajectory_variant": item.get("trajectory_variant", ""),
            "windows": item["windows"],
            "fault_onset_window": onset,
            "pre_coincidence_rate_hz": avg(pre, "coincidence_rate_hz"),
            "post_coincidence_rate_hz": avg(post, "coincidence_rate_hz"),
            "pre_singles_rate_a_hz": avg(pre, "singles_rate_a_hz"),
            "post_singles_rate_a_hz": avg(post, "singles_rate_a_hz"),
            "pre_singles_rate_b_hz": avg(pre, "singles_rate_b_hz"),
            "post_singles_rate_b_hz": avg(post, "singles_rate_b_hz"),
            "pre_peak_position_ps": avg(pre, "peak_position_ps"),
            "post_peak_position_ps": avg(post, "peak_position_ps"),
            "pre_peak_snr": avg(pre, "peak_snr"),
            "post_peak_snr": avg(post, "peak_snr"),
            "pre_car": avg(pre, "car"),
            "post_car": avg(post, "car"),
            "pre_zz_correlation": avg(pre, "zz_correlation"),
            "post_zz_correlation": avg(post, "zz_correlation"),
            "pre_link_temperature_a_c": avg(pre, "link_temperature_a_c"),
            "post_link_temperature_a_c": avg(post, "link_temperature_a_c"),
            "pre_link_temperature_b_c": avg(pre, "link_temperature_b_c"),
            "post_link_temperature_b_c": avg(post, "link_temperature_b_c"),
            "visibility_measurements": sum(int(r["visibility_measured"]) for r in rows),
            "xx_measurements": sum(int(r["xx_measured"]) for r in rows),
            "post_peak_position_abs_max_ps": extrema(post, "peak_position_ps", lambda v: max(abs(x) for x in v)),
            "post_peak_snr_min": extrema(post, "peak_snr", min),
            "post_car_min": extrema(post, "car", min),
            "post_coincidence_rate_min_hz": extrema(post, "coincidence_rate_hz", min),
            "post_singles_rate_a_max_hz": extrema(post, "singles_rate_a_hz", max),
            "post_singles_rate_b_max_hz": extrema(post, "singles_rate_b_hz", max),
            "post_singles_rate_a_min_hz": extrema(post, "singles_rate_a_hz", min),
            "post_singles_rate_b_min_hz": extrema(post, "singles_rate_b_hz", min),
            "post_zz_correlation_min": extrema(post, "zz_correlation", min),
            "post_link_temperature_a_min_c": extrema(post, "link_temperature_a_c", min),
            "post_link_temperature_b_min_c": extrema(post, "link_temperature_b_c", min),
        }
        summary["delta_coincidence_rate_hz"] = None if summary["pre_coincidence_rate_hz"] is None or summary["post_coincidence_rate_hz"] is None else summary["post_coincidence_rate_hz"] - summary["pre_coincidence_rate_hz"]
        summary["delta_singles_rate_a_hz"] = None if summary["pre_singles_rate_a_hz"] is None or summary["post_singles_rate_a_hz"] is None else summary["post_singles_rate_a_hz"] - summary["pre_singles_rate_a_hz"]
        summary["delta_singles_rate_b_hz"] = None if summary["pre_singles_rate_b_hz"] is None or summary["post_singles_rate_b_hz"] is None else summary["post_singles_rate_b_hz"] - summary["pre_singles_rate_b_hz"]
        summary["delta_peak_position_ps"] = None if summary["pre_peak_position_ps"] is None or summary["post_peak_position_ps"] is None else summary["post_peak_position_ps"] - summary["pre_peak_position_ps"]
        summary["delta_peak_snr"] = None if summary["pre_peak_snr"] is None or summary["post_peak_snr"] is None else summary["post_peak_snr"] - summary["pre_peak_snr"]
        summary["delta_car"] = None if summary["pre_car"] is None or summary["post_car"] is None else summary["post_car"] - summary["pre_car"]
        summary["delta_zz_correlation"] = None if summary["pre_zz_correlation"] is None or summary["post_zz_correlation"] is None else summary["post_zz_correlation"] - summary["pre_zz_correlation"]
        summary["delta_link_temperature_a_c"] = None if summary["pre_link_temperature_a_c"] is None or summary["post_link_temperature_a_c"] is None else summary["post_link_temperature_a_c"] - summary["pre_link_temperature_a_c"]
        summary["delta_link_temperature_b_c"] = None if summary["pre_link_temperature_b_c"] is None or summary["post_link_temperature_b_c"] is None else summary["post_link_temperature_b_c"] - summary["pre_link_temperature_b_c"]
        summaries.append(summary)

    by_label: dict[str, list[dict]] = {}
    for summary in summaries:
        by_label.setdefault(summary["label"], []).append(summary)

    label_rollup: list[dict] = []
    for label, items in sorted(by_label.items()):
        def label_avg(key: str) -> float | None:
            vals = [float(item[key]) for item in items if item.get(key) is not None]
            return mean(vals) if vals else None

        label_rollup.append(
            {
                "label": label,
                "samples": len(items),
                "avg_delta_coincidence_rate_hz": label_avg("delta_coincidence_rate_hz"),
                "avg_delta_singles_rate_a_hz": label_avg("delta_singles_rate_a_hz"),
                "avg_delta_singles_rate_b_hz": label_avg("delta_singles_rate_b_hz"),
                "avg_delta_peak_position_ps": label_avg("delta_peak_position_ps"),
                "avg_delta_peak_snr": label_avg("delta_peak_snr"),
                "avg_delta_car": label_avg("delta_car"),
                "avg_delta_zz_correlation": label_avg("delta_zz_correlation"),
                "avg_delta_link_temperature_a_c": label_avg("delta_link_temperature_a_c"),
                "avg_delta_link_temperature_b_c": label_avg("delta_link_temperature_b_c"),
            }
        )

    return {"per_sample": summaries, "by_label": label_rollup}


def _is_valid_summary(summary: dict) -> tuple[bool, list[str]]:
    label = str(summary["label"])
    d_peak = abs(float(summary.get("delta_peak_position_ps") or 0.0))
    d_peak_snr = float(summary.get("delta_peak_snr") or 0.0)
    d_car = float(summary.get("delta_car") or 0.0)
    d_corr = float(summary.get("delta_zz_correlation") or 0.0)
    d_coin = float(summary.get("delta_coincidence_rate_hz") or 0.0)
    d_sa = float(summary.get("delta_singles_rate_a_hz") or 0.0)
    d_sb = float(summary.get("delta_singles_rate_b_hz") or 0.0)
    d_temp = max(abs(float(summary.get("delta_link_temperature_a_c") or 0.0)), abs(float(summary.get("delta_link_temperature_b_c") or 0.0)))
    post_peak_abs = float(summary.get("post_peak_position_abs_max_ps") or 0.0)
    post_peak_snr_min = float(summary.get("post_peak_snr_min") or 0.0)
    post_car_min = float(summary.get("post_car_min") or 0.0)
    post_coin_min = float(summary.get("post_coincidence_rate_min_hz") or 0.0)
    post_sa_max = float(summary.get("post_singles_rate_a_max_hz") or 0.0)
    post_sb_max = float(summary.get("post_singles_rate_b_max_hz") or 0.0)
    post_sa_min = float(summary.get("post_singles_rate_a_min_hz") or 0.0)
    post_sb_min = float(summary.get("post_singles_rate_b_min_hz") or 0.0)
    post_corr_min = float(summary.get("post_zz_correlation_min") or 0.0)
    post_temp_min = min(
        float(summary.get("post_link_temperature_a_min_c") or 999.0),
        float(summary.get("post_link_temperature_b_min_c") or 999.0),
    )

    reasons: list[str] = []
    if label == "normal":
        if d_peak > 500.0:
            reasons.append("normal peak drift too large")
        if abs(d_coin) > 80.0:
            reasons.append("normal coincidence drift too large")
        if abs(d_corr) > 0.12:
            reasons.append("normal correlation drift too large")
    elif label == "synchronization_issue":
        if post_peak_snr_min > 6.0:
            reasons.append("synchronization peak degradation too small")
        if post_car_min > 500.0:
            reasons.append("synchronization CAR degradation too small")
        if abs(d_corr) > 0.25:
            reasons.append("synchronization correlation drift too large")
    elif label == "peak_shift_temperature":
        if post_peak_abs < 1200.0:
            reasons.append("temperature peak drift too small")
        if post_temp_min > 10.0:
            reasons.append("temperature telemetry drift too small")
        if abs(d_corr) > 0.18:
            reasons.append("temperature correlation drift too large")
    elif label == "polarization_drift":
        if post_corr_min > 0.2:
            reasons.append("polarization correlation drop too small")
        if d_peak > 1200.0:
            reasons.append("polarization peak drift too large")
    elif label == "raman_noise":
        if max(post_sa_max, post_sb_max) < 5_000.0:
            reasons.append("raman singles growth too small")
        if d_peak > 2_500.0:
            reasons.append("raman peak drift too large")
        if post_corr_min > 0.75:
            reasons.append("raman correlation degradation too small")
    elif label == "loss":
        if post_coin_min > 50.0:
            reasons.append("loss coincidence drop too small")
        if min(post_sa_min, post_sb_min) > min(float(summary.get("pre_singles_rate_a_hz") or 0.0), float(summary.get("pre_singles_rate_b_hz") or 0.0)) * 0.7:
            reasons.append("loss singles drop too small")
        if d_peak > 1_000.0:
            reasons.append("loss peak drift too large")
    return (len(reasons) == 0, reasons)


def _generate_validated_scenario(label: str, item_index: int, preset, skip_raw: bool, max_attempts: int = 8) -> tuple[object, list[dict], dict | None, dict]:
    last_result: tuple[object, list[dict], dict | None, dict] | None = None
    label_index = IMPAIRMENT_LABELS.index(label)
    base_seed = preset.base_seed + label_index * 10_000 + item_index
    for attempt in range(max_attempts):
        scenario = build_scenario(label, item_index, preset, seed=base_seed + attempt * 1_000_000)
        rows, raw = simulate_sequence(scenario, preset, include_raw=not skip_raw)
        validation = _summarize_validation(
            rows,
            [{"sample_id": scenario.sample_id, "label": scenario.impairment_label, "windows": len(rows), "fault_onset_window": scenario.fault_onset_window}],
        )
        summary = validation["per_sample"][0]
        is_valid, reasons = _is_valid_summary(summary)
        summary["validation_passed"] = int(is_valid)
        summary["validation_reasons"] = reasons
        summary["validation_attempt"] = attempt + 1
        if raw is not None:
            raw["validation"] = summary
        last_result = (scenario, rows, raw, summary)
        if is_valid:
            return scenario, rows, raw, summary
    assert last_result is not None
    return last_result


def _generate_unvalidated_scenario(label: str, item_index: int, preset, skip_raw: bool) -> tuple[object, list[dict], dict | None, dict]:
    label_index = IMPAIRMENT_LABELS.index(label)
    scenario = build_scenario(label, item_index, preset, seed=preset.base_seed + label_index * 10_000 + item_index)
    rows, raw = simulate_sequence(scenario, preset, include_raw=not skip_raw)
    validation = _summarize_validation(
        rows,
        [{"sample_id": scenario.sample_id, "label": scenario.impairment_label, "windows": len(rows), "fault_onset_window": scenario.fault_onset_window}],
    )
    summary = validation["per_sample"][0]
    summary["validation_passed"] = None
    summary["validation_reasons"] = ["validation disabled"]
    summary["validation_attempt"] = 1
    if raw is not None:
        raw["validation"] = summary
    return scenario, rows, raw, summary


def _generate_scenario_job(job: tuple[str, int, object, bool, bool]) -> tuple[object, list[dict], dict | None, dict]:
    label, item_index, preset, validation_off, skip_raw = job
    if validation_off:
        return _generate_unvalidated_scenario(label, item_index, preset, skip_raw)
    return _generate_validated_scenario(label, item_index, preset, skip_raw)


def _default_workers() -> int:
    cpu_total = os.cpu_count() or 1
    return max(1, cpu_total - 1)


def _make_logger(log_path: Path):
    def _log(message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    return _log


def _sample_id(label: str, item_index: int) -> str:
    return f"{label}_{item_index:03d}"


def _checkpoint_path(checkpoint_dir: Path, label: str, item_index: int) -> Path:
    return checkpoint_dir / f"{_sample_id(label, item_index)}.json"


def _write_checkpoint(checkpoint_dir: Path, result: tuple[object, list[dict], dict | None, dict]) -> Path:
    """Persist one completed scenario atomically so interrupted runs can resume."""

    scenario, rows, raw, validation_info = result
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{scenario.sample_id}.json"
    tmp_path = path.with_suffix(".json.tmp")
    payload = {
        "sample_id": scenario.sample_id,
        "label": scenario.impairment_label,
        "scenario": scenario.to_metadata(),
        "observable_rows": rows,
        "raw": raw,
        "validation_info": validation_info,
    }
    tmp_path.write_text(json.dumps(payload), encoding="utf-8")
    tmp_path.replace(path)
    return path


def _read_checkpoint(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _valid_checkpoint_ids(checkpoint_dir: Path) -> set[str]:
    if not checkpoint_dir.exists():
        return set()

    completed: set[str] = set()
    for path in checkpoint_dir.glob("*.json"):
        try:
            payload = _read_checkpoint(path)
        except (OSError, json.JSONDecodeError):
            continue
        sample_id = str(payload.get("sample_id", path.stem))
        if payload.get("observable_rows") and payload.get("scenario") and payload.get("validation_info"):
            completed.add(sample_id)
    return completed


def _aggregate_checkpoint(
    path: Path,
    *,
    output_dir: Path,
    raw_dir: Path,
    skip_raw: bool,
) -> tuple[dict, list[dict], dict, dict | None, dict]:
    """Read one checkpoint and convert it into final dataset rows."""

    payload = _read_checkpoint(path)
    scenario_metadata = payload["scenario"]
    rows = payload["observable_rows"]
    validation_info = payload["validation_info"]
    sample_id = str(scenario_metadata["sample_id"])
    label = str(scenario_metadata["impairment_label"])
    raw = payload.get("raw")

    sequence_row = {
        "sample_id": sample_id,
        "label": label,
        "trajectory_variant": scenario_metadata["trajectory_variant"],
        "windows": len(rows),
        "fault_onset_window": scenario_metadata["fault_onset_window"],
        "validation_attempt": validation_info["validation_attempt"],
        "validation_passed": validation_info["validation_passed"],
    }
    validation_attempt = {
        "sample_id": sample_id,
        "label": label,
        "trajectory_variant": scenario_metadata["trajectory_variant"],
        "validation_attempt": validation_info["validation_attempt"],
        "validation_passed": validation_info["validation_passed"],
        "validation_reasons": validation_info["validation_reasons"],
    }

    raw_index_row = None
    if not skip_raw and raw is not None:
        raw_path = raw_dir / f"{sample_id}.json"
        raw_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        raw_index_row = {
            "sample_id": sample_id,
            "label": label,
            "raw_path": str(raw_path.relative_to(output_dir)),
        }

    return scenario_metadata, rows, sequence_row, raw_index_row, validation_attempt


def main() -> None:
    args = parse_args()
    preset = _apply_overrides(args, get_preset(args.mode))
    workers = max(1, int(args.workers if args.workers is not None else _default_workers()))
    parallel_backend = str(args.parallel_backend)
    resume = bool(args.resume)
    skip_raw = bool(args.skip_raw)
    validation_off = bool(args.validation_off)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or (default_output_root() / f"{preset.name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "run.log"
    log = _make_logger(log_path)
    raw_dir = output_dir / "raw_sequences"
    if not skip_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    observable_rows: list[dict] = []
    sequence_rows: list[dict] = []
    raw_index: list[dict] = []
    validation_attempts: list[dict] = []

    all_jobs = [(label, item_index, preset, validation_off, skip_raw) for label in IMPAIRMENT_LABELS for item_index in range(preset.samples_per_label)]
    expected_sample_ids = [_sample_id(label, item_index) for label in IMPAIRMENT_LABELS for item_index in range(preset.samples_per_label)]
    completed_sample_ids = _valid_checkpoint_ids(checkpoint_dir) if resume else set()
    jobs = [job for job in all_jobs if _sample_id(job[0], job[1]) not in completed_sample_ids]
    total_jobs = len(all_jobs)
    pending_jobs = len(jobs)
    started = time.time()
    log(
        "Starting dataset generation "
        f"mode={preset.name} total_jobs={total_jobs} pending_jobs={pending_jobs} "
        f"resume_completed={len(completed_sample_ids)} workers={workers} parallel_backend={parallel_backend} "
        f"episode_duration_s={preset.episode_duration_s} window_duration_s={preset.window_duration_s} "
        f"source_frequency_hz={preset.source_frequency_hz} temperature_episode_span_s={preset.temperature_episode_span_s} "
        f"skip_raw={skip_raw} validation_off={validation_off} resume={resume}"
    )
    if pending_jobs == 0:
        log("All requested sample checkpoints already exist; aggregating final dataset files.")
    elif workers == 1:
        for completed, job in enumerate(jobs, start=1):
            result = _generate_scenario_job(job)
            _write_checkpoint(checkpoint_dir, result)
            elapsed = time.time() - started
            rate = completed / elapsed if elapsed > 0 else 0.0
            remaining = (pending_jobs - completed) / rate if rate > 0 else 0.0
            global_completed = len(completed_sample_ids) + completed
            log(
                f"Progress {global_completed}/{total_jobs} "
                f"({global_completed / total_jobs:.1%}) elapsed={elapsed/60:.1f}m eta={remaining/60:.1f}m "
                f"latest={result[0].sample_id}"
            )
    else:
        executor_cls = ThreadPoolExecutor if parallel_backend == "thread" else ProcessPoolExecutor
        with executor_cls(max_workers=workers) as executor:
            future_to_job = {executor.submit(_generate_scenario_job, job): job for job in jobs}
            completed = 0
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                label, item_index, *_ = job
                try:
                    result = future.result()
                except Exception as exc:
                    log(f"FAILED job label={label} item_index={item_index}: {exc!r}")
                    raise
                _write_checkpoint(checkpoint_dir, result)
                completed += 1
                elapsed = time.time() - started
                rate = completed / elapsed if elapsed > 0 else 0.0
                remaining = (pending_jobs - completed) / rate if rate > 0 else 0.0
                global_completed = len(completed_sample_ids) + completed
                log(
                    f"Progress {global_completed}/{total_jobs} "
                    f"({global_completed / total_jobs:.1%}) elapsed={elapsed/60:.1f}m eta={remaining/60:.1f}m "
                    f"latest={result[0].sample_id}"
                )

    missing = [sample_id for sample_id in expected_sample_ids if not (checkpoint_dir / f"{sample_id}.json").exists()]
    if missing:
        preview = ", ".join(missing[:10])
        raise RuntimeError(f"Missing {len(missing)} checkpoints after generation; first missing: {preview}")

    scenarios = []
    for label in IMPAIRMENT_LABELS:
        for item_index in range(preset.samples_per_label):
            checkpoint = _checkpoint_path(checkpoint_dir, label, item_index)
            scenario_metadata, rows, sequence_row, raw_index_row, validation_attempt = _aggregate_checkpoint(
                checkpoint,
                output_dir=output_dir,
                raw_dir=raw_dir,
                skip_raw=skip_raw,
            )
            scenarios.append(scenario_metadata)
            observable_rows.extend(rows)
            sequence_rows.append(sequence_row)
            validation_attempts.append(validation_attempt)
            if raw_index_row is not None:
                raw_index.append(raw_index_row)

    observable_df = pd.DataFrame(observable_rows).sort_values(["sample_id", "window_index"]).reset_index(drop=True)
    sequence_df = pd.DataFrame(sequence_rows).sort_values(["label", "sample_id"]).reset_index(drop=True)

    observable_df.to_csv(output_dir / "observable_windows.csv", index=False)
    observable_df.to_json(output_dir / "observable_windows.json", orient="records", indent=2)
    sequence_df.to_csv(output_dir / "sequence_labels.csv", index=False)
    validation = _summarize_validation(observable_rows, sequence_rows)
    pd.DataFrame(validation["per_sample"]).sort_values(["label", "sample_id"]).to_csv(output_dir / "validation_report.csv", index=False)
    pd.DataFrame(validation_attempts).sort_values(["label", "sample_id"]).to_csv(output_dir / "validation_attempts.csv", index=False)

    metadata = {
        "mode": preset.name,
        "created_at": datetime.now().isoformat(),
        "labels": IMPAIRMENT_LABELS,
        "samples_per_label": preset.samples_per_label,
        "episode_duration_s": preset.episode_duration_s,
        "window_duration_s": preset.window_duration_s,
        "temperature_episode_span_s": preset.temperature_episode_span_s,
        "windows_per_sample": preset.windows_per_sample,
        "source_frequency_hz": preset.source_frequency_hz,
        "visibility_measurement_stride_windows": preset.visibility_measurement_stride_windows,
        "workers": workers,
        "parallel_backend": parallel_backend,
        "resume": resume,
        "checkpointing": {
            "enabled": True,
            "checkpoints_dir": "checkpoints",
            "checkpoint_count": len(expected_sample_ids),
        },
        "skip_raw": skip_raw,
        "validation_off": validation_off,
        "total_sequences": len(sequence_rows),
        "total_windows": len(observable_rows),
        "visibility_angles_deg": list(preset.visibility_angles_deg),
        "histogram_settings": {
            "range_ps": preset.histogram.range_ps,
            "bin_width_ps": preset.histogram.bin_width_ps,
            "coincidence_window_ps": preset.histogram.coincidence_window_ps,
        },
        "ai_visible_observables": [
            "arm_length_a_m",
            "arm_length_b_m",
            "physical_time_s",
            "physical_time_h",
            "physical_window_start_s",
            "physical_window_end_s",
            "physical_window_spacing_s",
            "acquisition_window_duration_s",
            "peak_position_ps",
            "peak_width_ps",
            "peak_height",
            "peak_snr",
            "accidental_coincidence_rate_hz",
            "car",
            "coincidence_rate_hz",
            "coincidences_count",
            "singles_rate_a_hz",
            "singles_rate_b_hz",
            "coincidence_to_singles_a",
            "coincidence_to_singles_b",
            "hh_rate_z_hz",
            "hv_rate_z_hz",
            "vh_rate_z_hz",
            "vv_rate_z_hz",
            "zz_correlation",
            "pp_rate_x_hz",
            "pm_rate_x_hz",
            "mp_rate_x_hz",
            "mm_rate_x_hz",
            "xx_correlation",
            "xx_measured",
            "link_temperature_a_c",
            "link_temperature_b_c",
            "visibility",
            "visibility_measured",
            "delta_peak_position_ps",
            "delta_coincidence_rate_hz",
            "delta_singles_rate_a_hz",
            "delta_singles_rate_b_hz",
            "delta_zz_correlation",
            "rolling_peak_position_ps_mean_3",
            "rolling_coincidence_rate_hz_mean_3",
            "rolling_singles_rate_a_hz_mean_3",
            "rolling_singles_rate_b_hz_mean_3",
            "rolling_zz_correlation_mean_3",
        ],
        "hidden_metadata_not_for_ai_input": [
            "attenuation schedules",
            "classical coexistence powers",
            "birefringence settings",
            "seeds",
        ],
        "outputs": {
            "observable_windows_csv": "observable_windows.csv",
            "observable_windows_json": "observable_windows.json",
            "sequence_labels_csv": "sequence_labels.csv",
            "metadata_json": "metadata.json",
            "scenarios_json": "scenarios.json",
            "raw_index_json": "raw_index.json",
            "raw_sequences_dir": "raw_sequences",
            "checkpoints_dir": "checkpoints",
            "validation_report_csv": "validation_report.csv",
            "validation_attempts_csv": "validation_attempts.csv",
        },
    }

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "scenarios.json").write_text(
        json.dumps(scenarios, indent=2),
        encoding="utf-8",
    )
    (output_dir / "raw_index.json").write_text(json.dumps(raw_index, indent=2), encoding="utf-8")
    (output_dir / "validation_summary.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    total_elapsed = time.time() - started
    log(
        f"Completed dataset generation sequences={len(sequence_rows)} windows={len(observable_rows)} "
        f"elapsed={total_elapsed/60:.1f}m output_dir={output_dir}"
    )
    print(output_dir)


if __name__ == "__main__":
    main()
