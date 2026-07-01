"""Batch runner for HOM three-overlap validation.

This script runs the heavy SeQUeNCe simulations outside notebooks and writes
CSV files that notebooks can load for plotting.
"""

from __future__ import annotations

import argparse
import configparser
import fnmatch
import json
import math
import os
import re
import shutil
import tarfile
import tempfile
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "pyproject.toml").exists() and (p / "sequence").exists():
            return p
    raise RuntimeError("Could not find repository root")


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CONFIG_PATH = ROOT / "example" / "hom_validation" / "configs" / "hom_three_overlap_batch_config.ini"
DEFAULT_ENKI_PYTHON_MODULE = "python/3.10.9/anaconda"
DEFAULT_ENKI_TIME_LIMIT = "08:00:00"
DEFAULT_ENKI_PARTITION = "general"
DEFAULT_SYNC_EXCLUDES = [
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    "example/hom_validation/results",
]

from example.hom_validation.simulator import (  # noqa: E402
    CoincidenceConfig,
    SequenceHOMConfig,
    run_hom_sequence_delay_scan,
)


def _float_grid(start: float, stop: float, num: int) -> List[float]:
    return [float(x) for x in np.linspace(float(start), float(stop), int(num))]


def _int_grid(start: int, stop: int, num: int) -> List[int]:
    return [int(x) for x in np.linspace(int(start), int(stop), int(num), dtype=int)]


def _base_config(args: argparse.Namespace) -> SequenceHOMConfig:
    pulses_per_run = max(1, int(round(float(args.duration_s) * float(args.source_frequency_hz))))
    return SequenceHOMConfig(
        pulses_per_delay=pulses_per_run,
        emission_chunk_pulses=int(args.emission_chunk_pulses),
        source_frequency_hz=float(args.source_frequency_hz),
        mean_photon_num=float(args.mean_photon_num),
        source_bandwidth_nm=float(args.source_bandwidth_nm),
        wavelengths_nm_a=(float(args.lambda_a_nm), float(args.lambda_idler_a_nm)),
        wavelengths_nm_b=(float(args.lambda_b_nm), float(args.lambda_idler_b_nm)),
        photon_statistics=str(args.photon_statistics),
        use_sparse_emission=bool(args.use_sparse_emission),
        source_bell_state_a=str(args.bell_state_a),
        source_bell_state_b=str(args.bell_state_b),
        arm_length_m_a=float(args.arm_length_m_a),
        arm_length_m_b=float(args.arm_length_m_b),
        herald_length_m_a=float(args.herald_length_m_a),
        herald_length_m_b=float(args.herald_length_m_b),
        detector_eff_hom=float(args.detector_eff_hom),
        detector_eff_herald=float(args.detector_eff_herald),
        detector_jitter_ps=float(args.detector_jitter_ps),
        detector_dark_hz_hom=float(args.detector_dark_hz_hom),
        detector_dark_hz_herald=float(args.detector_dark_hz_herald),
        hom_match_window_ps=int(args.hom_match_window_ps),
        hom_coinc_window_ps=int(args.hom_window_ps),
        herald_mode=str(args.herald_mode),
        herald_basis=str(args.herald_basis),
        herald_channel_a=int(args.herald_channel_a),
        herald_channel_b=int(args.herald_channel_b),
        extra_overlap_scale=float(args.extra_overlap_scale),
        attenuation_db_per_m=float(args.attenuation_db_per_m),
        seed=int(args.seed),
    )


def _task_seed(base_seed: int, scan: str, point_index: int, run_index: int) -> int:
    scan_offsets = {"temporal": 10_000_000, "polarization": 20_000_000, "spectral": 30_000_000}
    return int(base_seed + scan_offsets[scan] + 10_000 * point_index + run_index)


def _run_task(task: Dict[str, Any]) -> Dict[str, Any]:
    cfg = SequenceHOMConfig(**task["cfg"])
    cfg.seed = int(task["seed"])
    hom_cfg = CoincidenceConfig(**task["hom_cfg"])
    herald_cfg = CoincidenceConfig(**task["herald_cfg"])
    scan = str(task["scan"])

    if scan == "temporal":
        delay_ps = int(task["value"])
        df, _ = run_hom_sequence_delay_scan(
            delays_ps=[delay_ps],
            cfg=cfg,
            hom_cfg=hom_cfg,
            herald_a_rel=herald_cfg,
            herald_b_rel=herald_cfg,
            store_streams=False,
        )
        row = df.iloc[0].to_dict()
        row["scan"] = scan
        row["run_index"] = int(task["run_index"])
        row["point_index"] = int(task["point_index"])
        row["seed"] = int(task["seed"])
        return row

    if scan == "polarization":
        angle_rad = float(task["value"])
        rotate_arm = int(task["rotate_arm"])
        if rotate_arm == 0:
            cfg.pol_rotation_arm0_rad = angle_rad
            cfg.pol_rotation_arm1_rad = 0.0
        else:
            cfg.pol_rotation_arm0_rad = 0.0
            cfg.pol_rotation_arm1_rad = angle_rad
        df, _ = run_hom_sequence_delay_scan(
            delays_ps=[0],
            cfg=cfg,
            hom_cfg=hom_cfg,
            herald_a_rel=herald_cfg,
            herald_b_rel=herald_cfg,
            store_streams=False,
        )
        row = df.iloc[0].to_dict()
        row["scan"] = scan
        row["run_index"] = int(task["run_index"])
        row["point_index"] = int(task["point_index"])
        row["seed"] = int(task["seed"])
        row["angle_rad"] = angle_rad
        row["angle_deg"] = float(np.degrees(angle_rad))
        return row

    if scan == "spectral":
        detuning_nm = float(task["value"])
        lambda_a_nm = float(cfg.wavelengths_nm_a[0])
        cfg.wavelengths_nm_b = (lambda_a_nm + detuning_nm, float(cfg.wavelengths_nm_b[1]))
        df, _ = run_hom_sequence_delay_scan(
            delays_ps=[0],
            cfg=cfg,
            hom_cfg=hom_cfg,
            herald_a_rel=herald_cfg,
            herald_b_rel=herald_cfg,
            store_streams=False,
        )
        row = df.iloc[0].to_dict()
        row["scan"] = scan
        row["run_index"] = int(task["run_index"])
        row["point_index"] = int(task["point_index"])
        row["seed"] = int(task["seed"])
        row["detuning_nm"] = detuning_nm
        return row

    raise ValueError(f"Unsupported scan: {scan}")


def _build_tasks(args: argparse.Namespace) -> List[Dict[str, Any]]:
    cfg = _base_config(args).__dict__.copy()
    hom_cfg = CoincidenceConfig(window_ps=int(args.hom_window_ps), offset_ps=int(args.hom_offset_ps)).__dict__.copy()
    herald_cfg = CoincidenceConfig(window_ps=int(args.herald_window_ps), offset_ps=None).__dict__.copy()
    tasks: List[Dict[str, Any]] = []

    scans = ["temporal", "polarization", "spectral"] if args.scan == "all" else [args.scan]
    for scan in scans:
        if scan == "temporal":
            values: Iterable[float | int] = _int_grid(-args.temporal_half_range_ps, args.temporal_half_range_ps, args.temporal_points)
        elif scan == "polarization":
            values = [math.radians(x) for x in _float_grid(0.0, 90.0, args.polarization_points)]
        elif scan == "spectral":
            values = _float_grid(-args.spectral_half_range_nm, args.spectral_half_range_nm, args.spectral_points)
        else:
            raise ValueError(f"Unsupported scan: {scan}")

        for point_index, value in enumerate(values):
            for run_index in range(int(args.runs_per_point)):
                tasks.append(
                    {
                        "scan": scan,
                        "value": value,
                        "point_index": int(point_index),
                        "run_index": int(run_index),
                        "seed": _task_seed(int(args.seed), scan, point_index, run_index),
                        "cfg": cfg,
                        "hom_cfg": hom_cfg,
                        "herald_cfg": herald_cfg,
                        "rotate_arm": int(args.rotate_arm),
                    }
                )
    num_shards = max(1, int(args.num_shards))
    shard_index = int(args.shard_index)
    if num_shards > 1:
        if shard_index < 0 or shard_index >= num_shards:
            raise ValueError(f"shard_index must be in [0, {num_shards}), got {shard_index}")
        tasks = [task for i, task in enumerate(tasks) if i % num_shards == shard_index]
    return tasks


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _weighted_mean(df: pd.DataFrame, col: str, weight_col: str) -> float:
    if col not in df or weight_col not in df:
        return float(df[col].mean()) if col in df else 0.0
    weights = df[weight_col].astype(float)
    total = float(weights.sum())
    if total <= 0:
        return 0.0
    return float((df[col].astype(float) * weights).sum() / total)


def _pulses_per_run(args: argparse.Namespace) -> int:
    return max(1, int(round(float(args.duration_s) * float(args.source_frequency_hz))))


def _aggregate_runs(
    df: pd.DataFrame,
    group_cols: List[str],
    pulses_per_run: int,
    source_frequency_hz: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    duration_per_run_s = _safe_div(float(pulses_per_run), float(source_frequency_hz))
    for keys, group in df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: Dict[str, Any] = dict(zip(group_cols, keys))
        row["runs"] = int(len(group))
        row["pulses_per_run"] = int(pulses_per_run)
        row["total_pulses"] = int(pulses_per_run * len(group))
        row["source_frequency_hz"] = float(source_frequency_hz)
        row["duration_per_run_s"] = float(duration_per_run_s)
        row["total_duration_s"] = float(duration_per_run_s * len(group))

        for col in group.columns:
            if col in group_cols or col in {"scan", "run_index", "point_index", "seed"}:
                continue
            if col.endswith("_count") or col.endswith("_pairs") or col in {"interfering_pairs"}:
                row[col] = int(group[col].sum())

        # Weighted overlap diagnostics.
        row["mean_total_overlap"] = _weighted_mean(group, "mean_total_overlap", "interfering_pairs")
        row["mean_temporal_overlap"] = _weighted_mean(group, "mean_temporal_overlap", "interfering_pairs")
        row["mean_spectral_overlap"] = _weighted_mean(group, "mean_spectral_overlap", "interfering_pairs")
        row["mean_polarization_overlap"] = _weighted_mean(group, "mean_polarization_overlap", "interfering_pairs")
        row["matched_channel_mean_polarization_overlap"] = _weighted_mean(
            group, "matched_channel_mean_polarization_overlap", "matched_channel_projected_pairs"
        )
        row["mismatched_channel_mean_polarization_overlap"] = _weighted_mean(
            group, "mismatched_channel_mean_polarization_overlap", "mismatched_channel_projected_pairs"
        )

        total_pulses = float(row["total_pulses"])
        total_duration_s = float(row["total_duration_s"])
        for count_col, rate_col in [
            ("hom_twofold_count", "hom_twofold_rate_per_pulse"),
            ("hom_fourfold_count", "hom_fourfold_rate_per_pulse"),
            ("selected_herald_fourfold_count", "selected_herald_fourfold_rate_per_pulse"),
            ("matched_herald_fourfold_count", "matched_herald_fourfold_rate_per_pulse"),
            ("mismatched_herald_fourfold_count", "mismatched_herald_fourfold_rate_per_pulse"),
            ("matched_projected_coincidence_count", "matched_projected_coincidence_rate_per_pulse"),
            ("mismatched_projected_coincidence_count", "mismatched_projected_coincidence_rate_per_pulse"),
            ("interfering_coincidence_count", "interfering_coincidence_rate_per_pulse"),
        ]:
            if count_col in row:
                row[rate_col] = _safe_div(float(row[count_col]), total_pulses)
                row[count_col.replace("_count", "_count_per_s")] = _safe_div(float(row[count_col]), total_duration_s)

        row["matched_projected_coincidence_prob"] = _safe_div(
            float(row.get("matched_projected_coincidence_count", 0)),
            float(row.get("matched_projected_interfering_pairs", 0)),
        )
        row["mismatched_projected_coincidence_prob"] = _safe_div(
            float(row.get("mismatched_projected_coincidence_count", 0)),
            float(row.get("mismatched_projected_interfering_pairs", 0)),
        )
        row["matched_channel_projected_coincidence_prob"] = _safe_div(
            float(row.get("matched_channel_projected_coincidence_count", 0)),
            float(row.get("matched_channel_projected_pairs", 0)),
        )
        row["mismatched_channel_projected_coincidence_prob"] = _safe_div(
            float(row.get("mismatched_channel_projected_coincidence_count", 0)),
            float(row.get("mismatched_channel_projected_pairs", 0)),
        )
        row["interfering_coincidence_prob"] = _safe_div(
            float(row.get("interfering_coincidence_count", 0)),
            float(row.get("interfering_pairs", 0)),
        )

        for col in [
            "fiber_dcd_a_ps_per_nm_km",
            "fiber_dcd_b_ps_per_nm_km",
            "raman_noise_rate_total_hz",
            "heraldA_offset_ps_used",
            "heraldB_offset_ps_used",
            "herald_mode",
            "herald_basis",
            "herald_channel_a",
            "herald_channel_b",
        ]:
            if col in group.columns:
                row[col] = group[col].iloc[0]
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def _write_outputs(results: pd.DataFrame, out_dir: Path, pulses_per_run: int, source_frequency_hz: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if results.empty:
        raise RuntimeError("No simulation results were produced")

    for scan, scan_df in results.groupby("scan", sort=True):
        if scan == "temporal":
            run_name = "hom_temporal_runs.csv"
            summary_name = "hom_temporal_delay_scan_validation.csv"
            group_cols = ["delay_ps"]
        elif scan == "polarization":
            run_name = "hom_polarization_runs.csv"
            summary_name = "hom_polarization_scan_validation.csv"
            group_cols = ["angle_rad", "angle_deg"]
        elif scan == "spectral":
            run_name = "hom_spectral_runs.csv"
            summary_name = "hom_spectral_scan_validation.csv"
            group_cols = ["detuning_nm"]
        else:
            continue

        scan_df.to_csv(out_dir / run_name, index=False)
        summary = _aggregate_runs(scan_df, group_cols, pulses_per_run, source_frequency_hz)
        summary.to_csv(out_dir / summary_name, index=False)
        print(f"wrote {out_dir / run_name}")
        print(f"wrote {out_dir / summary_name}")


def _run_local(args: argparse.Namespace) -> None:
    tasks = _build_tasks(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hom_batch_manifest.json").write_text(
        json.dumps({"args": vars(args), "task_count": len(tasks)}, indent=2, default=str),
        encoding="utf-8",
    )

    workers = int(args.workers or os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
    workers = max(1, min(workers, len(tasks)))
    print(f"running {len(tasks)} tasks with {workers} workers")

    completed_jsonl = out_dir / "hom_completed_rows.jsonl"
    failure_json = out_dir / "hom_failed_task.json"
    completed_jsonl.unlink(missing_ok=True)
    failure_json.unlink(missing_ok=True)

    def record_completed(row: Dict[str, Any]) -> None:
        rows.append(row)
        with completed_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()

    def write_partial_outputs() -> None:
        if not rows:
            return
        partial = pd.DataFrame(rows)
        partial.to_csv(out_dir / "hom_all_runs_partial.csv", index=False)

    rows: List[Dict[str, Any]] = []
    if workers == 1:
        for idx, task in enumerate(tasks, start=1):
            try:
                record_completed(_run_task(task))
            except BaseException as exc:
                failure_json.write_text(
                    json.dumps({"task": task, "error": repr(exc), "completed_rows": len(rows)}, indent=2, default=str),
                    encoding="utf-8",
                )
                write_partial_outputs()
                raise
            print(f"[{idx}/{len(tasks)}] {task['scan']} point={task['point_index']} run={task['run_index']}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_task = {executor.submit(_run_task, task): task for task in tasks}
            for idx, future in enumerate(as_completed(future_to_task), start=1):
                task = future_to_task[future]
                try:
                    record_completed(future.result())
                except BaseException as exc:
                    failure_json.write_text(
                        json.dumps({"task": task, "error": repr(exc), "completed_rows": len(rows)}, indent=2, default=str),
                        encoding="utf-8",
                    )
                    write_partial_outputs()
                    raise
                print(f"[{idx}/{len(tasks)}] {task['scan']} point={task['point_index']} run={task['run_index']}")

    all_results = pd.DataFrame(rows)
    all_results.to_csv(out_dir / "hom_all_runs.csv", index=False)
    _write_outputs(all_results, out_dir, _pulses_per_run(args), float(args.source_frequency_hz))
    print(f"wrote {out_dir / 'hom_all_runs.csv'}")


def _combine_shards(args: argparse.Namespace) -> None:
    shards_dir = Path(args.shards_dir)
    out_dir = Path(args.output_dir)
    frames = []
    used_files = []
    for shard_dir in sorted(shards_dir.glob("shard_*")):
        csv_path = shard_dir / "hom_all_runs.csv"
        partial_csv_path = shard_dir / "hom_all_runs_partial.csv"
        jsonl_path = shard_dir / "hom_completed_rows.jsonl"
        if csv_path.exists():
            frames.append(pd.read_csv(csv_path))
            used_files.append(csv_path)
        elif partial_csv_path.exists():
            frames.append(pd.read_csv(partial_csv_path))
            used_files.append(partial_csv_path)
        elif jsonl_path.exists():
            rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if rows:
                frames.append(pd.DataFrame(rows))
                used_files.append(jsonl_path)
    if not frames:
        raise FileNotFoundError(f"No shard result files found under {shards_dir}")
    all_results = pd.concat(frames, ignore_index=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(out_dir / "hom_all_runs.csv", index=False)
    _write_outputs(all_results, out_dir, _pulses_per_run(args), float(args.source_frequency_hz))
    print(f"combined {len(used_files)} shard result files into {out_dir}")
    if shards_dir.exists():
        shutil.rmtree(shards_dir)
        print(f"removed temporary shard directory {shards_dir}")


def _shell_quote(value: str) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _sync_excludes(args: argparse.Namespace) -> List[str]:
    patterns = getattr(args, "sync_excludes", DEFAULT_SYNC_EXCLUDES)
    if isinstance(patterns, str):
        patterns = [p.strip() for p in patterns.split(",") if p.strip()]
    return [str(p).strip().replace("\\", "/").strip("/") for p in patterns if str(p).strip()]


def _is_excluded(path: Path, patterns: List[str]) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    parts = set(path.relative_to(ROOT).parts)
    for pattern in patterns:
        if not pattern:
            continue
        if pattern in parts:
            return True
        if rel == pattern or rel.startswith(pattern.rstrip("/") + "/"):
            return True
        if fnmatch.fnmatch(rel, pattern):
            return True
    return False


def _make_repo_archive(args: argparse.Namespace) -> Path:
    patterns = _sync_excludes(args)
    tmp = tempfile.NamedTemporaryFile(prefix="hom_repo_", suffix=".tar.gz", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    with tarfile.open(tmp_path, "w:gz") as archive:
        for path in ROOT.rglob("*"):
            if _is_excluded(path, patterns):
                continue
            archive.add(path, arcname=path.relative_to(ROOT), recursive=False)
    return tmp_path


def _sync_project_to_enki(args: argparse.Namespace) -> None:
    remote_repo = str(args.remote_repo)
    if not remote_repo or remote_repo.lower() == "auto" or "$SLURM_SUBMIT_DIR" in remote_repo:
        raise ValueError("sync_project requires a resolved remote_repo path. Use remote_repo='auto' and let _run_enki resolve it.")

    archive = _make_repo_archive(args)
    remote_tmp = f"/tmp/hom_repo_{os.getpid()}.tar.gz"
    try:
        print(f"syncing local repo to {args.host}:{remote_repo}")
        subprocess.run(["ssh", args.host, f"mkdir -p {_shell_quote(remote_repo)}"], check=True)
        subprocess.run(["scp", str(archive), f"{args.host}:{remote_tmp}"], check=True)
        extract_cmd = (
            f"mkdir -p {_shell_quote(remote_repo)} && "
            f"tar -xzf {_shell_quote(remote_tmp)} -C {_shell_quote(remote_repo)} && "
            f"rm -f {_shell_quote(remote_tmp)}"
        )
        subprocess.run(["ssh", args.host, extract_cmd], check=True)
    finally:
        archive.unlink(missing_ok=True)


def _copy_file_to_enki(args: argparse.Namespace, local_path: Path, remote_rel_path: str) -> None:
    remote_repo = str(args.remote_repo)
    remote_path = f"{remote_repo.rstrip('/')}/{remote_rel_path}"
    remote_dir = str(Path(remote_path).parent).replace("\\", "/")
    subprocess.run(["ssh", args.host, f"mkdir -p {_shell_quote(remote_dir)}"], check=True)
    subprocess.run(["scp", str(local_path), f"{args.host}:{remote_path}"], check=True)


def _write_enki_slurm(args: argparse.Namespace) -> Path:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    script_path = out_dir / "run_hom_three_overlap_enki.slurm"
    local_args = [a for a in sys.argv[1:] if a not in ("--target", "enki", "--submit", "--no-submit")]
    # If launched from Windows, path arguments in sys.argv may contain backslashes.
    # The generated Slurm script runs on Linux, so normalize them before embedding.
    local_args = [a.replace("\\", "/") for a in local_args]
    for opt in (
        "--target",
        "--workers",
        "--shard-index",
        "--num-shards",
        "--combine-shards",
        "--shards-dir",
        "--output-dir",
        "--nodes",
        "--cpus-per-task",
        "--sync-project",
        "--no-sync-project",
        "--wait-for-completion",
        "--no-wait-for-completion",
        "--collect-outputs",
        "--no-collect-outputs",
    ):
        while opt in local_args:
            i = local_args.index(opt)
            del local_args[i : i + 1 if opt.startswith("--no-") or opt in ("--combine-shards", "--sync-project", "--wait-for-completion", "--collect-outputs") else i + 2]
    script_rel = Path(__file__).resolve().relative_to(ROOT).as_posix()
    slurm_tasks = max(1, int(args.nodes))
    cpus_per_task = max(1, int(args.cpus_per_task))
    python_cmd = str(args.python_cmd)
    partition = str(getattr(args, "partition", "") or "").strip()
    partition_line = f"#SBATCH --partition={partition}" if partition else ""
    time_limit = str(getattr(args, "time_limit", "") or "").strip()
    time_line = f"#SBATCH --time={time_limit}" if time_limit else ""
    module_load = str(getattr(args, "module_load", "") or "").strip()
    module_line = f"module load {module_load}" if module_load else ""
    dependency_lines = ""
    if _as_bool(getattr(args, "install_missing_dependencies", False)):
        dependency_lines = f"""
echo "Checking Python dependencies"
{python_cmd} - <<'PY' || ({python_cmd} -m pip install --user 'Cython<3' && {python_cmd} -m pip install --user --no-build-isolation 'qutip==4.7.6' 'qutip-qip==0.3.2')
import qutip
import qutip_qip
PY
"""
    shard_dir = f"{out_dir.as_posix()}/shards/shard_$SLURM_PROCID"
    run_cmd = " ".join(
        [
            python_cmd,
            script_rel,
            "--target",
            "local",
            *local_args,
            "--workers",
            str(cpus_per_task),
            "--shard-index",
            "$SLURM_PROCID",
            "--num-shards",
            "$SLURM_NTASKS",
            "--output-dir",
            shard_dir,
        ]
    )
    combine_cmd = " ".join(
        [
            python_cmd,
            script_rel,
            "--target",
            "local",
            *local_args,
            "--combine-shards",
            "--shards-dir",
            f"{out_dir.as_posix()}/shards",
            "--output-dir",
            out_dir.as_posix(),
        ]
    )
    remote_repo = str(args.remote_repo)
    script = f"""#!/bin/bash
#SBATCH --job-name=hom3
{partition_line}
#SBATCH --nodes={int(args.nodes)}
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task={cpus_per_task}
{time_line}
#SBATCH --output={out_dir.as_posix()}/hom3_%j.out
#SBATCH --error={out_dir.as_posix()}/hom3_%j.err

set -euo pipefail
cd {remote_repo}
{module_line}
{dependency_lines}
mkdir -p {out_dir.as_posix()}/shards
echo "Launching HOM shards: nodes=$SLURM_NNODES tasks=$SLURM_NTASKS cpus_per_task=$SLURM_CPUS_PER_TASK"
srun --nodes=$SLURM_NNODES --ntasks={slurm_tasks} --ntasks-per-node=1 bash -lc '{run_cmd}'
echo "Combining shard outputs"
{combine_cmd}
"""
    script_path.write_text(script, encoding="utf-8", newline="\n")
    print(f"wrote {script_path}")
    print(f"On Enki, from the repo root, run: sbatch {script_path.as_posix()}")
    return script_path


def _submit_enki_job(args: argparse.Namespace, script_path: Path) -> str:
    remote_repo = str(args.remote_repo)
    if remote_repo.lower() == "auto" or "$SLURM_SUBMIT_DIR" in remote_repo:
        raise ValueError(
            "Automatic SSH submission requires remote_repo to be an absolute path on Enki. "
            "Use submit=false and run sbatch from the repo root on Enki, or set remote_repo explicitly."
        )
    remote_cmd = f"cd {_shell_quote(remote_repo)} && sbatch --parsable {script_path.as_posix()}"
    print(f"submitting with ssh {args.host!r}: {remote_cmd}")
    result = subprocess.run(["ssh", args.host, remote_cmd], check=True, text=True, capture_output=True)
    output = result.stdout.strip()
    match = re.search(r"(\d+)", output)
    if not match:
        raise RuntimeError(f"Could not parse sbatch job id from output: {output!r}")
    job_id = match.group(1)
    print(f"submitted Enki job {job_id}")
    return job_id


def _wait_for_enki_job(args: argparse.Namespace, job_id: str) -> None:
    poll_seconds = max(5, int(args.poll_seconds))
    print(f"waiting for Enki job {job_id}; polling every {poll_seconds}s")
    while True:
        cmd = f"squeue -h -j {_shell_quote(job_id)}"
        result = subprocess.run(["ssh", args.host, cmd], text=True, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"squeue failed for job {job_id}")
        if not result.stdout.strip():
            break
        time.sleep(poll_seconds)
    print(f"job {job_id} is no longer in the Slurm queue")


def _collect_enki_outputs(args: argparse.Namespace, job_id: str | None = None) -> None:
    remote_repo = str(args.remote_repo)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    remote_output = f"{remote_repo.rstrip('/')}/{Path(args.output_dir).as_posix()}"
    remote_tmp = f"/tmp/hom_outputs_{job_id or os.getpid()}.tar.gz"
    local_tmp = out_dir / f"hom_outputs_{job_id or 'latest'}.tar.gz"
    print(f"collecting outputs from {args.host}:{remote_output}")
    pack_cmd = f"tar -czf {_shell_quote(remote_tmp)} -C {_shell_quote(remote_output)} ."
    subprocess.run(["ssh", args.host, pack_cmd], check=True)
    try:
        subprocess.run(["scp", f"{args.host}:{remote_tmp}", str(local_tmp)], check=True)
        with tarfile.open(local_tmp, "r:gz") as archive:
            archive.extractall(out_dir)
    finally:
        local_tmp.unlink(missing_ok=True)
        subprocess.run(["ssh", args.host, f"rm -f {_shell_quote(remote_tmp)}"], check=False)
    print(f"collected outputs into {out_dir}")


def _clean_old_slurm_logs(args: argparse.Namespace) -> None:
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("hom3_*.out", "hom3_*.err"):
        for path in out_dir.glob(pattern):
            path.unlink(missing_ok=True)

    remote_repo = str(args.remote_repo)
    if _as_bool(args.sync_project) or args.submit:
        remote_output = f"{remote_repo.rstrip('/')}/{Path(args.output_dir).as_posix()}"
        cmd = f"mkdir -p {_shell_quote(remote_output)} && rm -f {_shell_quote(remote_output)}/hom3_*.out {_shell_quote(remote_output)}/hom3_*.err"
        subprocess.run(["ssh", args.host, cmd], check=False)


def _resolve_remote_repo(args: argparse.Namespace) -> None:
    remote_repo = str(args.remote_repo or "").strip()
    if remote_repo and remote_repo.lower() != "auto":
        return
    result = subprocess.run(
        ["ssh", args.host, "printf %s \"$HOME\""],
        check=True,
        text=True,
        capture_output=True,
    )
    remote_home = result.stdout.strip()
    if not remote_home:
        raise RuntimeError(f"Could not resolve remote HOME on {args.host}")
    args.remote_repo = f"{remote_home.rstrip('/')}/{ROOT.name}"
    print(f"resolved remote_repo={args.remote_repo}")


def _run_enki(args: argparse.Namespace) -> None:
    _resolve_remote_repo(args)
    if _as_bool(getattr(args, "clean_old_logs", True)):
        _clean_old_slurm_logs(args)
    script_path = _write_enki_slurm(args)
    if _as_bool(args.sync_project):
        _sync_project_to_enki(args)
        script_abs = script_path if script_path.is_absolute() else ROOT / script_path
        script_rel = script_abs.relative_to(ROOT).as_posix()
        _copy_file_to_enki(args, script_path, script_rel)
    if not args.submit:
        return
    job_id = _submit_enki_job(args, script_path)
    if _as_bool(args.wait_for_completion):
        _wait_for_enki_job(args, job_id)
        if _as_bool(args.collect_outputs):
            _collect_enki_outputs(args, job_id)


def _load_config_defaults(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        return {}
    suffix = config_path.suffix.lower()
    if suffix == ".ini":
        data = _load_ini_config(config_path)
    else:
        raise ValueError(f"Unsupported config type {config_path.suffix!r}; use .ini")
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a top-level key/value table: {config_path}")
    return data


def _default_output_dir_for_config(config_path: Path) -> str:
    """Return results/<config_stem> next to this validation package."""
    resolved = config_path.resolve()
    package_dir = resolved.parent.parent if resolved.parent.name == "configs" else resolved.parent
    out_dir = package_dir / "results" / resolved.stem
    try:
        return out_dir.relative_to(ROOT).as_posix()
    except ValueError:
        return out_dir.as_posix()


def _load_ini_config(config_path: Path) -> Dict[str, Any]:
    """Load an INI config and infer simple Python value types."""
    parser = configparser.ConfigParser(
        interpolation=None,
        inline_comment_prefixes=("#", ";"),
        empty_lines_in_values=False,
    )
    parser.optionxform = str
    parser.read(config_path, encoding="utf-8-sig")

    data: Dict[str, Any] = {}
    for key, value in parser.defaults().items():
        data[key] = _parse_ini_value(value)
    for section in parser.sections():
        for key, value in parser.items(section):
            data[key] = _parse_ini_value(value)
    return data


def _parse_ini_value(value: str) -> Any:
    value = value.strip()
    lower = value.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if "\n" in value:
        return [line.strip() for line in value.splitlines() if line.strip()]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError as exc:
        if "," in value:
            return [part.strip() for part in value.split(",") if part.strip()]
        return value


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    pre_args, _ = pre.parse_known_args()
    config_path = Path(pre_args.config)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(config_path), help="INI config file. Defaults to hom_three_overlap_batch_config.ini.")
    p.add_argument("--target", choices=["local", "enki"], default="local")
    p.add_argument("--scan", choices=["all", "temporal", "polarization", "spectral"], default="all")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--runs-per-point", type=int, default=1)
    p.add_argument("--duration-s", type=float, default=1e-3, help="Physical simulated duration per run. Pulses are duration_s * source_frequency_hz.")
    p.add_argument("--emission-chunk-pulses", type=int, default=25_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--combine-shards", action="store_true")
    p.add_argument("--shards-dir", default=None)

    p.add_argument("--temporal-half-range-ps", type=int, default=800)
    p.add_argument("--temporal-points", type=int, default=17)
    p.add_argument("--polarization-points", type=int, default=13)
    p.add_argument("--spectral-half-range-nm", type=float, default=0.4)
    p.add_argument("--spectral-points", type=int, default=13)
    p.add_argument("--rotate-arm", type=int, choices=[0, 1], default=1)

    p.add_argument("--source-frequency-hz", type=float, default=8e7)
    p.add_argument("--mean-photon-num", type=float, default=0.01)
    p.add_argument("--source-bandwidth-nm", type=float, default=0.04)
    p.add_argument("--lambda-a-nm", type=float, default=1550.0)
    p.add_argument("--lambda-b-nm", type=float, default=1550.0)
    p.add_argument("--lambda-idler-a-nm", type=float, default=1550.0)
    p.add_argument("--lambda-idler-b-nm", type=float, default=1550.0)
    p.add_argument("--photon-statistics", choices=["thermal", "poisson"], default="thermal")
    p.add_argument("--use-sparse-emission", action="store_true")
    p.add_argument("--no-use-sparse-emission", action="store_false", dest="use_sparse_emission")
    p.add_argument("--bell-state-a", default="psi-")
    p.add_argument("--bell-state-b", default="psi-")

    p.add_argument("--arm-length-m-a", type=float, default=20_000.0)
    p.add_argument("--arm-length-m-b", type=float, default=20_000.0)
    p.add_argument("--herald-length-m-a", type=float, default=1.0)
    p.add_argument("--herald-length-m-b", type=float, default=1.0)
    p.add_argument("--attenuation-db-per-m", type=float, default=0.0002)

    p.add_argument("--detector-eff-hom", type=float, default=1.0)
    p.add_argument("--detector-eff-herald", type=float, default=1.0)
    p.add_argument("--detector-jitter-ps", type=float, default=0.0)
    p.add_argument("--detector-dark-hz-hom", type=float, default=0.0)
    p.add_argument("--detector-dark-hz-herald", type=float, default=0.0)
    p.add_argument("--hom-match-window-ps", type=int, default=1200)
    p.add_argument("--hom-window-ps", type=int, default=120)
    p.add_argument("--hom-offset-ps", type=int, default=0)
    p.add_argument("--herald-window-ps", type=int, default=300)
    p.add_argument("--herald-mode", choices=["projected", "hom_only"], default="projected")
    p.add_argument("--herald-basis", default="Z")
    p.add_argument("--herald-channel-a", type=int, default=0)
    p.add_argument("--herald-channel-b", type=int, default=0)
    p.add_argument("--extra-overlap-scale", type=float, default=1.0)

    p.add_argument("--host", default="enki")
    p.add_argument("--remote-repo", default="$SLURM_SUBMIT_DIR")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--no-submit", action="store_false", dest="submit")
    p.add_argument("--sync-project", action="store_true")
    p.add_argument("--no-sync-project", action="store_false", dest="sync_project")
    p.add_argument("--wait-for-completion", action="store_true")
    p.add_argument("--no-wait-for-completion", action="store_false", dest="wait_for_completion")
    p.add_argument("--collect-outputs", action="store_true")
    p.add_argument("--no-collect-outputs", action="store_false", dest="collect_outputs")
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--sync-excludes", nargs="*", default=DEFAULT_SYNC_EXCLUDES)
    p.add_argument("--install-missing-dependencies", action="store_true")
    p.add_argument("--no-install-missing-dependencies", action="store_false", dest="install_missing_dependencies")
    p.add_argument("--clean-old-logs", action="store_true", default=True)
    p.add_argument("--no-clean-old-logs", action="store_false", dest="clean_old_logs")
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--cpus-per-task", type=int, default=8)
    p.add_argument("--python-cmd", default="python3")
    p.add_argument("--module-load", default=DEFAULT_ENKI_PYTHON_MODULE)
    p.add_argument("--time-limit", default=DEFAULT_ENKI_TIME_LIMIT)
    p.add_argument("--partition", default=DEFAULT_ENKI_PARTITION)

    config_defaults = _load_config_defaults(config_path)
    valid_dests = {action.dest for action in p._actions}
    unknown = sorted(set(config_defaults) - valid_dests)
    if unknown:
        raise ValueError(f"Unknown config keys in {config_path}: {unknown}")
    p.set_defaults(**config_defaults)
    args = p.parse_args()
    args.config = str(config_path)
    if args.output_dir is None:
        args.output_dir = _default_output_dir_for_config(config_path)
    if args.shards_dir is None:
        args.shards_dir = str(Path(args.output_dir) / "shards")
    return args


def main() -> None:
    args = parse_args()
    if args.target == "enki":
        _run_enki(args)
    elif args.combine_shards:
        _combine_shards(args)
    else:
        _run_local(args)


if __name__ == "__main__":
    main()
