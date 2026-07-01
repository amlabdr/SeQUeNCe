"""Generate AI diagnosis datasets from HOM detector observations."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tarfile
import time
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

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

from example.hom_validation.ai_hom_dataset.config import (  # noqa: E402
    FAULT_LABELS,
    DatasetConfig,
    default_output_root,
    load_dataset_config,
)
from example.hom_validation.ai_hom_dataset.scenarios import build_scenario  # noqa: E402
from example.hom_validation.ai_hom_dataset.simulator import simulate_episode  # noqa: E402


DEFAULT_CONFIG = ROOT / "example" / "hom_validation" / "ai_hom_dataset" / "configs" / "hom_ai_dataset_config.ini"
DEFAULT_SYNC_EXCLUDES = [
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    "example/hom_validation/ai_hom_dataset/generated",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--target", choices=["local", "blackbird"], default=None)
    p.add_argument("--labels", nargs="*", default=None, help=f"Subset of labels. Available: {', '.join(FAULT_LABELS)}")
    p.add_argument("--dataset-size", type=int, default=None)
    p.add_argument("--episode-duration-s", type=float, default=None)
    p.add_argument("--window-duration-s", type=float, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--base-seed", type=int, default=None)
    p.add_argument("--shard-index", type=int, default=None)
    p.add_argument("--num-shards", type=int, default=None)
    p.add_argument("--checkpoint-only", action="store_true")
    p.add_argument("--combine-only", action="store_true")
    p.add_argument("--benchmark-one", action="store_true")
    p.add_argument("--submit", action="store_true")
    p.add_argument("--no-submit", action="store_false", dest="submit")
    p.add_argument("--wait-for-completion", action="store_true")
    p.add_argument("--collect-outputs", action="store_true")
    p.add_argument("--write-raw-streams", action="store_true")
    p.add_argument("--no-write-raw-streams", action="store_false", dest="write_raw_streams")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-resume", action="store_false", dest="resume")
    p.set_defaults(write_raw_streams=None, resume=None, submit=None)
    return p.parse_args()


def _apply_overrides(cfg: DatasetConfig, args: argparse.Namespace) -> DatasetConfig:
    gen_updates: dict[str, Any] = {}
    if args.dataset_size is not None:
        gen_updates["dataset_size"] = int(args.dataset_size)
    if args.episode_duration_s is not None:
        gen_updates["episode_duration_s"] = float(args.episode_duration_s)
    if args.window_duration_s is not None:
        gen_updates["window_duration_s"] = float(args.window_duration_s)
    if args.workers is not None:
        gen_updates["workers"] = int(args.workers)
    if args.base_seed is not None:
        gen_updates["base_seed"] = int(args.base_seed)
    if args.write_raw_streams is not None:
        gen_updates["write_raw_streams"] = bool(args.write_raw_streams)
    if args.resume is not None:
        gen_updates["resume"] = bool(args.resume)
    return replace(cfg, generation=replace(cfg.generation, **gen_updates)) if gen_updates else cfg


def _labels_from_args(args: argparse.Namespace) -> list[str]:
    if not args.labels:
        return list(FAULT_LABELS)
    labels = [str(v) for v in args.labels]
    unknown = sorted(set(labels) - set(FAULT_LABELS))
    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")
    return labels


def _job_seed(cfg: DatasetConfig, label: str, episode_index: int) -> int:
    return int(cfg.generation.base_seed + FAULT_LABELS.index(label) * 1_000_000 + episode_index * 10_000)


def _episode_id(label: str, episode_index: int) -> str:
    return f"{label}_{episode_index:05d}"


def _jobs_for_dataset(cfg: DatasetConfig, labels: list[str]) -> list[tuple[str, int, DatasetConfig]]:
    rng = random.Random(int(cfg.generation.base_seed))
    size = max(1, int(cfg.generation.dataset_size))
    return [(rng.choice(labels), episode_index, cfg) for episode_index in range(size)]


def _checkpoint_path(checkpoint_dir: Path, label: str, episode_index: int) -> Path:
    return checkpoint_dir / f"{_episode_id(label, episode_index)}.json"


def _run_job(job: tuple[str, int, DatasetConfig]) -> dict[str, Any]:
    label, episode_index, cfg = job
    seed = _job_seed(cfg, label, episode_index)
    scenario = build_scenario(label, episode_index, cfg, seed=seed)
    rows, truth_rows, raw = simulate_episode(
        scenario,
        cfg,
        include_raw=bool(cfg.generation.write_raw_streams),
    )
    return {
        "episode_id": scenario.episode_id,
        "label": label,
        "scenario": scenario.to_metadata(),
        "observable_rows": rows,
        "hidden_truth_rows": truth_rows,
        "raw": raw,
    }


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validation_summary(observable_rows: list[dict], label_rows: list[dict]) -> list[dict]:
    by_episode: dict[str, list[dict]] = {}
    for row in observable_rows:
        by_episode.setdefault(str(row["episode_id"]), []).append(row)

    out: list[dict] = []
    for label_row in label_rows:
        episode_id = str(label_row["episode_id"])
        rows = sorted(by_episode.get(episode_id, []), key=lambda r: int(r["window_index"]))
        onset = int(label_row["fault_onset_window"])
        pre = rows[:onset] or rows
        post = rows[onset:] or rows

        def avg(items: list[dict], key: str) -> float:
            vals = [float(item.get(key, 0.0)) for item in items]
            return sum(vals) / len(vals) if vals else 0.0

        row = {
            "episode_id": episode_id,
            "fault_class": label_row["fault_class"],
            "fault_onset_window": onset,
            "pre_hom_twofold_rate_hz": avg(pre, "hom_twofold_rate_hz"),
            "post_hom_twofold_rate_hz": avg(post, "hom_twofold_rate_hz"),
            "pre_hom1_rate_hz": avg(pre, "hom1_rate_hz"),
            "post_hom1_rate_hz": avg(post, "hom1_rate_hz"),
            "pre_hom2_rate_hz": avg(pre, "hom2_rate_hz"),
            "post_hom2_rate_hz": avg(post, "hom2_rate_hz"),
            "pre_peak_position_ps": avg(pre, "peak_position_ps"),
            "post_peak_position_ps": avg(post, "peak_position_ps"),
            "pre_peak_snr": avg(pre, "peak_snr"),
            "post_peak_snr": avg(post, "peak_snr"),
            "pre_car": avg(pre, "car"),
            "post_car": avg(post, "car"),
            "pre_accidental_rate_hz": avg(pre, "accidental_rate_hz"),
            "post_accidental_rate_hz": avg(post, "accidental_rate_hz"),
        }
        for key in [
            "hom_twofold_rate_hz",
            "hom1_rate_hz",
            "hom2_rate_hz",
            "peak_position_ps",
            "peak_snr",
            "car",
            "accidental_rate_hz",
        ]:
            row[f"delta_{key}"] = row[f"post_{key}"] - row[f"pre_{key}"]
        out.append(row)
    return out


def _default_output_dir(cfg: DatasetConfig) -> Path:
    configured = str(getattr(cfg.slurm, "output_dir", "") or "").strip()
    return Path(configured) if configured else default_output_root() / cfg.generation.dataset_name


def _filter_jobs_for_shard(jobs: list[tuple[str, int, DatasetConfig]], shard_index: int | None, num_shards: int | None):
    if shard_index is None and num_shards is None:
        return jobs
    shards = max(1, int(num_shards or 1))
    index = int(shard_index or 0)
    if index < 0 or index >= shards:
        raise ValueError(f"shard_index must be in [0, {shards}), got {index}")
    return [job for i, job in enumerate(jobs) if i % shards == index]


def _load_completed_payloads(
    jobs: list[tuple[str, int, DatasetConfig]],
    checkpoint_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    observable_rows: list[dict] = []
    truth_rows: list[dict] = []
    label_rows: list[dict] = []
    scenarios: list[dict] = []
    missing: list[str] = []

    for label, episode_index, _ in jobs:
        checkpoint = _checkpoint_path(checkpoint_dir, label, episode_index)
        if not checkpoint.exists():
            missing.append(checkpoint.stem)
            continue
        payload = _load_checkpoint(checkpoint)
        scenario = payload["scenario"]
        scenarios.append(scenario)
        observable_rows.extend(payload["observable_rows"])
        truth_rows.extend(payload["hidden_truth_rows"])
        label_rows.append(
            {
                "episode_id": scenario["episode_id"],
                "setup_id": scenario["setup_id"],
                "fault_class": scenario["fault_class"],
                "seed": scenario["seed"],
                "fault_onset_window": scenario["fault_onset_window"],
                "target_fault_level": scenario["target_fault_level"],
                "windows": len(scenario["windows"]),
            }
        )
    return observable_rows, truth_rows, label_rows, scenarios, missing


def _write_final_outputs(
    cfg: DatasetConfig,
    args: argparse.Namespace,
    labels: list[str],
    jobs: list[tuple[str, int, DatasetConfig]],
    output_dir: Path,
    started: float,
) -> None:
    checkpoint_dir = output_dir / "checkpoints"
    observable_rows, truth_rows, label_rows, scenarios, missing = _load_completed_payloads(jobs, checkpoint_dir)
    if missing:
        raise RuntimeError(f"Missing checkpoints after generation: {missing[:10]}")

    observable_df = pd.DataFrame(observable_rows).sort_values(["episode_id", "window_index"]).reset_index(drop=True)
    truth_df = pd.DataFrame(truth_rows).sort_values(["episode_id", "window_index"]).reset_index(drop=True)
    labels_df = pd.DataFrame(label_rows).sort_values(["fault_class", "episode_id"]).reset_index(drop=True)
    validation_rows = _validation_summary(observable_rows, label_rows)

    observable_df.to_csv(output_dir / "hom_observable_windows.csv", index=False)
    truth_df.to_csv(output_dir / "hidden_truth.csv", index=False)
    labels_df.to_csv(output_dir / "episode_labels.csv", index=False)
    pd.DataFrame(validation_rows).sort_values(["fault_class", "episode_id"]).to_csv(output_dir / "validation_summary.csv", index=False)
    pd.DataFrame([cfg.setup.__dict__]).to_csv(output_dir / "setup_params.csv", index=False)

    metadata = {
        "created_at": datetime.now().isoformat(),
        "config_path": str(args.config),
        "labels": labels,
        "dataset_name": cfg.generation.dataset_name,
        "dataset_size": int(cfg.generation.dataset_size),
        "episodes": len(label_rows),
        "windows": len(observable_rows),
        "episode_duration_s": cfg.generation.episode_duration_s,
        "window_duration_s": cfg.generation.window_duration_s,
        "hom_only": True,
        "ai_input_file": "hom_observable_windows.csv",
        "label_file": "episode_labels.csv",
        "hidden_truth_file": "hidden_truth.csv",
        "setup_params_file": "setup_params.csv",
        "validation_summary_file": "validation_summary.csv",
        "raw_streams_enabled": bool(cfg.generation.write_raw_streams),
        "elapsed_s": float(time.time() - started),
        "note": "AI inputs are HOM-only. Hidden truth contains simulator-only labels and must not be used as operational input.",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "scenarios.json").write_text(json.dumps(scenarios, indent=2), encoding="utf-8")
    (output_dir / "raw_index.json").write_text(json.dumps([], indent=2), encoding="utf-8")
    print(f"Wrote dataset to {output_dir}", flush=True)


def _shell_quote(value: str) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _uses_plink(cfg: DatasetConfig) -> bool:
    return bool(str(getattr(cfg.slurm, "plink_profile", "") or "").strip())


def _plink_base(cfg: DatasetConfig) -> list[str]:
    base = [str(cfg.slurm.plink_path), "-batch", "-load", str(cfg.slurm.plink_profile)]
    user = str(getattr(cfg.slurm, "plink_user", "") or "").strip()
    if user:
        base.extend(["-l", user])
    return base


def _pscp_base(cfg: DatasetConfig) -> list[str]:
    base = [str(cfg.slurm.pscp_path), "-batch", "-load", str(cfg.slurm.plink_profile)]
    user = str(getattr(cfg.slurm, "plink_user", "") or "").strip()
    if user:
        base.extend(["-l", user])
    return base


def _remote_spec(cfg: DatasetConfig, remote_path: str) -> str:
    return f"{cfg.slurm.host}:{remote_path}"


def _remote_run(cfg: DatasetConfig, command: str, **kwargs):
    if _uses_plink(cfg):
        return subprocess.run([*_plink_base(cfg), command], **kwargs)
    return subprocess.run(["ssh", cfg.slurm.host, command], **kwargs)


def _remote_copy_to(cfg: DatasetConfig, local_path: Path, remote_path: str) -> None:
    if _uses_plink(cfg):
        subprocess.run([*_pscp_base(cfg), str(local_path), _remote_spec(cfg, remote_path)], check=True)
    else:
        subprocess.run(["scp", str(local_path), _remote_spec(cfg, remote_path)], check=True)


def _remote_copy_from(cfg: DatasetConfig, remote_path: str, local_path: Path) -> None:
    if _uses_plink(cfg):
        subprocess.run([*_pscp_base(cfg), _remote_spec(cfg, remote_path), str(local_path)], check=True)
    else:
        subprocess.run(["scp", _remote_spec(cfg, remote_path), str(local_path)], check=True)


def _is_excluded(path: Path, patterns: list[str]) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    parts = set(path.relative_to(ROOT).parts)
    for pattern in patterns:
        clean = str(pattern).strip().replace("\\", "/").strip("/")
        if not clean:
            continue
        if clean in parts:
            return True
        if rel == clean or rel.startswith(clean.rstrip("/") + "/"):
            return True
        if fnmatch.fnmatch(rel, clean):
            return True
    return False


def _make_repo_archive() -> Path:
    tmp = tempfile.NamedTemporaryFile(prefix="hom_ai_repo_", suffix=".tar.gz", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    with tarfile.open(tmp_path, "w:gz") as archive:
        for path in ROOT.rglob("*"):
            if _is_excluded(path, DEFAULT_SYNC_EXCLUDES):
                continue
            archive.add(path, arcname=path.relative_to(ROOT), recursive=False)
    return tmp_path


def _resolve_remote_repo(cfg: DatasetConfig) -> str:
    remote_repo = str(cfg.slurm.remote_repo or "auto").strip()
    if remote_repo and remote_repo.lower() != "auto":
        return remote_repo
    result = _remote_run(
        cfg,
        "printf %s \"$HOME\"",
        check=True,
        text=True,
        capture_output=True,
    )
    return f"{result.stdout.strip().rstrip('/')}/{ROOT.name}"


def _sync_project_to_cluster(cfg: DatasetConfig, remote_repo: str) -> None:
    archive = _make_repo_archive()
    remote_tmp = f"/tmp/hom_ai_repo_{os.getpid()}.tar.gz"
    try:
        print(f"syncing local repo to {cfg.slurm.host}:{remote_repo}", flush=True)
        _remote_run(cfg, f"mkdir -p {_shell_quote(remote_repo)}", check=True)
        _remote_copy_to(cfg, archive, remote_tmp)
        cmd = (
            f"mkdir -p {_shell_quote(remote_repo)} && "
            f"tar -xzf {_shell_quote(remote_tmp)} -C {_shell_quote(remote_repo)} && "
            f"rm -f {_shell_quote(remote_tmp)}"
        )
        _remote_run(cfg, cmd, check=True)
    finally:
        archive.unlink(missing_ok=True)


def _copy_file_to_cluster(cfg: DatasetConfig, local_path: Path, remote_repo: str) -> None:
    rel = local_path.resolve().relative_to(ROOT).as_posix()
    remote_path = f"{remote_repo.rstrip('/')}/{rel}"
    remote_dir = str(Path(remote_path).parent).replace("\\", "/")
    _remote_run(cfg, f"mkdir -p {_shell_quote(remote_dir)}", check=True)
    _remote_copy_to(cfg, local_path, remote_path)


def _script_rel() -> str:
    return Path(__file__).resolve().relative_to(ROOT).as_posix()


def _config_rel(config_path: Path) -> str:
    return config_path.resolve().relative_to(ROOT).as_posix()


def _output_dir_for_slurm(cfg: DatasetConfig, args: argparse.Namespace) -> Path:
    return args.output_dir or _default_output_dir(cfg)


def _repo_relative_posix(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _generation_cli_flags(cfg: DatasetConfig) -> str:
    flags = [
        "--dataset-size",
        str(int(cfg.generation.dataset_size)),
        "--episode-duration-s",
        str(float(cfg.generation.episode_duration_s)),
        "--window-duration-s",
        str(float(cfg.generation.window_duration_s)),
        "--base-seed",
        str(int(cfg.generation.base_seed)),
    ]
    flags.append("--write-raw-streams" if cfg.generation.write_raw_streams else "--no-write-raw-streams")
    flags.append("--resume" if cfg.generation.resume else "--no-resume")
    return " ".join(flags)


def _write_slurm_scripts(cfg: DatasetConfig, args: argparse.Namespace, remote_repo: str) -> tuple[Path, Path, Path | None]:
    output_dir = _output_dir_for_slurm(cfg, args)
    output_dir.mkdir(parents=True, exist_ok=True)
    array_script = output_dir / "hom_ai_dataset_array.slurm"
    combine_script = output_dir / "hom_ai_dataset_combine.slurm"
    benchmark_script = output_dir / "hom_ai_dataset_benchmark.slurm" if args.benchmark_one else None

    array_tasks = max(1, int(cfg.slurm.array_tasks))
    nodes = max(1, int(cfg.slurm.nodes))
    cpus_per_node = max(1, int(cfg.slurm.cpus_per_node))
    shards_per_node = max(1, int(cfg.slurm.shards_per_node))
    ntasks_per_node = min(shards_per_node, cpus_per_node)
    cpus_per_task = max(1, cpus_per_node // ntasks_per_node)
    max_parallel = max(1, nodes * ntasks_per_node)

    config_rel = _config_rel(args.config)
    script_rel = _script_rel()
    out_rel = _repo_relative_posix(output_dir)
    python_cmd = str(cfg.slurm.python_cmd)
    generation_flags = _generation_cli_flags(cfg)
    module_line = f"module load {cfg.slurm.module_load}" if str(cfg.slurm.module_load).strip() else ""
    partition_line = f"#SBATCH --partition={cfg.slurm.partition}" if str(cfg.slurm.partition).strip() else ""
    time_line = f"#SBATCH --time={cfg.slurm.time_limit}" if str(cfg.slurm.time_limit).strip() else ""

    array_script.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=hom_ai
{partition_line}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
{time_line}
#SBATCH --array=0-{array_tasks - 1}%{max_parallel}
#SBATCH --output={out_rel}/hom_ai_%A_%a.out
#SBATCH --error={out_rel}/hom_ai_%A_%a.err

set -euo pipefail
cd {remote_repo}
{module_line}
mkdir -p {out_rel}
echo "Running HOM-AI shard $SLURM_ARRAY_TASK_ID/{array_tasks} cpus=$SLURM_CPUS_PER_TASK"
{python_cmd} {script_rel} --target local --config {config_rel} --output-dir {out_rel} {generation_flags} --workers $SLURM_CPUS_PER_TASK --shard-index $SLURM_ARRAY_TASK_ID --num-shards {array_tasks} --checkpoint-only
""",
        encoding="utf-8",
        newline="\n",
    )

    combine_script.write_text(
        f"""#!/bin/bash
#SBATCH --job-name=hom_ai_combine
{partition_line}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
{time_line}
#SBATCH --output={out_rel}/hom_ai_combine_%j.out
#SBATCH --error={out_rel}/hom_ai_combine_%j.err

set -euo pipefail
cd {remote_repo}
{module_line}
{python_cmd} {script_rel} --target local --config {config_rel} --output-dir {out_rel} {generation_flags} --combine-only
""",
        encoding="utf-8",
        newline="\n",
    )

    if benchmark_script is not None:
        selected_label = _labels_from_args(args)[0] if args.labels else "healthy"
        benchmark_script.write_text(
            f"""#!/bin/bash
#SBATCH --job-name=hom_ai_bench
{partition_line}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
{time_line}
#SBATCH --output={out_rel}/hom_ai_benchmark_%j.out
#SBATCH --error={out_rel}/hom_ai_benchmark_%j.err

set -euo pipefail
cd {remote_repo}
{module_line}
mkdir -p {out_rel}/benchmark
/usr/bin/time -v {python_cmd} {script_rel} --target local --config {config_rel} --output-dir {out_rel}/benchmark/one_episode --labels {selected_label} --dataset-size 1 --episode-duration-s {float(cfg.generation.episode_duration_s)} --window-duration-s {float(cfg.generation.window_duration_s)} --base-seed {int(cfg.generation.base_seed)} --workers 1 --no-write-raw-streams --no-resume --benchmark-one
""",
            encoding="utf-8",
            newline="\n",
        )

    print(f"wrote {array_script}", flush=True)
    print(f"wrote {combine_script}", flush=True)
    if benchmark_script is not None:
        print(f"wrote {benchmark_script}", flush=True)
    return array_script, combine_script, benchmark_script


def _clean_old_logs(cfg: DatasetConfig, output_dir: Path, remote_repo: str | None = None) -> None:
    for pattern in ("hom_ai_*.out", "hom_ai_*.err"):
        for path in output_dir.glob(pattern):
            path.unlink(missing_ok=True)
    if remote_repo:
        remote_output = f"{remote_repo.rstrip('/')}/{_repo_relative_posix(output_dir)}"
        cmd = f"mkdir -p {_shell_quote(remote_output)} && rm -f {_shell_quote(remote_output)}/hom_ai_*.out {_shell_quote(remote_output)}/hom_ai_*.err"
        _remote_run(cfg, cmd, check=False)


def _submit_slurm(cfg: DatasetConfig, remote_repo: str, array_script: Path, combine_script: Path) -> tuple[str, str]:
    array_rel = array_script.resolve().relative_to(ROOT).as_posix()
    combine_rel = combine_script.resolve().relative_to(ROOT).as_posix()
    array_cmd = f"cd {_shell_quote(remote_repo)} && sbatch --parsable {array_rel}"
    array_job = _remote_run(cfg, array_cmd, check=True, text=True, capture_output=True).stdout.strip()
    array_job_id = re.search(r"\d+", array_job).group(0)
    combine_cmd = f"cd {_shell_quote(remote_repo)} && sbatch --parsable --dependency=afterok:{array_job_id} {combine_rel}"
    combine_job = _remote_run(cfg, combine_cmd, check=True, text=True, capture_output=True).stdout.strip()
    combine_job_id = re.search(r"\d+", combine_job).group(0)
    print(f"submitted array job {array_job_id}", flush=True)
    print(f"submitted combine job {combine_job_id} afterok:{array_job_id}", flush=True)
    return array_job_id, combine_job_id


def _submit_benchmark(cfg: DatasetConfig, remote_repo: str, benchmark_script: Path) -> str:
    bench_rel = benchmark_script.resolve().relative_to(ROOT).as_posix()
    cmd = f"cd {_shell_quote(remote_repo)} && sbatch --parsable {bench_rel}"
    out = _remote_run(cfg, cmd, check=True, text=True, capture_output=True).stdout.strip()
    job_id = re.search(r"\d+", out).group(0)
    print(f"submitted benchmark job {job_id}", flush=True)
    return job_id


def _wait_for_job(cfg: DatasetConfig, job_id: str) -> None:
    poll = max(5, int(cfg.slurm.poll_seconds))
    while True:
        result = _remote_run(cfg, f"squeue -h -j {_shell_quote(job_id)}", text=True, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"squeue failed for job {job_id}")
        if not result.stdout.strip():
            return
        time.sleep(poll)


def _collect_outputs(cfg: DatasetConfig, output_dir: Path, remote_repo: str, tag: str = "latest") -> None:
    remote_output = f"{remote_repo.rstrip('/')}/{_repo_relative_posix(output_dir)}"
    remote_tmp = f"/tmp/hom_ai_outputs_{tag}.tar.gz"
    local_tmp = output_dir / f"hom_ai_outputs_{tag}.tar.gz"
    _remote_run(cfg, f"tar -czf {_shell_quote(remote_tmp)} -C {_shell_quote(remote_output)} .", check=True)
    try:
        _remote_copy_from(cfg, remote_tmp, local_tmp)
        with tarfile.open(local_tmp, "r:gz") as archive:
            archive.extractall(output_dir)
    finally:
        local_tmp.unlink(missing_ok=True)
        _remote_run(cfg, f"rm -f {_shell_quote(remote_tmp)}", check=False)


def _run_slurm_target(cfg: DatasetConfig, args: argparse.Namespace) -> None:
    remote_repo = _resolve_remote_repo(cfg)
    output_dir = _output_dir_for_slurm(cfg, args)
    if _as_bool(cfg.slurm.clean_old_logs):
        _clean_old_logs(cfg, output_dir, remote_repo if _as_bool(cfg.slurm.sync_project) or _as_bool(args.submit) else None)
    array_script, combine_script, benchmark_script = _write_slurm_scripts(cfg, args, remote_repo)
    if _as_bool(cfg.slurm.sync_project):
        _sync_project_to_cluster(cfg, remote_repo)
        _copy_file_to_cluster(cfg, array_script, remote_repo)
        _copy_file_to_cluster(cfg, combine_script, remote_repo)
        if benchmark_script is not None:
            _copy_file_to_cluster(cfg, benchmark_script, remote_repo)

    should_submit = _as_bool(args.submit) if args.submit is not None else _as_bool(cfg.slurm.submit)
    if not should_submit:
        print(f"On {cfg.slurm.host}, run: sbatch {array_script.as_posix()}", flush=True)
        print(f"Then combine with dependency, or run: sbatch {combine_script.as_posix()}", flush=True)
        return

    if args.benchmark_one:
        if benchmark_script is None:
            raise RuntimeError("Benchmark script was not generated")
        job_id = _submit_benchmark(cfg, remote_repo, benchmark_script)
        if _as_bool(args.wait_for_completion) or _as_bool(cfg.slurm.wait_for_completion):
            _wait_for_job(cfg, job_id)
            if _as_bool(args.collect_outputs) or _as_bool(cfg.slurm.collect_outputs):
                _collect_outputs(cfg, output_dir, remote_repo, tag=job_id)
        return

    _, combine_job_id = _submit_slurm(cfg, remote_repo, array_script, combine_script)
    if _as_bool(args.wait_for_completion) or _as_bool(cfg.slurm.wait_for_completion):
        _wait_for_job(cfg, combine_job_id)
        if _as_bool(args.collect_outputs) or _as_bool(cfg.slurm.collect_outputs):
            _collect_outputs(cfg, output_dir, remote_repo, tag=combine_job_id)


def main() -> None:
    args = parse_args()
    cfg = _apply_overrides(load_dataset_config(args.config), args)
    target = str(args.target or cfg.slurm.target or "local").lower()
    if target == "blackbird":
        _run_slurm_target(cfg, args)
        return

    labels = _labels_from_args(args)
    output_dir = args.output_dir or _default_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    raw_dir = output_dir / "raw"
    if cfg.generation.write_raw_streams:
        raw_dir.mkdir(parents=True, exist_ok=True)

    jobs = _jobs_for_dataset(cfg, labels)
    if args.combine_only:
        _write_final_outputs(cfg, args, labels, jobs, output_dir, time.time())
        return
    if args.benchmark_one:
        label = labels[0] if labels else "healthy"
        episode_index = 0
        if label not in labels:
            labels = [label]
        jobs = [(label, episode_index, cfg)]
    shard_jobs = _filter_jobs_for_shard(jobs, args.shard_index, args.num_shards)
    pending = []
    for label, episode_index, job_cfg in shard_jobs:
        checkpoint = _checkpoint_path(checkpoint_dir, label, episode_index)
        if cfg.generation.resume and checkpoint.exists():
            continue
        pending.append((label, episode_index, job_cfg))

    started = time.time()
    workers = max(1, int(cfg.generation.workers or os.cpu_count() or 1))
    print(
        f"Generating HOM dataset episodes={len(jobs)} shard_episodes={len(shard_jobs)} "
        f"pending={len(pending)} checkpoint=episode workers={workers} output={output_dir}",
        flush=True,
    )

    if workers == 1:
        for idx, job in enumerate(pending, start=1):
            payload = _run_job(job)
            _write_checkpoint(_checkpoint_path(checkpoint_dir, job[0], job[1]), payload)
            print(f"completed {idx}/{len(pending)} {payload['episode_id']}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_to_job = {pool.submit(_run_job, job): job for job in pending}
            for idx, future in enumerate(as_completed(future_to_job), start=1):
                job = future_to_job[future]
                payload = future.result()
                _write_checkpoint(_checkpoint_path(checkpoint_dir, job[0], job[1]), payload)
                print(f"completed {idx}/{len(pending)} {payload['episode_id']}", flush=True)

    if args.checkpoint_only:
        print(f"Wrote checkpoints only to {checkpoint_dir}", flush=True)
        return
    _write_final_outputs(cfg, args, labels, jobs, output_dir, started)


if __name__ == "__main__":
    main()
