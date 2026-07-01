"""Generate continuous, BSM-only fault-diagnosis datasets locally or on Slurm."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from example.hom_validation.ai_hom_dataset.generate_hom_dataset import (
    _copy_file_to_cluster,
    _remote_copy_from,
    _remote_run,
    _resolve_remote_repo,
    _shell_quote,
    _sync_project_to_cluster,
    _wait_for_job,
)

from example.entanglement_swapping_validation.ai_bsm_dataset.config import (
    FAULT_LABELS,
    DatasetConfig,
    load_config,
)
from example.entanglement_swapping_validation.ai_bsm_dataset.scenarios import build_scenario
from example.entanglement_swapping_validation.ai_bsm_dataset.simulator import simulate_episode


DEFAULT_CONFIG = Path(__file__).parent / "configs" / "bsm_dataset_config.ini"


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--target", choices=("local", "blackbird"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--episode-index", type=int)
    parser.add_argument("--combine-only", action="store_true")
    parser.add_argument("--submit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wait", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--collect-existing", action="store_true")
    parser.add_argument("--array-job-id")
    parser.add_argument("--combine-job-id")
    return parser.parse_args()


def output_dir(cfg, override):
    if override:
        return override.resolve()
    return Path(__file__).parent / "generated" / cfg.generation.dataset_name


def signature(cfg: DatasetConfig) -> str:
    payload = asdict(cfg)
    payload.pop("slurm", None)
    for key in ("dataset_name", "dataset_size", "workers", "resume"):
        payload["generation"].pop(key, None)
    payload = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def episode_label(index: int) -> str:
    return FAULT_LABELS[index % len(FAULT_LABELS)]


def episode_seed(cfg, index):
    return cfg.generation.base_seed + index * 10_000


def checkpoint_path(directory, index):
    return directory / "checkpoints" / f"episode_{index:06d}.json"


def write_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")
    temporary.replace(path)


def checkpoint_valid(path, cfg):
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("config_signature") == signature(cfg)
    except (OSError, ValueError):
        return False


def run_episode(index, cfg):
    label = episode_label(index)
    scenario = build_scenario(label, index, cfg, episode_seed(cfg, index))
    rows, truth, raw = simulate_episode(scenario, cfg)
    return {
        "episode_index": index,
        "config_signature": signature(cfg),
        "scenario": scenario.metadata(),
        "observable_rows": rows,
        "truth_rows": truth,
        "raw": raw,
    }


def run_local(cfg, directory, episode_index=None):
    directory.mkdir(parents=True, exist_ok=True)
    indices = [episode_index] if episode_index is not None else list(range(cfg.generation.dataset_size))
    pending = [
        index for index in indices
        if not (cfg.generation.resume and checkpoint_valid(checkpoint_path(directory, index), cfg))
    ]
    print(f"episodes={len(indices)} pending={len(pending)} workers={cfg.generation.workers}", flush=True)
    if cfg.generation.workers <= 1:
        for index in pending:
            write_atomic(checkpoint_path(directory, index), run_episode(index, cfg))
    else:
        with ProcessPoolExecutor(max_workers=cfg.generation.workers) as pool:
            futures = {pool.submit(run_episode, index, cfg): index for index in pending}
            for future in as_completed(futures):
                payload = future.result()
                write_atomic(checkpoint_path(directory, payload["episode_index"]), payload)
    if episode_index is None:
        combine(cfg, directory)


def combine(cfg, directory):
    missing = []
    stale = []
    expected = signature(cfg)
    paths = [checkpoint_path(directory, index) for index in range(cfg.generation.dataset_size)]
    missing = [index for index, path in enumerate(paths) if not path.exists()]
    if missing or stale:
        raise RuntimeError(
            f"Cannot combine: missing={missing[:10]} ({len(missing)}), "
            f"stale={stale[:10]} ({len(stale)})"
        )

    final_names = {
        "observable": "bsm_observable_windows.csv",
        "truth": "hidden_truth.csv",
        "labels": "episode_labels.csv",
        "scenarios": "scenarios.json",
    }
    temporary = {
        key: directory / f"{name}.tmp"
        for key, name in final_names.items()
    }
    handles = {}
    writers = {}
    windows = 0
    try:
        for key in ("observable", "truth", "labels"):
            handles[key] = temporary[key].open("w", encoding="utf-8", newline="")
        handles["scenarios"] = temporary["scenarios"].open("w", encoding="utf-8")
        handles["scenarios"].write("[\n")

        for index, path in enumerate(paths):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("config_signature") != expected:
                stale.append(index)
                continue

            scenario = payload["scenario"]
            label_row = {
                "episode_id": scenario["episode_id"],
                "fault_class": scenario["fault_class"],
                "seed": scenario["seed"],
                "fault_onset_s": scenario["onset_s"],
                "fault_ramp_duration_s": scenario["ramp_duration_s"],
                "target_fault_level": scenario["target"],
            }
            row_groups = {
                "observable": payload["observable_rows"],
                "truth": payload["truth_rows"],
                "labels": [label_row],
            }
            for key, rows in row_groups.items():
                if key not in writers and rows:
                    writers[key] = csv.DictWriter(handles[key], fieldnames=list(rows[0]))
                    writers[key].writeheader()
                if rows:
                    writers[key].writerows(rows)
            windows += len(payload["observable_rows"])
            if index:
                handles["scenarios"].write(",\n")
            json.dump(scenario, handles["scenarios"])

        handles["scenarios"].write("\n]\n")
    finally:
        for handle in handles.values():
            handle.close()

    if stale:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Cannot combine: {len(stale)} stale checkpoints; first: {stale[:10]}"
        )
    for key, name in final_names.items():
        temporary[key].replace(directory / name)

    pd.DataFrame([asdict(cfg.setup)]).to_csv(directory / "setup_params.csv", index=False)
    metadata = {
        "dataset_name": cfg.generation.dataset_name,
        "config_signature": expected,
        "episodes": cfg.generation.dataset_size,
        "windows": windows,
        "episode_duration_s": cfg.generation.episode_duration_s,
        "window_duration_s": cfg.generation.window_duration_s,
        "continuous_timeline_per_episode": True,
        "fault_labels": list(FAULT_LABELS),
        "ai_input_file": "bsm_observable_windows.csv",
        "note": "Only four BSM detector streams are model inputs; hidden truth is offline only.",
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Combined {cfg.generation.dataset_size} episodes into {directory}", flush=True)


def write_slurm(cfg, config_path, directory):
    directory.mkdir(parents=True, exist_ok=True)
    array = directory / "bsm_dataset_array.slurm"
    merger = directory / "bsm_dataset_combine.slurm"
    script_rel = Path(__file__).resolve().relative_to(ROOT).as_posix()
    config_rel = config_path.resolve().relative_to(ROOT).as_posix()
    output_rel = directory.resolve().relative_to(ROOT).as_posix()
    maximum = min(
        cfg.generation.dataset_size,
        cfg.slurm.nodes * cfg.slurm.cpus_per_node,
    )
    module = f"module load {cfg.slurm.module_load}" if cfg.slurm.module_load else ""
    array_text = f"""#!/bin/bash
#SBATCH --job-name=bsm_dataset
#SBATCH --partition={cfg.slurm.partition}
#SBATCH --time={cfg.slurm.time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-{cfg.generation.dataset_size - 1}%{maximum}
#SBATCH --output={output_rel}/logs/bsm_%A_%a.out
#SBATCH --error={output_rel}/logs/bsm_%A_%a.err
set -euo pipefail
{module}
cd "$SLURM_SUBMIT_DIR"
mkdir -p {output_rel}/logs
{cfg.slurm.python_cmd} {script_rel} --target local --config {config_rel} --output-dir {output_rel} --episode-index "$SLURM_ARRAY_TASK_ID"
"""
    combine_text = f"""#!/bin/bash
#SBATCH --job-name=bsm_combine
#SBATCH --partition={cfg.slurm.partition}
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output={output_rel}/logs/combine_%j.out
#SBATCH --error={output_rel}/logs/combine_%j.err
set -euo pipefail
{module}
cd "$SLURM_SUBMIT_DIR"
{cfg.slurm.python_cmd} {script_rel} --target local --config {config_rel} --output-dir {output_rel} --combine-only
"""
    for path, text in ((array, array_text), (merger, combine_text)):
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    return array, merger


def submit_job(wrapper, remote_repo, relative, dependency=None):
    option = f"--dependency=afterok:{dependency} " if dependency else ""
    result = _remote_run(
        wrapper,
        f"cd {_shell_quote(remote_repo)} && sbatch --parsable {option}{relative}",
        check=False, text=True, capture_output=True,
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    match = re.search(r"\d+", result.stdout)
    if not match:
        raise RuntimeError(f"Cannot parse job ID from {result.stdout!r}")
    return match.group()


def run_blackbird(args, cfg, directory):
    wrapper = SimpleNamespace(slurm=cfg.slurm)
    remote_repo = _resolve_remote_repo(wrapper)
    array, merger = write_slurm(cfg, args.config, directory)
    if cfg.slurm.sync_project:
        _sync_project_to_cluster(wrapper, remote_repo)
        _copy_file_to_cluster(wrapper, array, remote_repo)
        _copy_file_to_cluster(wrapper, merger, remote_repo)
    should_submit = cfg.slurm.submit if args.submit is None else args.submit
    if not should_submit:
        print(f"Wrote {array} and {merger}", flush=True)
        return
    array_rel = array.resolve().relative_to(ROOT).as_posix()
    merger_rel = merger.resolve().relative_to(ROOT).as_posix()
    array_id = submit_job(wrapper, remote_repo, array_rel)
    combine_id = submit_job(wrapper, remote_repo, merger_rel, array_id)
    write_atomic(directory / "submission.json", {
        "array_job_id": array_id,
        "combine_job_id": combine_id,
        "remote_repo": remote_repo,
    })
    print(f"submitted array job {array_id}", flush=True)
    print(f"submitted combine job {combine_id} afterok:{array_id}", flush=True)
    should_wait = cfg.slurm.wait_for_completion if args.wait is None else args.wait
    if should_wait:
        _wait_for_job(wrapper, combine_id)


def saved_jobs(directory):
    path = directory / "submission.json"
    return json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}


def status_remote(args, cfg, directory):
    wrapper = SimpleNamespace(slurm=cfg.slurm)
    saved = saved_jobs(directory)
    array_id = args.array_job_id or saved.get("array_job_id")
    combine_id = args.combine_job_id or saved.get("combine_job_id")
    if not array_id and not combine_id:
        raise RuntimeError("No saved job IDs; pass --array-job-id and --combine-job-id")
    ids = ",".join(item for item in (array_id, combine_id) if item)
    remote_repo = saved.get("remote_repo") or _resolve_remote_repo(wrapper)
    checkpoint_dir = str(directory.relative_to(ROOT)).replace(os.sep, "/") + "/checkpoints"
    result = _remote_run(
        wrapper,
        f"squeue -j {_shell_quote(ids)} "
        "-o '%.18i %.24j %.2t %.10M %.6D %.8C %R'; "
        "echo '--- accounting ---'; "
        f"sacct -j {_shell_quote(ids)} "
        "--format=JobID,State,Elapsed,ExitCode -n -X; "
        f"cd {_shell_quote(remote_repo)}; "
        f"echo checkpoints=$(find {_shell_quote(checkpoint_dir)} "
        f"-name 'episode_*.json' -type f 2>/dev/null | wc -l)/"
        f"{cfg.generation.dataset_size}",
        check=False, text=True, capture_output=True,
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip())
    print(result.stdout, end="")


def collect_existing(args, cfg, directory):
    wrapper = SimpleNamespace(slurm=cfg.slurm)
    saved = saved_jobs(directory)
    remote_repo = saved.get("remote_repo") or _resolve_remote_repo(wrapper)
    combine_id = args.combine_job_id or saved.get("combine_job_id") or "existing"
    relative = directory.resolve().relative_to(ROOT).as_posix()
    remote_output = f"{remote_repo.rstrip('/')}/{relative}"
    files = (
        "metadata.json", "bsm_observable_windows.csv", "episode_labels.csv",
        "hidden_truth.csv", "setup_params.csv", "scenarios.json",
    )
    ready = _remote_run(
        wrapper, f"test -f {_shell_quote(remote_output + '/metadata.json')}",
        check=False,
    )
    if ready.returncode:
        checkpoint_count = _remote_run(
            wrapper,
            f"find {_shell_quote(remote_output + '/checkpoints')} "
            "-name 'episode_*.json' -type f 2>/dev/null | wc -l",
            check=False,
            text=True,
            capture_output=True,
        ).stdout.strip()
        raise RuntimeError(
            "Remote combined outputs are not ready. "
            f"Completed checkpoints: {checkpoint_count}/"
            f"{cfg.generation.dataset_size}. Run --status for job details."
        )
    remote_tmp = f"/tmp/polarization_bsm_{combine_id}.tar.gz"
    local_tmp = directory / f"polarization_bsm_{combine_id}.tar.gz"
    names = " ".join(_shell_quote(name) for name in files)
    _remote_run(
        wrapper,
        f"tar -czf {_shell_quote(remote_tmp)} -C {_shell_quote(remote_output)} {names}",
        check=True,
    )
    directory.mkdir(parents=True, exist_ok=True)
    try:
        _remote_copy_from(wrapper, remote_tmp, local_tmp)
        with tarfile.open(local_tmp, "r:gz") as archive:
            archive.extractall(directory, filter="data")
    finally:
        local_tmp.unlink(missing_ok=True)
        _remote_run(wrapper, f"rm -f {_shell_quote(remote_tmp)}", check=False)
    print(f"Collected scientific outputs into {directory}", flush=True)


def main():
    args = arguments()
    cfg = load_config(args.config)
    directory = output_dir(cfg, args.output_dir)
    if args.status:
        status_remote(args, cfg, directory)
    elif args.collect_existing:
        collect_existing(args, cfg, directory)
    elif args.combine_only:
        combine(cfg, directory)
    elif (args.target or cfg.slurm.target) == "blackbird":
        run_blackbird(args, cfg, directory)
    else:
        run_local(cfg, directory, args.episode_index)


if __name__ == "__main__":
    main()
