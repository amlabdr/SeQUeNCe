"""Resumable local/Blackbird batch runner for full swapping tomography."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from example.entanglement_swapping_validation.batch_config import load_config
from example.entanglement_swapping_validation.full_simulator import (
    FullSwappingConfig,
    maximum_likelihood_tomography,
    run_full_tomography,
)
from example.hom_validation.ai_hom_dataset.generate_hom_dataset import (
    _copy_file_to_cluster,
    _remote_copy_from,
    _remote_run,
    _resolve_remote_repo,
    _shell_quote,
    _sync_project_to_cluster,
    _wait_for_job,
)


DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "full_swapping_validation.ini"


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--target", choices=("local", "blackbird"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-index", type=int)
    parser.add_argument("--combine-only", action="store_true")
    parser.add_argument("--submit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--wait", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--collect", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--collect-existing", action="store_true")
    parser.add_argument("--array-job-id")
    parser.add_argument("--combine-job-id")
    return parser.parse_args()


def _output_dir(batch, override: Path | None) -> Path:
    if override:
        return override.resolve()
    if str(batch.output_dir).strip():
        return (ROOT / str(batch.output_dir)).resolve()
    return Path(__file__).resolve().parent / "results" / batch.experiment_name


def _checkpoint(output_dir: Path, run_index: int) -> Path:
    return output_dir / "checkpoints" / f"run_{run_index:05d}.json"


def _config_signature(config: FullSwappingConfig) -> str:
    payload = json.dumps(config.__dict__, sort_keys=True, default=lambda value: value.__dict__)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _checkpoint_matches(path: Path, config: FullSwappingConfig) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get(
            "config_signature"
        ) == _config_signature(config)
    except (OSError, ValueError):
        return False


def _run_one(config: FullSwappingConfig, run_index: int):
    run_config = FullSwappingConfig(**{**config.__dict__, "seed": config.seed + 10_000 * run_index})
    frame = run_full_tomography(run_config)
    frame.insert(0, "run_index", run_index)
    return {
        "run_index": run_index,
        "config_signature": _config_signature(config),
        "rows": frame.to_dict("records"),
    }


def _write_json_atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=True), encoding="utf-8")
    temporary.replace(path)


def _submission_path(output_dir: Path) -> Path:
    return output_dir / "submission.json"


def _load_submission(output_dir: Path) -> dict:
    path = _submission_path(output_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_submission(output_dir: Path, array_id: str, combine_id: str, remote_repo: str):
    _write_json_atomic(_submission_path(output_dir), {
        "array_job_id": array_id,
        "combine_job_id": combine_id,
        "remote_repo": remote_repo,
    })


def _job_ids(args, output_dir: Path) -> tuple[str | None, str | None]:
    saved = _load_submission(output_dir)
    return (
        args.array_job_id or saved.get("array_job_id"),
        args.combine_job_id or saved.get("combine_job_id"),
    )


def _status_existing(args, batch, slurm, output_dir: Path):
    wrapper = SimpleNamespace(slurm=slurm)
    remote_repo = _resolve_remote_repo(wrapper)
    array_id, combine_id = _job_ids(args, output_dir)
    if not array_id and not combine_id:
        raise RuntimeError("No job IDs found. Pass --array-job-id and --combine-job-id.")
    ids = ",".join(value for value in (array_id, combine_id) if value)
    remote_output = f"{remote_repo.rstrip('/')}/{output_dir.resolve().relative_to(ROOT).as_posix()}"
    command = (
        f"squeue -j {_shell_quote(ids)} "
        "-o '%.18i %.24j %.2t %.10M %.6D %.8C %R'; "
        "echo '--- completed task states ---'; "
        f"sacct -j {_shell_quote(ids)} --format=JobID,State,Elapsed,ExitCode -n -X; "
        "echo '--- remote results ---'; "
        f"find {_shell_quote(remote_output)} -maxdepth 1 -type f "
        "\\( -name 'metadata.json' -o -name 'swapping_*.csv' -o -name 'swapping_*.npz' \\) "
        "-printf '%f\\n' 2>/dev/null | sort; "
        f"printf 'checkpoints='; find {_shell_quote(remote_output + '/checkpoints')} "
        "-maxdepth 1 -name 'run_*.json' -type f 2>/dev/null | wc -l"
    )
    result = _remote_run(wrapper, command, check=False, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Remote status query failed")
    print(result.stdout, end="")


def _collect_existing(args, slurm, output_dir: Path):
    wrapper = SimpleNamespace(slurm=slurm)
    saved = _load_submission(output_dir)
    remote_repo = saved.get("remote_repo") or _resolve_remote_repo(wrapper)
    _, combine_id = _job_ids(args, output_dir)
    remote_output = f"{remote_repo.rstrip('/')}/{output_dir.resolve().relative_to(ROOT).as_posix()}"
    ready = _remote_run(
        wrapper,
        f"test -f {_shell_quote(remote_output + '/metadata.json')}",
        check=False,
    )
    if ready.returncode != 0:
        raise RuntimeError("Remote final results are not ready; run --status first.")
    result_files = (
        "metadata.json",
        "swapping_basis_counts_per_run.csv",
        "swapping_basis_counts_aggregated.csv",
        "swapping_tomography_summary.csv",
        "swapping_density_matrices.npz",
    )
    tag = combine_id or "existing"
    remote_tmp = f"/tmp/full_swapping_outputs_{tag}.tar.gz"
    local_tmp = output_dir / f"full_swapping_outputs_{tag}.tar.gz"
    names = " ".join(_shell_quote(name) for name in result_files)
    _remote_run(
        wrapper,
        f"tar -czf {_shell_quote(remote_tmp)} -C {_shell_quote(remote_output)} {names}",
        check=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        _remote_copy_from(wrapper, remote_tmp, local_tmp)
        with tarfile.open(local_tmp, "r:gz") as archive:
            archive.extractall(output_dir)
    finally:
        local_tmp.unlink(missing_ok=True)
        _remote_run(wrapper, f"rm -f {_shell_quote(remote_tmp)}", check=False)
    print(f"Collected results into {output_dir}", flush=True)


def _combine(output_dir: Path, runs: int, config: FullSwappingConfig):
    payloads = []
    missing = []
    for run_index in range(runs):
        path = _checkpoint(output_dir, run_index)
        if path.exists():
            payloads.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            missing.append(run_index)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} checkpoints; first missing runs: {missing[:10]}")
    expected_signature = _config_signature(config)
    stale = [
        payload["run_index"]
        for payload in payloads
        if payload.get("config_signature") != expected_signature
    ]
    if stale:
        raise RuntimeError(
            f"Found {len(stale)} checkpoints from another configuration; "
            f"first stale runs: {stale[:10]}"
        )

    per_run = pd.DataFrame([row for payload in payloads for row in payload["rows"]])
    per_run.to_csv(output_dir / "swapping_basis_counts_per_run.csv", index=False)

    count_columns = ["count_00", "count_01", "count_10", "count_11", "fourfold_count", "bsm_bell_events"]
    aggregate = (
        per_run.groupby(
            ["source_a_state", "source_b_state", "bsm_state", "basis_a", "basis_d"],
            as_index=False,
        )[count_columns]
        .sum()
    )
    aggregate["correlation"] = (
        aggregate["count_00"] + aggregate["count_11"]
        - aggregate["count_01"] - aggregate["count_10"]
    ) / aggregate["fourfold_count"].replace(0, np.nan)
    aggregate["expectation_a"] = (
        aggregate["count_00"] + aggregate["count_01"]
        - aggregate["count_10"] - aggregate["count_11"]
    ) / aggregate["fourfold_count"].replace(0, np.nan)
    aggregate["expectation_d"] = (
        aggregate["count_00"] + aggregate["count_10"]
        - aggregate["count_01"] - aggregate["count_11"]
    ) / aggregate["fourfold_count"].replace(0, np.nan)
    aggregate.to_csv(output_dir / "swapping_basis_counts_aggregated.csv", index=False)

    summaries = []
    matrices = {}
    for bsm_state in ("psi_minus", "psi_plus"):
        rho, summary = maximum_likelihood_tomography(aggregate, bsm_state)
        summaries.append(summary)
        matrices[f"{bsm_state}_real"] = rho.real
        matrices[f"{bsm_state}_imag"] = rho.imag
    pd.DataFrame(summaries).to_csv(output_dir / "swapping_tomography_summary.csv", index=False)
    np.savez(output_dir / "swapping_density_matrices.npz", **matrices)
    (output_dir / "metadata.json").write_text(
        json.dumps({
            "completed_runs": runs,
            "config_signature": expected_signature,
            "duration_s_per_basis": config.duration_s,
            "source_frequency_hz": config.source_frequency_hz,
            "pulses_per_basis": config.pulses,
            "basis_settings_per_run": 9,
            "files": [
                "swapping_basis_counts_per_run.csv",
                "swapping_basis_counts_aggregated.csv",
                "swapping_tomography_summary.csv",
                "swapping_density_matrices.npz",
            ],
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Combined {runs} runs into {output_dir}", flush=True)


def _run_local(simulation, batch, output_dir: Path, run_index: int | None):
    output_dir.mkdir(parents=True, exist_ok=True)
    indices = [run_index] if run_index is not None else list(range(batch.runs))
    pending = [
        index for index in indices
        if not (
            batch.resume
            and _checkpoint_matches(_checkpoint(output_dir, index), simulation)
        )
    ]
    print(f"runs={len(indices)} pending={len(pending)} workers={batch.workers}", flush=True)
    if batch.workers <= 1:
        for index in pending:
            _write_json_atomic(_checkpoint(output_dir, index), _run_one(simulation, index))
    else:
        with ProcessPoolExecutor(max_workers=batch.workers) as executor:
            futures = {executor.submit(_run_one, simulation, index): index for index in pending}
            for future in as_completed(futures):
                payload = future.result()
                _write_json_atomic(_checkpoint(output_dir, payload["run_index"]), payload)
    if run_index is None:
        _combine(output_dir, batch.runs, simulation)


def _write_slurm(config_path: Path, output_dir: Path, batch, slurm):
    output_dir.mkdir(parents=True, exist_ok=True)
    script = output_dir / "full_swapping_array.slurm"
    combine = output_dir / "full_swapping_combine.slurm"
    rel_script = Path(__file__).resolve().relative_to(ROOT).as_posix()
    rel_config = config_path.resolve().relative_to(ROOT).as_posix()
    rel_output = output_dir.resolve().relative_to(ROOT).as_posix()
    maximum_parallel = min(batch.runs, slurm.nodes * slurm.cpus_per_node)
    module = f"module load {slurm.module_load}" if str(slurm.module_load).strip() else ""
    partition = f"#SBATCH --partition={slurm.partition}" if str(slurm.partition).strip() else ""
    time_limit = f"#SBATCH --time={slurm.time_limit}" if str(slurm.time_limit).strip() else ""
    script_text = f"""#!/bin/bash
#SBATCH --job-name=swap_tomo
{partition}
{time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-{batch.runs - 1}%{maximum_parallel}
#SBATCH --output={rel_output}/swap_%A_%a.out
#SBATCH --error={rel_output}/swap_%A_%a.err
set -euo pipefail
{module}
cd "$SLURM_SUBMIT_DIR"
{slurm.python_cmd} {rel_script} --target local --config {rel_config} --output-dir {rel_output} --run-index "$SLURM_ARRAY_TASK_ID"
"""
    combine_text = f"""#!/bin/bash
#SBATCH --job-name=swap_combine
{partition}
{time_limit}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output={rel_output}/swap_combine_%j.out
#SBATCH --error={rel_output}/swap_combine_%j.err
set -euo pipefail
{module}
cd "$SLURM_SUBMIT_DIR"
{slurm.python_cmd} {rel_script} --target local --config {rel_config} --output-dir {rel_output} --combine-only
"""
    with script.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(script_text)
    with combine.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(combine_text)
    return script, combine


def _submit_remote(wrapper, remote_repo: str, script_relative: str) -> str:
    result = _remote_run(
        wrapper,
        f"cd {_shell_quote(remote_repo)} && sbatch --parsable {script_relative}",
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown sbatch error").strip()
        raise RuntimeError(f"Remote sbatch failed for {script_relative}: {detail}")
    match = re.search(r"\d+", result.stdout)
    if match is None:
        raise RuntimeError(f"Could not parse Slurm job ID from: {result.stdout!r}")
    return match.group(0)


def _remote_preflight(wrapper, remote_repo: str, python_cmd: str) -> None:
    import_code = (
        "import numpy,pandas,scipy,sequence;"
        "from example.entanglement_swapping_validation.full_simulator "
        "import FullSwappingConfig"
    )
    command = (
        f"test -x {_shell_quote(python_cmd)} && "
        f"cd {_shell_quote(remote_repo)} && "
        f"{_shell_quote(python_cmd)} -c {_shell_quote(import_code)}"
    )
    result = _remote_run(
        wrapper,
        command,
        check=False,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "Python executable is missing").strip()
        raise RuntimeError(
            f"Blackbird Python preflight failed for {python_cmd}: {detail}"
        )


def _run_blackbird(args, batch, slurm, output_dir):
    wrapper = SimpleNamespace(slurm=slurm)
    remote_repo = _resolve_remote_repo(wrapper)
    script, combine = _write_slurm(args.config, output_dir, batch, slurm)
    if slurm.sync_project:
        _sync_project_to_cluster(wrapper, remote_repo)
        _copy_file_to_cluster(wrapper, script, remote_repo)
        _copy_file_to_cluster(wrapper, combine, remote_repo)
    submit = slurm.submit if args.submit is None else args.submit
    if not submit:
        print(f"Wrote {script} and {combine}; set submit=true to launch.", flush=True)
        return
    _remote_preflight(wrapper, remote_repo, str(slurm.python_cmd))
    script_rel = script.resolve().relative_to(ROOT).as_posix()
    combine_rel = combine.resolve().relative_to(ROOT).as_posix()
    array_id = _submit_remote(wrapper, remote_repo, script_rel)
    result = _remote_run(
        wrapper,
        f"cd {_shell_quote(remote_repo)} && sbatch --parsable --dependency=afterok:{array_id} {combine_rel}",
        check=False, text=True, capture_output=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown sbatch error").strip()
        raise RuntimeError(f"Remote combine sbatch failed: {detail}")
    match = re.search(r"\d+", result.stdout)
    if match is None:
        raise RuntimeError(f"Could not parse combine job ID from: {result.stdout!r}")
    combine_id = match.group(0)
    _save_submission(output_dir, array_id, combine_id, remote_repo)
    print(f"submitted array job {array_id}", flush=True)
    print(f"submitted combine job {combine_id} afterok:{array_id}", flush=True)
    wait = slurm.wait_for_completion if args.wait is None else args.wait
    collect = slurm.collect_outputs if args.collect is None else args.collect
    if wait:
        _wait_for_job(wrapper, combine_id)
        if collect:
            _collect_existing(args, slurm, output_dir)


def main():
    args = _arguments()
    simulation, batch, slurm = load_config(args.config)
    output_dir = _output_dir(batch, args.output_dir)
    target = args.target or slurm.target
    if args.status:
        _status_existing(args, batch, slurm, output_dir)
    elif args.collect_existing:
        _collect_existing(args, slurm, output_dir)
    elif args.combine_only:
        _combine(output_dir, batch.runs, simulation)
    elif target == "blackbird":
        _run_blackbird(args, batch, slurm, output_dir)
    else:
        _run_local(simulation, batch, output_dir, args.run_index)


if __name__ == "__main__":
    main()
