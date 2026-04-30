"""Run the queueing-model paper evaluation locally or on an SSH backend."""

from __future__ import annotations

import argparse
import os
import posixpath
import shlex
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
QUEUEING_MODULE = Path("example/queueing_model")
CORE_EXPERIMENT_MODULE = Path("example/entanglement_distribution_experiment")
WORKER_SCRIPT = QUEUEING_MODULE / "memory_allocation_paper_worker.py"
DEFAULT_OUTPUT_DIR = REPO_ROOT / QUEUEING_MODULE / "results"

INCLUDED_PATHS = [
    Path("sequence"),
    CORE_EXPERIMENT_MODULE / "eg_single_heralded_helpers.py",
    CORE_EXPERIMENT_MODULE / "memory_allocation_queue_sim.py",
    QUEUEING_MODULE / "queue_utils.py",
    WORKER_SCRIPT,
]

EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run memory-allocation paper evaluation.")
    parser.add_argument("--backend", choices=["local", "hpc"], default="local")
    parser.add_argument("--host", default="enki")
    parser.add_argument("--remote-root", default="~/sequence_memory_allocation_runs")
    parser.add_argument("--remote-python", default="auto")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--horizon-s", type=float, default=10.0)
    parser.add_argument("--lambda-num-runs", type=int, default=10)
    parser.add_argument("--lambda-horizon-s", type=float, default=10.0)
    parser.add_argument("--simulation-point-stride", type=int, default=20)
    parser.add_argument("--model-fidelity-points", type=int, default=100)
    parser.add_argument("--parallel-backend", choices=["process", "thread"], default="process")
    parser.add_argument("--no-parallelize-runs", action="store_true")
    parser.add_argument("--heartbeat-s", type=float, default=60.0)
    parser.add_argument("--ttl-diagnostic-runs", type=int, default=3)
    parser.add_argument("--ttl-diagnostic-horizon-s", type=float, default=2.0)
    parser.add_argument("--ttl-grid-min-ms", type=float, default=1.0)
    parser.add_argument("--ttl-grid-max-ms", type=float, default=9.0)
    parser.add_argument("--ttl-grid-points", type=int, default=9)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--keep-remote", action="store_true")
    parser.add_argument(
        "--install-command",
        default="python -m pip install numpy scipy pandas matplotlib networkx qutip qutip-qip gmpy2",
    )
    return parser.parse_args()


def quote(value: str) -> str:
    return shlex.quote(value)


def run(command: list[str], *, cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=cwd, text=True, encoding="utf-8", errors="replace", check=True)


def capture(command: list[str]) -> str:
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, file=sys.stderr, end="" if result.stdout.endswith("\n") else "\n")
        raise subprocess.CalledProcessError(result.returncode, command, output=result.stdout)
    return result.stdout


def ssh(host: str, command: str) -> list[str]:
    return ["ssh", host, command]


def remote_path_expression(path: str) -> str:
    if path == "~":
        return "$HOME"
    if path.startswith("~/"):
        return "$HOME/" + quote(path[2:])
    return quote(path)


def detect_remote_python(host: str) -> str:
    command = (
        "for py in python3.12 python3.11 python3.10; do "
        "if command -v \"$py\" >/dev/null 2>&1; then "
        "$py -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1 "
        "&& { printf '%s\\n' \"$py\"; exit 0; }; "
        "fi; done; "
        "printf 'No Python >=3.10 found.\\n' >&2; exit 1"
    )
    output = capture(ssh(host, command))
    for line in output.splitlines():
        if line.strip().startswith("python"):
            return line.strip()
    raise RuntimeError(f"Could not detect remote Python from output:\n{output}")


def detect_workers(host: str) -> int:
    output = capture(ssh(host, "nproc 2>/dev/null || getconf _NPROCESSORS_ONLN"))
    for line in output.splitlines():
        line = line.strip()
        if line.isdigit():
            return max(1, int(line))
    return 1


def should_exclude(path: Path) -> bool:
    rel = path.relative_to(REPO_ROOT)
    return any(part in EXCLUDED_DIR_NAMES for part in rel.parts)


def normalize_tarinfo(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo:
    tarinfo.uid = 0
    tarinfo.gid = 0
    tarinfo.uname = ""
    tarinfo.gname = ""
    tarinfo.mode = 0o755 if tarinfo.isdir() else 0o644
    return tarinfo


def make_archive() -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="sequence_memory_allocation_package_"))
    archive_path = temp_dir / "sequence_memory_allocation_project.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for include_path in INCLUDED_PATHS:
            root = REPO_ROOT / include_path
            if not root.exists():
                raise FileNotFoundError(root)
            paths = [root]
            if root.is_dir():
                paths.extend(root.rglob("*"))
            for path in paths:
                if should_exclude(path):
                    continue
                archive.add(path, arcname=path.relative_to(REPO_ROOT), recursive=False, filter=normalize_tarinfo)
    return archive_path


def worker_args(args: argparse.Namespace, output_dir: Path | str, workers: int) -> list[str]:
    values = [
        sys.executable,
        str(REPO_ROOT / WORKER_SCRIPT),
        "--output-dir",
        str(output_dir),
        "--workers",
        str(workers),
        "--num-runs",
        str(args.num_runs),
        "--horizon-s",
        str(args.horizon_s),
        "--lambda-num-runs",
        str(args.lambda_num_runs),
        "--lambda-horizon-s",
        str(args.lambda_horizon_s),
        "--simulation-point-stride",
        str(args.simulation_point_stride),
        "--model-fidelity-points",
        str(args.model_fidelity_points),
        "--parallel-backend",
        args.parallel_backend,
        *([] if not args.no_parallelize_runs else ["--no-parallelize-runs"]),
        "--heartbeat-s",
        str(args.heartbeat_s),
        "--ttl-diagnostic-runs",
        str(args.ttl_diagnostic_runs),
        "--ttl-diagnostic-horizon-s",
        str(args.ttl_diagnostic_horizon_s),
        "--ttl-grid-min-ms",
        str(args.ttl_grid_min_ms),
        "--ttl-grid-max-ms",
        str(args.ttl_grid_max_ms),
        "--ttl-grid-points",
        str(args.ttl_grid_points),
    ]
    if args.smoke:
        values.append("--smoke")
    return values


def run_local(args: argparse.Namespace) -> None:
    workers = args.workers or max(1, (os.cpu_count() or 2) - 2)
    run(worker_args(args, args.output_dir, workers))


def run_hpc(args: argparse.Namespace) -> None:
    host = args.host
    workers = args.workers or detect_workers(host)
    remote_python = detect_remote_python(host) if args.remote_python == "auto" else args.remote_python
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_memory_allocation_paper"

    remote_root_expr = remote_path_expression(args.remote_root)
    remote_run_dir_output = capture(
        ssh(
            host,
            f"set -e; mkdir -p {remote_root_expr}; remote_root=$(cd {remote_root_expr} && pwd); "
            f"run_dir=\"$remote_root/{run_name}\"; rm -rf \"$run_dir\"; mkdir -p \"$run_dir\"; printf '%s\\n' \"$run_dir\"",
        )
    )
    remote_run_dir = [line.strip() for line in remote_run_dir_output.splitlines() if line.strip().startswith("/")][-1]
    remote_project_dir = posixpath.join(remote_run_dir, "SeQUeNCe")
    remote_output_dir = posixpath.join(remote_project_dir, str(QUEUEING_MODULE / "results").replace("\\", "/"))

    archive_path = make_archive()
    remote_archive = posixpath.join(remote_run_dir, archive_path.name)
    try:
        print(f"Copying archive to {host}:{remote_archive}", flush=True)
        run(["scp", str(archive_path), f"{host}:{remote_archive}"])
        run(ssh(host, f"mkdir -p {quote(remote_project_dir)}; tar -xzf {quote(remote_archive)} -C {quote(remote_project_dir)}"))

        remote_worker_args = [
            "python",
            WORKER_SCRIPT.as_posix(),
            "--output-dir",
            remote_output_dir,
            "--workers",
            str(workers),
            "--num-runs",
            str(args.num_runs),
            "--horizon-s",
            str(args.horizon_s),
            "--lambda-num-runs",
            str(args.lambda_num_runs),
            "--lambda-horizon-s",
            str(args.lambda_horizon_s),
            "--simulation-point-stride",
            str(args.simulation_point_stride),
            "--model-fidelity-points",
            str(args.model_fidelity_points),
            "--parallel-backend",
            args.parallel_backend,
            *(["--no-parallelize-runs"] if args.no_parallelize_runs else []),
            "--heartbeat-s",
            str(args.heartbeat_s),
            "--ttl-diagnostic-runs",
            str(args.ttl_diagnostic_runs),
            "--ttl-diagnostic-horizon-s",
            str(args.ttl_diagnostic_horizon_s),
            "--ttl-grid-min-ms",
            str(args.ttl_grid_min_ms),
            "--ttl-grid-max-ms",
            str(args.ttl_grid_max_ms),
            "--ttl-grid-points",
            str(args.ttl_grid_points),
        ]
        if args.smoke:
            remote_worker_args.append("--smoke")
        remote_command = " ".join(quote(part) for part in remote_worker_args)
        setup = "\n".join(
            [
                "set -euo pipefail",
                f"cd {quote(remote_project_dir)}",
                f"{quote(remote_python)} -m venv .venv",
                ". .venv/bin/activate",
                "python -m ensurepip --upgrade >/dev/null 2>&1 || true",
                "python -m pip install --upgrade pip",
                args.install_command,
                remote_command,
            ]
        )
        run(ssh(host, f"bash -lc {quote(setup)}"))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        run(["scp", f"{host}:{remote_output_dir}/*.csv", str(args.output_dir)])
        run(["scp", f"{host}:{remote_output_dir}/*.json", str(args.output_dir)])
        print(f"Fetched CSV/JSON outputs to {args.output_dir}", flush=True)
    finally:
        try:
            archive_path.unlink(missing_ok=True)
            archive_path.parent.rmdir()
        except OSError:
            pass
        if not args.keep_remote:
            run(ssh(host, f"rm -rf -- {quote(remote_run_dir)}"))


def main() -> int:
    args = parse_args()
    if args.backend == "local":
        run_local(args)
    else:
        run_hpc(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
