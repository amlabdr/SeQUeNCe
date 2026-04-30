"""Run diagnosis dataset generation in a disposable SSH workspace.

This script is intentionally local-first: your laptop repository remains the
source of truth. The script packages the current working tree, copies it to an
HPC run directory, executes the generator there, fetches the finished dataset,
and removes the remote copy on success or Ctrl+C.
"""

from __future__ import annotations

import argparse
import os
import posixpath
import shlex
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_MODULE = Path("example/entanglement_distribution_experiment/ai_diagnosis_dataset")
DEFAULT_DATASET_NAME = "train_v5_60s_100khz_24h_temperature"

EXCLUDED_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".ipynb_checkpoints",
    "review_artifacts",
    "raw_sequences",
    "checkpoints",
}
EXCLUDED_TOP_LEVEL = {
    "generated",
}

INCLUDED_PATHS = [
    Path("sequence"),
    DATASET_MODULE,
]


@dataclass
class RemoteState:
    host: str
    remote_run_dir: str | None = None
    active_process: subprocess.Popen | None = None
    cleanup_started: bool = False


STATE = RemoteState(host="")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy this repo to an HPC over SSH, run dataset generation, fetch results, and clean up.")
    parser.add_argument("--host", default="enki", help="SSH host alias. Defaults to 'enki'.")
    parser.add_argument("--remote-root", default="~/sequence_dataset_runs", help="Remote parent directory for disposable run folders.")
    parser.add_argument("--remote-python", default="python3.12", help="Remote Python executable used to create the venv.")
    parser.add_argument("--workers", type=int, default=None, help="Remote workers. Defaults to all remote CPU cores.")
    parser.add_argument("--parallel-backend", choices=["process", "thread"], default="process", help="Generator parallel backend on the HPC.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME, help="Output dataset directory name.")
    parser.add_argument("--samples-per-label", type=int, default=500)
    parser.add_argument("--episode-duration-s", type=float, default=60.0)
    parser.add_argument("--window-duration-s", type=float, default=1.0)
    parser.add_argument("--source-frequency-hz", type=float, default=100_000.0)
    parser.add_argument("--visibility-stride-windows", type=int, default=30)
    parser.add_argument("--temperature-episode-span-s", type=float, default=86_400.0)
    parser.add_argument("--base-seed", type=int, default=300_000)
    parser.add_argument("--mode", choices=["sample", "full"], default="full", help="Generator preset mode.")
    parser.add_argument("--smoke", action="store_true", help="Run a small smoke dataset instead of the full settings.")
    parser.add_argument("--local-output-root", type=Path, default=REPO_ROOT / DATASET_MODULE / "generated", help="Local parent directory for fetched dataset.")
    parser.add_argument("--keep-remote", action="store_true", help="Keep the remote run directory after successful completion.")
    parser.add_argument("--cleanup-on-failure", action="store_true", help="Also delete the remote run directory if generation fails.")
    parser.add_argument("--skip-install", action="store_true", help="Skip remote virtualenv creation and dependency installation.")
    parser.add_argument(
        "--install-command",
        default="python -m pip install numpy scipy pandas matplotlib seaborn tqdm networkx qutip qutip-qip gmpy2",
        help="Command executed inside the activated remote venv to install dependencies/code.",
    )
    parser.add_argument("--archive-keep", action="store_true", help="Keep the temporary local tar archive for debugging.")
    return parser.parse_args()


def run_local(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, cwd=REPO_ROOT, text=True, check=check)


def capture_local(command: list[str], *, check: bool = True) -> str:
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)
    return result.stdout


def ssh_command(host: str, remote_command: str) -> list[str]:
    return ["ssh", host, remote_command]


def remote_quote(value: str) -> str:
    return shlex.quote(value)


def remote_path_expression(path: str) -> str:
    """Return a shell expression for a remote path, preserving ~/ expansion."""

    if path == "~":
        return "$HOME"
    if path.startswith("~/"):
        return "$HOME/" + remote_quote(path[2:])
    return remote_quote(path)


def remote_join(*parts: str) -> str:
    result = parts[0]
    for part in parts[1:]:
        result = posixpath.join(result, part)
    return result


def remote_rm_rf(host: str, remote_path: str) -> None:
    if not remote_path:
        return
    if "sequence_dataset_runs" not in remote_path or "run_" not in posixpath.basename(remote_path):
        print(f"Refusing cleanup of unexpected remote path: {remote_path}", file=sys.stderr)
        return
    command = f"rm -rf -- {remote_quote(remote_path)}"
    subprocess.run(ssh_command(host, command), cwd=REPO_ROOT)


def cleanup_remote(reason: str) -> None:
    if STATE.cleanup_started or STATE.remote_run_dir is None:
        return
    STATE.cleanup_started = True
    print(f"\nCleaning remote run directory because {reason}: {STATE.remote_run_dir}", flush=True)
    if STATE.active_process is not None and STATE.active_process.poll() is None:
        STATE.active_process.terminate()
        try:
            STATE.active_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            STATE.active_process.kill()
    remote_rm_rf(STATE.host, STATE.remote_run_dir)


def signal_handler(signum, _frame) -> None:
    cleanup_remote(f"signal {signum}")
    raise KeyboardInterrupt


def should_exclude(path: Path) -> bool:
    rel = path.relative_to(REPO_ROOT)
    parts = rel.parts
    if not parts:
        return False
    if parts[0] in EXCLUDED_TOP_LEVEL:
        return True
    if parts[:3] == ("example", "entanglement_distribution_experiment", "ai_diagnosis_dataset") and len(parts) >= 4 and parts[3] == "generated":
        return True
    return any(part in EXCLUDED_DIR_NAMES for part in parts)


def normalize_tarinfo(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo:
    tarinfo.uid = 0
    tarinfo.gid = 0
    tarinfo.uname = ""
    tarinfo.gname = ""
    if tarinfo.isdir():
        tarinfo.mode = 0o755
    else:
        tarinfo.mode = 0o644
    return tarinfo


def make_archive(keep: bool) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="sequence_hpc_package_"))
    archive_path = temp_dir / "sequence_project.tar.gz"
    print(f"Packaging project snapshot: {archive_path}", flush=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        for include_path in INCLUDED_PATHS:
            root = REPO_ROOT / include_path
            if not root.exists():
                raise FileNotFoundError(f"Required package path does not exist: {root}")
            paths = [root]
            if root.is_dir():
                paths.extend(root.rglob("*"))
            for path in paths:
                if should_exclude(path):
                    continue
                try:
                    archive.add(
                        path,
                        arcname=path.relative_to(REPO_ROOT),
                        recursive=False,
                        filter=normalize_tarinfo,
                    )
                except OSError as exc:
                    print(f"Skipping unreadable path while packaging: {path} ({exc})", file=sys.stderr, flush=True)
    if keep:
        print(f"Keeping archive: {archive_path}", flush=True)
    return archive_path


def detect_workers(host: str) -> int:
    output = capture_local(ssh_command(host, "nproc 2>/dev/null || getconf _NPROCESSORS_ONLN"), check=True)
    for line in output.splitlines():
        line = line.strip()
        if line.isdigit():
            return max(1, int(line))
    return 1


def print_remote_capabilities(host: str, workers: int) -> None:
    command = "hostname; uname -sr; printf 'cpus='; (nproc 2>/dev/null || getconf _NPROCESSORS_ONLN); awk '/MemTotal/ {printf \"mem_gib=%.1f\\n\", $2/1024/1024}' /proc/meminfo 2>/dev/null || true; command -v sbatch || true; command -v tmux || true"
    output = capture_local(ssh_command(host, command), check=False)
    print("Remote capability probe:", flush=True)
    for line in output.splitlines():
        if line.strip():
            print(f"  {line}", flush=True)
    print(f"Using workers={workers}", flush=True)


def remote_prepare(host: str, remote_root: str, run_name: str) -> tuple[str, str]:
    remote_root_expr = remote_path_expression(remote_root)
    command = (
        f"set -e; mkdir -p {remote_root_expr}; "
        f"remote_root=$(cd {remote_root_expr} && pwd); "
        f"run_dir=\"$remote_root/{run_name}\"; "
        f"rm -rf \"$run_dir\"; mkdir -p \"$run_dir\"; "
        f"printf '%s\\n' \"$run_dir\""
    )
    output = capture_local(ssh_command(host, command), check=True)
    lines = [line.strip() for line in output.splitlines() if line.strip().startswith("/")]
    if not lines:
        raise RuntimeError(f"Could not parse remote run directory from SSH output:\n{output}")
    remote_run_dir = lines[-1]
    remote_project_dir = remote_join(remote_run_dir, "SeQUeNCe")
    return remote_run_dir, remote_project_dir


def copy_archive(host: str, archive_path: Path, remote_run_dir: str) -> str:
    remote_archive = remote_join(remote_run_dir, "sequence_project.tar.gz")
    print(f"Copying archive to {host}:{remote_archive}", flush=True)
    run_local(["scp", str(archive_path), f"{host}:{remote_archive}"])
    return remote_archive


def extract_archive(host: str, remote_archive: str, remote_project_dir: str) -> None:
    command = (
        f"set -e; mkdir -p {remote_quote(remote_project_dir)}; "
        f"tar -xzf {remote_quote(remote_archive)} -C {remote_quote(remote_project_dir)}"
    )
    run_local(ssh_command(host, command))


def build_generator_args(args: argparse.Namespace, workers: int, remote_output_dir: str) -> list[str]:
    if args.smoke:
        mode = "sample"
        samples_per_label = 1
        episode_duration_s = 4.0
        source_frequency_hz = 1_000.0
        visibility_stride = 2
    else:
        mode = args.mode
        samples_per_label = args.samples_per_label
        episode_duration_s = args.episode_duration_s
        source_frequency_hz = args.source_frequency_hz
        visibility_stride = args.visibility_stride_windows

    return [
        "python",
        (DATASET_MODULE / "generate_dataset.py").as_posix(),
        "--mode",
        mode,
        "--samples-per-label",
        str(samples_per_label),
        "--episode-duration-s",
        str(episode_duration_s),
        "--window-duration-s",
        str(args.window_duration_s),
        "--source-frequency-hz",
        str(source_frequency_hz),
        "--visibility-stride-windows",
        str(visibility_stride),
        "--temperature-episode-span-s",
        str(args.temperature_episode_span_s),
        "--base-seed",
        str(args.base_seed),
        "--workers",
        str(workers),
        "--parallel-backend",
        args.parallel_backend,
        "--skip-raw",
        "--validation-off",
        "--output-dir",
        remote_output_dir,
    ]


def run_remote_generation(args: argparse.Namespace, host: str, remote_project_dir: str, workers: int, remote_dataset_dir: str) -> None:
    generator_args = build_generator_args(args, workers, remote_dataset_dir)
    generator_command = " ".join(remote_quote(part) for part in generator_args)
    setup_lines = [
        "set -euo pipefail",
        f"cd {remote_quote(remote_project_dir)}",
        "trap 'pkill -P $$ 2>/dev/null || true' HUP INT TERM",
    ]
    if not args.skip_install:
        setup_lines.extend(
            [
                f"{remote_quote(args.remote_python)} -m venv .venv",
                ". .venv/bin/activate",
                "python -m ensurepip --upgrade >/dev/null 2>&1 || true",
                "python -m pip install --upgrade pip",
                args.install_command,
            ]
        )
    else:
        setup_lines.append(". .venv/bin/activate")
    setup_lines.append(generator_command)
    remote_script = "\n".join(setup_lines)

    print("Starting remote dataset generation.", flush=True)
    print(f"Remote dataset dir: {remote_dataset_dir}", flush=True)
    STATE.active_process = subprocess.Popen(ssh_command(host, f"bash -lc {remote_quote(remote_script)}"), cwd=REPO_ROOT)
    return_code = STATE.active_process.wait()
    STATE.active_process = None
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, "remote generation")


def fetch_dataset(host: str, remote_dataset_dir: str, local_output_root: Path) -> Path:
    local_output_root.mkdir(parents=True, exist_ok=True)
    print(f"Fetching dataset to {local_output_root}", flush=True)
    run_local(["scp", "-r", f"{host}:{remote_dataset_dir}", str(local_output_root)])
    return local_output_root / posixpath.basename(remote_dataset_dir)


def main() -> int:
    args = parse_args()
    STATE.host = args.host
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    workers = int(args.workers) if args.workers is not None else detect_workers(args.host)
    print_remote_capabilities(args.host, workers)

    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{args.dataset_name}"
    remote_run_dir, remote_project_dir = remote_prepare(args.host, args.remote_root, run_name)
    STATE.remote_run_dir = remote_run_dir
    remote_dataset_dir = remote_join(remote_project_dir, str(DATASET_MODULE / "generated" / args.dataset_name).replace("\\", "/"))

    archive_path: Path | None = None
    try:
        archive_path = make_archive(args.archive_keep)
        remote_archive = copy_archive(args.host, archive_path, remote_run_dir)
        extract_archive(args.host, remote_archive, remote_project_dir)
        run_remote_generation(args, args.host, remote_project_dir, workers, remote_dataset_dir)
        local_dataset_dir = fetch_dataset(args.host, remote_dataset_dir, args.local_output_root)
        print(f"Fetched dataset: {local_dataset_dir}", flush=True)
        if not args.keep_remote:
            cleanup_remote("successful completion")
        else:
            print(f"Keeping remote run directory: {remote_run_dir}", flush=True)
        return 0
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if args.cleanup_on_failure:
            cleanup_remote("failure")
        else:
            print(f"Remote run directory preserved for debugging/resume: {remote_run_dir}", file=sys.stderr)
        return 1
    finally:
        if archive_path is not None and not args.archive_keep:
            try:
                archive_path.unlink(missing_ok=True)
                archive_path.parent.rmdir()
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
