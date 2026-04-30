"""Generate data products for the queueing-model paper-evaluation notebook."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
QUEUEING_DIR = REPO_ROOT / "example" / "queueing_model"
CORE_EXPERIMENT_DIR = REPO_ROOT / "example" / "entanglement_distribution_experiment"
for path in (REPO_ROOT, QUEUEING_DIR, CORE_EXPERIMENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from memory_allocation_queue_sim import (  # noqa: E402
    build_sequence_estimated_reference_config,
    reference_model_terms,
    reference_ttls_for_required_fidelity,
    run_allocation_sweep,
    run_memory_allocation_sequence_trial,
)
from eg_single_heralded_helpers import (  # noqa: E402
    collect_three_node_swap_time_trace,
    run_three_node_continuous_experiment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run memory-allocation paper evaluation sweeps.")
    parser.add_argument("--output-dir", type=Path, default=QUEUEING_DIR / "results")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
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
    parser.add_argument("--smoke", action="store_true", help="Use a very small run for backend testing.")
    return parser.parse_args()


def rows_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{k: v for k, v in row.items() if k != "trials"} for row in rows])


def simulation_fidelity_points(model_required_fidelities: np.ndarray, stride: int) -> np.ndarray:
    return model_required_fidelities[:: max(1, int(stride))]


def _executor_class(backend: str):
    return ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor


def _json_default(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_dataframe(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"saved {path} rows={len(df)}", flush=True)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    print(f"saved {path}", flush=True)


def three_node_components_from_config(
    config,
    *,
    left_cutoff_ratio: float = 1.0,
    right_cutoff_ratio: float = 1.0,
):
    left = replace(config.left_link_config(), cutoff_ratio=left_cutoff_ratio)
    right = replace(config.right_link_config(), cutoff_ratio=right_cutoff_ratio)
    swap = config.swap_config()
    return left, right, swap


def _continuous_target_task(args: tuple) -> dict:
    config, target_fidelity, runs, horizon_s, base_seed = args
    left, right, swap = three_node_components_from_config(config, left_cutoff_ratio=1.0, right_cutoff_ratio=1.0)
    out = run_three_node_continuous_experiment(
        left,
        right,
        swap,
        num_runs=runs,
        base_seed=base_seed,
        horizon_s=horizon_s,
        target_fidelity=target_fidelity,
    )
    return {
        "target_fidelity": target_fidelity,
        "throughput_mean": out["summary"]["throughput_hz"]["mean"],
        "throughput_std": out["summary"]["throughput_hz"]["std"],
        "fidelity_mean": out["summary"]["fidelity_mean"]["mean"],
        "fidelity_std": out["summary"]["fidelity_mean"]["std"],
        "latency_mean_s": out["summary"]["latency_mean_s"]["mean"],
        "latency_std_s": out["summary"]["latency_mean_s"]["std"],
        "pair_count_mean": out["summary"]["pair_count_mean"]["mean"],
        "pair_count_std": out["summary"]["pair_count_mean"]["std"],
    }


def build_requested_fidelity_sweep(
    config,
    *,
    target_fidelities: np.ndarray,
    runs: int,
    horizon_s: float,
    max_workers: int,
    parallel_backend: str,
) -> pd.DataFrame:
    tasks = [
        (config, float(target_fidelity), runs, horizon_s, 510_000 + idx * 1_000)
        for idx, target_fidelity in enumerate(target_fidelities)
    ]
    rows = []
    with _executor_class(parallel_backend)(max_workers=max_workers) as executor:
        futures = [executor.submit(_continuous_target_task, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows).sort_values("target_fidelity").reset_index(drop=True)


def _long_link_cutoff_task(args: tuple) -> dict:
    config, cutoff_time_ms, target_fidelity, runs, horizon_s, base_seed = args
    left_cutoff_ratio = (cutoff_time_ms * 1e-3) / config.coherence_time_s if config.coherence_time_s > 0 else 0.0
    left, right, swap = three_node_components_from_config(config, left_cutoff_ratio=left_cutoff_ratio, right_cutoff_ratio=1.0)
    out = run_three_node_continuous_experiment(
        left,
        right,
        swap,
        num_runs=runs,
        base_seed=base_seed,
        horizon_s=horizon_s,
        target_fidelity=target_fidelity,
    )
    return {
        "left_cutoff_time_ms": cutoff_time_ms,
        "left_cutoff_ratio": left_cutoff_ratio,
        "target_fidelity": target_fidelity,
        "throughput_mean": out["summary"]["throughput_hz"]["mean"],
        "throughput_std": out["summary"]["throughput_hz"]["std"],
        "fidelity_mean": out["summary"]["fidelity_mean"]["mean"],
        "fidelity_std": out["summary"]["fidelity_mean"]["std"],
        "latency_mean_s": out["summary"]["latency_mean_s"]["mean"],
        "latency_std_s": out["summary"]["latency_mean_s"]["std"],
    }


def build_long_link_cutoff_sweep(
    config,
    *,
    cutoff_times_ms: np.ndarray,
    target_fidelity: float,
    runs: int,
    horizon_s: float,
    max_workers: int,
    parallel_backend: str,
) -> pd.DataFrame:
    tasks = [
        (config, float(cutoff_ms), target_fidelity, runs, horizon_s, 610_000 + idx * 1_000)
        for idx, cutoff_ms in enumerate(cutoff_times_ms)
    ]
    rows = []
    with _executor_class(parallel_backend)(max_workers=max_workers) as executor:
        futures = [executor.submit(_long_link_cutoff_task, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows).sort_values("left_cutoff_time_ms").reset_index(drop=True)


def _run_ttl_trial_task(args: tuple) -> dict:
    (
        config,
        left_capacity,
        right_capacity,
        left_ttl_s,
        right_ttl_s,
        target_fidelity,
        multiplexing,
        horizon_s,
        seed,
    ) = args
    trial = run_memory_allocation_sequence_trial(
        config,
        left_capacity=left_capacity,
        right_capacity=right_capacity,
        left_ttl_s=left_ttl_s,
        right_ttl_s=right_ttl_s,
        target_fidelity=target_fidelity,
        multiplexing=multiplexing,
        horizon_s=horizon_s,
        seed=seed,
    )
    delivered_count = len(trial["delivery_fidelities"])
    below_target_count = sum(f < target_fidelity for f in trial["delivery_fidelities"])
    return {
        "throughput_hz": trial["throughput_hz"],
        "accepted_throughput_hz": trial["accepted_throughput_hz"],
        "fidelity_mean": trial["fidelity_mean"],
        "accepted_fidelity_mean": trial["accepted_fidelity_mean"],
        "delivered_count": delivered_count,
        "below_target_count": below_target_count,
    }


def run_ttl_motivation_trials(
    config,
    *,
    allocation=(1, 1),
    target_fidelity: float,
    runs: int,
    horizon_s: float,
    multiplexing: bool,
    max_workers: int,
    parallel_backend: str,
) -> pd.DataFrame:
    model_left_ttl_s, model_right_ttl_s = reference_ttls_for_required_fidelity(config, target_fidelity)
    scenarios = [
        ("no TTL", None, None),
        ("model TTL", model_left_ttl_s, model_right_ttl_s),
    ]
    tasks = []
    for scenario_idx, (label, left_ttl_s, right_ttl_s) in enumerate(scenarios):
        for run_idx in range(runs):
            tasks.append(
                (
                    label,
                    run_idx,
                    _run_ttl_trial_task,
                    (
                        config,
                        allocation[0],
                        allocation[1],
                        left_ttl_s,
                        right_ttl_s,
                        target_fidelity,
                        multiplexing,
                        horizon_s,
                        720_000 + 10_000 * allocation[0] + 1_000 * allocation[1] + 100 * scenario_idx + run_idx,
                    ),
                )
            )
    rows = []
    with _executor_class(parallel_backend)(max_workers=max_workers) as executor:
        future_map = {executor.submit(fn, fn_args): (label, run_idx) for label, run_idx, fn, fn_args in tasks}
        for future in as_completed(future_map):
            label, run_idx = future_map[future]
            result = future.result()
            rows.append(
                {
                    "allocation": f"{allocation[0]}:{allocation[1]}",
                    "scenario": label,
                    "run_idx": run_idx,
                    "rate_hz": result["throughput_hz"],
                    "accepted_rate_hz": result["accepted_throughput_hz"],
                    "fidelity": result["fidelity_mean"],
                    "accepted_fidelity": result["accepted_fidelity_mean"],
                    "below_target_count": result["below_target_count"],
                    "delivered_count": result["delivered_count"],
                }
            )
    return pd.DataFrame(rows)


def summarize_ttl_scenarios(
    config,
    *,
    allocation: tuple[int, int],
    target_fidelity: float,
    multiplexing: bool,
    runs: int,
    horizon_s: float,
    max_workers: int,
    parallel_backend: str,
) -> pd.DataFrame:
    model_left_ttl_s, model_right_ttl_s = reference_ttls_for_required_fidelity(config, target_fidelity)
    scenarios = [
        ("no TTL", None, None),
        ("model TTL", model_left_ttl_s, model_right_ttl_s),
    ]
    tasks = []
    for scenario_idx, (label, left_ttl_s, right_ttl_s) in enumerate(scenarios):
        for run_idx in range(runs):
            tasks.append(
                (
                    label,
                    run_idx,
                    _run_ttl_trial_task,
                    (
                        config,
                        allocation[0],
                        allocation[1],
                        left_ttl_s,
                        right_ttl_s,
                        target_fidelity,
                        multiplexing,
                        horizon_s,
                        920_000
                        + 100_000 * int(multiplexing)
                        + 10_000 * allocation[0]
                        + 1_000 * allocation[1]
                        + 100 * scenario_idx
                        + run_idx,
                    ),
                )
            )
    rows = []
    with _executor_class(parallel_backend)(max_workers=max_workers) as executor:
        future_map = {executor.submit(fn, fn_args): (label, run_idx) for label, run_idx, fn, fn_args in tasks}
        for future in as_completed(future_map):
            label, run_idx = future_map[future]
            result = future.result()
            rows.append(
                {
                    "allocation": f"{allocation[0]}:{allocation[1]}",
                    "multiplexing": multiplexing,
                    "target_fidelity": target_fidelity,
                    "scenario": label,
                    "run_idx": run_idx,
                    "rate_hz": result["throughput_hz"],
                    "accepted_rate_hz": result["accepted_throughput_hz"],
                    "below_target_rate_hz": result["throughput_hz"] - result["accepted_throughput_hz"],
                    "fidelity": result["fidelity_mean"],
                    "accepted_fidelity": result["accepted_fidelity_mean"],
                    "below_target_count": result["below_target_count"],
                    "delivered_count": result["delivered_count"],
                }
            )
    return pd.DataFrame(rows)


def _simulate_ttl_pair_task(args: tuple) -> dict:
    (
        config,
        left_capacity,
        right_capacity,
        left_ttl_ms,
        right_ttl_ms,
        target_fidelity,
        multiplexing,
        runs,
        horizon_s,
    ) = args
    delivered_rates = []
    accepted_rates = []
    delivered_fidelities = []
    below_target_fractions = []
    for run_idx in range(runs):
        trial = run_memory_allocation_sequence_trial(
            config,
            left_capacity=left_capacity,
            right_capacity=right_capacity,
            left_ttl_s=left_ttl_ms * 1e-3,
            right_ttl_s=right_ttl_ms * 1e-3,
            target_fidelity=target_fidelity,
            multiplexing=multiplexing,
            horizon_s=horizon_s,
            seed=810_000 + int(1000 * left_ttl_ms) + int(100 * right_ttl_ms) + run_idx,
        )
        delivered_rates.append(trial["throughput_hz"])
        accepted_rates.append(trial["accepted_throughput_hz"])
        delivered_fidelities.extend(trial["delivery_fidelities"])
        delivered_count = len(trial["delivery_fidelities"])
        below_target_count = sum(f < target_fidelity for f in trial["delivery_fidelities"])
        below_target_fractions.append(below_target_count / delivered_count if delivered_count else 0.0)
    return {
        "left_ttl_ms": left_ttl_ms,
        "right_ttl_ms": right_ttl_ms,
        "simulation_rate_hz": float(np.mean(delivered_rates)) if delivered_rates else 0.0,
        "simulation_accepted_rate_hz": float(np.mean(accepted_rates)) if accepted_rates else 0.0,
        "simulation_fidelity": float(np.nanmean(delivered_fidelities)) if delivered_fidelities else float("nan"),
        "simulation_below_target_fraction": float(np.mean(below_target_fractions)) if below_target_fractions else 0.0,
    }


def build_ttl_heatmap(
    config,
    *,
    left_capacity: int,
    right_capacity: int,
    target_fidelity: float,
    ttl_grid_ms: np.ndarray,
    multiplexing: bool,
    runs: int,
    horizon_s: float,
    max_workers: int,
    parallel_backend: str,
) -> pd.DataFrame:
    tasks = [
        (
            config,
            left_capacity,
            right_capacity,
            float(left_ttl_ms),
            float(right_ttl_ms),
            target_fidelity,
            multiplexing,
            runs,
            horizon_s,
        )
        for left_ttl_ms in ttl_grid_ms
        for right_ttl_ms in ttl_grid_ms
    ]
    rows = []
    with _executor_class(parallel_backend)(max_workers=max_workers) as executor:
        futures = [executor.submit(_simulate_ttl_pair_task, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
    return pd.DataFrame(rows)


def summarize_ttl_optimality(config, grid_df: pd.DataFrame, target_fidelity: float, allocation: tuple[int, int]) -> pd.Series:
    left_model_ttl_s, right_model_ttl_s = reference_ttls_for_required_fidelity(config, target_fidelity)
    best = grid_df.loc[grid_df["simulation_accepted_rate_hz"].idxmax()].copy()
    work_df = grid_df.copy()
    work_df["ttl_distance_ms"] = np.sqrt(
        (work_df["left_ttl_ms"] - left_model_ttl_s * 1e3) ** 2 + (work_df["right_ttl_ms"] - right_model_ttl_s * 1e3) ** 2
    )
    model_grid = work_df.sort_values(
        ["ttl_distance_ms", "simulation_below_target_fraction", "simulation_accepted_rate_hz"],
        ascending=[True, True, False],
    ).iloc[0].copy()
    best_rate = float(best["simulation_accepted_rate_hz"])
    model_rate = float(model_grid["simulation_accepted_rate_hz"])
    regret_pct = 100.0 * (best_rate - model_rate) / best_rate if best_rate > 0 else 0.0
    return pd.Series(
        {
            "allocation": f"{allocation[0]}:{allocation[1]}",
            "target_fidelity": target_fidelity,
            "model_left_ttl_ms": left_model_ttl_s * 1e3,
            "model_right_ttl_ms": right_model_ttl_s * 1e3,
            "nearest_grid_left_ttl_ms": model_grid["left_ttl_ms"],
            "nearest_grid_right_ttl_ms": model_grid["right_ttl_ms"],
            "model_grid_accepted_rate_hz": model_rate,
            "model_grid_below_target_pct": 100.0 * float(model_grid["simulation_below_target_fraction"]),
            "best_left_ttl_ms": best["left_ttl_ms"],
            "best_right_ttl_ms": best["right_ttl_ms"],
            "best_accepted_rate_hz": best_rate,
            "best_below_target_pct": 100.0 * float(best["simulation_below_target_fraction"]),
            "rate_regret_pct": regret_pct,
        }
    )


def main() -> int:
    args = parse_args()
    config = build_sequence_estimated_reference_config(
        lambda_num_runs=args.lambda_num_runs,
        lambda_horizon_s=args.lambda_horizon_s,
        lambda_max_workers=args.workers,
        lambda_parallel_backend=args.parallel_backend,
    )
    print(
        "Reference config: "
        f"left_lambda={config.left_rate_hz:.3f} Hz, "
        f"right_lambda={config.right_rate_hz:.3f} Hz, "
        f"left_raw_fidelity={config.left_raw_fidelity:.6f}, "
        f"right_raw_fidelity={config.right_raw_fidelity:.6f}",
        flush=True,
    )
    allocations = [(1, 1), (1, 5), (2, 4), (3, 3), (4, 2), (5, 1)]
    if args.smoke:
        allocations = [(1, 1), (3, 3)]
        args.num_runs = min(args.num_runs, 2)
        args.horizon_s = min(args.horizon_s, 0.5)
        args.simulation_point_stride = max(args.simulation_point_stride, 5)
        args.model_fidelity_points = min(args.model_fidelity_points, 15)
        args.ttl_diagnostic_runs = min(args.ttl_diagnostic_runs, 2)
        args.ttl_diagnostic_horizon_s = min(args.ttl_diagnostic_horizon_s, 0.5)
        args.ttl_grid_points = min(args.ttl_grid_points, 5)

    f_max = reference_model_terms(config)["F_max"]
    model_required_fidelities = np.linspace(0.75, f_max - 1e-4, args.model_fidelity_points)
    sim_required_fidelities = simulation_fidelity_points(model_required_fidelities, args.simulation_point_stride)
    final_target_fidelity = float(sim_required_fidelities[-1])
    behavior_trace_target_fidelity = final_target_fidelity
    cutoff_times_ms = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0], dtype=float)
    ttl_motivation_cases = [
        ((1, 1), True, "allocation 1:1"),
        ((3, 3), False, "allocation 3:3 | no multiplexing"),
        ((3, 3), True, "allocation 3:3 | multiplexing"),
        ((1, 1), True, "allocation 1:1 | doubled distance"),
        ((3, 3), False, "allocation 3:3 | no multiplexing | doubled distance"),
        ((3, 3), True, "allocation 3:3 | multiplexing | doubled distance"),
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing memory-allocation paper evaluation CSVs to {args.output_dir}", flush=True)

    model_no_mux_rows = run_allocation_sweep(
        config,
        allocations=allocations,
        required_fidelities=model_required_fidelities,
        multiplexing=False,
        num_runs=0,
        progress=True,
        include_trials=False,
    )
    save_dataframe(args.output_dir / "model_no_multiplexing.csv", rows_to_dataframe(model_no_mux_rows))
    model_mux_rows = run_allocation_sweep(
        config,
        allocations=allocations,
        required_fidelities=model_required_fidelities,
        multiplexing=True,
        num_runs=0,
        progress=True,
        include_trials=False,
    )
    save_dataframe(args.output_dir / "model_multiplexing.csv", rows_to_dataframe(model_mux_rows))
    no_mux_rows = run_allocation_sweep(
        config,
        allocations=allocations,
        required_fidelities=sim_required_fidelities,
        multiplexing=False,
        num_runs=args.num_runs,
        horizon_s=args.horizon_s,
        base_seed=310_000,
        max_workers=args.workers,
        parallel_backend=args.parallel_backend,
        parallelize_runs=not args.no_parallelize_runs,
        progress=True,
        heartbeat_s=args.heartbeat_s,
        include_trials=False,
    )
    save_dataframe(args.output_dir / "simulation_no_multiplexing.csv", rows_to_dataframe(no_mux_rows))
    mux_rows = run_allocation_sweep(
        config,
        allocations=allocations,
        required_fidelities=sim_required_fidelities,
        multiplexing=True,
        num_runs=args.num_runs,
        horizon_s=args.horizon_s,
        base_seed=410_000,
        max_workers=args.workers,
        parallel_backend=args.parallel_backend,
        parallelize_runs=not args.no_parallelize_runs,
        progress=True,
        heartbeat_s=args.heartbeat_s,
        include_trials=False,
    )
    save_dataframe(args.output_dir / "simulation_multiplexing.csv", rows_to_dataframe(mux_rows))
    requested_fidelity_df = build_requested_fidelity_sweep(
        config,
        target_fidelities=sim_required_fidelities,
        runs=args.ttl_diagnostic_runs,
        horizon_s=args.horizon_s,
        max_workers=args.workers,
        parallel_backend=args.parallel_backend,
    )
    save_dataframe(args.output_dir / "requested_fidelity_sweep_32_18_alloc_1_1.csv", requested_fidelity_df)
    long_link_cutoff_df = build_long_link_cutoff_sweep(
        config,
        cutoff_times_ms=cutoff_times_ms,
        target_fidelity=final_target_fidelity,
        runs=args.ttl_diagnostic_runs,
        horizon_s=args.horizon_s,
        max_workers=args.workers,
        parallel_backend=args.parallel_backend,
    )
    save_dataframe(args.output_dir / "long_link_cutoff_sweep_32_18_alloc_1_1.csv", long_link_cutoff_df)
    trace_left, trace_right, trace_swap = three_node_components_from_config(config, left_cutoff_ratio=1.0, right_cutoff_ratio=1.0)
    behavior_trace = collect_three_node_swap_time_trace(
        trace_left,
        trace_right,
        trace_swap,
        seed=470_000,
        stop_time_s=0.08,
        target_fidelity=behavior_trace_target_fidelity,
        max_seed_tries=40,
    )
    if not behavior_trace.get("success"):
        raise RuntimeError("Failed to collect 32 km / 18 km three-node behavior trace.")
    save_json(args.output_dir / "behavior_trace_32_18_alloc_1_1.json", behavior_trace)
    ttl_motivation_frames = []
    for allocation, multiplexing, case_label in ttl_motivation_cases:
        case_config = config
        if case_label.endswith("doubled distance"):
            case_config = build_sequence_estimated_reference_config(
                lambda_num_runs=args.lambda_num_runs,
                lambda_horizon_s=args.lambda_horizon_s,
                lambda_max_workers=args.workers,
                lambda_parallel_backend=args.parallel_backend,
                left_distance_km=64.0,
                right_distance_km=36.0,
                coherence_time_s=config.coherence_time_s,
                swap_success_prob=config.swap_success_prob,
            )
        for target_fidelity in sim_required_fidelities:
            case_df = summarize_ttl_scenarios(
                case_config,
                allocation=allocation,
                target_fidelity=float(target_fidelity),
                multiplexing=multiplexing,
                runs=args.ttl_diagnostic_runs,
                horizon_s=args.ttl_diagnostic_horizon_s,
                max_workers=args.workers,
                parallel_backend=args.parallel_backend,
            )
            case_df["case_label"] = case_label
            ttl_motivation_frames.append(case_df)
    ttl_motivation_df = pd.concat(ttl_motivation_frames, ignore_index=True)
    save_dataframe(args.output_dir / "ttl_motivation.csv", ttl_motivation_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
