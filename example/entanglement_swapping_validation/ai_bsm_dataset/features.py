"""Observable features from four polarization-BSM timestamp streams."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from example.hom_validation.ai_hom_dataset.features import (
    hom_lag_histogram,
    interarrival_stats,
)
from sequence.topology.optical_nodes import PolarizationBSMNode


DETECTORS = PolarizationBSMNode.DETECTOR_NAMES


def _slug(name: str) -> str:
    return name.lower()


def _greedy_bell_counts(streams: dict[str, list[int]], window_ps: int) -> dict[str, int]:
    clicks = sorted(
        (int(time_ps), index)
        for index, detector in enumerate(DETECTORS)
        for time_ps in streams[detector]
    )
    used: set[int] = set()
    counts = {"psi_minus": 0, "psi_plus": 0}
    for i, (time_i, detector_i) in enumerate(clicks):
        if i in used:
            continue
        best = None
        for j in range(i + 1, len(clicks)):
            if j in used:
                continue
            time_j, detector_j = clicks[j]
            if time_j - time_i > window_ps:
                break
            state = PolarizationBSMNode.classify_detector_pair(detector_i, detector_j)
            if state is not None and (best is None or time_j - time_i < best[0]):
                best = (time_j - time_i, j, state)
        if best is not None:
            used.update((i, best[1]))
            counts[best[2]] += 1
    return counts


def observable_features(
    streams: dict[str, list[int]],
    *,
    duration_s: float,
    coincidence_window_ps: int,
    histogram_range_ps: int,
    histogram_bin_width_ps: int,
) -> tuple[dict, dict]:
    duration = max(duration_s, 1e-12)
    features: dict[str, float | int] = {}
    raw_histograms = {}
    for detector in DETECTORS:
        prefix = _slug(detector)
        count = len(streams[detector])
        features[f"{prefix}_count"] = count
        features[f"{prefix}_rate_hz"] = count / duration
        features.update(interarrival_stats(streams[detector], prefix))

    rates = {detector: features[f"{_slug(detector)}_rate_hz"] for detector in DETECTORS}
    port3 = rates["D3H"] + rates["D3V"]
    port4 = rates["D4H"] + rates["D4V"]
    horizontal = rates["D3H"] + rates["D4H"]
    vertical = rates["D3V"] + rates["D4V"]
    features.update({
        "total_singles_rate_hz": sum(rates.values()),
        "port3_rate_hz": port3,
        "port4_rate_hz": port4,
        "horizontal_rate_hz": horizontal,
        "vertical_rate_hz": vertical,
        "port_asymmetry": (port3 - port4) / max(port3 + port4, 1e-12),
        "polarization_asymmetry": (horizontal - vertical) / max(horizontal + vertical, 1e-12),
    })

    pair_counts = {}
    for first, second in combinations(DETECTORS, 2):
        prefix = f"{_slug(first)}_{_slug(second)}"
        hist, raw = hom_lag_histogram(
            streams[first], streams[second],
            range_ps=histogram_range_ps,
            bin_width_ps=histogram_bin_width_ps,
            coincidence_window_ps=coincidence_window_ps,
            integration_time_s=duration,
        )
        pair_counts[prefix] = hist.coincidence_count_hist
        features.update({
            f"{prefix}_count": hist.coincidence_count_hist,
            f"{prefix}_rate_hz": hist.coincidence_count_hist / duration,
            f"{prefix}_peak_position_ps": hist.peak_position_ps,
            f"{prefix}_peak_width_ps": hist.peak_width_ps,
            f"{prefix}_peak_height": hist.peak_height,
            f"{prefix}_peak_snr": hist.peak_snr,
            f"{prefix}_accidental_rate_hz": hist.accidental_rate_hz,
            f"{prefix}_car": hist.car,
        })
        raw_histograms[prefix] = raw

    bell = _greedy_bell_counts(streams, coincidence_window_ps)
    accepted = bell["psi_minus"] + bell["psi_plus"]
    features.update({
        "psi_minus_count": bell["psi_minus"],
        "psi_minus_rate_hz": bell["psi_minus"] / duration,
        "psi_plus_count": bell["psi_plus"],
        "psi_plus_rate_hz": bell["psi_plus"] / duration,
        "accepted_bsm_count": accepted,
        "accepted_bsm_rate_hz": accepted / duration,
        "psi_minus_fraction": bell["psi_minus"] / max(accepted, 1),
        "psi_plus_fraction": bell["psi_plus"] / max(accepted, 1),
        "psi_balance": (bell["psi_minus"] - bell["psi_plus"]) / max(accepted, 1),
        "same_pol_cross_port_rate_hz": (
            pair_counts["d3h_d4h"] + pair_counts["d3v_d4v"]
        ) / duration,
    })
    return features, raw_histograms


DELTA_FEATURES = (
    "d3h_rate_hz", "d3v_rate_hz", "d4h_rate_hz", "d4v_rate_hz",
    "accepted_bsm_rate_hz", "psi_minus_rate_hz", "psi_plus_rate_hz",
    "port_asymmetry", "polarization_asymmetry", "same_pol_cross_port_rate_hz",
)


def add_deltas(rows: list[dict]) -> None:
    previous = None
    for row in rows:
        for key in DELTA_FEATURES:
            row[f"delta_{key}"] = (
                None if previous is None else float(row[key]) - float(previous[key])
            )
        previous = row
