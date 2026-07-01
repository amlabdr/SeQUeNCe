"""HOM-only feature extraction from detector timestamp streams."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from example.hom_validation.simulator import CoincidenceConfig, hom_twofold_events


@dataclass
class HOMHistogramFeatures:
    peak_position_ps: float
    peak_width_ps: float
    peak_height: float
    peak_snr: float
    coincidence_count_hist: int
    accidental_count_estimate: float
    accidental_rate_hz: float
    car: float


def _as_sorted_array(values: Sequence[int]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    if arr.size > 1 and np.any(arr[1:] < arr[:-1]):
        arr.sort()
    return arr


def lag_deltas_ps(t1_ps: Sequence[int], t2_ps: Sequence[int], range_ps: int) -> np.ndarray:
    t1 = _as_sorted_array(t1_ps)
    t2 = _as_sorted_array(t2_ps)
    deltas: list[int] = []
    j0 = 0
    for t in t1:
        while j0 < len(t2) and t2[j0] < t - range_ps:
            j0 += 1
        j = j0
        while j < len(t2) and t2[j] <= t + range_ps:
            deltas.append(int(t2[j] - t))
            j += 1
    return np.asarray(deltas, dtype=np.int64)


def _fwhm(bin_centers: np.ndarray, histogram: np.ndarray) -> float:
    if histogram.size == 0 or int(histogram.max()) <= 0:
        return 0.0
    peak = int(np.argmax(histogram))
    half = 0.5 * float(histogram[peak])
    left = peak
    right = peak
    while left > 0 and histogram[left] >= half:
        left -= 1
    while right < histogram.size - 1 and histogram[right] >= half:
        right += 1
    return float(abs(bin_centers[right] - bin_centers[left]))


def hom_lag_histogram(
    hom1_ps: Sequence[int],
    hom2_ps: Sequence[int],
    *,
    range_ps: int,
    bin_width_ps: int,
    coincidence_window_ps: int,
    integration_time_s: float,
    hom2_timestamp_offset_ps: int = 0,
) -> tuple[HOMHistogramFeatures, dict[str, list[float] | list[int]]]:
    hom2_shifted = [int(t) + int(hom2_timestamp_offset_ps) for t in hom2_ps]
    deltas = lag_deltas_ps(hom1_ps, hom2_shifted, range_ps)
    bins = np.arange(-int(range_ps), int(range_ps) + int(bin_width_ps), int(bin_width_ps))
    histogram, edges = np.histogram(deltas, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if histogram.size and int(histogram.max()) > 0:
        peak_idx = int(np.argmax(histogram))
        peak_position = float(centers[peak_idx])
        peak_height = float(histogram[peak_idx])
    else:
        peak_position = 0.0
        peak_height = 0.0

    coinc_mask = np.abs(centers) <= float(coincidence_window_ps)
    coincidence_count = int(histogram[coinc_mask].sum())
    background = histogram[~coinc_mask]
    background_level = float(background.mean()) if background.size else 0.0
    accidental_count = background_level * int(np.count_nonzero(coinc_mask))
    peak_snr = float(peak_height / max(background_level, 1.0))
    car = float(coincidence_count / max(accidental_count, 1.0))

    features = HOMHistogramFeatures(
        peak_position_ps=peak_position,
        peak_width_ps=_fwhm(centers, histogram),
        peak_height=peak_height,
        peak_snr=peak_snr,
        coincidence_count_hist=coincidence_count,
        accidental_count_estimate=float(accidental_count),
        accidental_rate_hz=float(accidental_count / max(float(integration_time_s), 1e-12)),
        car=car,
    )
    raw = {
        "bin_centers_ps": centers.astype(float).tolist(),
        "histogram": histogram.astype(int).tolist(),
    }
    return features, raw


def interarrival_stats(values: Sequence[int], prefix: str) -> dict[str, float]:
    arr = _as_sorted_array(values)
    if arr.size < 2:
        return {
            f"{prefix}_interarrival_mean_ps": 0.0,
            f"{prefix}_interarrival_std_ps": 0.0,
            f"{prefix}_interarrival_min_ps": 0.0,
            f"{prefix}_interarrival_max_ps": 0.0,
        }
    diffs = np.diff(arr).astype(float)
    return {
        f"{prefix}_interarrival_mean_ps": float(diffs.mean()),
        f"{prefix}_interarrival_std_ps": float(diffs.std(ddof=0)),
        f"{prefix}_interarrival_min_ps": float(diffs.min()),
        f"{prefix}_interarrival_max_ps": float(diffs.max()),
    }


def hom_observable_features(
    *,
    hom1_ps: Sequence[int],
    hom2_ps: Sequence[int],
    integration_time_s: float,
    histogram_range_ps: int,
    histogram_bin_width_ps: int,
    coincidence_window_ps: int,
    hom_offset_ps: int = 0,
    hom2_timestamp_offset_ps: int = 0,
) -> tuple[dict[str, float | int], dict[str, list[float] | list[int]]]:
    hom2_shifted = [int(t) + int(hom2_timestamp_offset_ps) for t in hom2_ps]
    # The shared simulator retains this legacy helper name for API compatibility.
    twofolds = hom_twofold_events(
        hom1_ps,
        hom2_shifted,
        CoincidenceConfig(window_ps=int(coincidence_window_ps), offset_ps=int(hom_offset_ps)),
        assume_sorted=True,
    )
    hist_features, raw_hist = hom_lag_histogram(
        hom1_ps,
        hom2_ps,
        range_ps=histogram_range_ps,
        bin_width_ps=histogram_bin_width_ps,
        coincidence_window_ps=coincidence_window_ps,
        integration_time_s=integration_time_s,
        hom2_timestamp_offset_ps=hom2_timestamp_offset_ps,
    )

    hom1_count = int(len(hom1_ps))
    hom2_count = int(len(hom2_ps))
    singles_sum = hom1_count + hom2_count
    twofold_count = int(len(twofolds))
    duration = max(float(integration_time_s), 1e-12)
    features: dict[str, float | int] = {
        "hom1_count": hom1_count,
        "hom2_count": hom2_count,
        "hom1_rate_hz": float(hom1_count / duration),
        "hom2_rate_hz": float(hom2_count / duration),
        "hom_singles_sum_rate_hz": float(singles_sum / duration),
        "hom_twofold_count": twofold_count,
        "hom_twofold_rate_hz": float(twofold_count / duration),
        **asdict(hist_features),
        **interarrival_stats(hom1_ps, "hom1"),
        **interarrival_stats(hom2_shifted, "hom2"),
    }
    return features, raw_hist


def add_temporal_derived_features(rows: list[dict]) -> list[dict]:
    previous: dict | None = None
    for row in rows:
        if previous is None:
            for key in [
                "hom1_rate_hz",
                "hom2_rate_hz",
                "hom_twofold_rate_hz",
                "peak_position_ps",
                "peak_width_ps",
                "peak_snr",
                "car",
                "accidental_rate_hz",
            ]:
                row[f"delta_{key}"] = None
        else:
            for key in [
                "hom1_rate_hz",
                "hom2_rate_hz",
                "hom_twofold_rate_hz",
                "peak_position_ps",
                "peak_width_ps",
                "peak_snr",
                "car",
                "accidental_rate_hz",
            ]:
                row[f"delta_{key}"] = float(row.get(key, 0.0)) - float(previous.get(key, 0.0))

        previous = row
    return rows
