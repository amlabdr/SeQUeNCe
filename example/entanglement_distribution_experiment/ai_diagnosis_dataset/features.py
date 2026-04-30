"""Feature extraction helpers for measurement logs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np


@dataclass
class HistogramFeatures:
    """Derived coincidence-histogram features."""

    peak_position_ps: float
    peak_width_ps: float
    peak_height: float
    peak_snr: float
    coincidence_rate_hz: float
    coincidences_count: int
    accidental_coincidences_count_estimate: float
    accidental_coincidence_rate_hz: float
    car: float


def flatten_detector_times(detector_times: Iterable[Iterable[int]]) -> list[int]:
    """Flatten per-detector click logs into a sorted list."""

    merged: list[int] = []
    for values in detector_times:
        merged.extend(int(v) for v in values)
    merged.sort()
    return merged


def calculate_fwhm_ps(bin_centers_ps: np.ndarray, histogram: np.ndarray) -> float:
    """Estimate FWHM from a histogram."""

    if histogram.size == 0 or np.max(histogram) <= 0:
        return 0.0

    peak_index = int(np.argmax(histogram))
    half_max = 0.5 * float(histogram[peak_index])

    left = peak_index
    while left > 0 and histogram[left] >= half_max:
        left -= 1

    right = peak_index
    while right < histogram.size - 1 and histogram[right] >= half_max:
        right += 1

    return float(abs(bin_centers_ps[right] - bin_centers_ps[left]))


def estimate_peak_position_ps(bin_centers_ps: np.ndarray, histogram: np.ndarray) -> float:
    """Estimate peak position with a small weighted centroid around the tallest bin."""

    if histogram.size == 0 or np.max(histogram) <= 0:
        return 0.0

    peak_index = int(np.argmax(histogram))
    left = max(0, peak_index - 1)
    right = min(histogram.size, peak_index + 2)
    weights = histogram[left:right].astype(float)
    centers = bin_centers_ps[left:right].astype(float)
    total = float(np.sum(weights))
    if total <= 0:
        return float(bin_centers_ps[peak_index])
    return float(np.sum(weights * centers) / total)


def coincidence_histogram(
    timestamps_a_ps: list[int],
    timestamps_b_ps: list[int],
    histogram_range_ps: int,
    bin_width_ps: int,
    coincidence_window_ps: int,
    integration_time_s: float,
    offset_b_ps: int = 0,
    fixed_peak_position_ps: float | None = None,
) -> tuple[HistogramFeatures, dict[str, list[float] | list[int] | float]]:
    """Build a coincidence histogram and summary features."""

    if not timestamps_a_ps or not timestamps_b_ps:
        empty_hist = [0] * max(1, int(np.ceil(2 * histogram_range_ps / bin_width_ps)))
        centers = list(
            np.linspace(-histogram_range_ps + 0.5 * bin_width_ps, histogram_range_ps - 0.5 * bin_width_ps, len(empty_hist))
        )
        features = HistogramFeatures(
            peak_position_ps=0.0,
            peak_width_ps=0.0,
            peak_height=0.0,
            peak_snr=0.0,
            coincidence_rate_hz=0.0,
            coincidences_count=0,
            accidental_coincidences_count_estimate=0.0,
            accidental_coincidence_rate_hz=0.0,
            car=0.0,
        )
        return features, {"bin_centers_ps": centers, "histogram": empty_hist}

    a = np.asarray(sorted(timestamps_a_ps), dtype=np.int64)
    b = np.asarray(sorted(int(v) + int(offset_b_ps) for v in timestamps_b_ps), dtype=np.int64)

    deltas: list[int] = []
    j0 = 0
    for t_a in a:
        while j0 < len(b) and b[j0] < t_a - histogram_range_ps:
            j0 += 1
        j = j0
        while j < len(b) and b[j] <= t_a + histogram_range_ps:
            deltas.append(int(t_a - b[j]))
            j += 1

    bins = max(1, int(np.ceil((2 * histogram_range_ps) / bin_width_ps)))
    histogram, bin_edges = np.histogram(deltas, bins=bins, range=(-histogram_range_ps, histogram_range_ps))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    peak_index = int(np.argmax(histogram))
    peak_position_ps = estimate_peak_position_ps(bin_centers, histogram) if fixed_peak_position_ps is None else float(fixed_peak_position_ps)
    peak_height = float(histogram[peak_index])
    peak_width_ps = calculate_fwhm_ps(bin_centers, histogram)

    win_low = peak_position_ps - 0.5 * coincidence_window_ps
    win_high = peak_position_ps + 0.5 * coincidence_window_ps
    coincidence_mask = (bin_centers >= win_low) & (bin_centers <= win_high)
    coincidences_count = int(np.sum(histogram[coincidence_mask]))
    background_bins = histogram[~coincidence_mask]
    background_level = float(np.mean(background_bins)) if background_bins.size else 0.0
    coincidence_bins = int(np.count_nonzero(coincidence_mask))
    accidental_coincidences_count_estimate = float(background_level * coincidence_bins)
    accidental_coincidence_rate_hz = float(accidental_coincidences_count_estimate / max(integration_time_s, 1e-12))
    snr = float(peak_height / max(background_level, 1.0))
    car = float(coincidences_count / max(accidental_coincidences_count_estimate, 1.0))
    coincidence_rate_hz = float(coincidences_count / max(integration_time_s, 1e-12))

    features = HistogramFeatures(
        peak_position_ps=peak_position_ps,
        peak_width_ps=peak_width_ps,
        peak_height=peak_height,
        peak_snr=snr,
        coincidence_rate_hz=coincidence_rate_hz,
        coincidences_count=coincidences_count,
        accidental_coincidences_count_estimate=accidental_coincidences_count_estimate,
        accidental_coincidence_rate_hz=accidental_coincidence_rate_hz,
        car=car,
    )
    raw = {
        "bin_centers_ps": bin_centers.astype(float).tolist(),
        "histogram": histogram.astype(int).tolist(),
        "window_low_ps": float(win_low),
        "window_high_ps": float(win_high),
        "accidental_coincidences_count_estimate": float(accidental_coincidences_count_estimate),
        "accidental_coincidence_rate_hz": float(accidental_coincidence_rate_hz),
        "car": float(car),
    }
    return features, raw


def estimate_visibility(angle_deg_to_rate: dict[float, float]) -> float:
    """Compute a simple fringe visibility from coincidence rates."""

    if not angle_deg_to_rate:
        return 0.0
    values = np.asarray(list(angle_deg_to_rate.values()), dtype=float)
    max_val = float(np.max(values))
    min_val = float(np.min(values))
    if max_val + min_val <= 0:
        return 0.0
    return float((max_val - min_val) / (max_val + min_val))


def histogram_features_to_dict(features: HistogramFeatures) -> dict[str, float | int]:
    """Convert histogram features into a row-friendly dictionary."""

    return asdict(features)
