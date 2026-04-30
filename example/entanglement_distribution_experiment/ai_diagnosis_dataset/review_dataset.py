"""Generate validation plots and a short review report for a generated dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


LABEL_ORDER = [
    "normal",
    "synchronization_issue",
    "peak_shift_temperature",
    "polarization_drift",
    "raman_noise",
    "loss",
]

FAULT_LABELS = [label for label in LABEL_ORDER if label != "normal"]

TRAJECTORY_VARIANTS = [
    "progressive",
    "recovering",
    "oscillatory",
    "intermittent",
]

FAULT_DISPLAY_NAMES = {
    "normal": "Normal",
    "synchronization_issue": "Clock sync problem",
    "peak_shift_temperature": "Temperature peak shift",
    "polarization_drift": "Polarization drift",
    "raman_noise": "Raman noise",
    "loss": "Extra loss",
}

TRAJECTORY_DISPLAY_NAMES = {
    "healthy_baseline": "Healthy baseline",
    "progressive": "Gets worse",
    "recovering": "Partly recovers",
    "oscillatory": "Moves up/down",
    "intermittent": "Bursty",
}

CORE_FEATURES = [
    "peak_position_ps",
    "peak_snr",
    "coincidence_rate_hz",
    "singles_rate_a_hz",
    "singles_rate_b_hz",
    "accidental_coincidence_rate_hz",
    "car",
    "zz_correlation",
]

REPRESENTATIVE_FEATURES = [
    "peak_position_ps",
    "peak_snr",
    "car",
    "coincidence_rate_hz",
    "singles_rate_a_hz",
    "singles_rate_b_hz",
    "zz_correlation",
    "link_temperature_a_c",
    "link_temperature_b_c",
]

PRETTY_NAMES = {
    "peak_position_ps": "Peak Position (ps)",
    "peak_snr": "Peak SNR",
    "coincidence_rate_hz": "Coincidence Rate (Hz)",
    "singles_rate_a_hz": "Singles Rate A (Hz)",
    "singles_rate_b_hz": "Singles Rate B (Hz)",
    "accidental_coincidence_rate_hz": "Accidental Coincidence Rate (Hz)",
    "car": "CAR",
    "zz_correlation": "ZZ Correlation",
    "link_temperature_a_c": "Link Temperature A (C)",
    "link_temperature_b_c": "Link Temperature B (C)",
    "physical_time_h": "Physical Time (h)",
    "delta_peak_position_ps": "Delta Peak Position (ps)",
    "delta_peak_snr": "Delta Peak SNR",
    "delta_car": "Delta CAR",
    "delta_coincidence_rate_hz": "Delta Coincidence Rate (Hz)",
    "delta_singles_rate_a_hz": "Delta Singles A (Hz)",
    "delta_singles_rate_b_hz": "Delta Singles B (Hz)",
    "delta_zz_correlation": "Delta ZZ Correlation",
}

CHANGE_FEATURES = [
    "delta_peak_position_ps",
    "delta_peak_snr",
    "delta_car",
    "delta_coincidence_rate_hz",
    "delta_singles_rate_a_hz",
    "delta_zz_correlation",
]

REPRESENTATIVE_REVIEW_FEATURES = [
    "peak_position_ps",
    "car",
    "coincidence_rate_hz",
    "singles_rate_a_hz",
    "zz_correlation",
]

FAULT_TRAJECTORY_FEATURE = {
    "synchronization_issue": "peak_snr",
    "peak_shift_temperature": "peak_position_ps",
    "polarization_drift": "zz_correlation",
    "raman_noise": "car",
    "loss": "coincidence_rate_hz",
}

PALETTE = {
    "normal": "#1b9e77",
    "synchronization_issue": "#d95f02",
    "peak_shift_temperature": "#7570b3",
    "polarization_drift": "#e7298a",
    "raman_noise": "#66a61e",
    "loss": "#e6ab02",
}

LABEL_ALIASES = {
    "timing_issue": "synchronization_issue",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create dataset-review plots for a generated AI diagnosis dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to a generated dataset directory containing observable_windows.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <dataset-dir>/review_artifacts.",
    )
    return parser.parse_args()


def load_dataset(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    observable = pd.read_csv(dataset_dir / "observable_windows.csv")
    sequence = pd.read_csv(dataset_dir / "sequence_labels.csv")
    validation = pd.read_csv(dataset_dir / "validation_report.csv")
    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))

    for frame in (observable, sequence, validation):
        if "label" in frame.columns:
            frame["label"] = frame["label"].replace(LABEL_ALIASES)

    sequence_columns = ["sample_id", "fault_onset_window"]
    if "trajectory_variant" in sequence.columns:
        sequence_columns.append("trajectory_variant")
    merged = observable.merge(
        sequence[sequence_columns],
        on="sample_id",
        how="left",
    )
    merged["label"] = pd.Categorical(merged["label"], categories=LABEL_ORDER, ordered=True)
    validation["label"] = pd.Categorical(validation["label"], categories=LABEL_ORDER, ordered=True)
    return merged, sequence, validation, metadata


def ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    converted = df.copy()
    for column in converted.columns:
        if column in {"sample_id", "label"}:
            continue
        try:
            converted[column] = pd.to_numeric(converted[column])
        except (TypeError, ValueError):
            continue
    return converted


def _display_label(label: str) -> str:
    return FAULT_DISPLAY_NAMES.get(str(label), str(label))


def _display_variant(variant: str) -> str:
    return TRAJECTORY_DISPLAY_NAMES.get(str(variant), str(variant))


def plot_dataset_composition(sequence: pd.DataFrame, output_dir: Path) -> None:
    """Plot sample counts by fault type and trajectory style."""

    sequence = sequence.copy()
    sequence["fault_name"] = sequence["label"].map(FAULT_DISPLAY_NAMES).fillna(sequence["label"])
    if "trajectory_variant" in sequence.columns:
        sequence["trajectory_name"] = sequence["trajectory_variant"].map(TRAJECTORY_DISPLAY_NAMES).fillna(sequence["trajectory_variant"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    counts = sequence["fault_name"].value_counts().reindex([FAULT_DISPLAY_NAMES[label] for label in LABEL_ORDER]).fillna(0)
    axes[0].barh(counts.index, counts.values, color=[PALETTE[label] for label in LABEL_ORDER])
    axes[0].set_title("How many episodes per fault type")
    axes[0].set_xlabel("Episodes")
    axes[0].grid(axis="x", alpha=0.25)

    if "trajectory_name" in sequence.columns:
        fault_rows = sequence[sequence["label"] != "normal"]
        variant_counts = pd.crosstab(fault_rows["fault_name"], fault_rows["trajectory_name"])
        variant_order = [_display_variant(v) for v in TRAJECTORY_VARIANTS if _display_variant(v) in variant_counts.columns]
        variant_counts = variant_counts.reindex([FAULT_DISPLAY_NAMES[label] for label in FAULT_LABELS]).fillna(0)
        bottom = None
        colors = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]
        for index, variant in enumerate(variant_order):
            values = variant_counts[variant].to_numpy()
            axes[1].barh(variant_counts.index, values, left=bottom, label=variant, color=colors[index % len(colors)])
            bottom = values if bottom is None else bottom + values
        axes[1].legend(fontsize=9)
    axes[1].set_title("Fault trajectory styles")
    axes[1].set_xlabel("Episodes")
    axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle("Dataset Composition", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "01_dataset_composition.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fault_change_summary(validation: pd.DataFrame, output_dir: Path) -> None:
    """Plot median pre/post changes per fault type."""

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    fault_rows = validation.copy()
    fault_rows["label"] = fault_rows["label"].astype(str)

    for ax, feature in zip(axes.flat, CHANGE_FEATURES):
        grouped = fault_rows.groupby("label", observed=False)[feature].median().reindex(LABEL_ORDER)
        labels = [_display_label(label) for label in grouped.index]
        ax.barh(labels, grouped.values, color=[PALETTE[label] for label in grouped.index])
        ax.axvline(0.0, color="#333333", linewidth=1)
        ax.set_title(PRETTY_NAMES[feature])
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle("Typical Change After Fault Starts", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "02_fault_change_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_representative_fault_examples(df: pd.DataFrame, validation: pd.DataFrame, output_dir: Path) -> None:
    """Plot one clear example episode for every fault type."""

    representatives = choose_representative_samples(validation)
    fig, axes = plt.subplots(len(LABEL_ORDER), len(REPRESENTATIVE_REVIEW_FEATURES), figsize=(22, 16), sharex="col")

    for row_index, label in enumerate(LABEL_ORDER):
        if label not in representatives:
            for col_index in range(len(REPRESENTATIVE_REVIEW_FEATURES)):
                axes[row_index, col_index].axis("off")
            continue

        sample_id = representatives[label]
        label_df = df[df["sample_id"] == sample_id].sort_values("window_index")
        onset = int(label_df["fault_onset_window"].iloc[0])
        variant = str(label_df["trajectory_variant"].iloc[0]) if "trajectory_variant" in label_df.columns else ""
        for col_index, feature in enumerate(REPRESENTATIVE_REVIEW_FEATURES):
            ax = axes[row_index, col_index]
            ax.plot(label_df["window_index"], label_df[feature], color=PALETTE[label], marker="o", linewidth=1.7, markersize=2.8)
            ax.axvline(onset, color="#444444", linestyle="--", linewidth=1)
            ax.set_title(PRETTY_NAMES[feature])
            if col_index == 0:
                ax.set_ylabel(f"{_display_label(label)}\n{_display_variant(variant)}\n{sample_id}", fontsize=9)
            if row_index == len(LABEL_ORDER) - 1:
                ax.set_xlabel("Measurement window")
            ax.grid(alpha=0.25)

    fig.suptitle("One Representative Episode Per Fault Type", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "03_representative_fault_examples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_fault_trajectory_styles(df: pd.DataFrame, validation: pd.DataFrame, output_dir: Path) -> None:
    """Plot the main signal for each fault across trajectory styles."""

    if "trajectory_variant" not in df.columns or "trajectory_variant" not in validation.columns:
        return

    fig, axes = plt.subplots(len(FAULT_LABELS), len(TRAJECTORY_VARIANTS), figsize=(18, 13), sharex=True)
    for row_index, label in enumerate(FAULT_LABELS):
        feature = FAULT_TRAJECTORY_FEATURE[label]
        for col_index, variant in enumerate(TRAJECTORY_VARIANTS):
            ax = axes[row_index, col_index]
            representatives = choose_representative_samples(validation, trajectory_variant=variant)
            sample_id = representatives.get(label)
            if sample_id is None:
                ax.axis("off")
                continue
            sample_df = df[df["sample_id"] == sample_id].sort_values("window_index")
            onset = int(sample_df["fault_onset_window"].iloc[0])
            ax.plot(sample_df["window_index"], sample_df[feature], color=PALETTE[label], marker="o", linewidth=1.7, markersize=2.5)
            ax.axvline(onset, color="#444444", linestyle="--", linewidth=1)
            if row_index == 0:
                ax.set_title(_display_variant(variant))
            if col_index == 0:
                ax.set_ylabel(f"{_display_label(label)}\n{PRETTY_NAMES[feature]}", fontsize=9)
            if row_index == len(FAULT_LABELS) - 1:
                ax.set_xlabel("Measurement window")
            ax.grid(alpha=0.25)

    fig.suptitle("Different Ways The Same Fault Can Evolve", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "04_fault_trajectory_styles.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_temperature_physical_time(df: pd.DataFrame, validation: pd.DataFrame, output_dir: Path) -> None:
    """Plot temperature-fault representatives against physical monitoring time."""

    if "physical_time_h" not in df.columns:
        return

    temp_validation = validation[validation["label"] == "peak_shift_temperature"]
    if temp_validation.empty:
        return

    features = [
        "link_temperature_a_c",
        "link_temperature_b_c",
        "peak_position_ps",
        "car",
    ]
    fig, axes = plt.subplots(len(TRAJECTORY_VARIANTS), len(features), figsize=(18, 11), sharex="col")

    for row_index, variant in enumerate(TRAJECTORY_VARIANTS):
        representatives = choose_representative_samples(validation, trajectory_variant=variant)
        sample_id = representatives.get("peak_shift_temperature")
        if sample_id is None:
            for col_index in range(len(features)):
                axes[row_index, col_index].axis("off")
            continue

        sample_df = df[df["sample_id"] == sample_id].sort_values("window_index")
        onset = int(sample_df["fault_onset_window"].iloc[0])
        onset_h = float(sample_df[sample_df["window_index"] == onset]["physical_time_h"].iloc[0])
        for col_index, feature in enumerate(features):
            ax = axes[row_index, col_index]
            ax.plot(sample_df["physical_time_h"], sample_df[feature], color=PALETTE["peak_shift_temperature"], marker="o", linewidth=1.8, markersize=3)
            ax.axvline(onset_h, color="#444444", linestyle="--", linewidth=1)
            ax.set_title(PRETTY_NAMES[feature])
            if col_index == 0:
                ax.set_ylabel(f"{_display_variant(variant)}\n{sample_id}", fontsize=9)
            if row_index == len(TRAJECTORY_VARIANTS) - 1:
                ax.set_xlabel("Physical time (hours)")
            ax.grid(alpha=0.25)

    fig.suptitle("Temperature Fault On The Physical Time Axis", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "05_temperature_physical_time.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_label_trajectories(df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(15, 15), sharex=True)
    grouped = (
        df.groupby(["label", "window_index"], observed=False)[CORE_FEATURES]
        .agg(["median", "min", "max"])
        .reset_index()
    )

    for ax, feature in zip(axes.flat, CORE_FEATURES):
        for label in LABEL_ORDER:
            label_rows = grouped[grouped["label"] == label]
            if label_rows.empty:
                continue
            x = label_rows["window_index"]
            median = label_rows[(feature, "median")]
            ymin = label_rows[(feature, "min")]
            ymax = label_rows[(feature, "max")]
            ax.plot(x, median, label=label, color=PALETTE[label], linewidth=2)
            ax.fill_between(x, ymin, ymax, color=PALETTE[label], alpha=0.12)
        ax.set_title(PRETTY_NAMES[feature])
        ax.set_xlabel("Window Index")
        ax.grid(alpha=0.25)

    axes[0, 0].legend(loc="best", fontsize=9)
    for ax in axes.flat[len(CORE_FEATURES):]:
        ax.axis("off")
    fig.suptitle("Core Observable Trajectories by Fault Type", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "01_label_trajectories.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def choose_representative_samples(validation: pd.DataFrame, trajectory_variant: str | None = None) -> dict[str, str]:
    representatives: dict[str, str] = {}
    for label in LABEL_ORDER:
        rows = validation[validation["label"] == label].copy()
        if trajectory_variant is not None:
            rows = rows[rows.get("trajectory_variant", "") == trajectory_variant]
        if rows.empty:
            continue
        if label == "normal":
            score = rows["delta_peak_position_ps"].abs().fillna(0.0) + rows["delta_zz_correlation"].abs().fillna(0.0) * 1000.0
            chosen = rows.iloc[int(score.argmin())]
        elif label == "synchronization_issue":
            chosen = rows.iloc[int(rows["delta_peak_snr"].fillna(math.inf).argmin())]
        elif label == "peak_shift_temperature":
            chosen = rows.iloc[int(rows["delta_peak_position_ps"].abs().fillna(-math.inf).argmax())]
        elif label == "polarization_drift":
            chosen = rows.iloc[int(rows["delta_zz_correlation"].fillna(math.inf).argmin())]
        elif label == "raman_noise":
            chosen = rows.iloc[int(rows["delta_singles_rate_a_hz"].fillna(-math.inf).argmax())]
        else:
            chosen = rows.iloc[int(rows["delta_coincidence_rate_hz"].fillna(math.inf).argmin())]
        representatives[label] = str(chosen["sample_id"])
    return representatives


def plot_representative_sequences(df: pd.DataFrame, validation: pd.DataFrame, output_dir: Path) -> None:
    representatives = choose_representative_samples(validation)
    fig, axes = plt.subplots(len(LABEL_ORDER), len(REPRESENTATIVE_FEATURES), figsize=(26, 20), sharex="col")

    for row_index, label in enumerate(LABEL_ORDER):
        if label not in representatives:
            for col_index in range(len(REPRESENTATIVE_FEATURES)):
                ax = axes[row_index, col_index]
                ax.axis("off")
            continue
        label_df = df[df["sample_id"] == representatives[label]].sort_values("window_index")
        onset = int(label_df["fault_onset_window"].iloc[0])
        for col_index, feature in enumerate(REPRESENTATIVE_FEATURES):
            ax = axes[row_index, col_index]
            ax.plot(label_df["window_index"], label_df[feature], color=PALETTE[label], marker="o", linewidth=1.8, markersize=3)
            ax.axvline(onset, color="#444444", linestyle="--", linewidth=1)
            ax.set_title(PRETTY_NAMES[feature])
            if col_index == 0:
                ax.set_ylabel(label)
            if row_index == len(LABEL_ORDER) - 1:
                ax.set_xlabel("Window Index")
            ax.grid(alpha=0.25)

    fig.suptitle("Representative Episode Per Label", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "02_representative_sequences.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_variant_trajectories(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot aggregate trajectories separately for each trajectory family."""

    if "trajectory_variant" not in df.columns:
        return

    for variant in TRAJECTORY_VARIANTS:
        variant_df = df[df["trajectory_variant"] == variant].copy()
        if variant_df.empty:
            continue

        fig, axes = plt.subplots(4, 2, figsize=(15, 15), sharex=True)
        grouped = (
            variant_df.groupby(["label", "window_index"], observed=False)[CORE_FEATURES]
            .agg(["median", "min", "max"])
            .reset_index()
        )

        for ax, feature in zip(axes.flat, CORE_FEATURES):
            for label in FAULT_LABELS:
                label_rows = grouped[grouped["label"] == label]
                if label_rows.empty:
                    continue
                x = label_rows["window_index"]
                median = label_rows[(feature, "median")]
                ymin = label_rows[(feature, "min")]
                ymax = label_rows[(feature, "max")]
                ax.plot(x, median, label=label, color=PALETTE[label], linewidth=2)
                ax.fill_between(x, ymin, ymax, color=PALETTE[label], alpha=0.12)
            ax.set_title(PRETTY_NAMES[feature])
            ax.set_xlabel("Window Index")
            ax.grid(alpha=0.25)

        axes[0, 0].legend(loc="best", fontsize=9)
        for ax in axes.flat[len(CORE_FEATURES):]:
            ax.axis("off")
        fig.suptitle(f"Core Observable Trajectories - {variant}", fontsize=15)
        fig.tight_layout()
        fig.savefig(output_dir / f"03_variant_trajectories_{variant}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def plot_representative_sequences_by_variant(df: pd.DataFrame, validation: pd.DataFrame, output_dir: Path) -> None:
    """Plot one representative episode per label for each trajectory family."""

    if "trajectory_variant" not in df.columns or "trajectory_variant" not in validation.columns:
        return

    for variant in TRAJECTORY_VARIANTS:
        representatives = choose_representative_samples(validation, trajectory_variant=variant)
        if not representatives:
            continue

        fig, axes = plt.subplots(len(FAULT_LABELS), len(REPRESENTATIVE_FEATURES), figsize=(30, 17), sharex="col")

        for row_index, label in enumerate(FAULT_LABELS):
            if label not in representatives:
                for col_index in range(len(REPRESENTATIVE_FEATURES)):
                    axes[row_index, col_index].axis("off")
                continue

            sample_id = representatives[label]
            label_df = df[df["sample_id"] == sample_id].sort_values("window_index")
            onset = int(label_df["fault_onset_window"].iloc[0])
            row_label = f"{label}\n{sample_id}"

            for col_index, feature in enumerate(REPRESENTATIVE_FEATURES):
                ax = axes[row_index, col_index]
                ax.plot(label_df["window_index"], label_df[feature], color=PALETTE[label], marker="o", linewidth=1.8, markersize=3)
                ax.axvline(onset, color="#444444", linestyle="--", linewidth=1)
                ax.set_title(PRETTY_NAMES[feature])
                if col_index == 0:
                    ax.set_ylabel(row_label)
                if row_index == len(FAULT_LABELS) - 1:
                    ax.set_xlabel("Window Index")
                ax.grid(alpha=0.25)

        fig.suptitle(f"Representative Episodes by Label - {variant}", fontsize=15)
        fig.tight_layout()
        fig.savefig(output_dir / f"04_representative_sequences_{variant}.png", dpi=180, bbox_inches="tight")
        plt.close(fig)


def write_summary(df: pd.DataFrame, validation: pd.DataFrame, metadata: dict, output_dir: Path) -> None:
    by_label = validation.groupby("label", observed=False).mean(numeric_only=True).reindex(LABEL_ORDER)
    lines = [
        "# Dataset Review Summary",
        "",
        f"- Total sequences: `{metadata['total_sequences']}`",
        f"- Total windows: `{metadata['total_windows']}`",
        f"- Episode duration: `{metadata['episode_duration_s']} s`",
        f"- Window duration: `{metadata['window_duration_s']} s`",
        f"- Temperature physical span: `{metadata.get('temperature_episode_span_s', 'not recorded')} s`",
        f"- Visibility stride: every `{metadata['visibility_measurement_stride_windows']}` windows",
        "",
        "## Plot Set",
        "",
        "- `01_dataset_composition.png`: number of generated episodes by fault type and trajectory style.",
        "- `02_fault_change_summary.png`: median before/after change after the fault starts.",
        "- `03_representative_fault_examples.png`: one clear example episode per fault type.",
        "- `04_fault_trajectory_styles.png`: the main signal for each fault across different trajectory styles.",
        "- `05_temperature_physical_time.png`: temperature fault examples plotted against physical monitoring time.",
        "",
        "## Label-Level Checks",
        "",
    ]

    for label in LABEL_ORDER:
        row = by_label.loc[label]
        lines.extend(
            [
                f"### `{label}`",
                f"- Avg delta peak position: `{row.get('delta_peak_position_ps', float('nan')):.2f} ps`",
                f"- Avg delta coincidence rate: `{row.get('delta_coincidence_rate_hz', float('nan')):.2f} Hz`",
                f"- Avg delta singles A: `{row.get('delta_singles_rate_a_hz', float('nan')):.2f} Hz`",
                f"- Avg delta singles B: `{row.get('delta_singles_rate_b_hz', float('nan')):.2f} Hz`",
                f"- Avg delta ZZ correlation: `{row.get('delta_zz_correlation', float('nan')):.4f}`",
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            "",
            "- `peak_position_ps` should stay near zero for normal, loss, Raman-noise, and most polarization-drift cases, and drift strongly for `peak_shift_temperature` and severe synchronization faults.",
            "- `peak_snr` and `car` should collapse for `synchronization_issue` because the coincidence peak becomes poorly defined.",
            "- `zz_correlation` should remain relatively stable for synchronization and loss faults, and degrade for polarization drift.",
            "- `accidental_coincidence_rate_hz` and `car` are expected to be most informative for Raman-noise episodes.",
            "",
        ]
    )

    (output_dir / "dataset_review_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = (args.output_dir or (dataset_dir / "review_artifacts")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    observable, _sequence, validation, metadata = load_dataset(dataset_dir)
    observable = ensure_numeric(observable)

    plot_dataset_composition(_sequence, output_dir)
    plot_fault_change_summary(validation, output_dir)
    plot_representative_fault_examples(observable, validation, output_dir)
    plot_fault_trajectory_styles(observable, validation, output_dir)
    plot_temperature_physical_time(observable, validation, output_dir)
    write_summary(observable, validation, metadata, output_dir)

    print(output_dir)


if __name__ == "__main__":
    main()
