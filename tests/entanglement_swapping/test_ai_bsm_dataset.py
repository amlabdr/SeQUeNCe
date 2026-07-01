from dataclasses import replace

from example.entanglement_swapping_validation.ai_bsm_dataset.config import (
    DatasetConfig,
    FaultConfig,
    FeatureConfig,
    GenerationConfig,
    SetupConfig,
    SlurmConfig,
)
from example.entanglement_swapping_validation.ai_bsm_dataset.features import (
    observable_features,
)
from example.entanglement_swapping_validation.ai_bsm_dataset.scenarios import (
    build_scenario,
)
from example.entanglement_swapping_validation.ai_bsm_dataset.simulator import (
    simulate_episode,
)


def small_config():
    return DatasetConfig(
        generation=GenerationConfig(
            dataset_size=1,
            episode_duration_s=0.004,
            window_duration_s=0.001,
            fault_update_interval_s=0.001,
            onset_min_fraction=0.2,
            onset_max_fraction=0.3,
            ramp_duration_min_s=0.001,
            ramp_duration_max_s=0.001,
        ),
        setup=SetupConfig(
            source_frequency_hz=10_000,
            mean_photon_num=0.1,
            signal_length_m_a=1,
            signal_length_m_b=1,
            attenuation_db_per_m=0,
            baseline_raman_power_mw=0,
            detector_efficiency=1,
            detector_dark_hz=0,
            detector_jitter_ps=0,
            emission_chunk_pulses=5,
        ),
        features=FeatureConfig(),
        faults=FaultConfig(),
        slurm=SlurmConfig(),
    )


def test_four_detector_pattern_features():
    streams = {
        "D3H": [100, 1000],
        "D3V": [2000],
        "D4H": [2010],
        "D4V": [105],
    }
    features, _ = observable_features(
        streams,
        duration_s=1,
        coincidence_window_ps=20,
        histogram_range_ps=100,
        histogram_bin_width_ps=10,
    )
    assert features["psi_minus_count"] == 2
    assert features["psi_plus_count"] == 0
    assert features["accepted_bsm_count"] == 2


def test_scenario_onset_is_inside_episode_and_ramped():
    cfg = small_config()
    scenario = build_scenario("temperature_change", 0, cfg, 123)
    assert 0 < scenario.onset_s < cfg.generation.episode_duration_s
    assert scenario.progress(scenario.onset_s - 1e-9) == 0
    assert scenario.progress(scenario.onset_s + scenario.ramp_duration_s) == 1


def test_continuous_episode_returns_all_windows_without_raw_streams():
    cfg = small_config()
    scenario = build_scenario("healthy", 0, cfg, 123)
    rows, truth, raw = simulate_episode(scenario, cfg)
    assert len(rows) == cfg.generation.windows_per_episode
    assert len(truth) == cfg.generation.windows_per_episode
    assert [row["window_index"] for row in rows] == [0, 1, 2, 3]
    assert raw is None
    assert rows[0]["delta_d3h_rate_hz"] is None
