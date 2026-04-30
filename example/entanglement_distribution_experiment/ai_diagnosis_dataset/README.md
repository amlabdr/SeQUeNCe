# AI Diagnosis Monitoring-Episode Dataset

This module generates a monitoring-episode dataset for diagnosing link impairments from receiver-visible measurements.

The design is intentionally sequence-based:

- one sample = one monitoring episode,
- one row = one monitoring window,
- one label = the impairment class for the episode,
- one `fault_onset_window` = when degradation begins for non-normal episodes.

## Design Choices

The exported AI-facing data includes only observables that a diagnosis agent could plausibly read from receiver nodes or attached operational telemetry.

Hidden simulator truth is stored separately for:

- reproducibility,
- debugging,
- validation.

The two arms are independent:

- arm A and arm B can have different lengths,
- different attenuation,
- different detector properties,
- different coexistence powers,
- different evolving hidden states.

## Episode Structure

Each episode contains:

- a configurable total duration,
- fixed-duration monitoring windows,
- a normal prefix before the fault onset,
- a degradation phase after the onset,
- passive monitoring windows at every step,
- sparse active diagnostics at a configurable stride.

Active diagnostics are not run every window. This is closer to an operational monitoring setting where passive data is frequent and active checks such as visibility scans are occasional.

The passive part of each window stays in the `Z` basis, so `HH/HV/VH/VV` and `zz_correlation` are available every window. `X`-basis statistics and visibility are only populated on sparse active-diagnostic windows.

## Labels

- `normal`
- `synchronization_issue`
- `polarization_drift`
- `raman_noise`
- `loss`
- `peak_shift_temperature`

## AI-Visible Window Features

The main training table is `observable_windows.csv`.

Columns:

- `sample_id`
- `label`
- `window_index`
- `timestamp_step`
- `physical_time_s`
- `physical_time_h`
- `physical_window_start_s`
- `physical_window_end_s`
- `physical_window_spacing_s`
- `acquisition_window_duration_s`
- `window_start_s`
- `window_end_s`
- `fault_active`
- `arm_length_a_m`
- `arm_length_b_m`
- `peak_position_ps`
- `peak_width_ps`
- `peak_height`
- `peak_snr`
- `accidental_coincidence_rate_hz`
- `car`
- `coincidence_rate_hz`
- `coincidences_count`
- `singles_rate_a_hz`
- `singles_rate_b_hz`
- `coincidence_to_singles_a`
- `coincidence_to_singles_b`
- `hh_rate_z_hz`
- `hv_rate_z_hz`
- `vh_rate_z_hz`
- `vv_rate_z_hz`
- `zz_correlation`
- `pp_rate_x_hz`
- `pm_rate_x_hz`
- `mp_rate_x_hz`
- `mm_rate_x_hz`
- `xx_correlation`
- `xx_measured`
- `link_temperature_a_c`
- `link_temperature_b_c`
- `visibility`
- `visibility_measured`
- `delta_peak_position_ps`
- `delta_coincidence_rate_hz`
- `delta_singles_rate_a_hz`
- `delta_singles_rate_b_hz`
- `delta_zz_correlation`
- `rolling_peak_position_ps_mean_3`
- `rolling_coincidence_rate_hz_mean_3`
- `rolling_singles_rate_a_hz_mean_3`
- `rolling_singles_rate_b_hz_mean_3`
- `rolling_zz_correlation_mean_3`

`visibility` is only populated on active-diagnostic windows. On passive-only windows it is left empty.

`window_start_s` and `window_end_s` are the simulated acquisition timeline used by SeQUeNCe. `physical_time_s` and `physical_time_h` are the monitoring timestamps exposed to the diagnosis task. For `peak_shift_temperature`, the monitoring timestamps can span 24 hours while each row still simulates only one short acquisition window.

`pp/pm/mp/mm` and `xx_correlation` are also only populated on active-diagnostic windows. On passive-only windows they are left empty and `xx_measured` is `0`.

`accidental_coincidence_rate_hz` is estimated from histogram sidebands outside the main coincidence peak.

`car` is the coincidence-to-accidental ratio estimated from the same histogram.

## Hidden Metadata

The following are kept out of the AI input table and saved in raw metadata:

- arm lengths,
- attenuation schedules,
- coexistence powers,
- birefringence parameters,
- seeds.

Per-link temperature is currently exported in the AI-facing table because it is treated as measurable telemetry for the diagnosis agent. Classical coexistence power remains in hidden metadata and raw scenario files, but is not exposed as an AI-input feature.

## Scenario Behavior

### `normal`

Stable monitoring with small random variation.

### `synchronization_issue`

Normal prefix followed by receiver-clock desynchronization that degrades or destroys the coincidence peak while leaving singles available. After onset, the desynchronization wanders in a degraded regime instead of following a purely monotonic ramp.

### `polarization_drift`

Normal prefix followed by analyzer mismatch and birefringence-related degradation that drifts within a degraded regime instead of only increasing in one direction.

### `raman_noise`

Normal prefix followed by classical coexistence power entering a degraded regime with continued fluctuation, raising background and accidental coincidence levels.

### `loss`

Normal prefix followed by attenuation entering a degraded regime with continued fluctuation instead of only increasing monotonically.

### `peak_shift_temperature`

Normal prefix followed by one arm cooling significantly, with the peak shifting in response to the changing link temperature. The temperature trajectory is allowed to drift and partially recover instead of remaining strictly monotonic.

Temperature episodes use `--temperature-episode-span-s` to separate physical monitoring time from simulation runtime. For example, `--episode-duration-s 60 --window-duration-s 1 --temperature-episode-span-s 86400` generates 60 one-second acquisition windows whose physical timestamps cover 24 hours.

## Output Files

Each run writes:

- `observable_windows.csv`
- `observable_windows.json`
- `sequence_labels.csv`
- `validation_report.csv`
- `metadata.json`
- `scenarios.json`
- `raw_index.json`
- `raw_sequences/`
- `checkpoints/`

`sequence_labels.csv` provides the per-episode label and `fault_onset_window`.

`validation_report.csv` summarizes pre-fault versus post-fault changes for each generated episode so the sample dataset can be sanity-checked before larger runs.

`raw_sequences/<sample_id>.json` contains:

- the hidden scenario definition,
- per-window hidden state,
- exported observable row,
- histogram arrays,
- visibility traces on active windows.

If `--skip-raw` is used, `raw_sequences/` is not created and `raw_index.json` is written as an empty list.

`checkpoints/<sample_id>.json` is written immediately after each episode finishes. If a run is interrupted, rerun the same command with the same `--output-dir`; existing checkpoints are detected automatically and only missing episodes are simulated. Use `--no-resume` only when you intentionally want to ignore existing checkpoints and regenerate everything.

## How To Run

Sample dataset:

```powershell
python example/entanglement_distribution_experiment/ai_diagnosis_dataset/generate_dataset.py --mode sample
```

To use multiple CPU cores during generation:

```powershell
python example/entanglement_distribution_experiment/ai_diagnosis_dataset/generate_dataset.py `
  --mode sample `
  --workers 19
```

To reduce production overhead:

```powershell
python example/entanglement_distribution_experiment/ai_diagnosis_dataset/generate_dataset.py `
  --mode full `
  --workers 19 `
  --parallel-backend thread `
  --skip-raw `
  --validation-off `
  --temperature-episode-span-s 86400
```

Larger dataset later:

```powershell
python example/entanglement_distribution_experiment/ai_diagnosis_dataset/generate_dataset.py --mode full
```

Training-sized dataset with 24-hour physical temperature episodes:

```powershell
python example/entanglement_distribution_experiment/ai_diagnosis_dataset/generate_dataset.py --mode full --samples-per-label 500 --episode-duration-s 60 --window-duration-s 1 --source-frequency-hz 100000 --visibility-stride-windows 30 --temperature-episode-span-s 86400 --base-seed 300000 --workers 18 --parallel-backend thread --skip-raw --validation-off --output-dir example/entanglement_distribution_experiment/ai_diagnosis_dataset/generated/train_v5_60s_100khz_24h_temperature
```

On Linux/HPC, prefer `--parallel-backend process` if memory is sufficient.

Dataset review plots:

```powershell
python example/entanglement_distribution_experiment/ai_diagnosis_dataset/review_dataset.py `
  --dataset-dir example/entanglement_distribution_experiment/ai_diagnosis_dataset/generated/train_v5_60s_100khz_24h_temperature
```

For an interactive review, open [dataset_review.ipynb](/c:/Users/ana35/OneDrive%20-%20NIST/Desktop/cooding/SeQUeNCe/example/entanglement_distribution_experiment/ai_diagnosis_dataset/dataset_review.ipynb). The expected plot set and interpretation guidance are documented in [PLOT_GUIDE.md](/c:/Users/ana35/OneDrive%20-%20NIST/Desktop/cooding/SeQUeNCe/example/entanglement_distribution_experiment/ai_diagnosis_dataset/PLOT_GUIDE.md).

## Presets

- `sample`: small validation set
- `full`: larger training-oriented set

The workflow should remain sample-first.

The sample preset currently uses:

- `20 s` total episode duration
- `1 s` window duration
- visibility measured every `10` windows

These are preset defaults and can be changed later in the generator configuration.
