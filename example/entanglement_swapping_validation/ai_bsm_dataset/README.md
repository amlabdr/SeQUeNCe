# Polarization-BSM Fault-Diagnosis Dataset

This package generates a supervised fault-diagnosis dataset for a
polarization-resolved Bell-state measurement (BSM) used in an entanglement
swapping setup.

The operational AI input is restricted to telemetry available at the central
BSM:

- `D3H`: spatial output 3, horizontal polarization
- `D3V`: spatial output 3, vertical polarization
- `D4H`: spatial output 4, horizontal polarization
- `D4V`: spatial output 4, vertical polarization

No Alice/David detection, herald/fourfold information, source pair IDs,
simulator overlap values, or injected fault parameters are used as model
inputs.

## Physical Setup

Two independent `SpdcSourceNode` sources emit polarization-entangled pairs.
One photon from each source travels through a 20 km `fiberQuantumChannel` to
`PolarizationBSMNode`. The BSM contains:

1. A non-polarizing 50:50 beam splitter
2. One H/V PBS on each beam-splitter output
3. Four threshold detectors: `D3H`, `D3V`, `D4H`, and `D4V`

The unused outer photons are routed directly to passive `PhotonSinkNode`
instances. They do not pass through outer fibers, are not measured, and do not
produce timestamps or operational data.

Recognizable detector patterns are:

| Detector pair | BSM classification |
|---|---|
| `D3H-D4V` | `psi_minus` |
| `D3V-D4H` | `psi_minus` |
| `D3H-D3V` | `psi_plus` |
| `D4H-D4V` | `psi_plus` |
| `D3H-D4H` | same-polarization diagnostic, inconclusive |
| `D3V-D4V` | same-polarization diagnostic, inconclusive |

The apparatus does not claim to distinguish `phi_plus` from `phi_minus` with
this passive linear-optical threshold-detector arrangement.

## Reference Setup

The generated `polarization_bsm_60s` dataset uses:

| Parameter | Value |
|---|---:|
| SPDC sources | 2 |
| Source repetition rate | 100 MHz |
| Mean pair number per pulse | 0.01 |
| Photon-pair statistics | Thermal |
| Effective post-filter bandwidth | 0.1 nm FWHM |
| Signal wavelength | 1550 nm |
| Signal-fiber length per source | 20 km |
| Fiber attenuation | 0.2 dB/km |
| Baseline Raman coexistence power | 0.02 mW |
| BSM detector efficiency | 95% |
| Detector timing jitter | 20 ps |
| Detector dark-count rate | 50 Hz |
| Internal photon matching window | 1200 ps |
| Observable coincidence window | 120 ps |
| PBS fidelity | 1.0 |

`source_bandwidth_nm` is interpreted as the effective photon bandwidth after
filtering, not the full raw SPDC phase-matching bandwidth.

## Episode Semantics

One episode is one continuous 60-second SeQUeNCe timeline:

- Both sources emit continuously throughout the episode.
- Fiber, source, BSM, and detector state persist across the episode.
- BSM telemetry is divided into 60 consecutive one-second feature windows.
- A one-second boundary does not restart the source or reconstruct the network.
- Completed BSM timestamp and interaction logs are drained after each window.
- Pending photons at the BSM are preserved across window boundaries.
- Raw timestamps are discarded after feature extraction unless explicitly
  enabled.

The dataset contains:

```text
10,000 episodes
60 windows per episode
600,000 observable rows
```

Episodes are independently reproducible from deterministic seeds and can run
in parallel. Windows within an episode remain sequential.

## Fault Evolution

Healthy episodes retain only baseline attenuation, Raman noise, detector
jitter, dark counts, and normal stochastic source behavior.

For a faulty 60-second episode:

1. Fault onset is sampled between 15% and 65% of the episode, or 9–39 seconds.
2. Ramp duration is sampled between 1 and 10 seconds.
3. The physical fault is updated every 0.1 simulated seconds.
4. The fault remains at its target after the ramp.

The onset may occur inside a one-second feature window.
`fault_active_fraction` records how much of that window occurs after onset.

## Fault Classes

- `healthy`: baseline setup without an injected fault.
- `temperature_change`: applies a signed temperature change to one signal
  fiber, affecting propagation delay, dispersion, and polarization evolution.
- `polarization_drift`: applies bend and twist evolution to one signal fiber.
- `spectral_detuning`: shifts the signal center wavelength of one source.
- `raman_noise`: increases classical coexistence/Raman power on one signal
  fiber above the common baseline.
- `attenuation_loss`: increases attenuation on one signal fiber.
- `source_brightness_loss`: reduces the mean pair number of one source.
- `sync_issue`: adds timestamp jitter to one randomly selected BSM detector
  stream. It is not supplied as a known correction.
- `source_clock_drift`: creates a growing relative source timing offset at the
  BSM.

The affected source, arm, or detector is stored in `scenarios.json` for
offline analysis. It is not an operational model input.

Classes are assigned deterministically in round-robin order. For 10,000
episodes, `healthy` has 1,112 episodes and each other class has 1,111.

## Output Files

### `bsm_observable_windows.csv`

The main 600,000-row table. Each row represents one one-second BSM monitoring
window.

#### Bookkeeping and supervised-label columns

```text
episode_id
setup_id
fault_class
window_index
window_start_s
window_end_s
window_duration_s
fault_active
fault_active_fraction
```

`fault_class`, `fault_active`, and `fault_active_fraction` are labels, not
model inputs. Split train/validation/test sets by `episode_id`, never by
individual rows, to prevent windows from one episode leaking across splits.

#### Detector singles

For every detector `d3h`, `d3v`, `d4h`, and `d4v`:

```text
<detector>_count
<detector>_rate_hz
<detector>_interarrival_mean_ps
<detector>_interarrival_std_ps
<detector>_interarrival_min_ps
<detector>_interarrival_max_ps
```

#### Aggregate singles and asymmetries

```text
total_singles_rate_hz
port3_rate_hz
port4_rate_hz
horizontal_rate_hz
vertical_rate_hz
port_asymmetry
polarization_asymmetry
```

The normalized asymmetries are:

```text
port_asymmetry = (port3 - port4) / (port3 + port4)
polarization_asymmetry = (H - V) / (H + V)
```

#### Six detector-pair feature groups

Features are computed for:

```text
d3h_d3v
d3h_d4h
d3h_d4v
d3v_d4h
d3v_d4v
d4h_d4v
```

Each pair contributes:

```text
<pair>_count
<pair>_rate_hz
<pair>_peak_position_ps
<pair>_peak_width_ps
<pair>_peak_height
<pair>_peak_snr
<pair>_accidental_rate_hz
<pair>_car
```

The lag histogram is formed from timestamp differences within the configured
`histogram_range_ps`. Coincidences are integrated inside
`coincidence_window_ps`. `car` is the coincidence-to-accidental ratio
estimated from histogram side bins.

#### BSM-pattern summaries

```text
psi_minus_count
psi_minus_rate_hz
psi_plus_count
psi_plus_rate_hz
accepted_bsm_count
accepted_bsm_rate_hz
psi_minus_fraction
psi_plus_fraction
psi_balance
same_pol_cross_port_rate_hz
```

`accepted_bsm_count` includes only recognizable `psi_minus` and `psi_plus`
two-detector patterns.

#### Window-to-window differences

```text
delta_d3h_rate_hz
delta_d3v_rate_hz
delta_d4h_rate_hz
delta_d4v_rate_hz
delta_accepted_bsm_rate_hz
delta_psi_minus_rate_hz
delta_psi_plus_rate_hz
delta_port_asymmetry
delta_polarization_asymmetry
delta_same_pol_cross_port_rate_hz
```

The first window of every episode has `NaN` deltas because no previous window
exists.

### `episode_labels.csv`

One row per episode:

```text
episode_id
fault_class
seed
fault_onset_s
fault_ramp_duration_s
target_fault_level
```

### `hidden_truth.csv`

Per-window simulator truth for validation only:

```text
episode_id
window_index
fault_class
fault_onset_s
fault_ramp_duration_s
target_fault_level
fault_progress
true_temperature_c
true_twist
true_bend_radius
true_detuning_nm
true_raman_extra_mw
true_extra_loss
true_brightness_factor
true_sync_jitter_ps
true_clock_offset_ps
```

Do not merge these columns into the operational feature matrix.

### Other files

- `setup_params.csv`: exact physical setup used for generation.
- `scenarios.json`: complete episode definitions, including affected
  source/arm/detector.
- `metadata.json`: dataset dimensions, configuration fingerprint, and label
  list.
- `checkpoints/*.json`: one atomic resumable checkpoint per completed episode.
- `submission.json`: most recent Slurm array/combine job IDs.

The final scientific dataset does not include Slurm `.out` or `.err` logs.

## Memory and Resume Behavior

Raw streams are disabled by default:

```ini
write_raw_streams = false
```

For each completed feature window, timestamp-derived features are retained and
the consumed BSM click/interaction buffers are cleared. Passive sinks retain
no photon objects or timestamps. This keeps memory approximately bounded by
one active window plus in-flight events.

Checkpoints include a physics-configuration fingerprint. `resume=true` reuses
only checkpoints generated with the same scientific configuration. The final
combine stage streams checkpoints one at a time instead of loading the entire
dataset into RAM.

## Validation and Plotting

After collecting the dataset, use:

- `bsm_dataset_validation.ipynb`: structural checks, window counts, feature
  availability, and hidden-truth response checks.
- `bsm_dataset_plots.ipynb`: class balance, BSM feature distributions, and
  continuous episode traces.

The notebooks load generated CSV files only; they do not run simulations.

## Run

Run commands from the repository root.

### Local

```powershell
python example/entanglement_swapping_validation/ai_bsm_dataset/generate_bsm_dataset.py `
  --config example/entanglement_swapping_validation/ai_bsm_dataset/configs/bsm_dataset_config.ini `
  --target local
```

### Blackbird Smoke Test

```powershell
python example/entanglement_swapping_validation/ai_bsm_dataset/generate_bsm_dataset.py `
  --config example/entanglement_swapping_validation/ai_bsm_dataset/configs/bsm_dataset_smoke_blackbird.ini `
  --target blackbird --submit --wait
```

### Full Blackbird Dataset

```powershell
python example/entanglement_swapping_validation/ai_bsm_dataset/generate_bsm_dataset.py `
  --config example/entanglement_swapping_validation/ai_bsm_dataset/configs/bsm_dataset_config.ini `
  --target blackbird --submit
```

### Check Status

```powershell
python example/entanglement_swapping_validation/ai_bsm_dataset/generate_bsm_dataset.py `
  --config example/entanglement_swapping_validation/ai_bsm_dataset/configs/bsm_dataset_config.ini `
  --status
```

If `submission.json` is unavailable:

```powershell
python example/entanglement_swapping_validation/ai_bsm_dataset/generate_bsm_dataset.py `
  --config example/entanglement_swapping_validation/ai_bsm_dataset/configs/bsm_dataset_config.ini `
  --status --array-job-id ARRAY_JOB_ID --combine-job-id COMBINE_JOB_ID
```

### Collect Results

After the combine job is `COMPLETED`:

```powershell
python example/entanglement_swapping_validation/ai_bsm_dataset/generate_bsm_dataset.py `
  --config example/entanglement_swapping_validation/ai_bsm_dataset/configs/bsm_dataset_config.ini `
  --collect-existing
```

Final files are downloaded to:

```text
example/entanglement_swapping_validation/ai_bsm_dataset/generated/<dataset_name>/
```

If generation is interrupted, submit the same configuration again. Valid
completed episode checkpoints are preserved and missing episodes are
regenerated.
