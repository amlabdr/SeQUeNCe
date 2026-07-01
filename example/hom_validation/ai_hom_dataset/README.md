# HOM-Only AI Diagnosis Dataset

This package generates HOM/HOM monitoring datasets for fault diagnosis. The
model-eligible physical features are derived only from the two HOM detector
timestamp streams. The same table also carries IDs and supervised labels such
as `fault_class` and `fault_active`; these are not model input features.

The idlers are sent to passive sinks: they are not projected or detected. The
simulator uses `herald_mode="hom_only"`, so idler detections, heralded
fourfolds, pair IDs, and simulator overlap truth are not operational inputs.

## Current Dataset Configuration

The default INI file is
`configs/hom_ai_dataset_config.ini`. It currently defines:

- dataset name: `hom_ai_dataset_60s_10k_blackbird`
- total dataset size: 10,000 episodes
- episode duration: 60 simulated seconds
- feature-window duration: 1 simulated second
- 60 HOM feature rows per episode
- random fault label per episode from the enabled label set
- deterministic episode/window seeds derived from `base_seed`
- raw HOM streams disabled
- episode-level resume checkpoints enabled
- Blackbird as the Slurm target

Labels are sampled independently and deterministically; the resulting class
counts are approximately, but not necessarily exactly, balanced. Supplying
`--labels` restricts the set from which labels are sampled.

## Episode Semantics

An episode is a shared hidden fault scenario represented by a sequence of
feature windows. The current implementation runs a separate SeQUeNCe HOM
simulation for each feature window, using the scenario state and deterministic
seed assigned to that window. It is therefore not one persistent 60-second
SeQUeNCe `Timeline` object.

For non-healthy episodes, the onset window is sampled uniformly from all
episode windows, including the first and last window. The fault is absent
before onset and switches directly to its sampled target level at onset; no
ramp or rolling fault model is currently used. Healthy episodes never activate
a fault.

## Fault Classes

- `healthy`: baseline attenuation, detector jitter, dark counts, and weak
  Raman/background noise only.
- `temperature_change`: applies a signed temperature shift to one randomly
  selected signal fiber. No timing correction is supplied to the HOM.
- `polarization_drift`: applies both fiber twist and bend to one randomly
  selected signal arm.
- `spectral_detuning`: shifts the signal center wavelength of one randomly
  selected source.
- `raman_noise`: adds classical coexistence power to one randomly selected
  signal arm, above the baseline Raman power present on both arms.
- `attenuation_loss`: adds attenuation to one randomly selected signal arm.
- `source_brightness_loss`: reduces the pair-generation brightness of one
  randomly selected source.
- `sync_issue`: adds timestamp jitter to one randomly selected HOM detector
  stream after detection. It is not treated as a known coincidence offset.
- `source_clock_drift`: creates a signed relative source-timing walk-off before
  the HOM. The offset grows with the simulated time elapsed since onset.

The current extra Raman fault range is `0.02` to `0.50` mW above the
`0.02` mW baseline. Baseline detector timing uncertainty is controlled by
`detector_jitter_ps`; there is no separate baseline synchronization-jitter
parameter.

## Output Files

The final combine step writes:

- `hom_observable_windows.csv`: AI input table, one row per episode window.
- `episode_labels.csv`: one row per episode with class, seed, onset, target,
  setup ID, and window count.
- `hidden_truth.csv`: per-window simulator truth for offline validation only.
- `setup_params.csv`: fixed reference setup parameters.
- `validation_summary.csv`: per-episode pre/post-onset feature summary.
- `metadata.json`: dataset configuration and output metadata.
- `scenarios.json`: complete hidden scenario definitions.
- `raw_index.json`: currently an empty compatibility index.
- `checkpoints/*.json`: one atomic checkpoint per completed episode.

`write_raw_streams=false` is the normal mode. If enabled, raw HOM timestamps
and histograms are embedded in episode checkpoint payloads; the current final
combine step does not export a separate consolidated raw-stream dataset.

The files that may reveal injected truth (`hidden_truth.csv`,
`episode_labels.csv`, `validation_summary.csv`, and `scenarios.json`) must not
be used as operational AI inputs.

## AI-Visible Features

`hom_observable_windows.csv` contains:

- HOM1 and HOM2 singles counts and rates
- summed singles rate
- HOM twofold count and rate
- lag-histogram peak position, FWHM width, height, and SNR
- histogram coincidence count
- estimated accidental count and rate
- coincidence-to-accidental ratio (`car`)
- HOM1 and HOM2 interarrival mean, standard deviation, minimum, and maximum
- window start/end, duration, index, setup ID, episode ID, class, and
  `fault_active` bookkeeping fields
- window-to-window `delta_*` features

The first row of each episode has `NaN` for every `delta_*` feature because no
previous window exists. Rolling-average features are not generated.

`fault_class` and `fault_active` are present for supervised training and
evaluation; exclude them from the model feature matrix.

## Local Run

From the repository root, explicitly override the configured Blackbird target:

```powershell
python example/hom_validation/ai_hom_dataset/generate_hom_dataset.py `
  --config example/hom_validation/ai_hom_dataset/configs/hom_ai_dataset_config.ini `
  --target local
```

Small smoke run:

```powershell
python example/hom_validation/ai_hom_dataset/generate_hom_dataset.py `
  --config example/hom_validation/ai_hom_dataset/configs/hom_ai_dataset_config.ini `
  --target local `
  --output-dir example/hom_validation/ai_hom_dataset/generated/smoke `
  --labels healthy temperature_change `
  --dataset-size 2 `
  --episode-duration-s 0.002 `
  --window-duration-s 0.001 `
  --workers 1 `
  --no-resume
```

## Blackbird Run

The default configuration creates 10,000 one-CPU Slurm array tasks, one task
per complete episode, with a concurrency planning cap of `32 * 316`. A
checkpoint is written only after an entire episode finishes. If a task reaches
the wall-time limit before finishing, that partial episode is lost; completed
episode checkpoints remain resumable.

Submit and synchronize from Windows using the configured PuTTY profile:

```powershell
python example/hom_validation/ai_hom_dataset/generate_hom_dataset.py `
  --config example/hom_validation/ai_hom_dataset/configs/hom_ai_dataset_config.ini `
  --target blackbird `
  --submit
```

Rerunning the same command reuses completed checkpoints when `resume=true`.
The combine job runs only after the array job completes successfully.

To execute generation directly inside an allocated Blackbird shell rather than
submitting another Slurm job, force the local execution path:

```bash
/home/ana35/.venvs/sequence-hom/bin/python \
  example/hom_validation/ai_hom_dataset/generate_hom_dataset.py \
  --config example/hom_validation/ai_hom_dataset/configs/hom_ai_dataset_config.ini \
  --target local \
  --workers "$SLURM_CPUS_PER_TASK"
```
