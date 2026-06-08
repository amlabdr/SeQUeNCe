# HOM Overlap Validation

This folder validates Hong-Ou-Mandel (HOM) interference with a full SeQUeNCe
simulation path:

- SPDC Bell-pair sources
- fiber quantum channels
- signal-arm waveplate rotation
- central HOM/BSM node
- polarization-resolved herald analyzers
- timestamp-based fourfold coincidence analysis

The validation focuses on three overlap contributions:

- temporal overlap by scanning relative signal delay
- polarization overlap by rotating one signal arm with a waveplate
- spectral overlap by detuning the source wavelength

## Files

- `hom_three_overlap_validation.ipynb`: plotting notebook for the final HOM validation results
- `simulator.py`: full SeQUeNCe HOM setup and coincidence-analysis helpers
- `run_hom_three_overlap_batch.py`: batch runner for local or Enki execution
- `configs/hom_three_overlap_batch_config.ini`: default batch-run configuration

## Config And Results Layout

Config files live under:

```text
configs/
```

By default, each config writes to a matching result folder:

```text
results/<config_file_stem>/
```

For example:

```text
configs/hom_three_overlap_batch_config.ini
```

writes to:

```text
results/hom_three_overlap_batch_config/
```

This makes it safe to keep multiple configs, such as one full scan config and
one temporal-only config, without mixing their CSV outputs.

You can still override the output location explicitly with `--output-dir` if
needed.

## Workflow

The notebook is intended to plot existing CSV results only. Heavy simulation
runs should be launched from the batch script:

```bash
python example/hom_swapping_validation/run_hom_three_overlap_batch.py
```

The runner loads `configs/hom_three_overlap_batch_config.ini` by default. Edit
that file to change scan ranges, run count, worker count, source parameters,
detector windows, or local/Enki execution settings.

Use `duration_s` to set the physical simulated time per run. The runner derives
the pulse count as:

```text
pulses_per_run = duration_s * source_frequency_hz
```

Do not set pulse counts manually.

## Enki Execution

For Enki, set these fields in the INI config:

- `target`: `enki`
- `nodes`: number of Slurm nodes
- `cpus_per_task`: worker processes per node
- `sync_project`: whether to sync this repo to Enki before submission
- `wait_for_completion`: whether the local script should poll the job
- `collect_outputs`: whether to copy outputs back after completion

The runner starts one Slurm task per node, shards independent scan-point/run
jobs across nodes, and each node uses `cpus_per_task` local worker processes.

## Outputs

Canonical per-run CSVs are written under the config-specific result folder:

- `hom_all_runs.csv`
- `hom_temporal_runs.csv`
- `hom_polarization_runs.csv`
- `hom_spectral_runs.csv`

Notebook-ready aggregated summaries are also written there:

- `hom_temporal_delay_scan_validation.csv`
- `hom_polarization_scan_validation.csv`
- `hom_spectral_scan_validation.csv`

Generated runtime artifacts such as `shards/`, `hom3_*.out`, `hom3_*.err`, and
smoke-test folders are not required for plotting.

## Plotting

After CSV generation, open:

```text
hom_three_overlap_validation.ipynb
```

The notebook currently loads:

```text
results/hom_three_overlap_batch_config/
```

and plots measurable fourfold coincidence counts/s with run-to-run standard
deviation. The polarization scan also includes normalized count curves for
comparison with the expected polarization dependence.

## Alternate Config

You can point to another config file:

```bash
python example/hom_swapping_validation/run_hom_three_overlap_batch.py --config example/hom_swapping_validation/configs/my_temporal_config.ini
```

That config will write to:

```text
results/my_temporal_config/
```

unless `--output-dir` is supplied.
