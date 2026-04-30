# Dataset Review Plot Guide

Use `dataset_review.ipynb` to check whether the generated measurement data is believable before training a diagnosis model. The notebook intentionally keeps only the plots that answer a clear validation question.

## Plot Set

1. `01_dataset_composition.png`
   - Checks how many episodes exist for each fault type.
   - Checks whether each non-normal fault has the expected trajectory styles.

2. `02_fault_change_summary.png`
   - Shows the typical before/after change after the fault starts.
   - Use this to confirm the main measurement signal moves in the expected direction.

3. `03_representative_fault_examples.png`
   - Shows one concrete episode per fault type.
   - The dashed line marks when the fault starts.

4. `04_fault_trajectory_styles.png`
   - Shows different ways the same fault can evolve.
   - This checks that the dataset is not only simple one-way ramps.

5. `05_temperature_physical_time.png`
   - Shows temperature fault examples on the physical time axis.
   - Use this to verify that temperature drift spans hours, not only the simulated acquisition seconds.

## What To Look For

- `normal`: stable peak, rate, CAR, and ZZ correlation.
- `synchronization_issue`: degraded peak quality and CAR.
- `peak_shift_temperature`: peak position follows link temperature change.
- `polarization_drift`: ZZ correlation degrades more than timing or rate.
- `raman_noise`: CAR drops and noise-related rates increase.
- `loss`: singles and coincidence rates drop while timing stays roughly centered.

## Run From Terminal

```powershell
python example/entanglement_distribution_experiment/ai_diagnosis_dataset/review_dataset.py `
  --dataset-dir example/entanglement_distribution_experiment/ai_diagnosis_dataset/generated/train_v5_60s_100khz_24h_temperature
```

The figures are written to `<dataset-dir>/review_artifacts/`.
