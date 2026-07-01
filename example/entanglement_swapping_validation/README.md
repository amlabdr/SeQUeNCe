# Entanglement-Swapping Validation

This example validates polarization entanglement swapping separately from the
HOM overlap validation.

Two Bell-pair sources prepare photons `(a,b)` and `(c,d)`. Photons `b,c` enter
`PolarizationBSMNode`; photons `a,d` are measured by polarization analyzers
only after conditioning on an accepted `psi_minus` or `psi_plus` detector
pattern.

The validator measures all nine `X/Y/Z` Pauli-basis combinations for `a,d`.
It reports the four conditional outcome counts, reconstructs the two-qubit
density matrix, and computes fidelity with the independently predicted Bell
state.

```python
from example.entanglement_swapping_validation.simulator import (
    SwappingValidationConfig,
    run_swapping_validation,
)

counts, summary = run_swapping_validation(
    "psi_minus",
    "psi_minus",
    config=SwappingValidationConfig(accepted_events_per_basis=500),
)
```

The leading minus sign shown for some swapped states in algebraic tables is a
global phase and is not physically observable.

## Full physical simulation

`full_simulator.py` wires two `SpdcSourceNode` instances through four
`fiberQuantumChannel` paths to `PolarizationBSMNode` and the Alice/David
polarization analyzers. Reported tomography counts are timestamp-derived
fourfolds; simulator pair IDs are not used operationally.

The outer paths include storage/optical delay so the BSM projection occurs
before analyzer detection in the event model.

Run locally or submit to Blackbird with:

```powershell
python example/entanglement_swapping_validation/run_full_swapping_batch.py `
  --config example/entanglement_swapping_validation/configs/full_swapping_validation.ini
```

Set `submit = true` in the INI file to synchronize, submit the resumable Slurm
array, run the dependent combine job, and optionally collect results.

After generation, open `full_swapping_results.ipynb`. That notebook only loads
the combined CSV/NPZ outputs and plots fourfolds, MLE fidelities, and density
matrices.
