# Queueing Model Package

This directory contains the maintained queueing-model workflow for the
three-node repeater study. It is separated from the broader
`entanglement_distribution_experiment` area so the paper material, support
scripts, and cached results live in one place.

## Layout

- [notebooks/queue_model_paper_evaluation.ipynb](/c:/Users/ana35/OneDrive%20-%20NIST/Desktop/cooding/SeQUeNCe/example/queueing_model/notebooks/queue_model_paper_evaluation.ipynb)
  Final paper-evaluation notebook.
- [notebooks/single_heralded_network_examples.ipynb](/c:/Users/ana35/OneDrive%20-%20NIST/Desktop/cooding/SeQUeNCe/example/queueing_model/notebooks/single_heralded_network_examples.ipynb)
  Reference notebook for time-domain single-heralded behavior.
- [memory_allocation_paper_worker.py](/c:/Users/ana35/OneDrive%20-%20NIST/Desktop/cooding/SeQUeNCe/example/queueing_model/memory_allocation_paper_worker.py)
  Batch worker that generates the paper figures.
- [run_memory_allocation_paper_evaluation.py](/c:/Users/ana35/OneDrive%20-%20NIST/Desktop/cooding/SeQUeNCe/example/queueing_model/run_memory_allocation_paper_evaluation.py)
  Local or SSH launcher for the paper-evaluation worker.
- [queue_utils.py](/c:/Users/ana35/OneDrive%20-%20NIST/Desktop/cooding/SeQUeNCe/example/queueing_model/queue_utils.py)
  Canonical queueing-model equations.
- [paper/](/c:/Users/ana35/OneDrive%20-%20NIST/Desktop/cooding/SeQUeNCe/example/queueing_model/paper)
  LaTeX sources for the paper draft and standalone evaluation section.
- [results/](/c:/Users/ana35/OneDrive%20-%20NIST/Desktop/cooding/SeQUeNCe/example/queueing_model/results)
  Cached CSV and JSON outputs used by the notebooks.

## Running the Paper Evaluation

From the repository root:

```powershell
python example/queueing_model/run_memory_allocation_paper_evaluation.py --backend local
```

For cached plotting in the notebook, set:

```python
RUN_BACKEND = "cached"
```

The launcher and notebook both depend on the shared simulation code that stays
under `example/entanglement_distribution_experiment`.
