# HPC Dataset Generation Guide

Run generation on the HPC filesystem, then copy the finished dataset directory back to your laptop. Do not stream the simulation through the laptop.

## One-Command Disposable Runner

From the local repository, this command packages the current working tree, copies it to `enki`, creates a disposable remote run directory, runs a small smoke dataset, fetches the dataset back, and removes the remote copy.

```powershell
python example/entanglement_distribution_experiment/ai_diagnosis_dataset/run_on_hpc.py --smoke
```

Full training dataset:

```powershell
python example/entanglement_distribution_experiment/ai_diagnosis_dataset/run_on_hpc.py
```

By default the script:

- uses SSH host `enki`
- uses remote Python `python3.12`
- detects all remote CPU cores and uses them as workers
- runs with `--parallel-backend process`
- installs only the runtime dependencies needed by the generator into a disposable remote venv
- fetches the dataset to `example/entanglement_distribution_experiment/ai_diagnosis_dataset/generated/`
- deletes the remote run directory on success
- deletes the remote run directory on `Ctrl+C`

If generation fails for another reason, the remote directory is preserved for debugging unless you pass `--cleanup-on-failure`.

Use fewer workers if the HPC is shared or the job is memory-bound:

```powershell
python example/entanglement_distribution_experiment/ai_diagnosis_dataset/run_on_hpc.py --workers 16
```

The manual workflow below is still useful if you prefer controlling the HPC session yourself.

## 1. Copy Or Clone The Repo

If the repo is on GitHub or another remote:

```bash
git clone <repo-url> SeQUeNCe
cd SeQUeNCe
```

If you need to copy the current local working tree from Windows PowerShell:

```powershell
scp -r -i C:\path\to\key C:\Users\ana35\OneDrive` -` NIST\Desktop\cooding\SeQUeNCe username@hpc-address:/path/to/work/
```

For large repos, `rsync` from Git Bash or WSL is usually better than `scp`.

## 2. Create The Python Environment On The HPC

```bash
cd /path/to/SeQUeNCe
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

If the HPC uses modules, load Python first, for example:

```bash
module load python
```

## 3. Run A Smoke Test

```bash
python example/entanglement_distribution_experiment/ai_diagnosis_dataset/generate_dataset.py \
  --mode sample \
  --samples-per-label 1 \
  --episode-duration-s 10 \
  --window-duration-s 1 \
  --source-frequency-hz 100000 \
  --visibility-stride-windows 5 \
  --temperature-episode-span-s 86400 \
  --base-seed 300000 \
  --workers 8 \
  --parallel-backend process \
  --skip-raw \
  --validation-off \
  --output-dir example/entanglement_distribution_experiment/ai_diagnosis_dataset/generated/hpc_smoke
```

## 4. Run The Full Dataset

Use `tmux` if the HPC does not have a scheduler:

```bash
tmux new -s diagnosis_dataset
```

Then run:

```bash
python example/entanglement_distribution_experiment/ai_diagnosis_dataset/generate_dataset.py \
  --mode full \
  --samples-per-label 500 \
  --episode-duration-s 60 \
  --window-duration-s 1 \
  --source-frequency-hz 100000 \
  --visibility-stride-windows 30 \
  --temperature-episode-span-s 86400 \
  --base-seed 300000 \
  --workers 32 \
  --parallel-backend process \
  --skip-raw \
  --validation-off \
  --output-dir example/entanglement_distribution_experiment/ai_diagnosis_dataset/generated/train_v5_60s_100khz_24h_temperature
```

Detach from tmux with `Ctrl-b`, then `d`. Reattach with:

```bash
tmux attach -t diagnosis_dataset
```

The generator writes per-sample checkpoints to `checkpoints/`. If the job stops, rerun the same command and it will resume from existing checkpoints.

## 5. Slurm Example

Create `run_diagnosis_dataset.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=diagnosis_dataset
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=diagnosis_dataset_%j.out
#SBATCH --error=diagnosis_dataset_%j.err

cd /path/to/SeQUeNCe
source .venv/bin/activate

python example/entanglement_distribution_experiment/ai_diagnosis_dataset/generate_dataset.py \
  --mode full \
  --samples-per-label 500 \
  --episode-duration-s 60 \
  --window-duration-s 1 \
  --source-frequency-hz 100000 \
  --visibility-stride-windows 30 \
  --temperature-episode-span-s 86400 \
  --base-seed 300000 \
  --workers 32 \
  --parallel-backend process \
  --skip-raw \
  --validation-off \
  --output-dir example/entanglement_distribution_experiment/ai_diagnosis_dataset/generated/train_v5_60s_100khz_24h_temperature
```

Submit and monitor:

```bash
sbatch run_diagnosis_dataset.sbatch
squeue -u "$USER"
tail -f diagnosis_dataset_<jobid>.out
```

## 6. Copy The Finished Dataset Back

From Windows PowerShell on your laptop:

```powershell
scp -r -i C:\path\to\key username@hpc-address:/path/to/SeQUeNCe/example/entanglement_distribution_experiment/ai_diagnosis_dataset/generated/train_v5_60s_100khz_24h_temperature example/entanglement_distribution_experiment/ai_diagnosis_dataset/generated/
```

If the dataset is large, prefer `rsync` from Git Bash or WSL:

```bash
rsync -avP -e "ssh -i /path/to/key" username@hpc-address:/path/to/SeQUeNCe/example/entanglement_distribution_experiment/ai_diagnosis_dataset/generated/train_v5_60s_100khz_24h_temperature/ example/entanglement_distribution_experiment/ai_diagnosis_dataset/generated/train_v5_60s_100khz_24h_temperature/
```
