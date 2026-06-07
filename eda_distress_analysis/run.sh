#!/usr/bin/env bash
# SBATCH settings - request an A100 GPU on the `gpu_a100_short` partition
# Adjust resource values (time, cpus, mem) as needed for your job
# Usage: sbatch run_distress_gpu_a100_short.sh
# Optional env vars: NOTEBOOK (path or name), VENV_PATH (virtualenv path)
# Example: NOTEBOOK=distress.ipynb VENV_PATH=~/venvs/bertopic_env sbatch run_distress_gpu_a100_short.sh

#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --job-name=distress_nb
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

set -euo pipefail

# Slurm submission directory; default to the script directory if not set
cd "$SLURM_SUBMIT_DIR"

# Notebook to run (default: distress.ipynb in the same folder as this script)
NOTEBOOK="${SLURM_SUBMIT_DIR}/distress.ipynb"

# Use the requested virtualenv (no extra checks)
VENV_PATH="${VENV_PATH:-$HOME/venv/smda_env}"
source "$VENV_PATH/bin/activate"

# Redirect all script stdout/stderr to a log file inside the submit dir
exec > "${SLURM_SUBMIT_DIR}/asset/distress_${SLURM_JOB_ID}.log" 2>&1

echo "Running notebook: $NOTEBOOK"
echo "Logs and outputs will be written to: ${SLURM_SUBMIT_DIR}/asset"

# Output filename includes Slurm job id for uniqueness
OUTBASE="${SLURM_SUBMIT_DIR}/asset/${SLURM_JOB_ID}.ipynb"

# Prefer papermill if available (shows progress and lets you pass parameters); otherwise use nbconvert
if python -c "import papermill" >/dev/null 2>&1; then
  python -m papermill "$NOTEBOOK" "$OUTBASE" --progress-bar --kernel smda_env
else
  python -m nbconvert --to notebook --execute "$NOTEBOOK" \
    --ExecutePreprocessor.kernel_name=smda_env \
    --ExecutePreprocessor.timeout=7200 \
    --output "$OUTBASE"
fi

echo "Done - executed notebook saved to $OUTBASE"
