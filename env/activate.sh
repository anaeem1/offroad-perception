# Source me (login node or inside sbatch jobs): source env/activate.sh
export OFFROAD_DATA_ROOT="${OFFROAD_DATA_ROOT:-/usace_share/Share_Abdullah/offroad_perception}"
export HF_HOME="$OFFROAD_DATA_ROOT/hf_cache"
export PIP_NO_CACHE_DIR=1
export PYTHONNOUSERSITE=1  # keep ~/.local site-packages out of the env
# Compute nodes are offline: only use what is already in HF_HOME there.
if [ -n "${SLURM_JOB_ID:-}" ]; then export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$OFFROAD_DATA_ROOT/env"
