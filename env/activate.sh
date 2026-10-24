# Source me (login node or inside sbatch jobs): source env/activate.sh
export OFFROAD_DATA_ROOT="${OFFROAD_DATA_ROOT:-/usace_share/Share_Abdullah/offroad_perception}"
export HF_HOME="$OFFROAD_DATA_ROOT/hf_cache"
export PIP_NO_CACHE_DIR=1
export PYTHONNOUSERSITE=1  # keep ~/.local site-packages out of the env
# Compute nodes are offline: only use what is already in HF_HOME there.
if [ -n "${SLURM_JOB_ID:-}" ]; then export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "$OFFROAD_DATA_ROOT/env"
# onnxruntime-gpu needs cuDNN 9 / cuBLAS 12 on the loader path; torch's pip wheels ship them. Without this,
# ORT silently falls back to the CPU provider (every replay before 2026-10-24 ran its models on CPU).
_NV="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
export LD_LIBRARY_PATH="$_NV/cudnn/lib:$_NV/cublas/lib:$_NV/cuda_runtime/lib:$_NV/cufft/lib:$_NV/curand/lib:$_NV/cuda_nvrtc/lib:${LD_LIBRARY_PATH:-}"
unset _NV
