#!/usr/bin/env bash
# Create the server env under the data root. Run on the login node: compute nodes have no internet.
set -euo pipefail
export PYTHONNOUSERSITE=1  # keep ~/.local site-packages out of the env
: "${OFFROAD_DATA_ROOT:=/usace_share/Share_Abdullah/offroad_perception}"
ENV="$OFFROAD_DATA_ROOT/env"
HERE="$(cd "$(dirname "$0")" && pwd)"
CONDA="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
"$CONDA" create -y -p "$ENV" -c conda-forge --override-channels python=3.10 pip
PIP_NO_CACHE_DIR=1 "$ENV/bin/pip" install -r "$HERE/requirements-server.txt"
"$ENV/bin/pip" freeze --exclude-editable > "$HERE/requirements-server.lock.txt"
echo "Done. Activate with: source env/activate.sh"
