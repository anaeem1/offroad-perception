#!/usr/bin/env bash
# Download the RELLIS-3D subset this project needs into $OFFROAD_DATA_ROOT.
# Idempotent: finished items leave a .done marker and are skipped on re-run; partial
# downloads resume. Each zip is integrity-tested, extracted, inventoried, then deleted.
# Run on the login node (compute nodes have no internet). Links: github.com/unmannedlab/RELLIS-3D
# Usage: scripts/download_rellis.sh [item ...]   (default: all items, in the order below)
set -euo pipefail
: "${OFFROAD_DATA_ROOT:=/usace_share/Share_Abdullah/offroad_perception}"
GDOWN="${GDOWN:-$OFFROAD_DATA_ROOT/env/bin/gdown}"
PY="${PY:-$OFFROAD_DATA_ROOT/env/bin/python}"
ZIPS="$OFFROAD_DATA_ROOT/rellis3d/_zips"
INVENTORY="$OFFROAD_DATA_ROOT/rellis3d/INVENTORY.tsv"
mkdir -p "$ZIPS"

# name | Google Drive file id | zip size (GB, for the disk check) | extract dir (relative to data root)
MANIFEST="
ontology          1K8Zf0ju_xI5lnx3NTDLJpVTs59wmGPI6  0.01 rellis3d/meta
image_split       1zHmnVaItcYJAWat3Yti1W_5Nfux194WQ  0.01 rellis3d/meta
lidar_split       1raQJPySyqDaHpc53KPnJVl3Bln6HlcVS  0.01 rellis3d/meta
cam_intrinsic     1NAigZTJYocRSOTfgFBddZYnDsI_CSpwK  0.01 rellis3d/calib
cam2lidar         19EOqWS9fDUFp4nsBrMCa69xs9LgIlS2e  0.01 rellis3d/calib
lidar_poses       1V3PT_NJhA41N7TBLp5AbW31d0ztQDQOX  0.01 rellis3d
lidar_labels      12bsblHXtob60KrjV7lGXUQTdC5PhV8Er  0.2  rellis3d
image_labels_id   16URBUQn_VOGvUqfms-0I8HHKMtjPHsu5  0.1  rellis3d
salsanext_weights 1DxuzlnFKnU8EpSuODRywJUrJlieUUheg  0.2  weights/salsanext_rellis
images            1F3Leu0H_m6aPVpZITragfreO_SGtL2yV  11   rellis3d
lidar_scans       1lDSVRf_kZrD0zHHMsKJ0V1GN9QATR4wH  14   rellis3d
bag_00004_merged  1AuEjX0do3jGZhGKPszSEUNoj85YswNya  14   bags/ros1/00004
bag_00001_merged  1geoU45pPavnabQ0arm4ILeHSsG3cU6ti  16   bags/ros1/00001
"

log() { echo "[$(date '+%F %T')] $*"; }

fetch() {
  local name="$1" id="$2" gb="$3" dest="$OFFROAD_DATA_ROOT/$4"
  local marker="$ZIPS/$name.done" zip="$ZIPS/$name.zip"
  if [ -f "$marker" ]; then log "skip $name (done)"; return; fi
  # Need room for the zip plus its extracted contents (lidar data roughly doubles).
  local need_kb free_kb
  need_kb=$(awk -v g="$gb" 'BEGIN{printf "%d", (g*3 + 5) * 1024 * 1024}')
  free_kb=$(df -Pk "$OFFROAD_DATA_ROOT" | awk 'NR==2{print $4}')
  if [ "$free_kb" -lt "$need_kb" ]; then
    log "ABORT $name: need ~$((need_kb/1024/1024)) GB free, have $((free_kb/1024/1024)) GB"; exit 1
  fi
  log "download $name ($gb GB) -> $zip"
  if ! "$GDOWN" --continue -O "$zip" "https://drive.google.com/uc?id=$id"; then
    log "FAILED $name: Google Drive refused or quota exceeded. Not trying mirrors; report and retry later."
    exit 2
  fi
  mkdir -p "$dest"
  log "extract $name -> $dest"
  # No unzip on the login node: use Python's zipfile, which CRC-checks every member as it extracts.
  local n_files bytes
  read -r n_files bytes < <("$PY" - "$zip" "$dest" <<'PYEOF'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    files = [i for i in z.infolist() if not i.is_dir()]
    z.extractall(sys.argv[2])
    print(len(files), sum(i.file_size for i in files))
PYEOF
  )
  printf '%s\t%s\t%s\t%s\t%s\n' "$name" "$id" "$(stat -c %s "$zip")" "$bytes" "$n_files" >> "$INVENTORY"
  rm -f "$zip"
  touch "$marker"
  log "done $name: $n_files files, $bytes bytes uncompressed"
}

[ -f "$INVENTORY" ] || printf 'item\tdrive_id\tzip_bytes\tuncompressed_bytes\tn_files\n' > "$INVENTORY"
want=" $* "
while read -r name id gb dest; do
  [ -z "$name" ] && continue
  if [ $# -eq 0 ] || [[ "$want" == *" $name "* ]]; then fetch "$name" "$id" "$gb" "$dest"; fi
done <<< "$MANIFEST"
log "all requested items done"
