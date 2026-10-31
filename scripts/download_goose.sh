#!/usr/bin/env bash
# Download the GOOSE subset the monitor study needs into $OFFROAD_DATA_ROOT/goose (licence: CC BY-SA 4.0).
# Idempotent: finished items leave a .done marker and are skipped on re-run; partial downloads resume.
# Zips are integrity-tested and extracted, then deleted. Run on the login node (compute nodes have no
# internet). Links are copied from goose-dataset.de/docs/setup/ and /docs/mucar3/ (read 2026-10-31).
# Usage: scripts/download_goose.sh [item ...]   (default: all items, in the order below)
set -euo pipefail
: "${OFFROAD_DATA_ROOT:=/usace_share/Share_Abdullah/offroad_perception}"
PY="${PY:-$OFFROAD_DATA_ROOT/env/bin/python}"
ROOT="$OFFROAD_DATA_ROOT/goose"
ZIPS="$ROOT/_zips"
mkdir -p "$ZIPS"
S=https://goose-dataset.de/storage
R=https://goose-dataset.de/docs/resources

# name | url | extract into (relative to $ROOT; "-" = keep the file as is)
MANIFEST="
calib_windshield_vis_yaml  $R/camera_parameters/mucar3/windshield_vis.yaml   calib/mucar3
calib_windshield_vis_K     $R/camera_parameters/mucar3/windshield_vis_K.dat  calib/mucar3
calib_windshield_vis_P     $R/camera_parameters/mucar3/windshield_vis_P.dat  calib/mucar3
calib_windshield_vis_R     $R/camera_parameters/mucar3/windshield_vis_R.dat  calib/mucar3
calib_windshield_vis_dist  $R/camera_parameters/mucar3/windshield_vis_dist.dat calib/mucar3
calib_tf_tree              $R/mucar3_tf/mucar3_tf.dot                        calib/mucar3
challenge_label_mapping    $R/challenge_label_mapping.csv                    meta
ddrnet_class_512           https://goose-dataset.de/models/ddrnet_class_512.pth  weights
goose_3d_val               $S/goose_3d_val.zip                               3d
goose_3d_test              $S/goose_3d_test.zip                              3d
goose_2d_val               $S/goose_2d_val.zip                               2d
goose_2d_test              $S/goose_2d_test.zip                              2d
goose_3d_train             $S/goose_3d_train.zip                             3d
"

want=("$@")
while read -r name url dest; do
  [ -z "${name:-}" ] && continue
  if [ ${#want[@]} -gt 0 ] && [[ ! " ${want[*]} " =~ " $name " ]]; then continue; fi
  [ -f "$ZIPS/$name.done" ] && { echo "have $name"; continue; }
  mkdir -p "$ROOT/$dest"
  file="$ZIPS/$(basename "$url")"
  echo "== $name <- $url"
  wget -q -c --tries=10 --timeout=60 -O "$file" "$url"
  if [[ "$file" == *.zip ]]; then
    "$PY" - "$file" "$ROOT/$dest" <<'PY'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
bad = z.testzip()
if bad:
    sys.exit(f"corrupt member {bad} in {sys.argv[1]}")
z.extractall(sys.argv[2])
print(f"extracted {len(z.namelist())} entries -> {sys.argv[2]}")
PY
    rm -f "$file"
  else
    mv "$file" "$ROOT/$dest/"
  fi
  touch "$ZIPS/$name.done"
done <<< "$MANIFEST"
du -sh "$ROOT"/* 2>/dev/null
