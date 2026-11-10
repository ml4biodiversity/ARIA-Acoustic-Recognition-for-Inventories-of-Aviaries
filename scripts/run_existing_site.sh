#!/usr/bin/env bash
# Full existing-site pipeline: BirdNET (clear stale + bare-analyze + merge),
# PERCH/Fusion inference, then final merge into _meta.xlsx.
#
# Use this when you have NEW RECORDINGS from an EXISTING, already-configured
# aviary/site (no new species, no config changes needed).
#
# Usage: bash run_existing_site.sh <dataset_name1> [<dataset_name2> ...]
# Example: bash run_existing_site.sh fl_cologne_zoo_aviary4_jan26_data

set -e

if [ "$#" -eq 0 ]; then
  echo "Usage: bash run_existing_site.sh <dataset_name1> [<dataset_name2> ...]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MERGE_SCRIPT="$SCRIPT_DIR/../src/inference/merge_model_predictions.py"

echo "############################################################"
echo "# STEP 1: BirdNET + PERCH/Fusion (per dataset)"
echo "############################################################"
bash "$SCRIPT_DIR/run_full_pipeline.sh" "$@"

echo ""
echo "############################################################"
echo "# STEP 2: Merge into final _meta.xlsx"
echo "############################################################"
python3 "$MERGE_SCRIPT" "$@"

echo ""
echo "############################################################"
echo "# ALL DONE: $@"
echo "############################################################"
