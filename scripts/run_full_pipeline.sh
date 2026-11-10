#!/usr/bin/env bash
# Full pipeline per dataset: clear stale BirdNET results, bare-analyze fresh,
# run_inference.py merge, then PERCH/Fusion inference - all in one go.
#
# Usage: bash run_full_pipeline.sh <dataset_name1> [<dataset_name2> ...]

set -e

if [ "$#" -eq 0 ]; then
  echo "Usage: bash run_full_pipeline.sh <dataset_name1> [<dataset_name2> ...]"
  exit 1
fi

ARIA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_ROOT="$ARIA_ROOT/results_inference_birdnet"
RESULTS_DIR="$ARIA_ROOT/results"
PREPROCESSED_ROOT="${PREPROCESSED_ROOT:-/home/zoodata/preprocessed}"
MODEL="$ARIA_ROOT/models/ZooCustom_v1.tflite"
RUN_INFERENCE_SCRIPT="$ARIA_ROOT/src/inference/run_inference.py"
TWO_TIER_DIR="$ARIA_ROOT/src/inference"

echo "Using PREPROCESSED_ROOT: $PREPROCESSED_ROOT"

for DATASET in "$@"; do
  SRC_DIR="$PREPROCESSED_ROOT/$DATASET"
  OUT_DIR="$RESULTS_ROOT/$DATASET"

  if [ ! -d "$SRC_DIR" ]; then
    echo "[ERR] Source not found: $SRC_DIR - skipping $DATASET"
    continue
  fi

  echo "============================================================"
  echo "[DATASET] $DATASET - STEP A: BirdNET"
  echo "============================================================"

  echo "[A1/3] Clearing any existing BirdNET result files..."
  find "$OUT_DIR" -iname "*.BirdNET.results.csv" -delete 2>/dev/null || true

  echo "[A2/3] Running bare birdnet_analyzer.analyze directly..."
  python3 -m birdnet_analyzer.analyze "$SRC_DIR" \
    -o "$OUT_DIR" \
    --classifier "$MODEL" \
    --min_conf 0.05 --overlap 0 --threads 40 --rtype csv

  EMPTY=$(find "$OUT_DIR" -iname "*.BirdNET.results.csv" -size -70c | wc -l)
  TOTAL=$(find "$OUT_DIR" -iname "*.BirdNET.results.csv" | wc -l)
  echo "  Result: $EMPTY/$TOTAL empty"

  echo "[A3/3] Running run_inference.py --only $DATASET for cage-filter/threshold/merge..."
  python3 "$RUN_INFERENCE_SCRIPT" --only "$DATASET" --results-root "$RESULTS_ROOT" --input-root "$PREPROCESSED_ROOT"

  echo ""
  echo "============================================================"
  echo "[DATASET] $DATASET - STEP B: PERCH/Fusion"
  echo "============================================================"

  python3 -c "
import sys
sys.path.insert(0, '$TWO_TIER_DIR')
from two_tier_inference_perchv2_with_voting_SAVE_PERCH_PREDICTIONS import TwoTierInferenceEnhanced
from pathlib import Path

RESULTS_DIR = Path('$RESULTS_DIR')
inf = TwoTierInferenceEnhanced(perch_predictions_top_k=10, save_perch_predictions=True, save_embeddings=False, verbose=False)
print('BirdNET species loaded:', len(inf.extractor.birdnet_species))

dataset_path = '$SRC_DIR'
inf.clear_collected_data()
inf.set_constraints_from_directory(dataset_path)
summary = inf.process_directory(dataset_path)
inf.print_summary(summary)
inf.save_results_csv(summary, str(RESULTS_DIR / '${DATASET}_ensemble.csv'))
inf.save_perch_predictions_csv(str(RESULTS_DIR / '${DATASET}_PERCH_predictions.csv'))
"

  echo ""
  echo "[DONE] $DATASET - full pipeline complete."
  echo "Species breakdown (BirdNET):"
  python3 -c "
import pandas as pd
df = pd.read_csv('$OUT_DIR/all_results.filtered.top3.csv')
print(df['Common name'].value_counts())
"
  echo ""
done

echo "============================================================"
echo "ALL DATASETS PROCESSED: $@"
echo "============================================================"
