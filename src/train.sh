#!/usr/bin/env bash
#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARIA_ROOT="$(dirname "$SCRIPT_DIR")"  # Parent of src/


# ===================== config =====================
SLIST="${SLIST:-$ARIA_ROOT/config/species_list.txt}"
SEG="${SEG:-$ARIA_ROOT/data/segments}"
WAV="${WAV:-$ARIA_ROOT/data/wav}"
CHK="${CHK:-$ARIA_ROOT/models}"
CURATED="${CURATED:-$ARIA_ROOT/data/train_curated}"
TEST_OUT="${TEST_OUT:-$ARIA_ROOT/data/test_curated}"
# thresholds for inclusion/balance
MIN_FILES_PER_CLASS="${MIN_FILES_PER_CLASS:-50}"   # exclude classes with < this many 3s clips
MAX_FILES_PER_CLASS="${MAX_FILES_PER_CLASS:-700}"  # cap per-class to reduce dominance (set -1 for no cap)
TEST_FRAC="${TEST_FRAC:-0.15}"                      # stratified test split fraction
PREP_MIN_FILES="${PREP_MIN_FILES:-10}" 
AUG_TARGET_MIN="${AUG_TARGET_MIN:-50}"
# noise class creation (non-event, suppressed at inference)
MAKE_NOISE="${MAKE_NOISE:-1}"                       # 1=enable, 0=disable
NOISE_LABEL="${NOISE_LABEL:-Noise}"                 # one of: Noise, Background, Other, Silence (BirdNET treats as non-event)
NOISE_TARGET="${NOISE_TARGET:-1800}"
NOISE_MIN_CONF="${NOISE_MIN_CONF:-0.15}"            # "no detection" threshold (lower -> stricter)
NOISE_SEG_LEN="${NOISE_SEG_LEN:-3.0}"
NOISE_STEP="${NOISE_STEP:-1.5}"
NOISE_DILATE="${NOISE_DILATE:-0.5}"                 # seconds to expand around detections when excluding
NOISE_MAX_PER_WAV="${NOISE_MAX_PER_WAV:-15}"

# training knobs (all are 1st-class BirdNET train flags)
CLASSIFIER_NAME="${CLASSIFIER_NAME:-ZooCustom_v1}"
EPOCHS="${EPOCHS:-200}"
BATCH_TR="${BATCH_TR:-112}"
LR="${LR:-6e-5}"
CROP_MODE="${CROP_MODE:-center}"                  
UPSAMPLING_RATIO="${UPSAMPLING_RATIO:-0.60}"         
UPSAMPLING_MODE="${UPSAMPLING_MODE:-linear}"        # repeat|linear|mean|smote
USE_FOCAL="${USE_FOCAL:-1}"                         # 1 enables focal loss
FOCAL_GAMMA="${FOCAL_GAMMA:-1.5}"
FOCAL_ALPHA="${FOCAL_ALPHA:-0.25}"
DROPOUT="${DROPOUT:-0.20}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0}"             # 1 enables
MIXUP="${MIXUP:-0}"                                 # 1 enables
HIDDEN_UNITS="${HIDDEN_UNITS:-512}"
VAL_SPLIT_FALLBACK="${VAL_SPLIT_FALLBACK:-0.2}"    

# system
PY_BIN="$(command -v python3 || command -v python)"
T="$(( $( { command -v nproc >/dev/null && nproc; } || { command -v sysctl >/dev/null && sysctl -n hw.ncpu; } || echo 4 ) ))"
mkdir -p "$CHK" "$CURATED"

command -v ffmpeg >/dev/null || { echo "ffmpeg not found"; exit 1; }
command -v $PY_BIN >/dev/null || { echo "python not found"; exit 1; }

# ===================== helpers (python) =====================
BIRDNET_REPO="${BIRDNET_REPO:-$HOME/BirdNET-Analyzer}"

if ! "$PY_BIN" -c "import birdnet_analyzer" 2>/dev/null; then
  if [ -d "$BIRDNET_REPO/birdnet_analyzer" ]; then
    export PYTHONPATH="$BIRDNET_REPO${PYTHONPATH:+:$PYTHONPATH}"
    echo "[INFO] Added \$BIRDNET_REPO to PYTHONPATH: $BIRDNET_REPO"
  else
    echo "[INFO] birdnet_analyzer not found; installing from PyPI into this env..."
    "$PY_BIN" -m pip install --upgrade pip
    "$PY_BIN" -m pip install --upgrade birdnet-analyzer
  fi
fi


PREP_PY="$ARIA_ROOT/src/train/birdnet/prep_curated.py"

NOISE_PY="$ARIA_ROOT/src/train/birdnet/make_noise_from_wav.py"


# ===================== steps =====================
if [ -d "$CURATED" ] && [ -d "$TEST_OUT" ] && [ "$(find "$CURATED" -type f -name '*.wav' | wc -l)" -gt 0 ]; then
  echo "[INFO] Existing curated train/test sets found at:"
  echo "       TRAIN: $CURATED"
  echo "       TEST : $TEST_OUT"
  echo "[INFO] Skipping re-creation step."
else
  echo "[STEP] Prepare curated train/test sets from $SEG using $SLIST"
  $PY_BIN "$PREP_PY" \
    --segments_root "$SEG" \
    --list_file "$SLIST" \
    --out_train "$CURATED" \
    --out_test "$TEST_OUT" \
    --min_files "$PREP_MIN_FILES" \
    --max_files "$MAX_FILES_PER_CLASS" \
    --test_frac "$TEST_FRAC"
fi

if [[ "$MAKE_NOISE" == "1" && -d "$WAV" ]]; then
  echo "[STEP] Create non-event class '$NOISE_LABEL' from windows with no detections"
  $PY_BIN "$NOISE_PY" \
  --wav_root "$WAV" \
  --out_dir "$CURATED" \
  --label "$NOISE_LABEL" \
  --target "$NOISE_TARGET" \
  --seg_len "$NOISE_SEG_LEN" \
  --step "$NOISE_STEP" \
  --max_per_wav "$NOISE_MAX_PER_WAV" \
  --threads "$T" \
  --exclude_from "$CURATED" "$TEST_OUT"
else
  echo "[INFO] Skipping noise creation (MAKE_NOISE=$MAKE_NOISE, WAV dir exists: $( [[ -d "$WAV" ]] && echo yes || echo no ))"
fi

AUG_PY="$ARIA_ROOT/src/train/birdnet/augment_like_lauha.py"
if [ ! -f "$AUG_PY" ]; then
  echo "[ERR] $AUG_PY not found (did you save the script?)."; exit 2
fi

echo "[STEP] Augment weak classes in $CURATED (spec-domain, paper-faithful) to reach MIN_FILES_PER_CLASS ($MIN_FILES_PER_CLASS)"
NOISE_DIR_CAND=""
for nn in "$CURATED/Noise" "$CURATED/Background" "$CURATED/Other" "$CURATED/Silence"; do
  [ -d "$nn" ] && NOISE_DIR_CAND="$nn" && break
done

python3 "$AUG_PY" \
  --train_root "$CURATED" \
  ${NOISE_DIR_CAND:+--noise_root "$NOISE_DIR_CAND"} \
  --target_min "$AUG_TARGET_MIN" \
  --max_per_src_frac 0.40 \
  --hard_cap_per_src 8 \
  --sr 48000

echo "[STEP] Train BirdNET custom classifier (zoo-focused, REPLACE mode)"
train_cmd=( "$PY_BIN" -m birdnet_analyzer.train "$CURATED"
  -o "$CHK/$CLASSIFIER_NAME"
  --epochs "$EPOCHS" -b "$BATCH_TR"
  --learning_rate "$LR"
  --crop_mode "$CROP_MODE" --overlap 0
  --model_save_mode replace
  --upsampling_ratio "$UPSAMPLING_RATIO" --upsampling_mode "$UPSAMPLING_MODE"
  --cache_mode save --cache_file "$CHK/${CLASSIFIER_NAME}_train_cache.npz"
  --dropout "$DROPOUT"
  --hidden_units "$HIDDEN_UNITS"
)

[[ "$LABEL_SMOOTHING" == "1" ]] && train_cmd+=( --label_smoothing )
[[ "$MIXUP" == "1"          ]] && train_cmd+=( --mixup )
[[ "$USE_FOCAL" == "1"      ]] && train_cmd+=( --focal-loss --focal-loss-gamma "$FOCAL_GAMMA" --focal-loss-alpha "$FOCAL_ALPHA" )
if [[ -d "$TEST_OUT" && -n "$(find -L "$TEST_OUT" -name '*.wav' -print -quit)" ]]; then
  train_cmd+=( --test_data "$TEST_OUT" )
else
  train_cmd+=( --val_split "$VAL_SPLIT_FALLBACK" )
fi


printf '[CMD] '; printf '%q ' "${train_cmd[@]}"; echo
"${train_cmd[@]}"

echo "[OK] Classifier should be at: $CHK/${CLASSIFIER_NAME}.tflite"

echo "[STEP] (Optional) Analyze with your classifier"
echo "$PY_BIN -m birdnet_analyzer.analyze $WAV -o $WAV --classifier $CHK/${CLASSIFIER_NAME}.tflite --min_conf 0.25 --overlap 0 --threads $T"

