#!/usr/bin/env bash
set -Eeuo pipefail

# ===================== config =====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARIA_ROOT="$(dirname "$SCRIPT_DIR")"  # Parent of src/

SLIST="${SLIST:-$ARIA_ROOT/config/species_list.txt}"
NOT_IN_BIRDNET="${NOT_IN_BIRDNET:-$ARIA_ROOT/config/birds_not_in_birdnet.txt}"
PERCH_LABELS_CSV="${PERCH_LABELS_CSV:-$ARIA_ROOT/config/perch_labels.csv}"
EBIRD_INDICES="${EBIRD_INDICES:-$ARIA_ROOT/config/ebird_indices.txt}"
BIRDNET_REPO="${BIRDNET_REPO:-$(python3 -c 'import birdnet_analyzer,os;print(os.path.dirname(os.path.dirname(birdnet_analyzer.__file__)))' 2>/dev/null || echo "$HOME/BirdNET-Analyzer")}"
RAW="${RAW:-$ARIA_ROOT/data/raw}"
WAV="${WAV:-$ARIA_ROOT/data/wav}"
SEG="${SEG:-$ARIA_ROOT/data/segments}"
CHK="${CHK:-$ARIA_ROOT/models}"                # output classifier .tflite
UNKNOWN_MODE="${UNKNOWN_MODE:-perch}"   # perch for unknowns
PERCH_THRESH="${PERCH_THRESH:-0.50}"
STEP_SECONDS="${STEP_SECONDS:-3.0}"     # hop for 3 s windows
PERCH_MARGIN="${PERCH_MARGIN:-0.10}"
export TFHUB_CACHE_DIR="$HOME/.cache/tfhub"
mkdir -p "$TFHUB_CACHE_DIR"



MIN_CONF_ANALYZE="${MIN_CONF_ANALYZE:-0.25}"     
MIN_CONF_SEGMENTS="${MIN_CONF_SEGMENTS:-0.40}"
SEG_LEN_SECONDS="${SEG_LEN_SECONDS:-3}"

# Speed knobs
OVERLAP="${OVERLAP:-0.0}"
BATCH_SIZE="${BATCH_SIZE:-16}"

# Behavior flags (env overrides)
SKIP_FETCH="${SKIP_FETCH:-0}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
REANALYZE="${REANALYZE:-0}"
FORCE_CSV="${FORCE_CSV:-0}"
ONLY_STEP="${ONLY_STEP:-}"

# ===================== sanity =====================
command -v ffmpeg >/dev/null || { echo "ffmpeg not found"; exit 1; }
command -v ffprobe >/dev/null || { echo "ffprobe not found"; exit 1; }
PY_BIN="$(command -v python || command -v python3 || true)"
[ -n "$PY_BIN" ] || { echo "python not found"; exit 1; }
[ -d "$BIRDNET_REPO" ] || { echo "BIRDNET_REPO not found: $BIRDNET_REPO"; exit 1; }
[ -f "$SLIST" ] || { echo "Species list not found: $SLIST"; exit 1; }
[ -f "$NOT_IN_BIRDNET" ] || { echo "Birds not in BirdNET list not found: $NOT_IN_BIRDNET"; exit 1; }

mkdir -p "$RAW" "$WAV" "$SEG" "$CHK"
export PYTHONPATH="${BIRDNET_REPO}${PYTHONPATH:+:$PYTHONPATH}"

# Check for required Python packages
"$PY_BIN" -c "import librosa, soundfile, numpy, scipy" 2>/dev/null || {
    echo "Missing required packages. Install with:"
    echo "pip install librosa soundfile numpy scipy"
    exit 1
}

# threads
T="$(( $( { command -v nproc >/dev/null && nproc; } || { command -v sysctl >/dev/null && sysctl -n hw.ncpu; } || echo 4 ) ))"

# absolute paths for mirrors
RAW_ABS="$(cd "$RAW" && pwd)"
WAV_ABS="$(cd "$WAV" && pwd)"

# ===================== helpers =====================
have_any_mp3() { find "$RAW_ABS" -type f -name '*.mp3' | head -n 1 | grep -q .; }
have_any_wav() { find "$WAV_ABS" -type f -name '*.wav' | head -n 1 | grep -q .; }
have_any_csv() {
    local first
    first="$(find "$WAV_ABS" -type f -name '*.BirdNET.results.csv' -print -quit 2>/dev/null || true)"
    [[ -n "$first" ]]
}

report_counts() {
    local n_tbl n_csv
    n_tbl="$(find "$WAV_ABS" -type f -name '*.BirdNET.selection.table.txt' | wc -l | awk '{print $1}')"
    n_csv="$(find "$WAV_ABS" -type f -name '*.BirdNET.results.csv' | wc -l | awk '{print $1}')"
    echo "[INFO] Tables: $n_tbl CSVs: $n_csv"
}

normalize_name() {
  local s="$*"
  # strip accents -> lowercase -> collapse non-alnum to single underscore
  s="$(printf '%s' "$s" | iconv -f utf8 -t ascii//TRANSLIT 2>/dev/null || printf '%s' "$s")"
  s="$(printf '%s' "$s" | tr '[:upper:]' '[:lower:]')"
  printf '%s' "$s" | sed -E 's/[^a-z0-9]+/_/g; s/^_+|_+$//g; s/_+/_/g'
}

build_not_in_birdnet_index() {
  # normalize the NOT_IN_BIRDNET file once and reuse
  if [[ -z "${NOT_IN_BIRDNET_IDX:-}" || ! -s "$NOT_IN_BIRDNET_IDX" ]]; then
    NOT_IN_BIRDNET_IDX="$(mktemp -t not_in_birdnet.XXXXXX)"
    iconv -f utf8 -t ascii//TRANSLIT "$NOT_IN_BIRDNET" 2>/dev/null \
      | tr '[:upper:]' '[:lower:]' \
      | sed -E 's/[^a-z0-9]+/_/g; s/^_+|_+$//g; s/_+/_/g' \
      | sort -u > "$NOT_IN_BIRDNET_IDX"
  fi
}

is_in_not_in_birdnet() {
  local needle; needle="$(normalize_name "$1")"
  grep -Fxqi "$needle" "$NOT_IN_BIRDNET_IDX"
}

convert_loop() {
    echo "[STEP] Converting MP3 -> WAV into $WAV_ABS"
    : > bad_mp3.list
    
    while IFS= read -r -d '' f; do
        rel="${f#"$RAW_ABS/"}"
        out="$WAV_ABS/${rel%.mp3}.wav"
        mkdir -p "$(dirname "$out")"
        [[ -f "$out" ]] && continue
        
        if ! ffprobe -v error -select_streams a:0 -show_entries stream=codec_name -of default=nw=1:nk=1 "$f" >/dev/null; then
            echo "!! unreadable or no audio: $f" | tee -a bad_mp3.list
            continue
        fi
        
        printf 'CONVERT: %q -> %q\n' "$f" "$out"
        if ! ffmpeg -hide_banner -loglevel error -y -i "$f" -ac 1 -ar 48000 "$out"; then
            echo "!! convert failed: $f" | tee -a bad_mp3.list
        fi
    done < <(find "$RAW_ABS" -type f -name '*.mp3' -print0)
}

already_done_species() {
  local species_name="$1"
  [[ -d "$SEG/$species_name" ]]
}

# Returns species lines from SLIST that do NOT yet have a directory in $1
new_species_for_dir() {
    local base_dir="$1"
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        local expected_dir="${line// /_}"
        if [[ ! -d "$base_dir/$expected_dir" ]]; then
            echo "$line"
        fi
    done < "$SLIST"
}


# ===================== steps =====================
do_fetch() {
    if [[ "$SKIP_FETCH" == "1" ]]; then
        echo "[STEP] Skipping fetch (SKIP_FETCH=1)"; return
    fi

    # Build a temp species list of only NEW species (not yet in RAW)
    local new_slist
    new_slist="$(mktemp -t new_species.XXXXXX)"
    new_species_for_dir "$RAW_ABS" > "$new_slist"

    local new_count
    new_count="$(grep -c . "$new_slist" || true)"

    if [[ "$new_count" -eq 0 ]]; then
        echo "[STEP] All species already fetched -> skipping fetch"
        rm -f "$new_slist"
        return
    fi

    echo "[STEP] Fetching $new_count NEW species from Xeno-canto into $RAW_ABS"
    "$PY_BIN" "$ARIA_ROOT/src/fetch/xc_fetch.py" \
        --species_list "$new_slist" --out_raw "$RAW_ABS"
    find "$RAW_ABS" -type f -name '*.mp3' -size -8k -print -delete || true
    rm -f "$new_slist"
}

do_convert() {
    if [[ "$SKIP_CONVERT" == "1" ]]; then
        echo "[STEP] Skipping convert (SKIP_CONVERT=1)"; return
    fi
    mkdir -p "$WAV_ABS"
    : > bad_mp3.list

    shopt -s nullglob
    for species_dir in "$RAW_ABS"/*/; do
        [[ -d "$species_dir" ]] || continue
        local sp_name
        sp_name="$(basename "$species_dir")"
        local wav_sp_dir="$WAV_ABS/$sp_name"

        # Count MP3s in this species
        local mp3_c wav_c
        mp3_c="$(find "$species_dir" -maxdepth 1 -name '*.mp3' | wc -l)"
        wav_c=0
        if [[ -d "$wav_sp_dir" ]]; then
            wav_c="$(find "$wav_sp_dir" -maxdepth 1 -name '*.wav' | wc -l)"
        fi

        if [[ "$mp3_c" -le "$wav_c" ]]; then
            continue  # all converted already
        fi

        echo "  Converting $sp_name ($mp3_c MP3s, $wav_c WAVs)"
        mkdir -p "$wav_sp_dir"

        while IFS= read -r -d '' f; do
            local base
            base="$(basename "${f%.mp3}.wav")"
            local out="$wav_sp_dir/$base"
            [[ -f "$out" ]] && continue
            if ! ffprobe -v error -select_streams a:0 \
                 -show_entries stream=codec_name \
                 -of default=nw=1:nk=1 "$f" >/dev/null 2>&1; then
                echo "!! unreadable: $f" | tee -a bad_mp3.list
                continue
            fi
            printf 'CONVERT: %s\n' "$base"
            if ! ffmpeg -hide_banner -loglevel error -y \
                 -i "$f" -ac 1 -ar 48000 "$out"; then
                echo "!! failed: $f" | tee -a bad_mp3.list
            fi
        done < <(find "$species_dir" -maxdepth 1 -name '*.mp3' -print0)
    done
    shopt -u nullglob
}

do_analyze() {
  build_not_in_birdnet_index
  echo "[STEP] Running BirdNET analysis (known species only)"
  shopt -s nullglob
  for species_dir in "$WAV_ABS"/*; do
    [[ -d "$species_dir" ]] || continue
    species_name="$(basename "$species_dir")"
    if [[ -d "$SEG/$species_name" ]]; then
        echo "  Skipping $species_name (already completed)"
        continue
        fi

    if is_in_not_in_birdnet "$species_name"; then
      echo "  Skipping BirdNET for $species_name (NOT in BirdNET)"
      continue
    fi

    # FIXED: Check REANALYZE flag before skipping
    if [[ "$REANALYZE" != "1" ]] && [ -n "$(find "$species_dir" -maxdepth 1 -name '*.BirdNET.results.csv' -print -quit)" ]; then
      echo "  Skipping $species_name (already analyzed)"
      continue
    fi


    echo "  Analyzing $species_name (IN BirdNET)"
    "$PY_BIN" -m birdnet_analyzer.analyze "$species_dir" \
      -o "$species_dir" --slist "$SLIST" --rtype csv \
      --min_conf "$MIN_CONF_ANALYZE" --overlap "$OVERLAP" \
      --batch_size "$BATCH_SIZE" --threads "$T" --lat -1 --lon -1 --week -1
  done
  shopt -u nullglob
}

do_segment_unknown() {
  build_not_in_birdnet_index
  echo "[STEP] Segmenting NOT-in-BirdNET species with Perch (species mode) -> $SEG"

  mkdir -p "$SEG"
  shopt -s nullglob
  local matched=0

  for species_dir in "$WAV_ABS"/*; do
    [[ -d "$species_dir" ]] || continue
    species_name="$(basename "$species_dir")"

    # Skip already completed species
    done_marker="$SEG/$species_name/.perch_done"

    if [[ -f "$done_marker" ]]; then
        echo " Skipping $species_name (already completed)"
        continue
    fi

    is_in_not_in_birdnet "$species_name" || continue
    matched=$((matched+1))

    # Extract common name
    common_name=$(echo "$species_name" | cut -d'_' -f3-)
    common_name="${common_name//_/ }"

    
    
    #out_dir="$SEG/$common_name"
    out_dir="$SEG/$species_name"
    mkdir -p "$out_dir"

    # # ✅ SKIP if already segmented
    # if [[ -n "$(find "$out_dir" -name '*.wav' -print -quit)" ]]; then
    # echo "  [SKIP] $species_name already has segments"
    # continue
    # fi

    echo "  [PERCH:species] Processing: $species_name -> '$common_name'"

    CUDA_VISIBLE_DEVICES=0 "$PY_BIN" "$ARIA_ROOT/src/segment/perch_segment.py" \
        --wav_dir "$species_dir" \
        --out_dir "$out_dir" \
        --species_name "$species_name" \
        --ebird_indices "$EBIRD_INDICES" \
        --seg_len "$SEG_LEN_SECONDS" \
        --step "$STEP_SECONDS" \
        --threshold "$PERCH_THRESH" \
        --margin "$PERCH_MARGIN" \
        --gpu \
        --batch_size 16

    # Check if out_dir is empty (no *.wav at top level)
    if [ -z "$(find "$out_dir" -maxdepth 1 -name '*.wav' -print -quit)" ]; then
      echo "  [WARN] Perch yielded 0 segments for $species_name; falling back to BirdNET (general mode)"
      
      # Run BirdNET analyze without slist for general detections
      "$PY_BIN" -m birdnet_analyzer.analyze "$species_dir" \
        -o "$species_dir" --rtype csv \
        --min_conf 0.1 --overlap "$OVERLAP" \
        --batch_size "$BATCH_SIZE" --threads "$T" --lat -1 --lon -1 --week -1

      # Segment without slist (will create subdirectories)
      "$PY_BIN" -m birdnet_analyzer.segments "$species_dir" \
        -o "$out_dir" --min_conf 0.2 \
        --seg_length "$SEG_LEN_SECONDS" --threads "$T"
      
      # FLATTEN: Move all WAVs from subdirectories to parent
      echo "  [INFO] Flattening BirdNET output subdirectories..."
      for subdir in "$out_dir"/*; do
        if [[ -d "$subdir" ]]; then
          mv "$subdir"/*.wav "$out_dir/" 2>/dev/null || true
          rmdir "$subdir" 2>/dev/null || true
        fi
      done
    else
      echo "  [INFO] Perch succeeded for $species_name; skipping BirdNET fallback"
    fi

    if [[ -n "$(find "$out_dir" -maxdepth 1 -name '*.wav' -print -quit)" ]]; then
        touch "$done_marker"
        echo "  [DONE] Marked $species_name as completed"
    else
        echo "  [WARN] No segments created for $species_name; not marking completed"
    fi                  
  done

  shopt -u nullglob
  [[ $matched -gt 0 ]] || echo "  [WARN] No folders matched NOT_IN_BIRDNET"
}

do_segments() {
    echo "[STEP] Creating training segments -> $SEG"
    mkdir -p "$SEG"

    # 1) Unknown species via Perch
    do_segment_unknown

    # 2) Known species via BirdNET's native segmenter
    for species_dir in "$WAV_ABS"/*; do
        [[ -d "$species_dir" ]] || continue
        species_name="$(basename "$species_dir")"

        done_marker="$SEG/$species_name/.birdnet_done"

        if [[ -f "$done_marker" ]]; then
            echo " Skipping $species_name (already completed)"
            continue
        fi

        is_in_not_in_birdnet "$species_name" && continue

        if [[ -d "$SEG/$species_name" ]] && [[ -n "$(find "$SEG/$species_name" -name '*.wav' -print -quit)" ]]; then
            echo " Skipping $species_name (segments already exist)"
            touch "$done_marker"
            continue
        fi

        echo " Segmenting $species_name (IN BirdNET) via native tool"
        "$PY_BIN" -m birdnet_analyzer.segments "$species_dir" \
            -o "$SEG" --min_conf "$MIN_CONF_SEGMENTS" \
            --seg_length "$SEG_LEN_SECONDS" --threads "$T"

        mkdir -p "$SEG/$species_name"
        if [[ -n "$(find "$SEG/$species_name" -maxdepth 1 -name '*.wav' -print -quit)" ]]; then
            touch "$done_marker"
            echo " [DONE] Marked $species_name as completed"
        else
            echo " [WARN] No segments produced for $species_name; NOT marking completed (will retry next run)"
        fi
    done

    echo "=== SEGMENT COUNTS ==="
    for d in "$SEG"/*; do
        [[ -d "$d" ]] || continue
        c=$(find "$d" -name '*.wav' | wc -l | awk '{print $1}')
        printf ' %-40s %6d\n' "$(basename "$d")" "$c"
    done
}


# Main execution
case "$ONLY_STEP" in
    fetch) do_fetch ;;
    convert) do_convert ;;
    analyze) do_analyze ;;
    segments) do_segments ;;
    "")
        do_fetch
        do_convert
        do_analyze
        do_segments
        ;;
    *) echo "UNKNOWN ONLY_STEP='$ONLY_STEP' (use: fetch|convert|analyze|segments|filter|augment|train)"; exit 1 ;;
esac