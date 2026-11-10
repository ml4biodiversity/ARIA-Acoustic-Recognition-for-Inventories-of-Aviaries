#!/usr/bin/env bash
# Mini retrain of BirdNET in a SCRATCH folder (real data/, models/, config/ are only read).
# Put in the ARIA folder, activate aria_unified, then:
#   bash smoke_retrain.sh                                  # default 4 species
#   bash smoke_retrain.sh "Cockatiel" "Mandarin Duck" ...  # your own species (common names)
# Env: N=80 clips per species, EPOCHS=3, OUT=~/smoke_train
set -Eeuo pipefail
ARIA="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N="${N:-80}"; EPOCHS="${EPOCHS:-3}"; OUT="${OUT:-$HOME/smoke_train}"
[ "$#" -gt 0 ] && SPECIES=("$@") || SPECIES=("Cockatiel" "Mandarin Duck" "Rose-ringed Parakeet" "Red-billed Blue-Magpie")
[ -e "$OUT" ] && { echo "[ABORT] $OUT exists - delete it first (rm -rf $OUT)"; exit 1; }
mkdir -p "$OUT/segments" "$OUT/models" "$OUT/nowav"
: > "$OUT/species_list.txt"

for sp in "${SPECIES[@]}"; do
  # all folders of this species (common-name folder AND Sci_Common folder), like the extractor groups them
  mapfile -t dirs < <(find "$ARIA/data/segments" -maxdepth 1 -mindepth 1 -type d \( -iname "*${sp// /_}*" -o -iname "*${sp}*" \) | sort)
  line="$(grep -i "_${sp}\$" "$ARIA/config/species_list.txt" | head -1 || true)"
  [ "${#dirs[@]}" -gt 0 ] && [ -n "$line" ] || { echo "[SKIP] $sp (no folder or no species_list line)"; continue; }
  name="$(basename "${dirs[0]}")"; for d in "${dirs[@]}"; do case "$(basename "$d")" in *_*) name="$(basename "$d")"; break;; esac; done
  mkdir -p "$OUT/segments/$name"
  for d in "${dirs[@]}"; do find "$d" -maxdepth 1 -name '*.wav' | head -"$N" | while read -r f; do ln -sf "$f" "$OUT/segments/$name/$(basename "$d")__$(basename "$f")"; done || true; done
  c="$(ls "$OUT/segments/$name" | wc -l)"
  if [ "$c" -lt 20 ]; then echo "[SKIP] $sp ($c clips only)"; rm -rf "$OUT/segments/$name"; continue; fi
  echo "$line" >> "$OUT/species_list.txt"
  echo "[USE] $sp -> $name ($c clips from ${#dirs[@]} folder(s))"
done
[ "$(wc -l < "$OUT/species_list.txt")" -ge 2 ] || { echo "[ABORT] need at least 2 species"; exit 1; }

STAMP="$OUT/real_files_before.txt"
ls -l --time-style=full-iso "$ARIA"/models/ZooCustom_v1.tflite "$ARIA"/models/ZooCustom_v1_Labels.txt > "$STAMP"

SLIST="$OUT/species_list.txt" SEG="$OUT/segments" WAV="$OUT/nowav" MAKE_NOISE=0 \
CURATED="$OUT/train_curated" TEST_OUT="$OUT/test_curated" CHK="$OUT/models" \
CLASSIFIER_NAME=SmokeTest EPOCHS="$EPOCHS" bash "$ARIA/src/train.sh" 2>&1 | tee "$OUT/train.log" | tail -25 || true

echo; echo "=== RESULT ==="
ls -l "$OUT/models/" | grep -i smoketest || true
[ -f "$OUT/models/SmokeTest.tflite" ] && echo "PASS: SmokeTest.tflite created" || echo "FAIL: no SmokeTest.tflite - see $OUT/train.log"
ls -l --time-style=full-iso "$ARIA"/models/ZooCustom_v1.tflite "$ARIA"/models/ZooCustom_v1_Labels.txt | diff -q - "$STAMP" >/dev/null \
  && echo "PASS: real models untouched" || echo "WARNING: real model files changed!"
