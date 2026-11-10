#!/usr/bin/env python3
"""
Make the ARIA scripts path-independent (no /home/P70096368/... any more).
Put this file in the ARIA folder and run from there.

  python3 make_portable.py            # DRY RUN: shows what would change
  python3 make_portable.py --apply    # writes (backup *.bak_portable next to each file)

Each edit is an exact text match; if a file does not contain the expected text
the script reports it and changes NOTHING in that file.
"""
import sys
from pathlib import Path

ARIA = Path(__file__).resolve().parent
APPLY = '--apply' in sys.argv
OLD = '/home/P70096368/secondclone/zoostuff/checkpoints/ARIA'

EDITS = {
 'run_full_pipeline.sh': [
  (f'RESULTS_ROOT="{OLD}/results_inference_birdnet"',
   'ARIA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\nRESULTS_ROOT="$ARIA_ROOT/results_inference_birdnet"'),
  (f'RESULTS_DIR="{OLD}/results"', 'RESULTS_DIR="$ARIA_ROOT/results"'),
  (f'MODEL="{OLD}/models/ZooCustom_v1.tflite"', 'MODEL="$ARIA_ROOT/models/ZooCustom_v1.tflite"'),
  (f'RUN_INFERENCE_SCRIPT="{OLD}/src/inference/run_inference.py"', 'RUN_INFERENCE_SCRIPT="$ARIA_ROOT/src/inference/run_inference.py"'),
  (f'TWO_TIER_DIR="{OLD}/src/inference"', 'TWO_TIER_DIR="$ARIA_ROOT/src/inference"'),
 ],
 'run_existing_site.sh': [
  (f'MERGE_SCRIPT="{OLD}/src/inference/merge_model_predictions.py"', 'MERGE_SCRIPT="$SCRIPT_DIR/src/inference/merge_model_predictions.py"'),
 ],
 'lookup_new_perch_mappings.py': [
  (f'FETCH_DIR = Path("{OLD}/src/fetch")', 'ARIA = Path(__file__).resolve().parent\nFETCH_DIR = ARIA / "src" / "fetch"'),
  (f'REVIEW_FILE = Path("{OLD}/perch_mapping_review.json")', 'REVIEW_FILE = ARIA / "perch_mapping_review.json"'),
 ],
 'src/train/fusion/train_fusion_perchv2_no_leakage.py': [
  ("while ARIA_ROOT.name != 'ARIA' and ARIA_ROOT != ARIA_ROOT.parent:",
   "while not ((ARIA_ROOT / 'models').is_dir() and (ARIA_ROOT / 'config').is_dir()) and ARIA_ROOT != ARIA_ROOT.parent:"),
  ("if ARIA_ROOT.name != 'ARIA':", "if not (ARIA_ROOT / 'models').is_dir():"),
 ],
 'src/fetch_and_segment.sh': [
  ('BIRDNET_REPO="${BIRDNET_REPO:-$HOME/BirdNET-Analyzer}"',
   'BIRDNET_REPO="${BIRDNET_REPO:-$(python3 -c \'import birdnet_analyzer,os;print(os.path.dirname(os.path.dirname(birdnet_analyzer.__file__)))\' 2>/dev/null || echo "$HOME/BirdNET-Analyzer")}"'),
 ],
 'aviary_species_value_counts.py': [
  (f"sys.path.insert(0, '{OLD}/src/inference')",
   "import os as _os\nARIA = Path(__file__).resolve().parent\nsys.path.insert(0, str(ARIA / 'src' / 'inference'))"),
  ('ap.add_argument("--preprocessed-root", required=True)',
   'ap.add_argument("--preprocessed-root", default=_os.environ.get("PREPROCESSED_ROOT", "/home/zoodata/preprocessed"))'),
  ('ap.add_argument("--config-json", required=True)',
   'ap.add_argument("--config-json", default=str(ARIA / "config" / "zoo_species_config.json"))'),
 ],
}

problems = 0
for name, edits in EDITS.items():
    p = ARIA / name
    if not p.exists():
        print(f'[MISSING] {name} (not in {ARIA}) - skipped'); problems += 1; continue
    t = p.read_text(encoding='utf-8')
    if OLD not in t and all(new in t for _, new in edits):
        print(f'[OK-ALREADY] {name}'); continue
    new_t, bad = t, []
    for old, new in edits:
        if old in new_t: new_t = new_t.replace(old, new, 1)
        else: bad.append(old)
    if bad:
        problems += 1
        print(f'[NO MATCH] {name}: expected text not found, file untouched:')
        for b in bad: print('     ', b)
        continue
    left = new_t.count('P70096368')
    print(f'[{"WRITE" if APPLY else "WOULD CHANGE"}] {name}: {len(edits)} edit(s); remaining hardcoded refs: {left}')
    if APPLY:
        p.with_name(p.name + '.bak_portable').write_text(t, encoding='utf-8')
        p.write_text(new_t, encoding='utf-8')

print('\nAlso scan for anything else hardcoded:')
import os
SKIP = {'data', 'models', '_archive', '.git', '__pycache__', 'results', 'results_inference_birdnet'}
for root, dirs, files in os.walk(ARIA):
    dirs[:] = [d for d in dirs if d not in SKIP and not d.startswith('models_backup') and not d.startswith('data_')]
    for n in sorted(files):
        f = Path(root) / n
        if f.suffix in ('.py', '.sh') and '.bak' not in n and n != 'make_portable.py' \
           and 'P70096368' in f.read_text(errors='ignore'):
            print('  still hardcoded:', f.relative_to(ARIA))
print('\n' + ('DONE.' if APPLY else 'DRY RUN - re-run with --apply.') + (f'  {problems} problem(s).' if problems else ''))
