#!/usr/bin/env python3
"""
Tidy the ARIA folder for GitHub. MOVES (never deletes) one-off files into _archive/.
Put in the ARIA folder.   python3 archive_unused.py          # dry run
                          python3 archive_unused.py --apply  # move + write .gitignore
Safety: a file is NOT moved if any file that stays refers to its name.
"""
import fnmatch, os, shutil, sys
from pathlib import Path
ARIA = Path(__file__).resolve().parent
APPLY = '--apply' in sys.argv
ARCH = ARIA / '_archive'
PATTERNS = ['audit_*.py', 'batch_*.py', 'check_*.py', 'find_*.py', 'fix_*.py', 'verify_*.py', 'scan_*.py',
            'spot_check_*.py', 'finalize_rerun_scope.py', 'dedup_perch_mapping.py', 'add_folder_patterns.py',
            'rename_beeksebergen_cage_keys.py', 'quick_test.sh', 'run_birdnet_safe.sh', 'thread_count_test.sh',
            'src/inference/add_to_db.sh', 'src/inference/batch_fix_birdnet.sh', 'src/inference/copy_prediction_excels.sh',
            'src/inference/make_aviary_meta_local.sh', '*.bak_portable',
            'patch_*.py', '*.bak', '*.bak[0-9]', '*.bak_*', '*.log', '*-old.py', '*_old.py', 'bad_mp3.list',
            'overnight_retrain.sh', 'bare_analyze_workaround.sh', 'not_in_birdnet_check.txt',
            'fetch_gladbach*', '*_sept2026_species.txt', 'gl_aviary*.txt', 'perch_mapping_review.json',
            'src/inference/*_meta.xlsx', 'src/bad_mp3.list', 'src/fetch/*.bak*',
            'src/inference/run_inference.py.bak*', 'src/inference/*.bak*', 'src/segment/*.bak*']
SKIP_DIRS = {'data', 'models', '_archive', '.git', '__pycache__', 'results', 'results_inference_birdnet'}
def walk():
    for root, dirs, files in os.walk(ARIA):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith('models_backup') and not d.startswith('data_')]
        for n in files: yield Path(root) / n
def cands():
    for p in sorted(walk()):
        rel = p.relative_to(ARIA)
        if not p.is_file() or rel.parts[0] in SKIP_DIRS or rel.parts[0].startswith('models_backup'): continue
        if p.name in ('archive_unused.py', 'make_portable.py', 'check_species_in_model.py'): continue
        r = rel.as_posix()
        if any(fnmatch.fnmatch(r, g) or fnmatch.fnmatch(p.name, g) for g in PATTERNS): yield p
move = list(cands()); mset = set(move)
keep_text = ''
for p in walk():
    rel = p.relative_to(ARIA)
    if p.is_file() and p.name not in ('archive_unused.py', 'make_portable.py') and p.suffix in ('.py', '.sh') and p not in mset and rel.parts[0] not in SKIP_DIRS and not rel.parts[0].startswith('models_backup'):
        keep_text += p.read_text(errors='ignore')
final = []
for p in move:
    if p.name in keep_text and p.suffix in ('.py', '.sh', '.txt', '.json'):
        print(f'  KEEP (still referenced): {p.relative_to(ARIA)}')
    else: final.append(p)
print(f'\n{len(final)} file(s) would move to _archive/:')
for p in final: print('  ', p.relative_to(ARIA))
if APPLY:
    for p in final:
        d = ARCH / p.relative_to(ARIA); d.parent.mkdir(parents=True, exist_ok=True); shutil.move(str(p), d)
    gi = ARIA / '.gitignore'
    if not gi.exists():
        gi.write_text('data/\nmodels/*\n!models/*.py\nmodels_backup*/\nresults/\nresults_inference_birdnet/\n_archive/\n'
                      '*.npy\n*.tflite\n*.pth\n*.wav\n*.mp3\n*.xlsx\n*.log\n*.bak*\n__pycache__/\n')
        print('wrote .gitignore')
    print('DONE.')
else: print('\nDRY RUN - add --apply.')
