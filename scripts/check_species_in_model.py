#!/usr/bin/env python3
"""
Generic pre-flight check - no editing needed, just pass folder names.

For each folder it prints:
  - which zoo / cage parse_folder_name() resolves it to (None = routing broken)
  - how many cage species the config lists
  - which of those species are NOT in the BirdNET model labels (will show 0 hits)
  - name-mismatch hints (same scientific name, different common name, or vice versa)

Usage:
  python3 check_species_in_model.py fl_gaia_taiga_sept2026_first_location fl_zoo_eindhoven_large_aviary_mar2025
  python3 check_species_in_model.py --all-pending      # every folder in PREPROCESSED_ROOT without a PERCH result yet
"""
import argparse, json, os, sys
from pathlib import Path

ARIA = Path(__file__).resolve().parent.parent   # scripts/ is inside the ARIA folder
sys.path.insert(0, str(ARIA / 'src' / 'inference'))
from run_inference import parse_folder_name, get_cage_birds, load_zoo_config, DEFAULT_CONFIG_JSON


def split(label):
    sci, _, com = label.partition('_')
    return sci.strip().lower(), com.strip().lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('folders', nargs='*')
    ap.add_argument('--all-pending', action='store_true')
    ap.add_argument('--labels', default=str(ARIA / 'models' / 'ZooCustom_v1_Labels.txt'))
    ap.add_argument('--root', default=os.environ.get('PREPROCESSED_ROOT', '/home/zoodata/preprocessed_new'))
    args = ap.parse_args()

    folders = list(args.folders)
    if args.all_pending:
        for d in sorted(Path(args.root).iterdir()):
            if d.is_dir() and d.name != 'meta' and not (ARIA / 'results' / f'{d.name}_PERCH_predictions.csv').exists():
                folders.append(d.name)
    if not folders:
        ap.error('give folder names or --all-pending')

    config = load_zoo_config(DEFAULT_CONFIG_JSON)
    wild = set(config.get('wild_birds', []))
    labels = [l.strip() for l in open(args.labels, encoding='utf-8') if l.strip()]
    lab_set = set(labels)
    lab_sci = {split(l)[0]: l for l in labels}
    lab_com = {split(l)[1]: l for l in labels}

    for name in folders:
        zoo, cage = parse_folder_name(name, config)
        birds = (get_cage_birds(config, zoo, cage) - wild) if zoo else set()
        print('=' * 78)
        print(f'{name}\n  -> zoo={zoo} cage={cage}  ({len(birds)} cage species)')
        if not zoo or not birds:
            print('  !! ROUTING BROKEN - fix config / zoo_patterns before running inference')
            continue
        # config entries may be 'Common name' or 'Scientific_Common name'
        def in_model(b):
            if b in lab_set:
                return True
            sci, com = split(b)
            if '_' not in b:
                return b.lower() in lab_com
            return sci in lab_sci and com in lab_com
        missing = sorted(b for b in birds if not in_model(b))
        if not missing:
            print('  OK: all cage species are in the model')
            continue
        print(f'  NOT in model labels ({len(missing)}):')
        for b in missing:
            sci, com = split(b)
            hint = ''
            if '_' in b and sci in lab_sci:
                hint = f'  <- name mismatch? model has "{lab_sci[sci]}"'
            elif '_' in b and com in lab_com:
                hint = f'  <- name mismatch? model has "{lab_com[com]}"'
            print(f'    {b}{hint}')


if __name__ == '__main__':
    main()
