#!/usr/bin/env python3
"""
For every dataset under preprocessed_root that has a *_meta.xlsx, print a
value_counts-style breakdown of ONLY the aviary/cage species (excluding
wild birds) across birdnet_prediction, perch_prediction, and
fusion_model_prediction columns.

Usage:
  python3 aviary_species_value_counts.py \
    --preprocessed-root /home/zoodata/preprocessed \
    --config-json config/zoo_species_config.json
"""
import argparse
import json
import re
from pathlib import Path
from collections import Counter
import pandas as pd
import sys

import os as _os
ARIA = Path(__file__).resolve().parent
sys.path.insert(0, str(ARIA / 'src' / 'inference'))
from run_inference import parse_folder_name, load_zoo_config, DEFAULT_CONFIG_JSON


def get_cage_birds(config, zoo_name, cage_name):
    # Normalize zoo-name aliases returned by parse_folder_name() to the
    # actual keys used in zoo_species_config.json's "zoos" section.
    config_zoo_key = zoo_name
    if zoo_name in ('cologne', 'cologne_zoo'):
        config_zoo_key = 'cologne'
    elif zoo_name == 'beeksebergen':
        config_zoo_key = 'beekse_bergen'

    if config_zoo_key not in config['zoos']:
        return set()
    zoo_data = config['zoos'][config_zoo_key]
    if cage_name and cage_name in zoo_data.get('cages', {}):
        return set(zoo_data['cages'][cage_name])
    return set(zoo_data.get('all_birds', []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preprocessed-root", default=_os.environ.get("PREPROCESSED_ROOT", "/home/zoodata/preprocessed"))
    ap.add_argument("--config-json", default=str(ARIA / "config" / "zoo_species_config.json"))
    ap.add_argument("--datasets", nargs="*", default=None,
                     help="Optional: only check these specific dataset names instead of scanning everything")
    args = ap.parse_args()

    with open(args.config_json, "r", encoding="utf-8") as f:
        config = json.load(f)
    wild_birds = set(config.get('wild_birds', []))

    root = Path(args.preprocessed_root)
    dataset_dirs = sorted([d for d in root.iterdir() if d.is_dir()])

    if args.datasets:
        dataset_dirs = [d for d in dataset_dirs if d.name in set(args.datasets)]
    for ds_dir in dataset_dirs:
        name = ds_dir.name
        meta_files = list(ds_dir.glob("*_meta.xlsx"))
        if not meta_files:
            continue

        zoo_name, cage_name = parse_folder_name(name, config)
        expected_cage_species = (get_cage_birds(config, zoo_name, cage_name) - wild_birds) if zoo_name else set()

        if not expected_cage_species:
            continue

        try:
            df = pd.read_excel(meta_files[0])
        except Exception as e:
            print(f"[ERR] Could not read {meta_files[0]}: {e}")
            continue

        print("\n" + "=" * 90)
        print(f"[DATASET] {name}")
        print(f"Expected aviary species: {sorted(expected_cage_species)}")
        print("=" * 90)

        for col in ['birdnet_prediction', 'perch_prediction', 'fusion_model_prediction']:
            if col not in df.columns:
                print(f"--- {col}: column not found ---")
                continue

            counts = Counter()
            for cell in df[col].astype(str):
                for sp in expected_cage_species:
                    if sp.lower() in cell.lower():
                        counts[sp] += cell.lower().count(sp.lower())

            print(f"\n--- {col} ---")
            if counts:
                for sp, c in counts.most_common():
                    print(f"    {sp:<40s} {c}")
            else:
                print("    (no aviary species found)")


if __name__ == "__main__":
    main()