#!/usr/bin/env python3
"""
Merge the shard_*.npy files made by shard_extract.py into ONE embeddings file
in the same format that extract_embeddings_better_working.py writes.

Species indices are rebuilt consistently (sorted species names), so the
shards can use different indices internally.

Checks printed at the end: total rows == sum of shards, embedding width,
species count, optional width check against the BirdNET labels file.

Usage:
  python3 merge_shards.py ~/shards_all -o ~/shards_all/merged.npy \
      --labels models/ZooCustom_v1_Labels.txt
  # compare two embedding files row by row (e.g. parallel vs single run):
  python3 merge_shards.py ~/shard_test -o ~/shard_test/merged.npy --compare ~/single_test/merged.npy

Put this file in the ARIA folder. It never overwrites an existing output
unless you pass --force (then a _BACKUP copy is made first).
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def load(p):
    return np.load(p, allow_pickle=True).item()


def row_keys(d):
    seg = d['segment_info']
    names = d['species_names']
    return [(names[i], seg[i]['file'], round(float(seg[i]['start_time']), 3)) for i in range(len(seg))]


def compare(a, b, label_a='merged', label_b='other'):
    ka, kb = row_keys(a), row_keys(b)
    idx_b = {k: i for i, k in enumerate(kb)}
    ia, ib = [], []
    for i, k in enumerate(ka):
        j = idx_b.get(k)
        if j is not None:
            ia.append(i)
            ib.append(j)
    print(f'\nCOMPARE {label_a} ({len(ka)} rows) vs {label_b} ({len(kb)} rows)')
    print(f'  rows found in both : {len(ia)}')
    print(f'  only in {label_a:7s}: {len(ka) - len(ia)}')
    print(f'  only in {label_b:7s}: {len(kb) - len(ia)}')
    if ia:
        diff = np.abs(a['embeddings'][ia] - b['embeddings'][ib])
        print(f'  max abs difference : {diff.max():.3e}')
        print(f'  rows differing >1e-4: {int((diff.max(axis=1) > 1e-4).sum())}')
    ok = len(ia) == len(ka) == len(kb) and (not ia or diff.max() < 1e-4)
    print('  RESULT:', 'IDENTICAL (within 1e-4)' if ok else 'DIFFERENT - do not use')
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('shards_dir')
    ap.add_argument('-o', '--output', required=True)
    ap.add_argument('--labels', default=None, help='BirdNET labels file to check the width against')
    ap.add_argument('--compare', default=None, help='another embeddings .npy to compare with the merged result')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()

    sdir = Path(a.shards_dir).expanduser()
    out = Path(a.output).expanduser()
    files = sorted(sdir.glob('shard_*.npy'), key=lambda p: int(p.stem.split('_')[1]) if p.stem.split('_')[1].isdigit() else 0)
    files = [f for f in files if not f.stem.endswith('_BACKUP')]
    if not files:
        sys.exit(f'[ABORT] no shard_*.npy in {sdir}')

    plan = sdir / 'plan.json'
    if plan.exists():
        n_plan = len(json.loads(plan.read_text()))
        if n_plan != len(files):
            sys.exit(f'[ABORT] plan has {n_plan} shards but {len(files)} shard files found - a shard is missing/failed')

    if out.exists() and not a.force:
        sys.exit(f'[ABORT] {out} exists (use --force to overwrite; a backup is made)')

    parts = []
    for f in files:
        d = load(f)
        n = len(d['embeddings'])
        if not (n == len(d['labels']) == len(d['species_names']) == len(d['filenames']) == len(d['segment_info'])):
            sys.exit(f'[ABORT] {f.name}: inconsistent lengths inside the file')
        parts.append((f, d))
        print(f'  {f.name}: {n} rows, width {d["embeddings"].shape[1]}, {len(set(d["species_names"]))} species')

    widths = {d['embeddings'].shape[1] for _, d in parts}
    if len(widths) != 1:
        sys.exit(f'[ABORT] shards have different embedding widths: {widths}')
    width = widths.pop()
    meta0 = dict(parts[0][1].get('metadata', {}))
    for f, d in parts:
        m = d.get('metadata', {})
        for k in ('birdnet_dim', 'perch_dim', 'total_dim'):
            if m.get(k) != meta0.get(k):
                sys.exit(f'[ABORT] {f.name}: metadata {k} differs ({m.get(k)} vs {meta0.get(k)})')

    # species -> shards (a species must be in one shard only)
    seen = {}
    for f, d in parts:
        for s in set(d['species_names']):
            seen.setdefault(s, []).append(f.name)
    dup = {s: v for s, v in seen.items() if len(v) > 1}
    if dup:
        print(f'[INFO] {len(dup)} species are split over several shards (expected for very large species): '
              f'{[(s, len(v)) for s, v in list(dup.items())[:6]]}')

    names_sorted = sorted(seen)
    species_to_idx = {n: i for i, n in enumerate(names_sorted)}
    idx_to_species = {i: n for n, i in species_to_idx.items()}

    emb = np.concatenate([d['embeddings'] for _, d in parts]).astype(np.float32)
    species_names, filenames, seginfo, failed, nosegs = [], [], [], [], []
    for _, d in parts:
        species_names += list(d['species_names'])
        filenames += list(d['filenames'])
        seginfo += list(d['segment_info'])
        failed += list(d.get('failed_files', []))
        nosegs += list(d.get('files_with_no_segments', []))
    labels = np.array([species_to_idx[n] for n in species_names])

    meta = meta0
    meta['total_segments'] = len(emb)
    merged = {
        'embeddings': emb, 'labels': labels, 'species_names': species_names,
        'filenames': filenames, 'segment_info': seginfo,
        'species_to_idx': species_to_idx, 'idx_to_species': idx_to_species,
        'failed_files': failed, 'files_with_no_segments': nosegs, 'metadata': meta,
    }

    expected_rows = sum(len(d['embeddings']) for _, d in parts)
    assert len(emb) == expected_rows == len(labels) == len(species_names)

    if out.exists() and a.force:
        shutil.copy(out, out.with_name(out.stem + '_BACKUP.npy'))
        print(f'backup: {out.stem}_BACKUP.npy')
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, merged)

    chk = load(out)
    print('\nMERGED')
    print(f'  output          : {out}')
    print(f'  rows            : {len(chk["embeddings"])}  (sum of shards: {expected_rows})')
    print(f'  width           : {chk["embeddings"].shape[1]}   (birdnet {meta.get("birdnet_dim")} + perch {meta.get("perch_dim")})')
    print(f'  species         : {len(chk["species_to_idx"])}')
    print(f'  failed files    : {len(failed)}   files without segments: {len(nosegs)}')
    ok = len(chk['embeddings']) == expected_rows
    if a.labels:
        n_lab = sum(1 for line in open(a.labels, encoding='utf-8') if line.strip())
        good = (meta.get('birdnet_dim') == n_lab)
        print(f'  labels file     : {n_lab} labels -> birdnet width {"OK" if good else "MISMATCH"}')
        ok = ok and good
    if a.compare:
        ok = compare(chk, load(Path(a.compare).expanduser()), 'merged', 'other') and ok
    print('\nALL CHECKS PASSED' if ok else '\nCHECKS FAILED - do not use this file')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
