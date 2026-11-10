#!/usr/bin/env python3
"""
Run the existing embedding extraction in PARALLEL shards.

Splits the species folders of data/segments into N groups and runs
src/segment/extract_embeddings_better_working.py once per group (one process
each, own output file). Afterwards use merge_shards.py to join the pieces.

* Species folders that belong to the same species (e.g. "Mandarin Duck" and
  "Aix_galericulata_Mandarin_Duck") are grouped exactly like the extraction
  script groups them, so they get the same species label.
* A species that is bigger than one worker's share is split by FILES into
  several chunks (spread over several workers). Every chunk keeps the folder
  name the extractor would use, so the label stays identical.

Put this file in the ARIA folder (next to config/ and src/).

Usage (dry run is the default - nothing is created):
  python3 shard_extract.py --out-dir ~/shards_all --workers 7 --gpus 0,1,2,3,4,5,6
  python3 shard_extract.py --out-dir ~/shards_all --workers 7 --gpus 0,1,2,3,4,5,6 --apply

Test on a few species only (substring of the common name, comma separated):
  python3 shard_extract.py --out-dir ~/sh_test --workers 2 \
      --only "mandarin duck,cockatiel,rose-ringed parakeet" --apply

GPU options (TensorFlow grabs a whole GPU per process):
  --gpus none      force CPU for all workers (about 6x slower here)
  --gpus 0,1,2     worker i uses GPU ids[i % 3]

Re-run only failed shards:
  python3 shard_extract.py --out-dir ~/shards_all --apply --only-shards 3,5
"""
import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ARIA = Path(__file__).resolve().parent.parent
EXTRACT = ARIA / 'src' / 'segment' / 'extract_embeddings_better_working.py'


def normalize(name):
    # identical to extract_embeddings_better_working.py
    return (name.lower().replace("'", "").replace("’", "")
            .replace("-", " ").replace("_", " ").strip())


def group_species(root: Path):
    groups = {}
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        key = normalize(d.name.split("_", 2)[-1])   # same rule as the extractor
        groups.setdefault(key, []).append(d)
    return groups


def canonical_name(dirs):
    # same rule as the extractor: first folder with "_" in its name, else the first folder
    return next((d for d in dirs if "_" in d.name), dirs[0]).name


def fmt(sec):
    sec = int(sec)
    return f"{sec // 3600}h{(sec % 3600) // 60:02d}m{sec % 60:02d}s"


def build_plan(root, workers, only):
    groups = group_species(root)
    species = []
    for key, dirs in groups.items():
        if only and not any(f in key for f in only):
            continue
        files = sorted(str(f) for d in dirs for f in d.glob('*.wav'))
        if not files:
            continue
        marker = any((d / '.embeddings_done').exists() for d in dirs)
        species.append({'key': key, 'dirs': [str(d) for d in dirs], 'canon': canonical_name(dirs),
                        'files': files, 'marker': marker})
    total = sum(len(s['files']) for s in species)
    target = max(1, math.ceil(total / workers))

    # split species bigger than one worker's share into file chunks
    items = []
    for s in species:
        n = len(s['files'])
        if workers > 1 and n > target:
            k = math.ceil(n / target)
            step = math.ceil(n / k)
            for c in range(k):
                part = s['files'][c * step:(c + 1) * step]
                items.append({'key': s['key'], 'canon': s['canon'], 'dirs': s['dirs'], 'marker': s['marker'],
                              'wavs': len(part), 'files': part, 'chunk': f'{c + 1}/{k}'})
        else:
            items.append({'key': s['key'], 'canon': s['canon'], 'dirs': s['dirs'], 'marker': s['marker'],
                          'wavs': n, 'files': None, 'chunk': ''})

    # greedy balance: biggest first into the least loaded shard
    items.sort(key=lambda x: -x['wavs'])
    shards = [{'items': [], 'wavs': 0} for _ in range(workers)]
    for it in items:
        s = min(shards, key=lambda s: s['wavs'])
        s['items'].append(it)
        s['wavs'] += it['wavs']
    return [s for s in shards if s['items']], len(species)


def make_shard_dir(sd: Path, shard):
    sd.mkdir()
    for it in shard['items']:
        if it['files'] is None:                       # whole species: link the folders
            for d in it['dirs']:
                (sd / Path(d).name).symlink_to(d)
        else:                                         # file chunk: one folder with the canonical name
            cd = sd / it['canon']
            cd.mkdir(exist_ok=True)
            for f in it['files']:
                link = cd / Path(f).name
                n = 1
                while link.exists() or link.is_symlink():   # same file name in two folders
                    link = cd / f'{Path(f).stem}__dup{n}{Path(f).suffix}'
                    n += 1
                link.symlink_to(f)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--segments-root', default=str(ARIA / 'data' / 'segments'))
    ap.add_argument('--out-dir', required=True, help='folder for shard folders, shard_*.npy and logs')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--threads-per-worker', type=int, default=2,
                    help='CPU threads each process may use (default 2)')
    ap.add_argument('--only', default='', help='comma separated substrings of common names')
    ap.add_argument('--only-shards', default='', help='re-run only these shard numbers, e.g. 3,5')
    ap.add_argument('--gpus', default='',
                    help='GPU handling: empty = leave as is; "none" = force CPU; "0,1,2" = give workers '
                         'these GPUs round-robin')
    ap.add_argument('--python', default=sys.executable)
    ap.add_argument('--apply', action='store_true', help='really create shards and start the processes')
    a = ap.parse_args()

    root = Path(a.segments_root)
    out = Path(a.out_dir).expanduser().resolve()
    plan_file = out / 'plan.json'
    only = [normalize(s) for s in a.only.split(',') if s.strip()]

    if not EXTRACT.exists():
        sys.exit(f'[ABORT] not found: {EXTRACT}  (is this file inside the ARIA folder?)')

    if a.only_shards:
        if not plan_file.exists():
            sys.exit(f'[ABORT] {plan_file} not found - cannot re-run shards')
        shards = json.loads(plan_file.read_text())
        wanted = {int(x) for x in a.only_shards.split(',')}
        n_species = None
    else:
        if not root.is_dir():
            sys.exit(f'[ABORT] segments root not found: {root}')
        print(f'Scanning {root} ...')
        shards, n_species = build_plan(root, a.workers, only)
        wanted = set(range(len(shards)))

    total = sum(s['wavs'] for s in shards)
    print(f'\nPlan: {len(shards)} shard(s), {total} wav files' +
          (f', {n_species} species groups' if n_species else ''))
    for i, s in enumerate(shards):
        mark = '' if i in wanted else '   (not re-run)'
        print(f'  shard_{i}: {len(s["items"]):3d} item(s), {s["wavs"]:7d} wavs{mark}')
    big = sorted([it for s in shards for it in s['items'] if it['wavs'] >= 5000], key=lambda x: -x['wavs'])
    if big:
        print('\nLarge species (chunk = part of a species split over several workers):')
        for it in big[:14]:
            print(f'  {it["wavs"]:7d}  {it["canon"]}  {("chunk " + it["chunk"]) if it["chunk"] else ""}')
    skipped = sorted({it['key'] for s in shards for it in s['items'] if it['marker']})
    if skipped:
        print('\n[WARN] these species have a .embeddings_done marker and the extractor will SKIP them:')
        for k in skipped:
            print('   ', k)
        print('       delete the markers (find data/segments -name .embeddings_done -delete) if you want them included')

    if not a.apply:
        print('\nDRY RUN - nothing created. Add --apply to start.')
        return

    if not a.only_shards:
        if out.exists() and any(out.iterdir()):
            sys.exit(f'[ABORT] {out} is not empty (use a new --out-dir, or --only-shards to re-run)')
        out.mkdir(parents=True, exist_ok=True)
        print('\nCreating shard folders (links) ...')
        for i, s in enumerate(shards):
            make_shard_dir(out / f'shard_{i}', s)
        slim = [{'wavs': s['wavs'], 'items': [{k: v for k, v in it.items() if k != 'files'} for it in s['items']]}
                for s in shards]
        plan_file.write_text(json.dumps(slim, indent=1))
    else:
        for i in wanted:
            for f in (out / f'shard_{i}.npy', out / f'shard_{i}_BACKUP.npy'):
                if f.exists():
                    f.unlink()
                    print(f'removed old {f.name}')

    env = os.environ.copy()
    t = str(a.threads_per_worker)
    for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
              'TF_NUM_INTRAOP_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[k] = t

    procs = {}
    t0 = time.time()
    for i in sorted(wanted):
        log = open(out / f'shard_{i}.log', 'w')
        cmd = [a.python, str(EXTRACT), str(out / f'shard_{i}'), '-o', str(out / f'shard_{i}.npy')]
        wenv = dict(env)
        gpu_note = ''
        if a.gpus.strip().lower() == 'none':
            wenv['CUDA_VISIBLE_DEVICES'] = ''
            gpu_note = '  [CPU only]'
        elif a.gpus.strip():
            ids = [g.strip() for g in a.gpus.split(',') if g.strip()]
            wenv['CUDA_VISIBLE_DEVICES'] = ids[i % len(ids)]
            gpu_note = f'  [GPU {wenv["CUDA_VISIBLE_DEVICES"]}]'
        procs[i] = (subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=wenv), log, time.time())
        print(f'started shard_{i}{gpu_note}  (log: {out / f"shard_{i}.log"})')

    def stop(*_):
        print('\nstopping all shards ...')
        for p, _, _ in procs.values():
            p.terminate()
        sys.exit(130)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    done = {}
    last_print = 0
    while len(done) < len(procs):
        time.sleep(20)
        for i, (p, log, ts) in procs.items():
            if i not in done and p.poll() is not None:
                done[i] = p.returncode
                log.close()
                print(f'[{fmt(time.time() - t0)}] shard_{i} finished, exit code {p.returncode}, took {fmt(time.time() - ts)}')
        if len(done) < len(procs) and time.time() - last_print > 300:
            last_print = time.time()
            print(f'[{fmt(time.time() - t0)}] {len(done)}/{len(procs)} shards done', flush=True)

    bad = [i for i, rc in done.items() if rc != 0]
    print(f'\nTOTAL ELAPSED {fmt(time.time() - t0)}')
    if bad:
        print(f'[FAILED] shards {bad} - see their .log files, then re-run with --only-shards {",".join(map(str, bad))}')
        sys.exit(1)
    missing = [i for i in wanted if not (out / f'shard_{i}.npy').exists()]
    if missing:
        print(f'[WARN] no output file for shards {missing} (nothing extracted?) - check logs')
        sys.exit(1)
    print(f'All shards OK. Next:\n  python3 scripts/merge_shards.py {out} -o <merged.npy> --labels models/ZooCustom_v1_Labels.txt')


if __name__ == '__main__':
    main()
