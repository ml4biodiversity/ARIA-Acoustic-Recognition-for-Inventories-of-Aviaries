#!/usr/bin/env python3
"""
Generic: register a NEW zoo / cage set so a folder name routes correctly.
Only needed for a new zoo or new cage names - NOT for new recordings of an
existing, already-configured aviary.

DRY RUN by default (prints what it would do). Add --apply to write.
Backups are made automatically.

Examples:
  # new zoo key, species copied from an existing zoo, two cages
  python3 add_zoo_config.py --zoo gaia_savannah --copy-from gaia_zoo_savannah \
      --cages first_location second_location --register-pattern --apply

  # new zoo with its own species file (one "Scientific name_Common name" per line)
  python3 add_zoo_config.py --zoo mochengladbach --species-file gladbach_aviary2.txt \
      --cages aviary2 kittas --register-pattern --apply

  # different species per cage + own routing text per cage
  python3 add_zoo_config.py --zoo mochengladbach --cages aviary1 aviary2 \
      --cage-species aviary1=a1.txt aviary2=a2.txt \
      --cage-pattern aviary1=kittas aviary2=sept2026 --register-pattern

--cages: cage keys. The folder-name substring used for routing is the cage key
         itself (e.g. cage 'kittas' matches folders containing 'kittas'),
         unless you give --cage-pattern CAGE=TEXT. Order matters if one folder
         name contains several patterns: list the most specific cage first.
--cage-species: CAGE=FILE (one species per line, '#' lines ignored) instead of
         --species-file; all_birds becomes the union of the cages.
--register-pattern: also adds the zoo key to zoo_patterns in run_inference.py
         (needed when the zoo key is not already in that list).
"""
import argparse, json, shutil, sys
from datetime import datetime
from pathlib import Path

ARIA = Path(__file__).resolve().parent   # script lives in the ARIA folder


def backup(p: Path):
    dst = p.with_name(p.name + f'.bak_{datetime.now():%Y%m%d_%H%M%S}')
    shutil.copy2(p, dst)
    print(f'  backup: {dst}')


def parse_pairs(items, what):
    out = {}
    for it in items or []:
        if '=' not in it:
            sys.exit(f'{what}: expected CAGE=VALUE, got "{it}"')
        k, v = it.split('=', 1)
        out[k.strip()] = v.strip()
    return out


def read_list(path):
    return [l.strip() for l in open(path, encoding='utf-8')
            if l.strip() and not l.strip().startswith('#')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zoo', required=True)
    ap.add_argument('--copy-from')
    ap.add_argument('--species-file')
    ap.add_argument('--cages', nargs='+', required=True)
    ap.add_argument('--cage-species', nargs='+', metavar='CAGE=FILE',
                    help='species file per cage, e.g. aviary1=a1.txt aviary2=a2.txt (instead of --species-file)')
    ap.add_argument('--cage-pattern', nargs='+', metavar='CAGE=TEXT',
                    help='folder-name text that routes to a cage (default: the cage key itself)')
    ap.add_argument('--register-pattern', action='store_true')
    ap.add_argument('--after', default='gaia_zoo_savannah',
                    help="existing zoo_patterns entry to insert after (default gaia_zoo_savannah)")
    ap.add_argument('--config', default=str(ARIA / 'config' / 'zoo_species_config.json'))
    ap.add_argument('--run-inference', default=str(ARIA / 'src' / 'inference' / 'run_inference.py'))
    ap.add_argument('--overwrite', action='store_true',
                    help='replace an existing zoo block + folder_patterns entry (use to fix/redo)')
    ap.add_argument('--apply', action='store_true')
    a = ap.parse_args()

    if sum([bool(a.copy_from), bool(a.species_file), bool(a.cage_species)]) != 1:
        sys.exit('give exactly one of --copy-from, --species-file or --cage-species')
    cage_files = parse_pairs(a.cage_species, '--cage-species')
    cage_pat = parse_pairs(a.cage_pattern, '--cage-pattern')
    bad = [c for c in list(cage_files) + list(cage_pat) if c not in a.cages]
    if bad:
        sys.exit(f'cage(s) {bad} used in --cage-species/--cage-pattern but not listed in --cages')

    cfg_path, ri_path = Path(a.config), Path(a.run_inference)
    cfg = json.load(open(cfg_path, encoding='utf-8'))

    if a.zoo in cfg['zoos'] and not a.overwrite:
        sys.exit(f'zoo "{a.zoo}" already exists - use --overwrite to replace it')
    if cage_files:
        missing = [c for c in a.cages if c not in cage_files]
        if missing:
            sys.exit(f'--cage-species needs a file for every cage, missing: {missing}')
        cage_lists = {c: read_list(cage_files[c]) for c in a.cages}
        birds = []
        for c in a.cages:
            for b in cage_lists[c]:
                if b not in birds:
                    birds.append(b)
    else:
        if a.copy_from:
            if a.copy_from not in cfg['zoos']:
                sys.exit(f'--copy-from "{a.copy_from}" not in config zoos')
            birds = list(cfg['zoos'][a.copy_from].get('all_birds', []))
        else:
            birds = read_list(a.species_file)
        cage_lists = {c: list(birds) for c in a.cages}

    print(f'zoo={a.zoo}: {len(birds)} species in total')
    for c in a.cages:
        print(f'  cage {c:12s} routes on folder text "{cage_pat.get(c, c)}": {len(cage_lists[c])} species')
        for b in cage_lists[c]:
            print(f'      - {b}')
    cfg['zoos'][a.zoo] = {'all_birds': birds, 'cages': cage_lists}
    cfg.setdefault('folder_patterns', {})[a.zoo] = {'cages': {c: [cage_pat.get(c, c)] for c in a.cages}}

    ri_text = ri_path.read_text(encoding='utf-8')
    new_ri = None
    if a.register_pattern:
        if f"'{a.zoo}'" in ri_text:
            print('zoo_patterns already contains it - skipping code edit')
        else:
            marker = f"        '{a.after}',\n"
            if marker not in ri_text:
                sys.exit(f"marker '{a.after}' not found in run_inference.py - pick another with --after")
            new_ri = ri_text.replace(marker, marker + f"        '{a.zoo}',\n", 1)
            print(f"will add '{a.zoo}' to zoo_patterns after '{a.after}'")

    if not a.apply:
        print('\nDRY RUN - nothing written. Re-run with --apply.')
        return
    backup(cfg_path)
    json.dump(cfg, open(cfg_path, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    if new_ri:
        backup(ri_path)
        ri_path.write_text(new_ri, encoding='utf-8')
    print('written.')


if __name__ == '__main__':
    main()
