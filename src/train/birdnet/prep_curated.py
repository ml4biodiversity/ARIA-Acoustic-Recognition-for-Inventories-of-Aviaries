import argparse, os, sys, shutil, random, re, unicodedata, csv
from pathlib import Path

# ----------------------- utils -----------------------

def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def read_list(list_path: Path):
    labels = []
    with open(list_path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#"): continue
            if "_" not in line:
                print(f"[WARN] Skipping invalid line (no underscore): {line}")
                continue
            labels.append(line)
    return labels

# def build_source_index(seg_root: Path):
#     idx = {}
#     for d in seg_root.iterdir():
#         if d.is_dir():
#             idx[norm(d.name)] = d
#     return idx

# def choose_source_dir(seg_root: Path, label: str, src_index: dict):
#     # prefer exact Scientific_Common folder
#     sci_com = label
#     if (seg_root / sci_com).is_dir():
#         return seg_root / sci_com
#     # otherwise try matching by common name only
#     common = label.split("_",1)[1]
#     key = norm(common)
#     return src_index.get(key, None)
def build_source_index(seg_root: Path):
    """Index every folder by multiple possible keys: its full name,
    and (if it looks like Scientific_Common) just the common-name portion.
    Each key maps to a LIST of candidate folders, since duplicates can
    exist (e.g. a Scientific_Common folder and a separate bare-name
    folder for the same species)."""
    idx = {}

    def add(key, d):
        idx.setdefault(key, []).append(d)

    for d in seg_root.iterdir():
        if not d.is_dir():
            continue
        add(norm(d.name), d)
        parts = d.name.split("_")
        if len(parts) >= 3:
            add(norm(" ".join(parts[2:])), d)
        if len(parts) >= 2:
            add(norm(" ".join(parts[1:])), d)
    return idx


def _wav_count(d: Path) -> int:
    return sum(1 for _ in d.glob("*.wav"))


def choose_source_dir(seg_root: Path, label: str, src_index: dict):
    # prefer exact Scientific_Common folder
    sci_com = label
    if (seg_root / sci_com).is_dir():
        return seg_root / sci_com

    # otherwise try matching by common name only - if multiple
    # candidates exist (duplicate folders), pick the one with the
    # most real .wav files, not just whichever came first.
    common = label.split("_", 1)[1]
    key = norm(common)
    candidates = src_index.get(key, [])
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=_wav_count)

def list_wavs(p: Path):
    return sorted(x for x in p.rglob("*.wav"))

def write_summary_csv(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label","used_for_train","used_for_test","excluded_reason"])
        w.writerows(rows)

# ----------------- recording-wise split -----------------
# Remove typical "segment markers" appended by pipelines: _seg0001, _clip12, _start12.3, _offset3.0s, etc.
SEG_MARKERS = re.compile(r'(?:^|[_-])(seg|clip|chunk|part|start|offset|t)\d+(?:\.\d+)?s?', re.I)
TRAILING_NUM  = re.compile(r'([_-])\d+(?:\.\d+)?s?$', re.I)

def recording_key(path: Path) -> str:
    """Heuristic key for the original recording this 3s clip came from."""
    s = path.stem
    s = SEG_MARKERS.sub('', s)
    s = TRAILING_NUM.sub('', s)
    s = s.strip('_-')
    return s or path.stem

def group_by_recording(files):
    groups = {}
    for f in files:
        k = recording_key(f)
        groups.setdefault(k, []).append(f)
    return groups  # {key: [files_from_that_recording]}

def truncate_by_groups(group_items, max_files: int):
    """Keep whole groups until reaching <= max_files (may end below the cap)."""
    if max_files <= 0: return group_items
    kept = []
    total = 0
    for k, lst in group_items:
        n = len(lst)
        if total + n > max_files:
            continue  # skip to preserve group integrity
        kept.append((k, lst))
        total += n
    return kept

def split_groups(group_items, test_frac: float):
    """Assign whole groups to test until ≈ test_frac of clips reached."""
    total = sum(len(lst) for _, lst in group_items)
    if total == 0:
        return set()
    target = int(round(test_frac * total))
    test_keys, test_count = set(), 0
    for k, lst in group_items:
        if test_count < target:
            test_keys.add(k)
            test_count += len(lst)
        else:
            break
    # Edge case: ensure non-empty train or test if possible
    if 0 < test_count < total:
        return test_keys
    
    # If everything fell in one side (e.g., a single group), force at least one group into the smaller side
    if test_count == 0 and group_items:
        test_keys.add(group_items[0][0])
    return test_keys

# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segments_root", required=True)
    ap.add_argument("--list_file", required=True)
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_test", required=True)
    ap.add_argument("--min_files", type=int, default=100)
    ap.add_argument("--max_files", type=int, default=2000)
    ap.add_argument("--test_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    seg_root = Path(args.segments_root)
    out_train = Path(args.out_train); out_train.mkdir(parents=True, exist_ok=True)
    out_test  = Path(args.out_test);  out_test.mkdir(parents=True, exist_ok=True)

    labels = read_list(Path(args.list_file))
    src_index = build_source_index(seg_root)

    summary_rows=[]
    excluded_path = out_train.parent / "excluded_classes.txt"
    excl = []

    for label in labels:
        src_dir = choose_source_dir(seg_root, label, src_index)
        if not src_dir or not src_dir.exists():
            excl.append(f"{label}\tMISSING_SOURCE_DIR")
            summary_rows.append([label, 0, 0, "missing_source"])
            continue

        wavs = list_wavs(src_dir)
        n_raw = len(wavs)
        if n_raw < args.min_files:
            excl.append(f"{label}\tTOO_FEW({n_raw}<{args.min_files})")
            summary_rows.append([label, 0, 0, f"too_few({n_raw})"])
            continue

        # ---- recording-wise grouping ----
        groups = group_by_recording(wavs)
        group_items = list(groups.items())
        random.shuffle(group_items)

        # ---- optional cap (by groups) ----
        if args.max_files > 0:
            group_items = truncate_by_groups(group_items, args.max_files)

        n_after = sum(len(lst) for _, lst in group_items)
        if n_after < args.min_files:
            excl.append(f"{label}\tTOO_FEW_AFTER_CAP({n_after}<{args.min_files})")
            summary_rows.append([label, 0, 0, f"too_few_after_cap({n_after})"])
            continue

        # ---- leak-free split (by groups) ----
        test_keys = split_groups(group_items, args.test_frac)
        train_count = sum(len(lst) for k, lst in group_items if k not in test_keys)
        if train_count == 0 and group_items:
            # move the largest TEST group back to TRAIN
            largest = max(((k, lst) for k, lst in group_items if k in test_keys),
                        key=lambda kv: len(kv[1]),
                        default=None)
            if largest:
                test_keys.discard(largest[0])

        train_dir = out_train / label
        test_dir  = out_test  / label
        train_dir.mkdir(parents=True, exist_ok=True)
        test_dir.mkdir(parents=True, exist_ok=True)

        used_train=used_test=0
        for k, lst in group_items:
            dst_base = test_dir if k in test_keys else train_dir
            for src in lst:
                dst = dst_base / src.name
                try:
                    os.symlink(os.path.abspath(src), dst)
                except FileExistsError:
                    continue
                except Exception:
                    try:
                        shutil.copy2(src, dst)
                    except (shutil.SameFileError, FileExistsError):
                        continue
                if dst_base is test_dir: used_test += 1
                else: used_train += 1

        summary_rows.append([label, used_train, used_test, ""])

    # bookkeeping
    (out_train / "labels_used.txt").write_text(
        "\n".join([r[0] for r in summary_rows if (r[1] or r[2])]) + "\n",
        encoding="utf-8"
    )
    (out_train / "README.txt").write_text(
        "Prepared dataset for BirdNET training.\n"
        "Leak-free split: grouping by original recording key.\n"
        "Labels are folder names (Scientific_Common).\n",
        encoding="utf-8"
    )
    with open(excluded_path,"w",encoding="utf-8") as f:
        f.write("\n".join(excl) + ("\n" if excl else ""))

    write_summary_csv(out_train.parent / "prep_summary.csv", summary_rows)
    print(f"[OK] Prepared train: {out_train}, test: {out_test}")
    print(f"[OK] Excluded classes list: {excluded_path}")
    print(f"[OK] Summary CSV: {out_train.parent / 'prep_summary.csv'}")

if __name__ == "__main__":
    main()

