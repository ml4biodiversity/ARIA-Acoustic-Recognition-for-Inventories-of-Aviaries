#!/usr/bin/env python3
import os, sys, argparse, subprocess, csv, math, json, re, tempfile
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
from typing import Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
ARIA_ROOT = SCRIPT_DIR.parent.parent  # src/inference -> src -> ARIA

# ============= Default paths relative to ARIA =============
DEFAULT_RESULTS_ROOT = ARIA_ROOT / "results_inference_birdnet"
DEFAULT_MODEL = ARIA_ROOT / "models" / "ZooCustom_v1" / "ZooCustom_v1.tflite"
DEFAULT_LABELS = ARIA_ROOT / "models" / "ZooCustom_v1" / "ZooCustom_v1_Labels.txt"
DEFAULT_EVAL_CSV = ARIA_ROOT / "models" / "ZooCustom_v1" / "ZooCustom_v1_evaluation.csv"
DEFAULT_CONFIG_JSON = ARIA_ROOT / "config" / "zoo_species_config.json"


# ============= Temperature Scaling Functions =============

def inverse_sigmoid(conf: float) -> float:
    """
    Convert sigmoid output back to logit.
    Handles edge cases near 0 and 1.
    """
    # Clip to avoid log(0) or division by zero
    conf = np.clip(conf, 1e-7, 1 - 1e-7)
    return np.log(conf / (1 - conf))

def sigmoid(logit: float) -> float:
    """Apply sigmoid function."""
    return 1.0 / (1.0 + np.exp(-logit))

def apply_temperature_scaling(conf: float, temperature: float) -> float:
    """
    Apply temperature scaling to a confidence value.
    
    Args:
        conf: Original confidence (0-1)
        temperature: T > 1.0 reduces confidence, T < 1.0 increases it
    
    Returns:
        Scaled confidence
    """
    if temperature == 1.0:
        return conf
    
    logit = inverse_sigmoid(conf)
    scaled_logit = logit / temperature
    return sigmoid(scaled_logit)

def scale_csv_confidences(csv_path: Path, temperature: float, output_path: Path = None) -> int:
    """
    Apply temperature scaling to all confidences in a CSV file.
    
    Args:
        csv_path: Path to input CSV
        temperature: Temperature value (>1.0 reduces confidence)
        output_path: Path for output CSV (if None, overwrites input)
    
    Returns:
        Number of rows processed
    """
    if not csv_path.exists():
        return 0
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    if not rows or len(rows) <= 1:
        return 0
    
    header = rows[0]
    conf_idx = find_col_index(header, ["Confidence"])
    
    if conf_idx is None:
        print(f"[WARN] No Confidence column in {csv_path}")
        return 0
    
    # Apply scaling to each row
    scaled_rows = []
    for r in rows[1:]:
        try:
            original_conf = float(r[conf_idx])
            scaled_conf = apply_temperature_scaling(original_conf, temperature)
            r[conf_idx] = f"{scaled_conf:.4f}"
            scaled_rows.append(r)
        except (ValueError, IndexError):
            # Keep row as-is if confidence can't be parsed
            scaled_rows.append(r)
    
    # Write output
    output = output_path or csv_path
    output.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(scaled_rows)
    
    return len(scaled_rows)

def compute_saturation_metrics(csv_path: Path) -> dict:
    """
    Compute saturation metrics for a CSV file.
    
    Returns:
        dict with saturation percentages at different thresholds
    """
    if not csv_path.exists():
        return {}
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    if not rows or len(rows) <= 1:
        return {}
    
    header = rows[0]
    conf_idx = find_col_index(header, ["Confidence"])
    
    if conf_idx is None:
        return {}
    
    confidences = []
    for r in rows[1:]:
        try:
            confidences.append(float(r[conf_idx]))
        except (ValueError, IndexError):
            continue
    
    if not confidences:
        return {}
    
    total = len(confidences)
    
    return {
        "total": total,
        "mean": np.mean(confidences),
        "median": np.median(confidences),
        "sat_0.9999": sum(1 for c in confidences if c >= 0.9999) / total * 100,
        "sat_0.99": sum(1 for c in confidences if c >= 0.99) / total * 100,
        "sat_0.95": sum(1 for c in confidences if c >= 0.95) / total * 100,
        "range_low": sum(1 for c in confidences if c < 0.60) / total * 100,
        "range_mid": sum(1 for c in confidences if 0.60 <= c < 0.85) / total * 100,
        "range_high": sum(1 for c in confidences if c >= 0.85) / total * 100,
    }

def find_optimal_temperature(csv_path: Path, target_saturation: float = 0.30) -> Tuple[float, dict]:
    """
    Find optimal temperature to achieve target saturation level.
    
    Args:
        csv_path: Path to CSV with original confidences
        target_saturation: Target percentage of detections at >=0.9999 (0.0-1.0)
    
    Returns:
        (optimal_temperature, metrics_dict)
    """
    if not csv_path.exists():
        return 1.0, {}
    
    print(f"[INFO] Finding optimal temperature for {csv_path.name}...")
    
    # Load original confidences
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    if not rows or len(rows) <= 1:
        return 1.0, {}
    
    header = rows[0]
    conf_idx = find_col_index(header, ["Confidence"])
    
    if conf_idx is None:
        return 1.0, {}
    
    original_confs = []
    for r in rows[1:]:
        try:
            original_confs.append(float(r[conf_idx]))
        except (ValueError, IndexError):
            continue
    
    if not original_confs:
        return 1.0, {}
    
    original_confs = np.array(original_confs)
    
    # Test different temperatures
    temperatures = np.arange(1.0, 3.0, 0.1)
    best_temp = 1.0
    best_diff = float('inf')
    results = []
    
    for T in temperatures:
        # Apply temperature scaling
        scaled_confs = np.array([apply_temperature_scaling(c, T) for c in original_confs])
        
        # Compute saturation
        sat_0_9999 = (scaled_confs >= 0.9999).sum() / len(scaled_confs)
        
        # Find temperature closest to target
        diff = abs(sat_0_9999 - target_saturation)
        
        results.append({
            "temperature": T,
            "sat_0.9999": sat_0_9999 * 100,
            "sat_0.99": (scaled_confs >= 0.99).sum() / len(scaled_confs) * 100,
            "mean": scaled_confs.mean(),
            "median": np.median(scaled_confs),
        })
        
        if diff < best_diff:
            best_diff = diff
            best_temp = T
    
    # Print results table
    print("\nTemperature Search Results:")
    print("T     | Sat≥0.9999 | Sat≥0.99 | Mean  | Median")
    print("-" * 55)
    for r in results[::2]:  # Print every other row to reduce output
        print(f"{r['temperature']:.1f}   | {r['sat_0.9999']:6.1f}%   | {r['sat_0.99']:5.1f}%  | {r['mean']:.3f} | {r['median']:.3f}")
    
    best_metrics = [r for r in results if r['temperature'] == best_temp][0]
    print(f"\n[OPTIMAL] T={best_temp:.1f} gives {best_metrics['sat_0.9999']:.1f}% saturation (target: {target_saturation*100:.0f}%)")
    
    return best_temp, best_metrics


def scale_individual_csvs(out_dir: Path, temperature: float) -> int:
    """
    Apply temperature scaling to all individual BirdNET result CSVs.
    
    Args:
        out_dir: Directory containing subdirectories with .BirdNET.results.csv files
        temperature: Temperature value for scaling
    
    Returns:
        Number of files processed
    """
    csv_files = list(out_dir.rglob("*.BirdNET.results.csv"))
    
    # Exclude merged files
    csv_files = [f for f in csv_files if f.name not in [
        "all_results.csv", 
        "all_results.scaled.csv", 
        "all_results.filtered.csv",
        "all_results.filtered.top3.csv"
    ]]
    
    print(f"[INFO] Scaling {len(csv_files)} individual CSV files with T={temperature:.2f}...")
    
    scaled_count = 0
    for csv_file in csv_files:
        rows_scaled = scale_csv_confidences(csv_file, temperature, output_path=csv_file)
        if rows_scaled > 0:
            scaled_count += 1
    
    print(f"[OK] Scaled {scaled_count}/{len(csv_files)} individual CSVs")
    return scaled_count

# ---------------- Load JSON Config ----------------

def load_zoo_config(json_path: Path) -> dict:
    """Load the zoo species configuration from JSON."""
    if not json_path.exists():
        print(f"[ERR] Config JSON not found: {json_path}", file=sys.stderr)
        sys.exit(1)
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_folder_name(folder_name: str, config: dict = None):
    """
    Config-driven zoo/cage detection. Reads zoo/cage routing rules from
    zoo_species_config.json's 'folder_patterns' section when available,
    falling back to generic heuristics (date pattern, aug25, flamingo,
    vulture/prox, storage) for zoos without explicit patterns defined.

    To add a new zoo/cage routing rule in the future, edit
    zoo_species_config.json's 'folder_patterns' section - no code change
    needed here.
    """
    name = folder_name.lower().replace('.zip', '').replace('.tar', '')
    if name.startswith('fl_'):
        name = name[3:]
    if name.startswith('5sec_'):
        name = name[5:]

    zoo_patterns = [
        'beekse_bergen',
        'beeksebergen',
        'zoo_eindhoven',
        'zoo_helsinki',
        'gaia_zoo_congo',
        'gaia_zoo_taiga',
        'gaia_zoo_savannah',
        'mochengladbach',
        'gaia_taiga',
        'gaia_savannah',
        'zoo_parc',
        'avifauna',
        'cologne',
        'cologne_zoo',
        'blijdorp',
        'aachen',
        'antwerpen',
        'duisburg',
        'ziezoo'
    ]

    zoo_name = None
    for pattern in zoo_patterns:
        if pattern in name:
            zoo_name = pattern
            break

    if not zoo_name:
        return None, None

    remainder = name.replace(zoo_name, '').strip('_')

    # Normalize cologne_zoo -> cologne for config lookup, since both map
    # to the same folder_patterns entry.
    config_zoo_key = 'cologne' if zoo_name in ('cologne', 'cologne_zoo') else zoo_name
    if zoo_name == 'beeksebergen':
        config_zoo_key = 'beekse_bergen'

    cage = None
    folder_patterns = (config or {}).get('folder_patterns', {})
    zoo_pattern_config = folder_patterns.get(config_zoo_key)

    if zoo_pattern_config:
        # Config-driven lookup
        for cage_name, substrings in zoo_pattern_config.get('cages', {}).items():
            if any(s in remainder for s in substrings):
                cage = cage_name
                break
    else:
        # Generic fallback heuristics (unchanged from before, for zoos
        # without explicit folder_patterns entries)
        if re.search(r'\d{8}', remainder):
            cage = re.search(r'\d{8}', remainder).group()
        elif 'aug25' in remainder:
            cage = 'aug25'
        elif 'flamingo' in remainder:
            cage = 'flamingos'
        elif 'vulture' in remainder or 'prox' in remainder:
            cage = 'vultures'
        elif 'storage' in remainder:
            cage = 'storage'
        else:
            cage = None

    return zoo_name, cage

def get_cage_birds(config: dict, zoo_name: str, cage_name: str) -> set:
    """
    Get the set of bird common names for a specific cage.
    Returns empty set if zoo not found.
    """
    config_zoo_key = zoo_name
    if zoo_name in ('cologne', 'cologne_zoo'):
        config_zoo_key = 'cologne'
    elif zoo_name == 'beeksebergen':
        config_zoo_key = 'beekse_bergen'

    if config_zoo_key not in config['zoos']:
        return set()
    
    zoo_data = config['zoos'][config_zoo_key]
    
    # If cage specified and exists, use cage-specific birds
    if cage_name and cage_name in zoo_data.get('cages', {}):
        return set(zoo_data['cages'][cage_name])
    
    # Otherwise use all birds for that zoo
    return set(zoo_data.get('all_birds', []))

# def build_common_to_label_mapping(eval_csv: Path) -> dict:
#     """
#     Build mapping from common name to full label (Scientific_Common).
#     Uses the evaluation CSV which has the Class column.
#     """
#     if not eval_csv.exists():
#         print(f"[WARN] Eval CSV not found at {eval_csv}", file=sys.stderr)
#         return {}
    
#     mapping = {}
#     with open(eval_csv, "r", encoding="utf-8") as f:
#         reader = csv.reader(f)
#         rows = list(reader)
    
#     if not rows:
#         return {}
    
#     header = rows[0]
#     class_idx = None
#     for i, col in enumerate(header):
#         if col.lower() == "class":
#             class_idx = i
#             break
    
#     if class_idx is None:
#         print("[WARN] No 'Class' column in eval CSV", file=sys.stderr)
#         return {}
    
#     for r in rows[1:]:
#         if not r or r[0] == "OVERALL (Macro-avg)":
#             continue
#         try:
#             label = r[class_idx].strip()
#             # Extract common name (after underscore)
#             if "_" in label:
#                 common = label.split("_", 1)[1]
#                 mapping[common] = label
#         except Exception:
#             continue
    
#     print(f"[INFO] Built mapping for {len(mapping)} species from eval CSV")
#     return mapping
# Known BirdNET-label vs config-name variants that refer to the same species.
# Keys and values are already run through the base normalization below.
_NAME_ALIASES = {
    "redshank": "common redshank",
    "turtle dove": "european turtle dove",
    "eastern crested guineafowl": "crested guineafowl",
    "eurasian stone curlew": "stone curlew",
}

def normalize_name(s: str) -> str:
    import unicodedata
    s = s.lower().strip()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.replace("-", " ").replace("_", " ").replace("'", "")
    s = " ".join(s.split())
    return _NAME_ALIASES.get(s, s)


def build_common_to_label_mapping(train_root: Path) -> dict:
    """
    Build mapping from common name → full label using TRAINING FOLDERS.
    This is robust and does NOT depend on eval CSV.
    """
    mapping = {}

    if not train_root.exists():
        print(f"[ERR] Train root not found: {train_root}")
        return mapping

    for d in train_root.iterdir():
        if not d.is_dir():
            continue

        label = d.name  # e.g. "Turdus merula_Common Blackbird"

        if "_" in label:
            common = label.split("_", 1)[1]
            key = normalize_name(common)
            mapping[key] = label

    print(f"[INFO] Built mapping for {len(mapping)} species from train_curated")
    return mapping

def create_cage_labels_file(cage_birds: set, wild_birds: set, 
                            common_to_label: dict, output_path: Path) -> int:
    """
    Create a temporary labels file with only cage birds + wild birds.
    
    Returns: number of species in the file
    """
    all_birds = cage_birds | wild_birds
    
    labels = []
    missing = []
    
    for common_name in all_birds:
        # if common_name in common_to_label:
        #     labels.append(common_to_label[common_name])
        # else:
        #     missing.append(common_name)
        key = normalize_name(common_name)
        if key in common_to_label:
            labels.append(common_to_label[key])
        else:
            missing.append(common_name)
    
    if missing:
        print(f"[WARN] Could not find labels for: {', '.join(missing)}")
    
    # Write labels file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for label in sorted(labels):
            f.write(label + '\n')
    
    print(f"[INFO] Created cage-specific labels file with {len(labels)} species")
    return len(labels)

# -------------------- utils --------------------

def ensure_birdnet_in_path(env: dict) -> dict:
    """Guarantee birdnet_analyzer is importable for the subprocess."""
    try:
        import importlib
        importlib.import_module("birdnet_analyzer")
        return env
    except Exception:
        repo = os.environ.get("BIRDNET_REPO") or os.path.join(Path.home(), "BirdNET-Analyzer")
        if os.path.isdir(repo):
            env["PYTHONPATH"] = repo + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
            print(f"[INFO] Using BIRDNET_REPO at {repo}")
        else:
            print("[WARN] 'birdnet_analyzer' not importable and no repo found at:", repo, file=sys.stderr)
            print("[WARN] Consider: pip install birdnet-analyzer", file=sys.stderr)
        return env

def class_key(scientific: str, common: str) -> str:
    # Must match your training label convention "Scientific_Common"
    s = (scientific or "").strip()
    c = (common or "").strip()
    return f"{s}_{c}" if s and c else f"{s}{c}"

def find_col_index(header, candidates):
    """Return first matching column index from candidate names (case-insensitive)."""
    low = [h.lower() for h in header]
    for cand in candidates:
        if cand.lower() in low:
            return low.index(cand.lower())
    return None

# ---------------- BirdNET run + merge ----------------

def run_analyze(input_dir: Path, out_dir: Path, model: Path, labels: Path,
                min_conf: float, overlap: float, threads: int, 
                use_neutral_priors: bool, allowed_species: set):  # ← Add this
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Run BirdNET normally
    cmd = [
        sys.executable, "-m", "birdnet_analyzer.analyze", str(input_dir),
        "-o", str(out_dir),
        "--classifier", str(model),
        # NOTE: --slist doesn't restrict outputs with custom classifiers
        "--min_conf", str(min_conf),
        "--overlap", str(overlap),
        "--threads", str(threads),
        "--rtype", "csv",
        "--skip_existing_results",
    ]
    if use_neutral_priors:
        cmd += ["--lat", "-1", "--lon", "-1", "--week", "-1"]
    
    env = ensure_birdnet_in_path(os.environ.copy())
    subprocess.run(cmd, check=True, env=env)
    
    # ← NEW: Immediately filter each CSV to only allowed species
    for csv_file in out_dir.rglob("*.BirdNET.results.csv"):
        filter_csv_in_place(csv_file, allowed_species)



def filter_csv_in_place(csv_path: Path, allowed_species: set):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows or len(rows) <= 1:
        return

    header = rows[0]

    common_idx = None
    for i, col in enumerate(header):
        if col.lower() in ["common name", "common_name"]:
            common_idx = i
            break

    if common_idx is None:
        return

    allowed_norm = {normalize_name(x) for x in allowed_species}

    kept = []
    for r in rows[1:]:
        species = normalize_name(r[common_idx])

        if species in allowed_norm:
            kept.append(r)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(kept)

def merge_csvs(search_root: Path, fallback_root: Path, merged_path: Path):
    # Prefer CSVs under the results dir; if none, look in the input dir.
    csvs = list(search_root.rglob("*.BirdNET.results.csv"))
    if not csvs:
        csvs = list(fallback_root.rglob("*.BirdNET.results.csv"))
        if csvs:
            print(f"[WARN] Results landed next to inputs; merging from {fallback_root}")
    if not csvs:
        print("[WARN] No CSVs found to merge.")
        return 0

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    with open(merged_path, "w", newline="", encoding="utf-8") as out_f:
        w = None
        for f in sorted(csvs):
            with open(f, "r", encoding="utf-8") as in_f:
                reader = csv.reader(in_f)
                rows = list(reader)
                if not rows:
                    continue
                header, data = rows[0], rows[1:]
                if w is None:
                    w = csv.writer(out_f); w.writerow(header)
                for r in data:
                    w.writerow(r)
                total_rows += len(data)
    print(f"[OK] Merged {total_rows} rows into {merged_path}")
    return total_rows

# --------------- thresholds & filtering ---------------

def default_eval_csv_for(model_path: Path) -> Path:
    # checkpoints/ZooCustom_v1.tflite -> checkpoints/ZooCustom_v1_evaluation.csv
    name = model_path.name
    if name.endswith(".tflite"):
        stem = name[:-7]  # drop ".tflite"
        return model_path.parent / f"{stem}_evaluation.csv"
    return model_path.with_name(model_path.name + "_evaluation.csv")

def load_optimal_thresholds(eval_csv: Path) -> dict:
    """Return { 'Scientific_Common': threshold } from the evaluation CSV."""
    if not eval_csv.exists():
        print(f"[WARN] Eval CSV not found at {eval_csv}; will use global --min-conf.", file=sys.stderr)
        return {}
    out = {}
    with open(eval_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return {}

    header = rows[0]
    class_idx = find_col_index(header, ["Class"])
    thr_idx   = find_col_index(header, ["Optimal Threshold", "Optimal threshold", "optimal_threshold"])
    if class_idx is None or thr_idx is None:
        print("[WARN] Could not find 'Class' or 'Optimal Threshold' columns in eval CSV; using global cutoff.", file=sys.stderr)
        return {}

    for r in rows[1:]:
        try:
            label = r[class_idx].strip()
            thr = float(r[thr_idx]) if r[thr_idx] not in ("", None) else math.nan
            if label and not math.isnan(thr):
                out[label] = thr
        except Exception:
            continue
    print(f"[INFO] Loaded {len(out)} per-class optimal thresholds from {eval_csv}")
    return out



def apply_threshold_filter(all_csv: Path, filtered_csv: Path, per_class_thr: dict, fallback_min_conf: float):
    if not all_csv.exists():
        print(f"[WARN] Missing results file: {all_csv}")
        return 0,0

    with open(all_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return 0, 0

    header = rows[0]
    sci_idx = find_col_index(header, ["Scientific name", "Scientific", "Scientific Name"])
    com_idx = find_col_index(header, ["Common name", "Common", "Common Name"])
    conf_idx = find_col_index(header, ["Confidence"])
    if sci_idx is None or com_idx is None or conf_idx is None:
        print("[ERR] Results CSV missing required columns (Scientific name, Common name, Confidence).", file=sys.stderr)
        return 0,0

    kept = []
    for r in rows[1:]:
        try:
            conf = float(r[conf_idx])
        except Exception:
            continue
        key = class_key(r[sci_idx], r[com_idx])
        thr = per_class_thr.get(key, fallback_min_conf)
        if conf >= thr:
            kept.append(r)

    filtered_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(filtered_csv, "w", newline="", encoding="utf-8") as out_f:
        w = csv.writer(out_f)
        w.writerow(header)
        w.writerows(kept)

    print(f"[OK] Threshold-filtered {len(kept)}/{len(rows)-1} rows -> {filtered_csv}")
    return len(rows)-1, len(kept)

def categorize_detections(try_csv: Path, cage_birds: set, wild_birds: set):
    """Categorize try detections into cage birds vs wild birds."""
    if not try_csv.exists():
        return 0, 0
    
    with open(try_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    if not rows:
        return 0, 0
    
    header = rows[0]
    com_idx = find_col_index(header, ["Common name", "Common", "Common Name"])
    
    if com_idx is None:
        return 0, 0
    
    cage_count = 0
    wild_count = 0
    
    for r in rows[1:]:
        common = (r[com_idx] or "").strip()
        if common in cage_birds:
            cage_count += 1
        elif common in wild_birds:
            wild_count += 1
    
    return cage_count, wild_count

def topk_per_window(filtered_csv: Path, out_csv: Path, k: int):
    """Keep only top-K rows per (file,start,end) window."""
    if k <= 0:
        return 0,0
    with open(filtered_csv, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return 0,0

    header = rows[0]
    start_idx = find_col_index(header, ["Start (s)", "Start"])
    end_idx   = find_col_index(header, ["End (s)", "End"])
    conf_idx  = find_col_index(header, ["Confidence"])
    file_idx  = find_col_index(header, ["File name", "Filename", "File"])
    if None in (start_idx, end_idx, conf_idx, file_idx):
        print("[WARN] topK skipped: required columns not found.", file=sys.stderr)
        return 0,0

    buckets = defaultdict(list)
    for r in rows[1:]:
        try:
            key = (r[file_idx], r[start_idx], r[end_idx])
            conf = float(r[conf_idx])
        except Exception:
            continue
        buckets[key].append((conf, r))

    kept = []
    for key, lst in buckets.items():
        lst.sort(key=lambda x: x[0], reverse=True)
        kept.extend([r for _, r in lst[:k]])

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(kept)

    print(f"[OK] Top-{k} per window: {len(kept)}/{len(rows)-1} rows -> {out_csv}")
    return len(rows)-1, len(kept)

# ------------------------ main ------------------------

def main():
    ap = argparse.ArgumentParser(description="Run BirdNET inference with cage-specific species lists (Option B).")
    ap.add_argument("--input-root", default="/home/zoodata/preprocessed",
                    help="Root folder containing dataset directories.")
    ap.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT),
                    help="Where to store inference outputs.")
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--eval-csv", default=str(DEFAULT_EVAL_CSV),
                    help="Path to *_evaluation.csv. If empty, will infer from --model.")
    ap.add_argument("--config-json", default=str(DEFAULT_CONFIG_JSON),
                    help="Path to zoo_species_config.json")
    ap.add_argument("--min-conf", type=float, default=0.05,
                    help="Global minimum confidence for BirdNET inference.")
    ap.add_argument("--overlap", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=40)
    ap.add_argument("--use-neutral-priors", action="store_true",
                    help="Run analyze with --lat -1 --lon -1 --week -1 (neutral priors).")
    ap.add_argument("--topk", type=int, default=3,
                    help="Optional: keep only top-K species per (file,start,end) window after thresholding.")
    ap.add_argument("--only", action="append", default=[],
                    help="Process only these dataset folder names (can be given multiple times).")
    ap.add_argument("--name-prefix", default="", help="Only process dataset names starting with this prefix.")
    ap.add_argument("--apply-temperature", action="store_true", default=True,
                help="Apply temperature scaling to confidences.")
    ap.add_argument("--temperature", type=float, default=1.8,
                help="Temperature value (>1.0 reduces confidence). If not specified, will find optimal.")
    ap.add_argument("--target-saturation", type=float, default=0.30,
                    help="Target saturation level (0-1) for automatic temperature finding.")
    args = ap.parse_args()

    input_root = Path(args.input_root)
    results_root = Path(args.results_root)
    model = Path(args.model)
    labels = Path(args.labels)
    config_json = Path(args.config_json)

    if not input_root.is_dir():
        sys.exit(f"[ERR] Input root not found: {input_root}")
    results_root.mkdir(parents=True, exist_ok=True)

    # Load JSON config
    config = load_zoo_config(config_json)
    wild_birds_set = set(config.get('wild_birds', []))
    print(f"[INFO] Loaded config with {len(config['zoos'])} zoos and {len(wild_birds_set)} wild bird species")

    # Build mapping from common names to full labels
    eval_csv = Path(args.eval_csv) if args.eval_csv else default_eval_csv_for(model)
    #common_to_label = build_common_to_label_mapping(eval_csv)
    train_root = ARIA_ROOT / "data" / "train_curated"
    common_to_label = build_common_to_label_mapping(train_root)
    per_class_thr = load_optimal_thresholds(eval_csv)

    # choose datasets
    candidates = [d for d in input_root.iterdir() if d.is_dir()]
    if args.name_prefix:
        candidates = [d for d in candidates if d.name.startswith(args.name_prefix)]
    if args.only:
        allow = set(args.only)
        candidates = [d for d in candidates if d.name in allow]
    datasets = sorted(candidates, key=lambda p: p.name)
    if not datasets:
        sys.exit("[ERR] No dataset directories to process.")

    print(f"[INFO] Will process {len(datasets)} dataset(s).")
    summary_rows = []
    total_files = 0
    
    for ds in datasets:
        print("\n" + "="*80)
        print(f"[DATASET] {ds}")
        
        # Parse folder name to get zoo and cage
        zoo_name, cage_name = parse_folder_name(ds.name, config)
        print(f"[INFO] Parsed: zoo={zoo_name}, cage={cage_name}")
        
        if not zoo_name:
            print(f"[WARN] Could not parse zoo name from {ds.name}")
            print("[WARN] Running without cage-specific filtering...")
            zoo_name = "unknown"
            cage_name = None
        
        # Get cage-specific birds
        cage_birds = get_cage_birds(config, zoo_name, cage_name)
        print(f"[INFO] Cage has {len(cage_birds)} bird species")
        print(f"[INFO] Wild birds: {len(wild_birds_set)} species")
        print(f"[INFO] Total species to check: {len(cage_birds) + len(wild_birds_set)}")
        
        # Create temporary cage-specific labels file
        out_dir = results_root / ds.name
        temp_labels = out_dir / "cage_specific_labels.txt"
        num_labels = create_cage_labels_file(cage_birds, wild_birds_set, 
                                            common_to_label, temp_labels)
        
        if num_labels == 0:
            print(f"[ERR] No valid labels created for {ds.name}, skipping...")
            continue
        
        try:
            # Run BirdNET with cage-specific labels
            run_analyze(ds, out_dir, model, temp_labels,  
                args.min_conf, args.overlap, args.threads,
                use_neutral_priors=args.use_neutral_priors,
                allowed_species=cage_birds | wild_birds_set)
        except subprocess.CalledProcessError as e:
            print(f"[ERR] Inference failed for {ds}: {e}", file=sys.stderr)
            continue
    
        if args.apply_temperature:
            # Determine temperature
            optimal_temp = args.temperature
            
            if optimal_temp is None:
                # Need to merge first to find optimal temperature
                print("[INFO] Merging CSVs to find optimal temperature...")
                temp_merged = out_dir / "all_results_temp.csv"
                merge_csvs(out_dir, ds, temp_merged)
                
                optimal_temp, temp_metrics = find_optimal_temperature(
                    temp_merged, 
                    target_saturation=args.target_saturation
                )
                print(f"[INFO] Found optimal temperature: {optimal_temp:.2f}")
                temp_merged.unlink()  # Delete temporary file
            
            # Apply temperature to individual CSVs
            print(f"[INFO] Applying temperature scaling (T={optimal_temp:.2f}) to individual CSVs...")
            scale_individual_csvs(out_dir, optimal_temp)
        
        # ============= Now merge the SCALED individual CSVs =============
        all_csv = out_dir / "all_results.csv"
        merged_rows = merge_csvs(out_dir, ds, all_csv)
        
        # Compute metrics (all_results.csv now contains scaled confidences)
        if args.apply_temperature:
            metrics = compute_saturation_metrics(all_csv)
            print(f"[METRICS] Saturation ≥0.9999: {metrics.get('sat_0.9999', 0):.1f}%")
            print(f"[METRICS] Saturation ≥0.99: {metrics.get('sat_0.99', 0):.1f}%")
            print(f"[METRICS] Mean confidence: {metrics.get('mean', 0):.3f}")

        # Apply threshold filter
        filtered_csv = out_dir / "all_results.filtered.csv"
        before, after = apply_threshold_filter(all_csv, filtered_csv, per_class_thr, args.min_conf)
        # Optional top-K
        try_csv = filtered_csv
        if args.topk > 0 and filtered_csv.exists():
            topk_csv = out_dir / f"all_results.filtered.top{args.topk}.csv"
            tbefore, tafter = topk_per_window(filtered_csv, topk_csv, args.topk)
            print(f"[INFO] After thresholds: {after}; after top-{args.topk}: {tafter}")
            try_csv = topk_csv

        # Categorize detections
        cage_count, wild_count = categorize_detections(try_csv, cage_birds, wild_birds_set)
        total_kept = cage_count + wild_count
        
        pct = lambda x: (100.0 * x / total_kept) if total_kept > 0 else 0.0

        print(f"[METRICS] {ds.name} [{zoo_name}/{cage_name or 'all'}]")
        print(f"  Total kept: {total_kept}")
        print(f"  Cage birds: {cage_count} ({pct(cage_count):.1f}%)")
        print(f"  Wild birds: {wild_count} ({pct(wild_count):.1f}%)")

        # Store results
        summary_rows.append({
            "dataset": ds.name,
            "zoo": zoo_name,
            "cage": cage_name or "all",
            "species_checked": num_labels,
            "detections_kept": total_kept,
            "cage_birds_count": cage_count,
            "wild_birds_count": wild_count,
            "cage_birds_pct": f"{pct(cage_count):.2f}",
            "wild_birds_pct": f"{pct(wild_count):.2f}",
            "cage_species_defined": len(cage_birds),
        })
        
        n_wavs = sum(1 for _ in ds.rglob("*.wav"))
        print(f"[INFO] WAVs in input: {n_wavs}, merged detections: {merged_rows}, try kept: {total_kept}")
        total_files += n_wavs
    
    # Write summary
    summary_path = results_root / "optionB_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "dataset", "zoo", "cage", "species_checked", "detections_kept",
            "cage_birds_count", "cage_birds_pct",
            "wild_birds_count", "wild_birds_pct",
            "cage_species_defined"
        ])
        for row in summary_rows:
            w.writerow([
                row["dataset"], row["zoo"], row["cage"], row["species_checked"],
                row["detections_kept"],
                row["cage_birds_count"], row["cage_birds_pct"],
                row["wild_birds_count"], row["wild_birds_pct"],
                row["cage_species_defined"],
            ])
    print(f"[OK] Wrote summary → {summary_path}")

    print("\n[DONE] Inference finished.")
    print(f"[SUMMARY] Input root: {input_root}")
    print(f"[SUMMARY] Results root: {results_root}")
    print(f"[SUMMARY] Total WAVs seen (approx): {total_files}")
    print("\n[OPTION B] BirdNET ran with cage-specific + wild bird species only!")
    print("[OPTION B] This is much faster and more accurate than checking all species.")

if __name__ == "__main__":
    main()