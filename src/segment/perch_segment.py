#!/usr/bin/env python3
import argparse, tempfile, shutil, os
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import random
import tensorflow as tf
import kagglehub
from packaging import version
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
MIN_TF = "2.20.0"
if version.parse(tf.__version__) < version.parse(MIN_TF):
    raise RuntimeError(
        f"Perch 2.0 requires TensorFlow >= {MIN_TF} (found {tf.__version__}). "
        "Upgrade TF or use a different environment."
    )

# ------------------------ mapping helpers ------------------------
def extract_scientific_name(name: str) -> str:
    """Extract 'Genus species' from folder name like 'Genus_species_Common_Name'"""
    parts = name.split('_')
    if len(parts) >= 2:
        # Take first two parts as 'Genus species'
        return f"{parts[0]} {parts[1]}".strip().lower()
    # Fallback: just use first part
    return parts[0].strip().lower()


def parse_ebird_mapping(indices_file: Path) -> dict:
    mapping = {
        'scientific_to_ebird': {},
        'ebird_to_scientific': {},
        'ebird_to_index': {},
        'scientific_to_index': {},
    }
    if not indices_file.exists():
        print(f"[WARN] eBird indices file not found: {indices_file}")
        return mapping
    with open(indices_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('#')
            if len(parts) < 2:
                continue
            try:
                index = int(parts[0].strip())
                info_part = parts[1].strip()
                if '[' in info_part and ']' in info_part:
                    ebird_code = info_part[info_part.index('[')+1:info_part.index(']')].lower()
                    name_part = info_part[:info_part.index('[')].strip()
                    if ' - ' in name_part:
                        scientific, _common = name_part.split(' - ', 1)
                        scientific = scientific.strip().lower()
                        mapping['scientific_to_ebird'][scientific] = ebird_code
                        mapping['ebird_to_scientific'][ebird_code] = scientific
                        mapping['ebird_to_index'][ebird_code] = index
                        mapping['scientific_to_index'][scientific] = index
            except Exception:
                continue
    print(f"[INFO] Loaded {len(mapping['scientific_to_ebird'])} species mappings from {indices_file}")
    return mapping

def find_species_info(folder_name: str, ebird_mapping: dict):
    scientific = extract_scientific_name(folder_name)
    ebird_code = ebird_mapping.get('scientific_to_ebird', {}).get(scientific)
    perch_index = None
    if ebird_code:
        perch_index = ebird_mapping.get('ebird_to_index', {}).get(ebird_code)
    return ebird_code, perch_index, scientific

# ------------------------ class lists ------------------------
def load_inat_class_names(model_path: str):
    """Return iNaturalist-ordered class names from assets/labels.csv."""
    model_dir = Path(model_path)
    for csv_file in [model_dir / "assets" / "labels.csv", model_dir / "labels.csv"]:
        if csv_file.exists():
            try:
                df = pd.read_csv(csv_file)
            except Exception:
                df = pd.read_csv(csv_file, engine="python")
            if 'label' in df.columns:
                names = df['label'].astype(str).str.lower().tolist()
            else:
                names = df.iloc[:, 0].astype(str).str.lower().tolist()
            print(f"[INFO] Loaded {len(names)} class names from {csv_file} (iNat order)")
            return names
    print("[ERROR] Could not load class names from labels.csv")
    return None

def try_load_ebird_to_inat_map(model_path: str) -> dict:
    """ Best-effort eBird->iNat mapping. If it fails, we just return {} and will fall back to direct scientific-name matching. """
    model_dir = Path(model_path)
    cand = [model_dir / "assets" / "perch_v2_ebird_classes.csv",
            model_dir / "perch_v2_ebird_classes.csv"]
    for csv_file in cand:
        if not csv_file.exists():
            continue
        try:
            # Try multiple separators; if still 1 column, let it pass (we'll ignore)
            for sep in [',', ';', '\t']:
                try:
                    df = pd.read_csv(csv_file, sep=sep)
                    if df.shape[1] >= 2:
                        break
                except Exception:
                    continue
            else:
                # last attempt with engine='python'
                df = pd.read_csv(csv_file, engine="python")
            if df.shape[1] < 2:
                raise RuntimeError("file looks like a single column")
            norm = [str(c).strip().lower() for c in df.columns]
            df.columns = norm
            eb_col = 'ebird_code' if 'ebird_code' in norm else norm[0]
            lb_col = 'label' if 'label' in norm else (norm[1] if len(norm) > 1 else None)
            if lb_col is None:
                raise RuntimeError("no label column found")
            m = dict(zip(df[eb_col].astype(str).str.lower(), df[lb_col].astype(str).str.lower()))
            print(f"[INFO] Loaded eBird→iNat map ({len(m)} rows) from {csv_file}")
            return m
        except Exception as e:
            print(f"[WARN] Could not parse {csv_file}: {e}")
    print("[WARN] eBird mapping CSV not available; will use scientific-name fallback")
    return {}

# ------------------------ model wrapper ------------------------
class KagglePerchModel:
    def __init__(self, use_gpu=True):
        print("[INFO] Loading Perch model from Kaggle...")
        if use_gpu:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        print(f"[INFO] Enabled memory growth for GPU: {gpu}")
                    except Exception as e:
                        print(f"GPU config warning: {e}")
        has_gpu = bool(tf.config.list_physical_devices('GPU')) and use_gpu
        model_id = ("google/bird-vocalization-classifier/tensorFlow2/perch_v2" if has_gpu else "google/bird-vocalization-classifier/tensorFlow2/perch_v2_cpu")
        model_path = kagglehub.model_download(model_id)
        print(f"[INFO] Model path: {model_path} (variant: {'GPU' if has_gpu else 'CPU'})")
        self.class_names = load_inat_class_names(model_path)
        if self.class_names is None:
            raise RuntimeError("Could not load Perch class names from labels.csv")
        self.inat_index = {name: i for i, name in enumerate(self.class_names)}
        self.ebird_to_inat = try_load_ebird_to_inat_map(model_path)
        self.model = tf.saved_model.load(model_path)
        print(f"[INFO] Signatures: {list(self.model.signatures.keys())}")
        print(f"[INFO] Loaded model with {len(self.class_names)} classes")
        print(f"[DEBUG] First 5 classes: {self.class_names[:5]}")

    def preprocess_audio(self, audio_path, target_length=160000):
        try:
            audio, sr = sf.read(audio_path, dtype='float32', always_2d=False)
            if isinstance(audio, np.ndarray) and audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 32000:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=32000)
            if len(audio) > target_length:
                audio = audio[:target_length]
            elif len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            max_abs = np.max(np.abs(audio))
            if max_abs > 0:
                audio = audio / max_abs * 0.9
            else:
                print(f"[WARN] Silent audio in {audio_path}; skipping normalization")
            return audio.astype(np.float32)
        except Exception as e:
            print(f"[ERROR] Failed to preprocess {audio_path}: {e}")
            return None

    def warmup_model(self):
        """Warmup the model with a dummy batch to trigger XLA compilation"""
        print("[INFO] Warming up model to trigger XLA compilation...")
        try:
            dummy_audio = np.random.randn(2, 160000).astype(np.float32)  # 2 samples
            serving_fn = self.model.signatures['serving_default']
            input_keys = list(serving_fn.structured_input_signature[1].keys())
            input_name = input_keys[0] if input_keys else 'inputs'
            
            _ = serving_fn(**{input_name: tf.constant(dummy_audio)})
            print("[INFO] Model warmup complete")
        except Exception as e:
            print(f"[WARN] Warmup failed (may still work): {e}")

    def predict_batch(self, audio_paths, batch_size=16) -> pd.DataFrame:
        all_probs, valid_paths = [], []
        serving_fn = self.model.signatures['serving_default']
        input_keys = list(serving_fn.structured_input_signature[1].keys())
        input_name = input_keys[0] if input_keys else 'inputs'
        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i+batch_size]
            batch_audio = []
            for p in batch_paths:
                a = self.preprocess_audio(p)
                if a is not None:
                    batch_audio.append(a)
                    valid_paths.append(p)
            if not batch_audio:
                continue
            try:
                batch_tensor = tf.constant(np.stack(batch_audio, axis=0), dtype=tf.float32)
                outputs = serving_fn(**{input_name: batch_tensor})
                if 'label' not in outputs:
                    raise RuntimeError(f"Expected 'label' in outputs, got {list(outputs.keys())}")
                logits = outputs['label']
                probs = tf.nn.softmax(logits, axis=-1).numpy()
                all_probs.append(probs)
                print(f" [DEBUG] Processed batch {(i//batch_size)+1}: {len(batch_audio)} files")
            except Exception as e:
                print(f" [ERROR] Batch {(i//batch_size)+1} failed: {e}")
                continue
        if not all_probs:
            return pd.DataFrame()
        probs_all = np.concatenate(all_probs, axis=0)
        df = pd.DataFrame(probs_all, columns=self.class_names)
        df['file_path'] = valid_paths
        return df

# ------------------------ windowing helpers ------------------------
def make_windows(wav_path: Path, step: float, window_size: float = 5.0):
    try:
        with sf.SoundFile(str(wav_path)) as f:
            duration = f.frames / f.samplerate
    except Exception as e:
        print(f"[ERROR] Cannot read {wav_path}: {e}")
        return []
    starts, t = [], 0.0
    while t + window_size <= duration:
        starts.append(round(t, 2))
        t += step
    return starts

def extract_audio_segment(wav_path, start_time, duration):
    try:
        with sf.SoundFile(str(wav_path)) as f:
            start_sample = int(start_time * f.samplerate)
            num_samples = int(duration * f.samplerate)
            if start_sample >= f.frames:
                return np.array([]), f.samplerate
            f.seek(start_sample)
            actual = min(num_samples, f.frames - start_sample)
            audio = f.read(actual, dtype='float32', always_2d=False)
            if isinstance(audio, np.ndarray) and audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio, f.samplerate
    except Exception as e:
        print(f"[ERROR] Failed to extract segment: {e}")
        return np.array([]), 0

# ------------------------ main ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--species_name", required=True)
    ap.add_argument("--labels_csv", default=None, help="(unused)")
    ap.add_argument("--ebird_indices", default="your_species_indices.txt")
    ap.add_argument("--seg_len", type=float, default=3.0)
    ap.add_argument("--step", type=float, default=2.5)
    ap.add_argument("--threshold", type=float, default=0.1, help="Probability threshold 0-1")
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--no_fuzzy", action="store_true", help="Disable fuzzy matching; error if no exact match")
    args = ap.parse_args()
    wav_dir = Path(args.wav_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Kaggle Perch segmentation for {args.species_name}")
    print(f" Threshold: {args.threshold} (probability)")
    ebird_mapping = parse_ebird_mapping(Path(args.ebird_indices))
    ebird_code, _perch_index, scientific_name = find_species_info(args.species_name, ebird_mapping=ebird_mapping)
    if not scientific_name:
        scientific_name = extract_scientific_name(args.species_name)
    if ebird_code:
        print(f"[INFO] Target species: {scientific_name} (eBird: {ebird_code})")
    else:
        print(f"[INFO] Target species: {scientific_name} (no eBird code in indices file)")
    # Load model
    model = KagglePerchModel(use_gpu=args.gpu)
    model.warmup_model()
    # --- Determine target column ---
    target_col_idx = None
    target_col_name = None
    # (A) Try eBird -> iNat map if present
    if ebird_code and ebird_code in model.ebird_to_inat:
        inat_label = model.ebird_to_inat[ebird_code]
        if inat_label in model.inat_index:
            target_col_idx = model.inat_index[inat_label]
            target_col_name = model.class_names[target_col_idx]
            print(f"[INFO] Mapping via eBird: '{ebird_code}' -> iNat '{inat_label}' -> column '{target_col_name}' (idx {target_col_idx})")
        else:
            print(f"[WARN] iNat label '{inat_label}' from eBird map not in class list; will try scientific-name fallback")
    # (B) Fallback: direct scientific-name match against labels.csv (with fuzzy support)
    if target_col_idx is None:
        import difflib  # Standard lib; for fuzzy matching
        
        sci = scientific_name.lower().strip()
        if sci in model.inat_index:
            target_col_idx = model.inat_index[sci]
            target_col_name = model.class_names[target_col_idx]
            print(f"[INFO] Using exact scientific-name match: '{sci}' -> column '{target_col_name}' (idx {target_col_idx})")
        elif not args.no_fuzzy:
            # Fuzzy matching: find best candidate
            def similarity(a: str, b: str) -> float:
                return difflib.SequenceMatcher(None, a, b).ratio()
            
            # Pre-filter candidates (genus match or prefix) for efficiency
            base = sci.replace('-', ' ').replace('_', ' ')
            genus = base.split(' ')[0]
            candidates = [n for n in model.class_names 
                          if genus in n or n.startswith(base[:6])]  # Slightly longer prefix for better filtering
            
            if candidates:
                matches = [(name, similarity(sci, name)) for name in candidates]
                high_sim_matches = [(name, sim) for name, sim in matches if sim > 0.5]
                
                if high_sim_matches:
                    best = max(high_sim_matches, key=lambda x: x[1])
                    target_col_name = best[0]
                    target_col_idx = model.inat_index[target_col_name]
                    print(f"[INFO] Using fuzzy scientific-name match: '{sci}' -> '{target_col_name}' (similarity {best[1]:.3f}, idx {target_col_idx})")
                else:
                    # No high-similarity match; debug and error out
                    print(f"[ERROR] No close match for '{sci}' in model classes (threshold 0.5).")
                    sorted_matches = sorted(matches, key=lambda x: x[1], reverse=True)[:10]
                    if sorted_matches:
                        print(f"[DEBUG] Top 10 similarity matches: {[(n, f'{s:.3f}') for n, s in sorted_matches]}")
                    return
            else:
                print(f"[ERROR] No candidates for '{sci}' (genus: '{genus}').")
                return
        else:
            print(f"[ERROR] No exact match for '{sci}' and --no_fuzzy enabled.")
            return
    if target_col_idx is None:
        print(f"[ERROR] Failed to determine target column for {scientific_name}")
        return
    # --- Process files ---
    tmp_root = Path(tempfile.mkdtemp(prefix="kaggle_perch_"))
    kept_total = 0
    try:
        wav_files = sorted(wav_dir.glob("*.wav"))

        max_wavs = int(os.environ.get("MAX_WAVS_PER_SPECIES", "4000"))

        if max_wavs > 0 and len(wav_files) > max_wavs:
            random.seed(42)
            wav_files = sorted(random.sample(wav_files, max_wavs))
            print(f"[INFO] Limiting to {len(wav_files)} WAV files via MAX_WAVS_PER_SPECIES={max_wavs}")
        else:
            print(f"[INFO] Processing {len(wav_files)} WAV files...")
        for i, wav_file in enumerate(wav_files, 1):
            print(f"\n[Processing {i}/{len(wav_files)}] {wav_file.name}")

            # ✅ SKIP if already processed
            existing = list(out_dir.glob(f"{wav_file.stem}_t*.wav"))
            if existing:
                print(f" [SKIP] Already processed ({len(existing)} segments exist)")
                continue
            
            starts = make_windows(wav_file, args.step, 5.0)
            if not starts:
                print(" [SKIP] File too short")
                continue
            tmp_dir = tmp_root / wav_file.stem
            tmp_dir.mkdir(exist_ok=True)
            temp_files = []
            for j, start_time in enumerate(starts):
                audio, sr = extract_audio_segment(wav_file, start_time, 5.0)
                if len(audio) > 0:
                    temp_path = tmp_dir / f"window_{j:04d}.wav"
                    try:
                        sf.write(temp_path, audio, sr)
                    except Exception as e:
                        print(f"[WARN] Failed writing {temp_path}: {e}")
                        continue
                    temp_files.append((temp_path, start_time))
            if not temp_files:
                continue
            temp_paths = [str(p) for (p, _t) in temp_files]
            predictions = model.predict_batch(temp_paths, args.batch_size)
            if predictions.empty:
                print(" [WARN] No predictions returned")
                continue
            target_scores = predictions.iloc[:, target_col_idx].values
            keep_mask = target_scores >= args.threshold
            if target_scores.size:
                print(f" Target scores: min={target_scores.min():.3f}, max={target_scores.max():.3f}")
                print(f" Windows passing: {int(keep_mask.sum())}/{len(keep_mask)}")
            kept = 0
            for idx, (_temp_path, start_time) in enumerate(list(temp_files)):
                if idx >= len(keep_mask) or not keep_mask[idx]:
                    continue
                # center a 3 s crop within the 5 s window (1.0–4.0 s)
                segment_start = start_time + 1.0
                audio_seg, sr = extract_audio_segment(wav_file, segment_start, args.seg_len)
                if len(audio_seg) > 0:
                    prob = float(target_scores[idx])
                    out_name = f"{wav_file.stem}_t{segment_start:08.2f}_p{prob:.3f}.wav"
                    sf.write(out_dir / out_name, audio_seg, sr)
                    kept += 1
            kept_total += kept
            print(f" Kept {kept} segments")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"\n[DONE] Saved {kept_total} total segments to {out_dir}")

if __name__ == "__main__":
    main()