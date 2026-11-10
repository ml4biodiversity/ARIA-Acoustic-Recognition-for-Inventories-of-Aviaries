#!/usr/bin/env python3
import argparse, random, math
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import librosa.feature
import librosa.feature.inverse

# ---- Constants ----
MEL_N_MELS = 128
MEL_PWR    = 2.0      # power mel (|A|^2)
N_FFT      = 2048
HOP        = 512
FMIN       = 0
# FMAX uses Nyquist via librosa mel_to_audio defaults

def list_wavs(p: Path):
    return sorted(x for x in p.glob("*.wav"))

def rms(x):
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float64)))

def safe_write_wav(path: Path, y, sr):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), np.ascontiguousarray(y), sr)

# --------- interpolation helpers (axis-aware) ----------
def interp_axis0(S, out_H):
    """Resize along frequency axis (rows) with linear interpolation."""
    H, W = S.shape
    if out_H == H:
        return S
    x_old = np.arange(H, dtype=np.float32)
    x_new = np.linspace(0, H - 1, out_H, dtype=np.float32)
    out = np.empty((out_H, W), dtype=S.dtype)
    for j in range(W):
        out[:, j] = np.interp(x_new, x_old, S[:, j])
    return out

def interp_axis1(S, out_W):
    """Resize along time axis (cols) with linear interpolation."""
    H, W = S.shape
    if out_W == W:
        return S
    x_old = np.arange(W, dtype=np.float32)
    x_new = np.linspace(0, W - 1, out_W, dtype=np.float32)
    out = np.empty((H, out_W), dtype=S.dtype)
    for i in range(H):
        out[i, :] = np.interp(x_new, x_old, S[i, :])
    return out

# --------------- paper ops on mel power ----------------
def y_stretch(S, drop_rate=0.10):
    """Frequency stretch: crop top/bottom then resize back"""
    H, W = S.shape
    low  = np.random.randint(0, max(1, int(drop_rate * H)))
    high = H - np.random.randint(0, max(1, int(drop_rate * H)))
    if high - low < 2:
        return S
    S2 = S[low:high, :]
    return interp_axis0(S2, H)

def x_stretch(S, drop_rate=0.10):
    """Time stretch: crop left/right then resize back"""
    H, W = S.shape
    left  = np.random.randint(0, max(1, int(drop_rate * W)))
    right = W - np.random.randint(0, max(1, int(drop_rate * W)))
    if right - left < 2:
        return S
    S2 = S[:, left:right]
    return interp_axis1(S2, W)

def y_shift(S, shift_rate=0.10):
    """Frequency shift by rolling part and shuffling the exposed strip"""
    H, W = S.shape
    k = np.random.randint(0, max(1, int(shift_rate * H)))
    if k == 0:
        return S
    out = np.empty_like(S)
    if np.random.rand() < 0.5:  # move up
        out[0:H-k, :] = S[k:H, :]
        empty = S[H-k:H, :].copy()
        np.random.shuffle(empty)
        out[H-k:H, :] = empty
    else:  # move down
        out[k:H, :] = S[0:H-k, :]
        empty = S[0:k, :].copy()
        np.random.shuffle(empty)
        out[0:k, :] = empty
    return out

def x_shift(S, shift_rate=0.10):
    """Time shift applied to either left or right half """
    H, W = S.shape
    cut = np.random.randint(0, W)
    L, R = S[:, :cut], S[:, cut:]
    if np.random.rand() < 0.5:
        ow = R.shape[1]
        k = np.random.randint(0, max(1, int(shift_rate * ow)))
        if k > 0:
            R = np.concatenate([R[:, k:], R[:, :k]], axis=1) if np.random.rand() < 0.5 \
                else np.concatenate([R[:, ow-k:], R[:, :ow-k]], axis=1)
        out = np.concatenate([L, R], axis=1)
    else:
        ow = L.shape[1]
        k = np.random.randint(0, max(1, int(shift_rate * ow)))
        if k > 0:
            L = np.concatenate([L[:, k:], L[:, :k]], axis=1) if np.random.rand() < 0.5 \
                else np.concatenate([L[:, ow-k:], L[:, :ow-k]], axis=1)
        out = np.concatenate([L, R], axis=1)
    return out

def raise_to_power(S, max_power=2.0):
    """Elementwise power with Beta(2,2)*max_power """
    p = np.random.beta(2, 2) * max_power
    return np.power(S, p, dtype=S.dtype)

def frequency_mask(S, lamb=0.5, max_height=0.2):
    """Poisson number of horizontal masks; clip height to <= max_height*S.shape[0]."""
    H, W = S.shape
    r = np.random.poisson(lamb)
    if r > 0:
        for j in range(r):
            low, high = np.sort((j * H / max(1, r)) + np.random.randint(0, max(1, H // max(1, r)), 2))
            if np.random.rand() < 0.5:
                high = min(high, low + max_height * H)
            else:
                low = max(low, high - max_height * H)
            S[int(low):int(high), :] = S.min()
    return S

def time_mask(S, lamb=0.5, max_len=0.25):
    """Poisson number of vertical masks; clip width to <= max_len*S.shape[1]."""
    H, W = S.shape
    r = np.random.poisson(lamb)
    if r > 0:
        for j in range(r):
            st, en = np.sort((j * W / max(1, r)) + np.random.randint(0, max(1, W // max(1, r)), 2))
            if np.random.rand() < 0.5:
                en = min(en, st + max_len * W)
            else:
                st = max(st, en - max_len * W)
            S[:, int(st):int(en)] = S.min()
    return S

def add_noise_mel(S, Sn, noise_to_signal_max_ratio=3.0):
    ratio = np.random.random() * noise_to_signal_max_ratio
    return (S + ratio * Sn) / (ratio + 1.0)

def add_vocal_mel(S, Sv, beta1=4, beta2=5):
    lam = np.random.beta(beta1, beta2)
    return lam * S + (1.0 - lam) * Sv

# -------- wav<->mel converters (power mel) --------
def wav_to_mel(y, sr):
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP,
        n_mels=MEL_N_MELS, power=MEL_PWR, fmin=FMIN
    )
    return S

def mel_to_wav(S, sr, length):
    y = librosa.feature.inverse.mel_to_audio(
        M=S, sr=sr, n_fft=N_FFT, hop_length=HOP,
        power=MEL_PWR, fmin=FMIN, length=length
    )
    return y

def augment_one(y, sr, noise_pool):
    """
    Augmentation optimized for focal gamma=1.8.
    Provides moderate regularization to balance learning and generalization.
    
    Expected transforms per sample: ~2.22
    """
    
    # Skip augmentation: 14% of samples remain clean
    if np.random.rand() < 0.14:
        return y
    
    # Convert to mel spectrogram
    S = wav_to_mel(y, sr)
    
    # Spatial transforms: p=0.34-0.35, strength=0.095
    if np.random.rand() < 0.34: S = y_stretch(S, 0.095)
    if np.random.rand() < 0.35: S = x_stretch(S, 0.095)
    if np.random.rand() < 0.34: S = y_shift(S, 0.095)
    if np.random.rand() < 0.35: S = x_shift(S, 0.095)
    
    # Power variation: p=0.35, max=2.18
    if np.random.rand() < 0.35: 
        S = raise_to_power(S, 2.18)
    
    # Noise mixing: p=0.35, ratio=2.75
    if (noise_pool is not None) and len(noise_pool) > 0 and np.random.rand() < 0.35:
        pn = random.choice(noise_pool)
        n, nsr = sf.read(pn, always_2d=False)
        if n.ndim > 1: 
            n = n[:, 0]
        if nsr != sr: 
            n = librosa.resample(n, orig_sr=nsr, target_sr=sr)
        if len(n) < len(y):
            n = np.pad(n, (0, len(y) - len(n)))
        elif len(n) > len(y):
            s0 = np.random.randint(0, len(n) - len(y) + 1)
            n = n[s0:s0 + len(y)]
        Sn = wav_to_mel(n, sr)
        S = add_noise_mel(S, Sn, 2.75)
    
    # SpecAugment masking: λ=0.32
    if np.random.rand() < 0.5:
        S = frequency_mask(S, lamb=0.32, max_height=0.16)
    else:
        S = time_mask(S, lamb=0.32, max_len=0.175)
    
    # Convert back to waveform with RMS normalization
    y_out = mel_to_wav(S, sr, length=len(y))
    r0 = rms(y) + 1e-9
    r1 = rms(y_out) + 1e-9
    y_out = np.clip(y_out * (r0 / r1), -1.0, 1.0)
    
    return y_out
    
# ------------------------------- main -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", required=True)
    ap.add_argument("--noise_root", required=False)
    ap.add_argument("--target_min", type=int, default=50)
    ap.add_argument("--max_per_src_frac", type=float, default=0.40,
                    help="Max fraction of NEW synthetic clips per class that can come from one recording.")
    ap.add_argument("--hard_cap_per_src", type=int, default=8,
                    help="Absolute per-recording cap; relaxed only if target cannot be met.")
    ap.add_argument("--sr", type=int, default=48000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    noise_pool = []
    if args.noise_root:
        noise_pool = [str(p) for p in Path(args.noise_root).glob("*.wav")]

    train_root = Path(args.train_root)
    for cdir in sorted([d for d in train_root.iterdir() if d.is_dir()]):
        cname = cdir.name
        if cname.lower() in ("noise", "background", "other", "silence"):
            continue

        files = list_wavs(cdir)
        n0 = len(files)
        if n0 == 0:
            print(f"[SKIP] {cname}: no originals.")
            continue

        need = max(0, args.target_min - n0)
        if need == 0:
            print(f"[OK] {cname}: {n0} >= target.")
            continue

        print(f"[AUG] {cname}: {n0} -> {args.target_min} (create {need})")

        # Load all sources, resample and enforce exact 3.0 s length
        Y = []
        L = int(args.sr * 3.0)
        for p in files:
            y, sr = sf.read(p, always_2d=False)
            if y.ndim > 1: y = y[:, 0]
            if sr != args.sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=args.sr)
                sr = args.sr
            if len(y) < L: y = np.pad(y, (0, L - len(y)))
            elif len(y) > L: y = y[:L]
            Y.append((p, y))

        # Diversity control: per-recording cap
        per_src_cap = max(1, min(args.hard_cap_per_src, int(math.ceil(args.max_per_src_frac * need))))
        made = 0
        per = {p: 0 for p, _ in Y}
        cursor = 0

        while made < need:
            p, y = Y[cursor]
            if per[p] < per_src_cap:
                y_aug = augment_one(y, args.sr, noise_pool)
                out = cdir / f"{Path(p).stem}_aug{made+1:04d}.wav"
                safe_write_wav(out, y_aug, args.sr)
                per[p] += 1
                made += 1

            cursor = (cursor + 1) % len(Y)

            # If we hit all caps but still need more, relax by +1 and continue.
            if cursor == 0 and made < need and all(v >= per_src_cap for v in per.values()):
                per_src_cap += 1

        print(f"[DONE] {cname}: created {made} augmented clips (final per-recording cap used: {per_src_cap})")

    print("[OK] Augmentation complete.")

if __name__ == "__main__":
    main()
