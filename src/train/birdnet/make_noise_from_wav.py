import argparse, subprocess, os, csv, random, sys, re
from pathlib import Path
import soundfile as sf

# --- must match the recording_key heuristic used during prep ---
SEG_MARKERS = re.compile(r'(?:^|[_-])(seg|clip|chunk|part|start|offset|t)\d+(?:\.\d+)?s?', re.I)
TRAILING_NUM  = re.compile(r'([_-])\d+(?:\.\d+)?s?$', re.I)
def recording_key_from_stem(stem: str) -> str:
    s = SEG_MARKERS.sub('', stem)
    s = TRAILING_NUM.sub('', s)
    s = s.strip('_-')
    return s or stem

def analyze_if_needed(wav_root: Path, slist: Path, min_conf: float, threads: int):
    any_csv = next(wav_root.rglob("*.BirdNET.results.csv"), None)
    if any_csv:
        return
    if not slist or not slist.exists():
        return
    cmd = [
        sys.executable, "-m", "birdnet_analyzer.analyze", str(wav_root),
        "-o", str(wav_root),
        "--rtype","csv","--min_conf", str(min_conf),
        "--slist", str(slist),
        "--threads", str(threads),
        "--overlap","0","--batch_size","8","--lat","-1","--lon","-1","--week","-1"
    ]
    subprocess.run(cmd, check=True)

def windows(duration: float, seg_len: float, step: float):
    t=0.0
    while t + seg_len <= duration + 1e-6:
        yield (t, t+seg_len)
        t += step

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav_root", required=True)
    ap.add_argument("--slist", required=False)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--label", default="Noise")
    ap.add_argument("--target", type=int, default=2000)
    ap.add_argument("--min_conf", type=float, default=0.15)
    ap.add_argument("--seg_len", type=float, default=3.0)
    ap.add_argument("--step", type=float, default=1.5)
    ap.add_argument("--dilate", type=float, default=0.5)  
    ap.add_argument("--max_per_wav", type=int, default=10)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude_from", nargs="*", default=[])
    args = ap.parse_args()
    random.seed(args.seed)

    wav_root = Path(args.wav_root)
    out_dir = Path(args.out_dir)/args.label
    out_dir.mkdir(parents=True, exist_ok=True)


    if args.slist:
        analyze_if_needed(wav_root, Path(args.slist), args.min_conf, args.threads)

    # 1) Build a set of "recording keys" used by curated segments (train + test)
    used_recording_keys = set()
    for pdir in args.exclude_from:
        p = Path(pdir)
        if not p.exists(): continue
        for seg in p.rglob("*.wav"):
            used_recording_keys.add(recording_key_from_stem(seg.stem))

    # 2) Collect noise *only* from WAV files whose recording key is NOT used
    candidates = []
    for wav in wav_root.rglob("*.wav"):
        try:
            info = sf.info(str(wav)); dur = info.frames / info.samplerate
        except Exception:
            continue
        rk = recording_key_from_stem(wav.stem)
        if rk in used_recording_keys:
            continue  # never sample noise from recordings that yielded curated segments
        for (s,e) in windows(dur, args.seg_len, args.step):
            candidates.append((wav, s, args.seg_len))

    if not candidates:
        print("[WARN] No noise candidates found (try pointing --wav_root to other ambience).")
        return

    random.shuffle(candidates)
    selected=[]; per={}
    for wav, start, seglen in candidates:
        if len(selected) >= args.target: break
        if per.get(wav,0) >= args.max_per_wav: continue
        selected.append((wav, start, seglen))
        per[wav] = per.get(wav,0) + 1

    for i,(wav, start, seglen) in enumerate(selected,1):
        out = out_dir / f"{wav.stem}_noise_{i:05d}.wav"
        cmd = ["ffmpeg","-hide_banner","-loglevel","error","-y","-ss",f"{start:.3f}",
               "-t",f"{seglen:.3f}","-i",str(wav), "-ac","1","-ar","48000", str(out)]
        subprocess.run(cmd, check=True)

    print(f"[OK] Wrote {len(selected)} noise segments to {out_dir}")

if __name__ == "__main__":
    main()
