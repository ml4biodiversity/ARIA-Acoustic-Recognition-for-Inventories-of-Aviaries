# ARIA: Acoustic Recognition for Inventories of Aviaries

A bioacoustic machine learning pipeline for zoo soundscape monitoring, combining BirdNET and PERCH models for accurate bird species detection.


## Quick Start

### 1. Clone BirdNET Repository
```bash
cd ~ && git clone https://github.com/kahst/BirdNET-Analyzer.git
```

### 2. Create Unified Environment
```bash
micromamba create -n aria_unified python=3.10 -y
micromamba activate aria_unified
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
cat >> ~/.bashrc << 'BASHRC'
export BIRDNET_REPO="$HOME/BirdNET-Analyzer"
export PYTHONPATH="$BIRDNET_REPO:$PYTHONPATH"
export XC_API_KEY="your_key_from_xeno-canto.org"
BASHRC
source ~/.bashrc
```

#### 3.1 Install FFmpeg
```bash
# Option 1: Via micromamba (recommended)
micromamba install -c conda-forge ffmpeg

# Option 2: System-wide (requires sudo)
sudo apt-get install ffmpeg  # Ubuntu/Debian
brew install ffmpeg          # macOS
```

### 4. Verify Installation
```bash
bash check_dependencies.sh
```

### 5. Run Pipeline
```bash
bash src/fetch_and_segment.sh        # Fetch & prepare data
bash src/train.sh                     # Train BirdNET classifier
python3 src/inference/run_inference.py  # Run inference
```


---

## Project Structure
```
ARIA/
├── config/
│   ├── species_list.txt              # Target species (Genus species_Common Name)
│   ├── birds_not_in_birdnet.txt     # Species requiring PERCH
│   ├── zoo_species_config.json      # Zoo-specific species lists
│   ├── perch_labels.csv & ebird_indices.txt
├── src/
│   ├── fetch/xc_fetch.py            # Xeno-canto downloader
│   ├── segment/perch_segment.py     # PERCH segmentation
│   ├── train/                       # prep_curated.py, augment, noise generation
│   ├── inference/                   # run_inference.py, merge_predictions.py
│   ├── fetch_and_segment.sh         # Data pipeline
│   └── train.sh                     # Training pipeline
├── data/                            # Raw, WAV, segments, train/test (gitignored)
├── models/                          # Trained .tflite models
├── results_inference_birdnet/       # Inference outputs (gitignored)
└── prediction_summaries/            # Excel reports (gitignored)
```

---

## Requirements

### System
- **Python 3.10** (required for compatibility)
- **FFmpeg**: `sudo apt-get install ffmpeg`
- **Git**: For cloning BirdNET-Analyzer
- **GPU**: Optional (NVIDIA with CUDA for PERCH acceleration)

### Python Environment
Single unified environment supporting both BirdNET and PERCH:
- **aria_unified**: TensorFlow 2.20.0 (training, inference, and PERCH segmentation)

See `requirements.txt` for all dependencies.

### Why Cloned BirdNET Repo?
The `birdnet-analyzer` pip package requires Python ≥3.11, but we need Python 3.10 for TF compatibility. So we clone the repo and use `PYTHONPATH`. Also for retraining.

---

## Configuration

### 1. Species List
Edit `config/species_list.txt`:
```
Aegypius monachus_Cinereous Vulture
Anser cygnoid_Swan Goose
Anser erythropus_Lesser White-fronted Goose
Anthropoides virgo_Demoiselle Crane
Ardea goliath_Goliath Heron
Aythya baeri_Baer's Pochard
Aythya fuligula_Tufted Duck
Balearica pavonina_Black Crowned Crane
```

### 2. Unknown Species
Edit `config/birds_not_in_birdnet.txt` for species not in BirdNET's taxonomy:
```
Balearica pavonina_Black Crowned Crane

```
These use PERCH segmentation instead.

### 3. Zoo Configuration
Edit `config/zoo_species_config.json`:
```json
{
  "wild_birds": [
    "Common Blackbird",
    "House Sparrow"
  ],
  "zoos": {
    "beekse_bergen": {
      "all_birds": [
        "Eurasian Griffon",
        "Greater Flamingo",
        "Hadada Ibis"
      ],
      "cages": {
        "flamingos": [
          "Greater Flamingo",
          "Hadada Ibis"
        ],
      }
    }
  }
}
```

---

## Usage

### Workflow Overview
1. **Fetch data** → Download from Xeno-canto
2. **Segment** → Create 3s training clips (BirdNET or PERCH)
3. **Train** → Custom BirdNET classifier
4. **Inference** → Detect species in zoo recordings
5. **Reports** → Excel summaries with predictions

### Phase 1: Data Collection
```bash
micromamba activate aria_unified

# Full pipeline
bash src/fetch_and_segment.sh

# Or step-by-step
ONLY_STEP=fetch bash src/fetch_and_segment.sh     # Download MP3s
ONLY_STEP=convert bash src/fetch_and_segment.sh   # Convert to WAV
ONLY_STEP=analyze bash src/fetch_and_segment.sh   # BirdNET analysis
ONLY_STEP=segments bash src/fetch_and_segment.sh  # Create 3s clips
```

**Output**: `data/segments/[species]/` with 3-second WAV clips

### Phase 2: Training
```bash
micromamba activate aria_unified
bash src/train.sh
```

**Features**:
- Leak-free train/test split (15% test, by recording)
- Noise class generation (non-event suppression)
- Advanced augmentation (mel-spectrogram, focal γ=1.8)
- Focal loss (γ=1.5, α=0.25) for long-tailed distributions
- Linear upsampling (60%) for minority classes

**Output**: `models/ZooCustom_v1/ZooCustom_v1.tflite`


### Phase 3: Inference
```bash
micromamba activate aria_unified

# Standard inference (T=1.8 temperature scaling)
python3 src/inference/run_inference.py

# Custom parameters
python3 src/inference/run_inference.py \
    --temperature 1.8 \
    --min-conf 0.05 \
    --topk 3 \
    --only fl_beekse_bergen_20250404
```

**Features**:
- Cage-specific species filtering (only checks relevant species)
- Per-class optimal thresholds from evaluation CSV
- Temperature scaling (T=1.8) reduces confidence saturation
- Top-K filtering per time window
- Automatic wild bird detection

**Output**: `results_inference_birdnet/[dataset]/all_results.filtered.top3.csv`

### Phase 4: Reports
```bash
micromamba activate aria_unified

# Auto-discover datasets from /home/zoodata/preprocessed/
python3 src/inference/merge_model_predictions.py

# Custom paths
python3 src/inference/merge_model_predictions.py \
    --preprocessed-root /path/to/data \
    --output-dir custom_reports
```

**Output**: Excel files in `prediction_summaries/` with BirdNET, PERCH, and Fusion predictions

---

## Key Parameters

### Data Pipeline (`fetch_and_segment.sh`)
```bash
MIN_CONF_ANALYZE=0.25        # BirdNET analysis threshold
MIN_CONF_SEGMENTS=0.40       # Segmentation threshold
PERCH_THRESH=0.30            # PERCH probability threshold
SEG_LEN_SECONDS=3            # Segment length
STEP_SECONDS=1.5             # Window hop
```

### Training (`train.sh`)
```bash
MIN_FILES_PER_CLASS=50       # Minimum clips per species
MAX_FILES_PER_CLASS=700      # Maximum clips per species
AUG_TARGET_MIN=50            # Augmentation target
EPOCHS=200                   # Training epochs
BATCH_TR=112                 # Batch size
LR=6e-5                      # Learning rate
FOCAL_GAMMA=1.5              # Focal loss gamma
UPSAMPLING_RATIO=0.60        # Minority class upsampling
```

### Inference (`run_inference.py`)
```bash
--temperature 1.8            # Confidence scaling (>1.0 reduces saturation)
--min-conf 0.05              # Global threshold
--topk 3                     # Top-K per window
--threads 40                 # CPU parallelization
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'birdnet_analyzer'"
```bash
# Ensure environment variables are set
export BIRDNET_REPO="$HOME/BirdNET-Analyzer"
export PYTHONPATH="$BIRDNET_REPO:$PYTHONPATH"
source ~/.bashrc

# Test
python -m birdnet_analyzer.analyze --help
```

### "XC_API_KEY not set"
```bash
# Get key from: https://xeno-canto.org/api/2/
export XC_API_KEY="your_key_here"
echo 'export XC_API_KEY="your_key_here"' >> ~/.bashrc
```

### "ffmpeg not found"
```bash
sudo apt-get install ffmpeg  # Ubuntu
brew install ffmpeg          # macOS
```

### CUDA Out of Memory (PERCH)
```bash
# Reduce batch size
--batch_size 8

# Or use CPU mode (remove --gpu flag)
```

### Wrong Python Version
```bash
python --version  # Must be 3.10.x

# If wrong, recreate environment
micromamba remove -n aria_unified --all
micromamba create -n aria_unified python=3.10 -y
```

### Verify Everything
```bash
bash check_dependencies.sh
```

---

## Environment Variables

Add to `~/.bashrc`:
```bash
# BirdNET setup
export BIRDNET_REPO="$HOME/BirdNET-Analyzer"
export PYTHONPATH="$BIRDNET_REPO:$PYTHONPATH"

# Xeno-canto API (get from: https://xeno-canto.org/api/2/)
export XC_API_KEY="your_key_here"

# Optional
export TFHUB_CACHE_DIR="$HOME/.cache/tfhub"
export CUDA_VISIBLE_DEVICES=0  # GPU selection
```

---


**Last Updated**: February 2026  
