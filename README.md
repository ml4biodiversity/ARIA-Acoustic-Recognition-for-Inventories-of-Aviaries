# ARIA: Acoustic Recognition for Inventories of Aviaries

Bioacoustic pipeline for zoo aviary monitoring. For every recording it runs three models and merges the results into one Excel file per recording folder:

- **BirdNET** (our retrained classifier `ZooCustom_v1`, knows only the trained species)
- **PERCH v2** (large general bird model; top guesses, also for untrained species)
- **Fusion** (small network combining BirdNET and PERCH)

**Step-by-step instructions for colleagues: see `aria_new_data_protocol.docx`** (new recordings, new aviaries, new species).

---

## Setup (once)

```bash
# 1. BirdNET code (used via PYTHONPATH, the pip package needs Python >= 3.11)
cd ~ && git clone https://github.com/kahst/BirdNET-Analyzer.git

# 2. Environment (Python 3.10)
micromamba create -n aria_unified python=3.10 -y
micromamba activate aria_unified
pip install -r requirements.txt
micromamba install -c conda-forge ffmpeg

# 3. Environment variables (add to ~/.bashrc)
export BIRDNET_REPO="$HOME/BirdNET-Analyzer"
export PYTHONPATH="$BIRDNET_REPO:$PYTHONPATH"
export XC_API_KEY="your_key_from_xeno-canto.org"     # only needed for downloads

# 4. Check
bash scripts/check_dependencies.sh
python3 -c "import perch_hoplite"     # must not fail, otherwise the PERCH step crashes
```

**Model files are not in git** (too large, see `.gitignore`). Create them with the retraining steps below, or copy `models/` from a colleague:
`models/ZooCustom_v1.tflite`, `models/ZooCustom_v1_Labels.txt`, `models/Fusion_Test/` (`fusion_model_perchv2.pth`, `architecture_info.json`).
Recordings and training clips (`data/`) are not in git either.

All scripts find their own location, so the folder can be anywhere. Run everything from the ARIA folder.

---

## Everyday use: new recordings

```bash
micromamba activate aria_unified
export PREPROCESSED_ROOT=/path/to/preprocessed        # folder that contains the recording folders
python3 scripts/check_species_in_model.py <folder_name>       # routing + species check
bash scripts/run_existing_site.sh <folder_name> [<folder_name> ...]
python3 scripts/aviary_species_value_counts.py --datasets <folder_name> ...
```

Result: `<folder>/<folder>_meta.xlsx` (columns `birdnet_prediction`, `perch_prediction`, `fusion_model_prediction`, `overlap`), plus
`results/<folder>_ensemble.csv`, `results/<folder>_PERCH_predictions.csv`, `results_inference_birdnet/<folder>/`.
The `_meta.xlsx` files are written after ALL listed folders have finished.

## New aviary / zoo

```bash
python3 scripts/add_zoo_config.py --zoo <zoo_key> --cages <cage1> <cage2> \
  --cage-species <cage1>=cage1.txt <cage2>=cage2.txt --register-pattern          # dry run
# add --apply to write; --cage-pattern CAGE=TEXT sets the folder-name text that routes to a cage
```

## New species (retrain, about 2 days)

1. Add `Scientific name_Common name` to `config/species_list.txt`; if base BirdNET does not know the species (check by genus in its V2.4 labels), also add it to `config/birds_not_in_birdnet.txt`.
2. `python3 scripts/lookup_new_perch_mappings.py "<Scientific name_Common name>"` (PERCH mapping; candidates go to `config/perch_mapping_review.json`, accept with `scripts/accept_perch_candidate.py`).
3. `bash src/fetch_and_segment.sh` (download from Xeno-canto, convert, cut clips). Species with fewer than ~10 clips cannot be trained.
4. Move old `data/train_curated` and `data/test_curated` aside, then `bash src/train.sh` (BirdNET retrain -> `models/ZooCustom_v1.tflite`).
5. Re-extract ALL embeddings (the width changes with the number of labels):
   `scripts/shard_extract.py` -> `scripts/merge_shards.py --labels models/ZooCustom_v1_Labels.txt` -> copy to `models/Fusion_Test/all_embeddings_final.npy`.
6. `python3 src/train/fusion/train_fusion_perchv2_no_leakage.py` and `python3 src/train/fusion/generate_architecture_info.py`.
7. Register the species in the cage (`scripts/add_zoo_config.py`) and run inference on the new folders only.

Safe mini test of the retraining chain (scratch folder, nothing real is touched): `bash scripts/smoke_retrain.sh`.

---

## Project structure

```
ARIA/
├── README.md  requirements.txt  .gitignore
├── config/                       # species lists, zoo config, PERCH helper files
├── scripts/                      # everything you run by hand
│   ├── run_existing_site.sh, run_full_pipeline.sh   # inference + merge for recording folders
│   ├── check_species_in_model.py                    # routing / species coverage check
│   ├── add_zoo_config.py                            # register a new zoo / cage
│   ├── lookup_new_perch_mappings.py, accept_perch_candidate.py
│   ├── aviary_species_value_counts.py
│   ├── shard_extract.py, merge_shards.py            # parallel embedding extraction
│   └── smoke_retrain.sh, check_dependencies.sh, activate_aria.sh
├── src/
│   ├── fetch/                    # xc_fetch.py, PERCH species mappings
│   ├── segment/                  # PERCH segmentation, embedding extraction
│   ├── train/birdnet|fusion/     # BirdNET prep/augment, Fusion training
│   ├── inference/                # run_inference.py, two-tier PERCH/Fusion inference, merge
│   ├── fetch_and_segment.sh
│   └── train.sh
└── data/  models/  results/      # not in git
```

## Key settings

`src/train.sh` and `src/fetch_and_segment.sh` read all their settings from environment variables (see the top of each file), e.g. `EPOCHS`, `MAX_FILES_PER_CLASS`, `MIN_CONF_SEGMENTS`, `PERCH_THRESH`, `SLIST`, `SEG`, `CHK`. This allows test runs in scratch folders without touching real data.

## Troubleshooting

- **No module named perch_hoplite**: the prompt must show `(aria_unified)`.
- **No module named birdnet_analyzer**: set `BIRDNET_REPO` and `PYTHONPATH` (Setup, step 3).
- **XC_API_KEY not set**: get a key at xeno-canto.org and export it.
- **ffmpeg not found**: `micromamba install -c conda-forge ffmpeg`.
- **zoo=None in scripts/check_species_in_model.py**: the folder name is not registered yet, use `scripts/add_zoo_config.py`.
- **CUDA out of memory / broken GPU**: pick working GPUs with `--gpus` in `scripts/shard_extract.py`; check with `nvidia-smi`.

**Last updated**: November 2026