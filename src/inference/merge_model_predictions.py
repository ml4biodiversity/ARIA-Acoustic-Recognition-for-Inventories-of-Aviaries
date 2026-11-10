#!/usr/bin/env python3
"""
Merge BirdNET, PERCH, and Fusion predictions into each dataset's metadata file.

USAGE:
  python3 merge_model_predictions.py <dataset_name1> [<dataset_name2> ...]
  python3 merge_model_predictions.py --all-with-results   # process every dataset that has PERCH+Fusion result files

No more manual dict editing - paths are computed automatically from the
dataset name using the standard naming convention:
  PERCH:  {ARIA_ROOT}/results/{dataset_name}_PERCH_predictions.csv
  Fusion: {ARIA_ROOT}/results/{dataset_name}_ensemble.csv
"""
import argparse
import os
import sys
import pandas as pd
from pathlib import Path

# ============================================================
# ROOTS
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
ARIA_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_PREPROCESSED_ROOT = Path(os.environ.get("PREPROCESSED_ROOT", "/home/zoodata/preprocessed"))
RESULTS_ROOT = ARIA_ROOT / "results_inference_birdnet"
RESULTS_DIR = ARIA_ROOT / "results"


def get_perch_path(dataset_name: str) -> Path:
    return RESULTS_DIR / f"{dataset_name}_PERCH_predictions.csv"


def get_fusion_path(dataset_name: str) -> Path:
    return RESULTS_DIR / f"{dataset_name}_ensemble.csv"


def discover_all_datasets_with_results():
    """Find every dataset name that has both a PERCH and a Fusion result file."""
    perch_files = {
        f.name.replace("_PERCH_predictions.csv", "")
        for f in RESULTS_DIR.glob("*_PERCH_predictions.csv")
    }
    fusion_files = {
        f.name.replace("_ensemble.csv", "")
        for f in RESULTS_DIR.glob("*_ensemble.csv")
    }
    return sorted(perch_files & fusion_files)


def merge_dataset(dataset_name: str):
    print("\n" + "=" * 60)
    print(f"[DATASET] {dataset_name}")

    perch_path = get_perch_path(dataset_name)
    fusion_path = get_fusion_path(dataset_name)

    if not perch_path.exists():
        print(f"  - SKIP: no PERCH results found at {perch_path}")
        return
    if not fusion_path.exists():
        print(f"  - SKIP: no Fusion results found at {fusion_path}")
        return

    dataset_dir = DEFAULT_PREPROCESSED_ROOT / dataset_name
    if not dataset_dir.exists():
        print(f"  - SKIP: preprocessed folder not found at {dataset_dir}")
        return

    # 1) metadataspeechless
    metadata_files = list(dataset_dir.glob("*metadataspeechless*.xlsx"))
    # 2) metadata_speechless
    if not metadata_files:
        metadata_files = list(dataset_dir.glob("*metadata_speechless*.xlsx"))
    # 3) fallback to metadata.xlsx
    if not metadata_files:
        metadata_files = [
            f for f in dataset_dir.glob("*metadata.xlsx")
            if "_meta.xlsx" not in f.name
        ]
    if not metadata_files:
        print("  - SKIP: No metadata file found")
        return
    metadata_path = metadata_files[0]
    print(f"  - Metadata: {metadata_path}")
    meta_df = pd.read_excel(metadata_path)

    # ========================================================
    # CLEAN FILENAMES
    # ========================================================
    meta_df["filename_clean"] = (
        meta_df["filename"]
        .astype(str)
        .apply(lambda x: Path(x).name)
    )

    # ========================================================
    # PERCH
    # ========================================================
    print(f"  - Loading PERCH: {perch_path}")
    perch_df = pd.read_csv(perch_path)
    perch_df["formatted_prediction"] = (
        perch_df["species_full"].astype(str)
        + " ("
        + perch_df["confidence"].round(4).astype(str)
        + ")"
    )
    perch_grouped = (
        perch_df
        .sort_values(["file_name", "segment_idx", "rank"])
        .groupby(["file_name", "segment_idx"])["formatted_prediction"]
        .apply("\n".join)
        .reset_index()
        .rename(columns={"formatted_prediction": "perch_prediction"})
    )
    perch_grouped["file_name_clean"] = (
        perch_grouped["file_name"].astype(str).apply(lambda x: Path(x).name)
    )
    meta_filenames = set(meta_df["filename_clean"])
    perch_filenames = set(perch_grouped["file_name_clean"])
    overlap_count = len(meta_filenames & perch_filenames)
    print(f"  - [DIAG] PERCH filename match: {overlap_count}/{len(meta_filenames)}")
    if overlap_count == 0 and len(meta_filenames) > 0 and len(perch_filenames) > 0:
        print(f"  - [DIAG] Sample metadata filename: {sorted(meta_filenames)[0]}")
        print(f"  - [DIAG] Sample PERCH filename:     {sorted(perch_filenames)[0]}")
    meta_df = meta_df.merge(
        perch_grouped[["file_name_clean", "perch_prediction"]],
        how="left", left_on="filename_clean", right_on="file_name_clean"
    )

    # ========================================================
    # FUSION
    # ========================================================
    print(f"  - Loading Fusion: {fusion_path}")
    fusion_df = pd.read_csv(fusion_path)
    fusion_df["formatted_prediction"] = (
        fusion_df["Species"].astype(str)
        + " ("
        + fusion_df["Confidence"].round(4).astype(str)
        + ")"
    )
    fusion_grouped = (
        fusion_df
        .sort_values(["File", "Segment"])
        .groupby(["File", "Segment"])["formatted_prediction"]
        .apply("\n".join)
        .reset_index()
        .rename(columns={"formatted_prediction": "fusion_model_prediction"})
    )
    fusion_grouped["File_clean"] = (
        fusion_grouped["File"].astype(str).apply(lambda x: Path(x).name)
    )
    meta_filenames = set(meta_df["filename_clean"])
    fusion_filenames = set(fusion_grouped["File_clean"])
    overlap_count = len(meta_filenames & fusion_filenames)
    print(f"  - [DIAG] Fusion filename match: {overlap_count}/{len(meta_filenames)}")
    if overlap_count == 0 and len(meta_filenames) > 0 and len(fusion_filenames) > 0:
        print(f"  - [DIAG] Sample metadata filename: {sorted(meta_filenames)[0]}")
        print(f"  - [DIAG] Sample Fusion filename:   {sorted(fusion_filenames)[0]}")
    meta_df = meta_df.merge(
        fusion_grouped[["File_clean", "fusion_model_prediction"]],
        how="left", left_on="filename_clean", right_on="File_clean"
    )

    # ========================================================
    # BIRDNET
    # ========================================================
    birdnet_path = RESULTS_ROOT / dataset_name / "all_results.filtered.top3.csv"
    if birdnet_path.exists():
        print(f"  - Loading BirdNET: {birdnet_path}")
        birdnet_df = pd.read_csv(birdnet_path)
        birdnet_df["filename_clean"] = birdnet_df["File"].apply(lambda x: Path(str(x)).name)
        birdnet_df["formatted_prediction"] = (
            birdnet_df["Scientific name"].astype(str)
            + " "
            + birdnet_df["Common name"].astype(str)
            + " ("
            + birdnet_df["Confidence"].round(4).astype(str)
            + ")"
        )
        birdnet_grouped = (
            birdnet_df
            .groupby("filename_clean")["formatted_prediction"]
            .apply("\n".join)
            .reset_index()
            .rename(columns={"formatted_prediction": "birdnet_prediction"})
        )
        meta_filenames = set(meta_df["filename_clean"])
        birdnet_filenames = set(birdnet_grouped["filename_clean"])
        overlap_count = len(meta_filenames & birdnet_filenames)
        print(f"  - [DIAG] BirdNET filename match: {overlap_count}/{len(meta_filenames)}")
        if overlap_count == 0 and len(meta_filenames) > 0 and len(birdnet_filenames) > 0:
            print(f"  - [DIAG] Sample metadata filename: {sorted(meta_filenames)[0]}")
            print(f"  - [DIAG] Sample BirdNET filename:  {sorted(birdnet_filenames)[0]}")
        meta_df = meta_df.merge(birdnet_grouped, how="left", on="filename_clean")
    else:
        print(f"  - No BirdNET results found at {birdnet_path}")
        meta_df["birdnet_prediction"] = ""

    # ========================================================
    # OVERLAP
    # ========================================================
    overlaps = []
    for _, row in meta_df.iterrows():
        overlap_items = []
        perch_text = str(row.get("perch_prediction", ""))
        birdnet_text = str(row.get("birdnet_prediction", ""))
        fusion_text = str(row.get("fusion_model_prediction", ""))
        for line in perch_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line in birdnet_text or line in fusion_text:
                overlap_items.append(line)
        overlaps.append("\n".join(sorted(set(overlap_items))))
    meta_df["overlap"] = overlaps

    # ========================================================
    # CLEANUP
    # ========================================================
    meta_df.drop(
        columns=["filename_clean", "file_name_clean", "File_clean"],
        inplace=True, errors="ignore"
    )
    wanted_cols = ["perch_prediction", "birdnet_prediction", "overlap", "fusion_model_prediction"]
    for col in wanted_cols:
        if col not in meta_df.columns:
            meta_df[col] = ""
    original_cols = [c for c in meta_df.columns if c not in wanted_cols]
    meta_df = meta_df[original_cols + wanted_cols]
    meta_df.fillna("", inplace=True)

    # ========================================================
    # SAVE (with permission-error fallback)
    # ========================================================
    output_path = dataset_dir / f"{dataset_name}_meta.xlsx"
    try:
        meta_df.to_excel(output_path, index=False)
        print(f"  - Saved: {output_path}")
    except PermissionError:
        fallback_dir = ARIA_ROOT / "results_meta_fallback"
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fallback_path = fallback_dir / f"{dataset_name}_meta.xlsx"
        meta_df.to_excel(fallback_path, index=False)
        print(f"  - No write permission at {output_path}")
        print(f"  - Saved instead to: {fallback_path}")


def main():
    parser = argparse.ArgumentParser(description="Merge BirdNET/PERCH/Fusion predictions into dataset metadata")
    parser.add_argument("datasets", nargs="*", help="Dataset name(s) to process")
    parser.add_argument("--all-with-results", action="store_true",
                         help="Process every dataset that has both PERCH and Fusion result files")
    args = parser.parse_args()

    if args.all_with_results:
        dataset_names = discover_all_datasets_with_results()
        print(f"[INFO] Discovered {len(dataset_names)} datasets with both PERCH and Fusion results")
    elif args.datasets:
        dataset_names = args.datasets
    else:
        print("Usage: python3 merge_model_predictions.py <dataset_name1> [<dataset_name2> ...]")
        print("       python3 merge_model_predictions.py --all-with-results")
        sys.exit(1)

    print(f"[INFO] Processing datasets: {dataset_names}")

    for dataset_name in dataset_names:
        merge_dataset(dataset_name)

    print("\n[DONE]")


if __name__ == "__main__":
    main()