#!/usr/bin/env python3
"""
Extract embeddings using birdnetlib properly - FIXED VERSION.
SAVES IMMEDIATELY after extraction to prevent data loss.
"""

import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
import json
import traceback
warnings.filterwarnings('ignore')

from integrated_embedding_extractor_perchv2_with_raw_embeddings import IntegratedEmbeddingExtractor

SCRIPT_DIR = Path(__file__).parent.resolve()  # src/segment/
ARIA_ROOT = SCRIPT_DIR.parent.parent          # ARIA/

#Default paths relative to ARIA root
DEFAULT_PATHS = {
    'birdnet_model': ARIA_ROOT / 'models/ZooCustom_v1.tflite',
    'birdnet_labels': ARIA_ROOT / 'models/ZooCustom_v1_Labels.txt',
    'perch_mapping': ARIA_ROOT / 'src/fetch/perch_v2_zoo_species_mapping.json',
    'output_dir': ARIA_ROOT / 'models/Fusion_Test',
    'output_file': ARIA_ROOT / 'models/Fusion_Test/all_embeddings_final.npy'}

def normalize_species_name(name):
    return (
        name.lower()
        .replace("'", "")
        .replace("’", "")
        .replace("-", " ")
        .replace("_", " ")
        .strip()
    )


def extract_embeddings_final(
    audio_root: str,
    output_file: str = str(DEFAULT_PATHS['output_file']),
    birdnet_model: str = str(DEFAULT_PATHS['birdnet_model']),
    birdnet_labels: str = str(DEFAULT_PATHS['birdnet_labels']),
    perch_species_mapping: str = str(DEFAULT_PATHS['perch_mapping']),
    temperature: float = 1.8,
    min_conf: float = 0.01,
    use_constrained_perch: bool = True,
    segment_length: float = 3.0,
    overlap: float = 0.0
):

    print("=" * 60)
    print("INCREMENTAL EMBEDDING EXTRACTION")
    print("=" * 60)

    output_path = Path(output_file).resolve()

    print(f"\n📁 USING EMBEDDINGS FILE:")
    print(output_path)

    # ============================================================
    # LOAD EXISTING EMBEDDINGS
    # ============================================================

    existing_data = None
    existing_species = set()
    existing_species_to_idx = {}

    if output_path.exists():

        print(f"\n📂 Existing embeddings found:")
        print(f"   {output_path}")

        try:
            existing_data = np.load(output_path, allow_pickle=True).item()
            if 'embeddings' not in existing_data:
                raise ValueError("Invalid embedding file format")

            existing_species = set(
                normalize_species_name(s)
                for s in existing_data.get('species_names', [])
            )

            existing_species_to_idx = existing_data.get(
                'species_to_idx',
                {}
            )

            print(f"   ✓ Existing species: {len(existing_species)}")
            print(f"   ✓ Existing embeddings: {len(existing_data['embeddings'])}")

        except Exception as e:
            print(f"\n❌ Failed to load existing embeddings:")
            print(e)
            return

    # ============================================================
    # INITIALIZE EXTRACTOR
    # ============================================================

    extractor = IntegratedEmbeddingExtractor(
        birdnet_model_path=birdnet_model,
        birdnet_labels_path=birdnet_labels,
        perch_species_mapping_path=perch_species_mapping,
        min_conf=min_conf,
        temperature=temperature,
        use_constrained_perch=use_constrained_perch,
        segment_length=segment_length,
        overlap=overlap
    )

    if use_constrained_perch and extractor.perch_species_indices:
        birdnet_dim = extractor.n_birdnet_species
        perch_dim = len(extractor.perch_species_indices)
    else:
        birdnet_dim = extractor.n_birdnet_species
        perch_dim = 1536

    total_dim = birdnet_dim + perch_dim

    # ============================================================
    # FIND ONLY NEW SPECIES
    # ============================================================

    audio_root = Path(audio_root)
    print(f"\n🔍 Scanning species folders...")

    # ---- Group folders by normalized common name first, so duplicate
    # folders for the same species (e.g. "Scientific_Common" AND a
    # separate bare "Common Name" folder) get merged instead of one
    # silently shadowing/losing the other's real data. ----
    groups = {}
    for species_dir in sorted(audio_root.iterdir()):
        if not species_dir.is_dir():
            continue
        species_name = species_dir.name
        common_name = species_name.split("_", 2)[-1]
        normalized_species = normalize_species_name(common_name)
        groups.setdefault(normalized_species, []).append(species_dir)

    audio_files = []
    species_names = []
    existing_files = set(existing_data.get("filenames", [])) if existing_data else set()

    for normalized_species, dirs in groups.items():
        if any((d / ".embeddings_done").exists() for d in dirs):
            print(f"   ⏭ Skipping completed species: {dirs[0].name}")
            continue
        if normalized_species in existing_species:
            print(f"   ⏭ Skipping existing species: {dirs[0].name}")
            continue

        canonical_dir = next((d for d in dirs if "_" in d.name), dirs[0])
        canonical_label = canonical_dir.name

        wav_files = []
        for d in dirs:
            wav_files.extend(d.glob("*.wav"))

        if len(wav_files) == 0:
            continue

        if len(dirs) > 1:
            print(f"   🔗 Merged {len(dirs)} duplicate folders for '{canonical_label}' "
                  f"({', '.join(d.name for d in dirs)})")
        print(f"   🆕 New species: {canonical_label} ({len(wav_files)} files)")

        for wav_file in wav_files:
            if wav_file.name in existing_files:
                continue
            audio_files.append(wav_file)
            species_names.append(canonical_label)

    print(f"\n✓ New files to process: {len(audio_files)}")
    if len(audio_files) == 0:
        print("\n✅ No new species found.")
        return

    # ============================================================
    # EXTRACT EMBEDDINGS
    # ============================================================

    all_embeddings = []
    all_species_names = []
    all_filenames = []
    all_segment_info = []

    failed_files = []
    files_with_no_segments = []

    with tqdm(total=len(audio_files), desc="Processing") as pbar:

        for audio_path, species_name in zip(audio_files, species_names):

            try:

                embeddings = extractor.extract_embeddings_from_file(
                    str(audio_path)
                )

                if len(embeddings) == 0:
                    files_with_no_segments.append(str(audio_path))
                    pbar.update(1)
                    continue

                for emb_dict in embeddings:

                    all_embeddings.append(emb_dict['combined'])

                    all_species_names.append(species_name)

                    all_filenames.append(audio_path.name)

                    all_segment_info.append({
                        'file': audio_path.name,
                        'start_time': emb_dict['start_time'],
                        'end_time': emb_dict['end_time']
                    })

                pbar.set_postfix({
                'species': species_name[:20],
                'segments': len(embeddings)
            })

                pbar.update(1)

            except Exception as e:

                failed_files.append((str(audio_path), str(e)))
                pbar.update(1)

    # ============================================================
    # CHECK EXTRACTION
    # ============================================================

    if len(all_embeddings) == 0:
        print("\n❌ No embeddings extracted.")
        return

    all_embeddings = np.array(all_embeddings, dtype=np.float32)

    # ============================================================
    # PRESERVE LABEL MAPPINGS
    # ============================================================

    species_to_idx = dict(existing_species_to_idx)

    next_idx = max(species_to_idx.values(), default=-1) + 1

    for species in sorted(set(all_species_names)):

        if species not in species_to_idx:
            species_to_idx[species] = next_idx
            next_idx += 1

    idx_to_species = {
        idx: species
        for species, idx in species_to_idx.items()
    }

    labels = np.array([
        species_to_idx[name]
        for name in all_species_names
    ])

    # ============================================================
    # BUILD NEW DATA
    # ============================================================

    embedding_data = {

        'embeddings': all_embeddings,
        'labels': labels,
        'species_names': all_species_names,
        'filenames': all_filenames,
        'segment_info': all_segment_info,
        'species_to_idx': species_to_idx,
        'idx_to_species': idx_to_species,
        'failed_files': failed_files,
        'files_with_no_segments': files_with_no_segments,

        'metadata': {
            'birdnet_dim': birdnet_dim,
            'perch_dim': perch_dim,
            'total_dim': total_dim,
            'total_segments': len(all_embeddings)
        }
    }

    # ============================================================
    # MERGE WITH EXISTING DATA
    # ============================================================

    if existing_data is not None:

        print("\n🔄 Merging with existing embeddings...")

        # Ensure embedding dimensions match
        old_dim = existing_data['embeddings'].shape[1]
        new_dim = embedding_data['embeddings'].shape[1]

        if old_dim != new_dim:
            raise ValueError(
                f"Embedding dimension mismatch! "
                f"Old: {old_dim}, New: {new_dim}"
            )

        embedding_data['embeddings'] = np.concatenate([
            existing_data['embeddings'],
            embedding_data['embeddings']
        ])

        embedding_data['labels'] = np.concatenate([
            existing_data['labels'],
            embedding_data['labels']
        ])

        embedding_data['species_names'] = (
            existing_data['species_names']
            + embedding_data['species_names']
        )

        embedding_data['filenames'] = (
            existing_data['filenames']
            + embedding_data['filenames']
        )

        embedding_data['segment_info'] = (
            existing_data['segment_info']
            + embedding_data['segment_info']
        )

        embedding_data['failed_files'] = (
            existing_data.get('failed_files', [])
            + embedding_data['failed_files']
        )

        embedding_data['files_with_no_segments'] = (
            existing_data.get('files_with_no_segments', [])
            + embedding_data['files_with_no_segments']
        )

        embedding_data['species_to_idx'] = species_to_idx
        embedding_data['idx_to_species'] = idx_to_species

    # ============================================================
    # BACKUP
    # ============================================================

    if output_path.exists():

        import shutil

        backup_path = output_path.parent / (
            output_path.stem + "_BACKUP.npy"
        )

        shutil.copy(output_path, backup_path)

        print(f"\n💾 Backup created:")
        print(f"   {backup_path}")

    # ============================================================
    # SAVE
    # ============================================================

    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_path, embedding_data)

    # ============================================================
    # VERIFY
    # ============================================================

    loaded = np.load(output_path, allow_pickle=True).item()

    print("\n✅ SAVE SUCCESSFUL")
    print(f"   Total embeddings: {len(loaded['embeddings'])}")
    print(f"   Total species: {len(loaded['species_to_idx'])}")
    print(f"   Saved to: {output_path}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract embeddings - saves immediately to prevent data loss"
    )
    
    parser.add_argument(
        'audio_root',
        help='Root directory containing species subdirectories with audio files'
    )
    
    parser.add_argument(
        '-o', '--output',
        default=str(ARIA_ROOT / 'models/Fusion_Test/all_embeddings_final.npy'),
        help='Output file for embeddings'
    )
    
    parser.add_argument(
        '--birdnet-model',
        default=str(ARIA_ROOT / 'models/ZooCustom_v1.tflite'),
        help='Path to BirdNET TFLite model'
    )
    
    parser.add_argument(
        '--birdnet-labels',
        default=str(ARIA_ROOT / 'models/ZooCustom_v1_Labels.txt'),
        help='Path to BirdNET labels (if None, derives from model path)'
    )
    
    
    parser.add_argument(
        '--perch-mapping',
        default=str(ARIA_ROOT / 'src/fetch/perch_v2_zoo_species_mapping.json'),
        help='Path to PERCH species mapping'
    )
    
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.8,
        help='Temperature scaling for BirdNET'
    )
    
    parser.add_argument(
        '--min-conf',
        type=float,
        default=0.01,
        help='Minimum confidence for BirdNET detections'
    )
    
    parser.add_argument(
        '--segment-length',
        type=float,
        default=3.0,
        help='Segment length in seconds'
    )
    
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.0,
        help='Overlap between segments in seconds'
    )
    
    parser.add_argument(
        '--use-full-perch',
        action='store_true',
        help='Use full 1536-dim PERCH 2.0 embeddings instead of constrained logits'
    )
    
    args = parser.parse_args()
    
    extract_embeddings_final(
        audio_root=args.audio_root,
        output_file=args.output,
        birdnet_model=args.birdnet_model,
        birdnet_labels=args.birdnet_labels,
        perch_species_mapping=args.perch_mapping,
        temperature=args.temperature,
        min_conf=args.min_conf,
        use_constrained_perch=not args.use_full_perch,
        segment_length=args.segment_length,
        overlap=args.overlap
    )


if __name__ == "__main__":
    main()
