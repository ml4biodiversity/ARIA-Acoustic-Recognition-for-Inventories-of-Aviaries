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
    'birdnet_model': ARIA_ROOT / 'models/ZooCustom_v1/ZooCustom_v1.tflite',
    'birdnet_labels': ARIA_ROOT / 'config/WholeBirdsList.txt',
    'perch_mapping': ARIA_ROOT / 'src/fetch/perch_v2_zoo_species_mapping.json',
    'output_dir': ARIA_ROOT / 'models/Fusion_Test',
    'output_file': ARIA_ROOT / 'models/Fusion_Test/all_embeddings_final.npy'}


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
    """
    Extract embeddings using birdnetlib properly.
    SAVES IMMEDIATELY after extraction to prevent data loss.
    """
    
    print("="*60)
    print("EXTRACTING EMBEDDINGS - FIXED VERSION")
    print("="*60)
    print("\n⚠️  IMPORTANT: Will save immediately after extraction")
    print("  to prevent data loss from any errors")
    
    # initialize extractor
    print(f"\n1. Initializing embedding extractor...")
    
    # derive labels path if needed
    if birdnet_labels is None:
        birdnet_labels = birdnet_model.replace('.tflite', '_Labels.txt')
    
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
    
    # determine embedding dimensions
    if use_constrained_perch and extractor.perch_species_indices:
        birdnet_dim = extractor.n_birdnet_species
        perch_dim = len(extractor.perch_species_indices)
    else:
        birdnet_dim = extractor.n_birdnet_species
        perch_dim = 1536  # PERCH 2.0 full embedding
    
    total_dim = birdnet_dim + perch_dim
    print(f"\n   Embedding dimensions:")
    print(f"   BirdNET: {birdnet_dim} (probabilities)")
    print(f"   PERCH: {perch_dim} ({'constrained logits' if use_constrained_perch else 'full embeddings'})")
    print(f"   Expected total: {total_dim}")
    
    # find all audio files
    print(f"\n2. Finding audio files in: {audio_root}")
    audio_root = Path(audio_root)
    
    audio_files = []
    species_names = []
    
    for species_dir in sorted(audio_root.iterdir()):
        if species_dir.is_dir():
            species_name = species_dir.name
            
            # find all .wav files
            wav_files = list(species_dir.glob("*.wav"))
            
            if len(wav_files) > 0:
                print(f"   {species_name}: {len(wav_files)} files")
                
                for wav_file in wav_files:
                    audio_files.append(wav_file)
                    species_names.append(species_name)
    
    print(f"\n   ✓ Found {len(audio_files)} audio files")
    print(f"   ✓ From {len(set(species_names))} species")
    
    if len(audio_files) == 0:
        print(f"\n❌ ERROR: No audio files found in {audio_root}")
        print(f"   Make sure directory contains subdirectories with .wav files")
        return
    
    # extract embeddings
    print(f"\n3. Extracting embeddings...")
    print(f"   Segment length: {segment_length}s")
    print(f"   Overlap: {overlap}s")
    
    all_embeddings = []
    all_species_names = []
    all_filenames = []
    all_segment_info = []
    failed_files = []
    files_with_no_segments = []
    
    with tqdm(total=len(audio_files), desc="Processing files", unit="file") as pbar:
        for audio_path, species_name in zip(audio_files, species_names):
            try:
                # extract embeddings for this file
                embeddings = extractor.extract_embeddings_from_file(str(audio_path))
                
                if len(embeddings) == 0:
                    files_with_no_segments.append(str(audio_path))
                    pbar.update(1)
                    pbar.set_postfix({'species': species_name[:20], 'segments': 0, 'status': 'no_segments'})
                    continue
                
                # store each segment's embedding
                for emb_dict in embeddings:
                    all_embeddings.append(emb_dict['combined'])
                    all_species_names.append(species_name)
                    all_filenames.append(audio_path.name)
                    all_segment_info.append({
                        'file': audio_path.name,
                        'start_time': emb_dict['start_time'],
                        'end_time': emb_dict['end_time']
                    })
                
                pbar.update(1)
                pbar.set_postfix({'species': species_name[:20], 'segments': len(embeddings)})
                
            except Exception as e:
                failed_files.append((str(audio_path), str(e)))
                pbar.update(1)
                pbar.set_postfix({'species': species_name[:20], 'status': 'failed'})
    
    print(f"\n   Extraction complete!")
    print(f"   • Successfully processed: {len(audio_files) - len(failed_files)} files")
    print(f"   • Failed files: {len(failed_files)}")
    print(f"   • Files with no segments: {len(files_with_no_segments)}")
    print(f"   • Total segments extracted: {len(all_embeddings)}")
    
    # check if we have any embeddings
    if len(all_embeddings) == 0:
        print(f"\n❌ ERROR: No embeddings were extracted!")
        print(f"   Possible causes:")
        print(f"   1. Audio files are too short (< {segment_length}s)")
        print(f"   2. BirdNET confidence threshold too high (current: {min_conf})")
        print(f"   3. Audio files are corrupt or empty")
        
        if failed_files:
            print(f"\n   Failed files details:")
            for filepath, error in failed_files[:5]:
                print(f"      {Path(filepath).name}: {error[:100]}")
        
        # still save empty data so we know what happened
        embedding_data = {
            'embeddings': np.array([]),
            'labels': np.array([]),
            'species_names': [],
            'filenames': [],
            'segment_info': [],
            'failed_files': failed_files,
            'files_with_no_segments': files_with_no_segments,
            'metadata': {
                'extraction_failed': True,
                'reason': 'No embeddings extracted',
                'total_files_attempted': len(audio_files),
                'failed_files': len(failed_files),
                'files_with_no_segments': len(files_with_no_segments)
            }
        }
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, embedding_data)
        print(f"\n   Saved empty embedding file for debugging to: {output_path}")
        return
    
    # convert to arrays
    all_embeddings = np.array(all_embeddings, dtype=np.float32)
    
    # create species mapping
    unique_species = sorted(set(all_species_names))
    species_to_idx = {species: idx for idx, species in enumerate(unique_species)}
    idx_to_species = {idx: species for species, idx in species_to_idx.items()}
    
    # convert species names to labels
    labels = np.array([species_to_idx[name] for name in all_species_names])
    
    # prepare embedding data
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
            'actual_dim': all_embeddings.shape[1] if len(all_embeddings.shape) > 1 else 0,
            'temperature': temperature,
            'min_conf': min_conf,
            'segment_length': segment_length,
            'overlap': overlap,
            'constrained_perch': use_constrained_perch,
            'uses_probabilities': True,
            'multi_label': True,
            'birdnet_normalized': False,
            'uses_birdnetlib': True,
            'total_files': len(audio_files),
            'successful_files': len(audio_files) - len(failed_files) - len(files_with_no_segments),
            'total_segments': len(all_embeddings)
        }
    }
    
    # SAVE IMMEDIATELY - before any verification that could crash
    print(f"\n4. SAVING EMBEDDINGS IMMEDIATELY...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # backup old embeddings if they exist
    if output_path.exists():
        backup_path = output_path.parent / f"{output_path.stem}_BACKUP.npy"
        import shutil
        shutil.copy(output_path, backup_path)
        print(f"   💾 Backed up old embeddings to: {backup_path.name}")
    
    np.save(output_path, embedding_data)
    print(f"   ✅ SAVED successfully to: {output_path}")
    print(f"   Your data is safe!")
    
    # now we can safely do verification and printing
    print(f"\n5. Verification (data already saved)...")
    
    try:
        print(f"   ✓ Embeddings shape: {all_embeddings.shape}")
        print(f"   ✓ Total segments: {len(all_embeddings)}")
        
        if len(all_embeddings.shape) > 1:
            print(f"   ✓ Embedding dimension: {all_embeddings.shape[1]}")
            
            if all_embeddings.shape[1] == total_dim:
                print(f"   ✓ Dimension check passed: {all_embeddings.shape[1]} dims")
            else:
                print(f"   ⚠️  Dimension mismatch: got {all_embeddings.shape[1]}, expected {total_dim}")
        
        print(f"   ✓ Labels shape: {labels.shape}")
        print(f"   ✓ Species count: {len(unique_species)}")
        
        print(f"\n   Species mapping:")
        for species, idx in sorted(species_to_idx.items())[:5]:
            count = np.sum(labels == idx)
            print(f"       {idx:3d}. {species:40s} {count:5d} segments")
        
        if len(unique_species) > 5:
            print(f"       ... and {len(unique_species) - 5} more")
        
        if failed_files:
            print(f"\n   ⚠️  Failed files: {len(failed_files)}")
            for filepath, error in failed_files[:5]:
                print(f"      {Path(filepath).name}: {error[:100]}")
        
        if files_with_no_segments:
            print(f"\n   ⚠️  Files with no segments: {len(files_with_no_segments)}")
            for filepath in files_with_no_segments[:5]:
                print(f"      {Path(filepath).name}")
        
        # load and verify saved file
        print(f"\n6. Verifying saved file...")
        loaded = np.load(output_path, allow_pickle=True).item()
        print(f"   ✓ File loads correctly")
        print(f"   ✓ Contains {len(loaded['embeddings'])} embeddings")
        
    except Exception as e:
        print(f"\n   ⚠️  Error during verification (but data is already saved): {e}")
        traceback.print_exc()
    
    print(f"\n{'='*60}")
    print("✅ EMBEDDING EXTRACTION COMPLETE!")
    print(f"{'='*60}")
    print(f"\n📊 Summary:")
    print(f"  • Files processed: {len(audio_files)}")
    print(f"  • Successful: {len(audio_files) - len(failed_files) - len(files_with_no_segments)}")
    print(f"  • Total segments: {len(all_embeddings)}")
    if len(audio_files) > 0 and len(all_embeddings) > 0:
        print(f"  • Average segments per file: {len(all_embeddings) / (len(audio_files) - len(failed_files) - len(files_with_no_segments)):.1f}")
    
    print(f"\n✅ Data saved to: {output_path}")
    print(f"   Your embeddings are safe!")


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
        default='zoostuff/checkpoints/ARIA/models/Fusion_Test/all_embeddings_final.npy',
        help='Output file for embeddings'
    )
    
    parser.add_argument(
        '--birdnet-model',
        default=str(ARIA_ROOT / 'models/ZooCustom_v1/ZooCustom_v1.tflite'),
        help='Path to BirdNET TFLite model'
    )
    
    parser.add_argument(
        '--birdnet-labels',
        default=str(ARIA_ROOT / 'models/ZooCustom_v1/ZooCustom_v1_Labels.txt'),
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
