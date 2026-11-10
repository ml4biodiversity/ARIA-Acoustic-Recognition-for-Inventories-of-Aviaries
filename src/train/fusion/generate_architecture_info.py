#!/usr/bin/env python3
"""
Generate architecture_info.json from existing embeddings and datasets
"""

import numpy as np
import json
from pathlib import Path


SCRIPT_DIR = Path(__file__).parent.resolve()

# Find ARIA root by walking up until we find 'ARIA' folder
ARIA_ROOT = SCRIPT_DIR
while ARIA_ROOT.name != 'ARIA' and ARIA_ROOT != ARIA_ROOT.parent:
    ARIA_ROOT = ARIA_ROOT.parent

if ARIA_ROOT.name != 'ARIA':
    raise RuntimeError(f"Could not find ARIA root from {SCRIPT_DIR}")

# Default paths relative to ARIA root
DEFAULT_PATHS = {
    'model_dir': ARIA_ROOT / 'models/Fusion_Test',
}



def generate_architecture_info(model_dir: Path = None):
    """Generate architecture_info.json from embeddings and model."""
    
    if model_dir is None:
        model_dir = DEFAULT_PATHS['model_dir']
    else:
        model_dir = Path(model_dir)
    
    print("="*60)
    print("GENERATING ARCHITECTURE INFO")
    print("="*60)
    print(f"\n📁 ARIA Root: {ARIA_ROOT}")
    print(f"📁 Model Dir: {model_dir}")

    # Load embeddings to get metadata
    embedding_data_path = model_dir / 'all_embeddings_final.npy'
    if not embedding_data_path.exists():
        # Try alternative name
        embedding_data_path = model_dir / 'embeddings.npy'
    
    if not embedding_data_path.exists():
        print(f"❌ Embeddings not found in: {model_dir}")
        print(f"   Looked for: all_embeddings_final.npy or embeddings.npy")
        exit(1)

    embedding_data = np.load(embedding_data_path, allow_pickle=True).item()

    # Get dimensions
    embeddings = embedding_data['embeddings']
    total_dim = embeddings.shape[1]

    print(f"✓ Loaded embeddings: {embeddings.shape}")
    print(f"   From: {embedding_data_path.name}")

    # Get metadata
    metadata = embedding_data.get('metadata', {})

    if metadata:
        print(f"\n✓ Found metadata:")
        BIRDNET_DIM = metadata.get('birdnet_dim', 87)
        PERCH_DIM = metadata.get('perch_dim', total_dim - BIRDNET_DIM)
        CONSTRAINED = metadata.get('constrained_perch', PERCH_DIM < 200)
        TEMPERATURE = metadata.get('temperature', 1.8)
    else:
        print(f"\n⚠️  No metadata, guessing dimensions...")
        if total_dim < 500:
            CONSTRAINED = True
            BIRDNET_DIM = 87
            PERCH_DIM = total_dim - BIRDNET_DIM
        else:
            CONSTRAINED = False
            PERCH_DIM = 1536
            BIRDNET_DIM = total_dim - PERCH_DIM
        TEMPERATURE = 1.8

    # Determine architecture
    if CONSTRAINED:
        HIDDEN_DIMS = [256, 128, 64]
        ARCHITECTURE_TYPE = 'constrained'
    else:
        HIDDEN_DIMS = [1024, 512, 256]
        ARCHITECTURE_TYPE = 'unconstrained'

    print(f"   BirdNET dim: {BIRDNET_DIM}")
    print(f"   PERCH dim: {PERCH_DIM}")
    print(f"   Total dim: {total_dim}")
    print(f"   Architecture: {ARCHITECTURE_TYPE}")
    print(f"   Hidden dims: {HIDDEN_DIMS}")

    # Load datasets to get species mapping
    datasets_path = model_dir / 'datasets.npy'
    if datasets_path.exists():
        datasets = np.load(datasets_path, allow_pickle=True).item()
        idx_to_species = datasets['idx_to_species']
        
        if isinstance(list(idx_to_species.keys())[0], str):
            idx_to_species = {int(k): v for k, v in idx_to_species.items()}
        
        print(f"\n✓ Loaded datasets")
        print(f"   Species: {len(idx_to_species)}")
    else:
        # Fall back to embeddings
        idx_to_species = embedding_data.get('idx_to_species', {})
        if idx_to_species and isinstance(list(idx_to_species.keys())[0], str):
            idx_to_species = {int(k): v for k, v in idx_to_species.items()}
        
        print(f"\n⚠️  Datasets not found, using embeddings")
        print(f"   Species: {len(idx_to_species)}")

    n_species = len(idx_to_species)

    # Check if model exists
    model_path = model_dir / 'fusion_model_perchv2.pth'
    if not model_path.exists():
        print(f"\n⚠️  Model not found: {model_path}")
        print(f"   Architecture info will be created anyway (model_path will be placeholder)")
        model_path_str = str(model_path) + " (NOT YET CREATED)"
    else:
        print(f"\n✓ Model exists: {model_path}")
        model_path_str = str(model_path)

    # Create architecture info
    arch_info = {
        'total_dim': int(total_dim),
        'birdnet_dim': int(BIRDNET_DIM),
        'perch_dim': int(PERCH_DIM),
        'architecture': {
            'type': ARCHITECTURE_TYPE,
            'hidden_dims': HIDDEN_DIMS,
            'birdnet_dim': int(BIRDNET_DIM),
            'perch_dim': int(PERCH_DIM),
            'total_dim': int(total_dim)
        },
        'n_species': int(n_species),
        'test_accuracy': None,  # Will be filled by test script
        'model_path': model_path_str,
        'temperature': float(TEMPERATURE),
        'uses_probabilities': True,
        'multi_label': True,
        'split_method': 'file_level_stratified',
        'no_data_leakage': True,
        'species_mapping': {
            'idx_to_species': {str(k): v for k, v in idx_to_species.items()},
            'species_to_idx': {v: str(k) for k, v in idx_to_species.items()}
        }
    }

    # Save
    output_path = model_dir / 'architecture_info.json'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(arch_info, f, indent=2)

    print(f"\n✅ Architecture info saved!")
    print(f"   Path: {output_path}")

    # Verify
    print(f"\n🔍 Verification:")
    print(f"   Total dim: {arch_info['total_dim']}")
    print(f"   BirdNET: {arch_info['birdnet_dim']} (probabilities)")
    print(f"   PERCH: {arch_info['perch_dim']} (constrained: {CONSTRAINED})")
    print(f"   Architecture: {arch_info['architecture']['type']}")
    print(f"   Hidden layers: {arch_info['architecture']['hidden_dims']}")
    print(f"   Output species: {arch_info['n_species']}")
    print(f"   Model: {arch_info['model_path']}")

    print(f"\n✨ Now you can run: python test_fusion_perchv2.py")
    
    return arch_info


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate architecture_info.json from embeddings"
    )
    
    parser.add_argument(
        '-d', '--model-dir',
        default=None,
        help='Model directory containing embeddings and model files'
    )
    
    args = parser.parse_args()
    
    generate_architecture_info(model_dir=args.model_dir)


if __name__ == "__main__":
    main()