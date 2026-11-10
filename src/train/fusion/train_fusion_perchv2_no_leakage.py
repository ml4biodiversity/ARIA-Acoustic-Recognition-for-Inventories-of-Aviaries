#!/usr/bin/env python3
"""
Retrain fusion model with FILE-LEVEL splits (no data leakage)
Uses BirdNET probabilities (not logits) for consistency
"""

import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).parent.resolve()

# Find ARIA root by walking up until we find the 'models' and 'config' folders
ARIA_ROOT = SCRIPT_DIR
while not ((ARIA_ROOT / 'models').is_dir() and (ARIA_ROOT / 'config').is_dir()) and ARIA_ROOT != ARIA_ROOT.parent:
    ARIA_ROOT = ARIA_ROOT.parent

if not (ARIA_ROOT / 'models').is_dir():
    raise RuntimeError(f"Could not find ARIA root from {SCRIPT_DIR}")
    
# Add models directory to path for imports
MODELS_DIR = ARIA_ROOT / 'models'
sys.path.insert(0, str(MODELS_DIR))

# Default paths relative to ARIA root
DEFAULT_PATHS = {
    'embeddings': ARIA_ROOT / 'models/Fusion_Test/all_embeddings_final.npy',
    'output_dir': ARIA_ROOT / 'models/Fusion_Test',
}


from dual_embedding_classifier import (
    DualEmbeddingClassifier,
    DualEmbeddingTrainer,
    BirdEmbeddingDataset
)


def train_fusion_model(
    embedding_data_path: str = None,
    output_dir: str = None,
    n_epochs: int = 100,
    batch_size: int = None,  # Auto-detect based on architecture
    learning_rate: float = None,  # Auto-detect based on architecture
):
    """
    Train fusion model with file-level splits (no data leakage).
    
    Args:
        embedding_data_path: Path to embeddings .npy file
        output_dir: Directory to save model and artifacts
        n_epochs: Number of training epochs
        batch_size: Batch size (auto-detected if None)
        learning_rate: Learning rate (auto-detected if None)
    """
    
    # Resolve default paths
    if embedding_data_path is None:
        embedding_data_path = str(DEFAULT_PATHS['embeddings'])
    if output_dir is None:
        output_dir = DEFAULT_PATHS['output_dir']
    else:
        output_dir = Path(output_dir)
    
    print("="*60)
    print("RETRAINING WITH FILE-LEVEL SPLITS (NO DATA LEAKAGE)")
    print("="*60)
    print(f"\n📁 ARIA Root: {ARIA_ROOT}")
    print(f"📁 Script Dir: {SCRIPT_DIR}")
    print("\nKey improvements:")
    print("  • File-level train/val/test splits")
    print("  • No segments from same file in multiple splits")
    print("  • Uses BirdNET probabilities (not logits)")
    print("  • Consistent temperature scaling")

    # Load embeddings
    print("\n📁 Loading embeddings...")
    print(f"   Path: {embedding_data_path}")

    if not Path(embedding_data_path).exists():
        print(f"   ⚠️  Embeddings not found at: {embedding_data_path}")
        print(f"   Please run: python extract_embeddings_perchv2.py")
        exit(1)

    embedding_data = np.load(embedding_data_path, allow_pickle=True).item()

    # Verify file tracking exists
    if 'filenames' not in embedding_data:
        print("❌ ERROR: No 'filenames' key in embeddings!")
        print("   Cannot do file-level splits without file tracking")
        exit(1)

    embeddings = embedding_data['embeddings']
    labels = embedding_data['labels']
    filenames = embedding_data['filenames']

    print(f"   Total embeddings: {len(embeddings)}")
    print(f"   Total files tracked: {len(filenames)}")

    # Verify counts match
    if len(embeddings) != len(labels) or len(embeddings) != len(filenames):
        print(f"❌ ERROR: Length mismatch!")
        print(f"   Embeddings: {len(embeddings)}")
        print(f"   Labels: {len(labels)}")
        print(f"   Filenames: {len(filenames)}")
        exit(1)

    print(f"   ✓ All counts match")

    # Get metadata
    total_dim = embeddings.shape[1]
    metadata = embedding_data.get('metadata', {})

    if metadata:
        print(f"\n📊 Metadata:")
        print(f"   BirdNET dim: {metadata.get('birdnet_dim', 'unknown')}")
        print(f"   PERCH dim: {metadata.get('perch_dim', 'unknown')}")
        print(f"   Temperature: {metadata.get('temperature', 'unknown')}")
        print(f"   Uses probabilities: {metadata.get('uses_probabilities', False)}")
        print(f"   Multi-label: {metadata.get('multi_label', False)}")
        
        if not metadata.get('uses_probabilities', False):
            print(f"\n⚠️  WARNING: Embeddings may not use probabilities!")

    # Detect architecture
    if metadata:
        BIRDNET_DIM = metadata.get('birdnet_dim', 87)
        PERCH_DIM = metadata.get('perch_dim', total_dim - BIRDNET_DIM)
        CONSTRAINED = metadata.get('constrained_perch', PERCH_DIM < 200)
    else:
        if total_dim < 500:
            CONSTRAINED = True
            BIRDNET_DIM = 87
            PERCH_DIM = total_dim - BIRDNET_DIM
        else:
            CONSTRAINED = False
            PERCH_DIM = 1536
            BIRDNET_DIM = total_dim - PERCH_DIM

    if CONSTRAINED:
        HIDDEN_DIMS = [256, 128, 64]
        ARCHITECTURE_TYPE = 'constrained'
    else:
        HIDDEN_DIMS = [1024, 512, 256]
        ARCHITECTURE_TYPE = 'unconstrained'

    print(f"\n✓ Detected architecture:")
    print(f"   Type: {ARCHITECTURE_TYPE.upper()}")
    print(f"   BirdNET: {BIRDNET_DIM} dims (probabilities)")
    print(f"   PERCH: {PERCH_DIM} dims")
    print(f"   Total: {total_dim} dims")
    print(f"   Hidden layers: {HIDDEN_DIMS}")

    # Get species mapping
    if 'idx_to_species' in embedding_data:
        idx_to_species = embedding_data['idx_to_species']
    else:
        unique_labels = np.unique(labels)
        idx_to_species = {int(i): f"Species_{i}" for i in unique_labels}

    if isinstance(list(idx_to_species.keys())[0], str):
        idx_to_species = {int(k): v for k, v in idx_to_species.items()}

    n_species = len(idx_to_species)

    print(f"\n📊 Dataset Info:")
    print(f"   Output species: {n_species}")
    print(f"   Total segments: {len(embeddings)}")

    # GROUP BY FILES
    print(f"\n🗂️  Grouping segments by source file...")

    file_to_indices = defaultdict(list)
    for idx, filename in enumerate(filenames):
        file_to_indices[filename].append(idx)

    unique_files = list(file_to_indices.keys())
    print(f"   Unique files: {len(unique_files)}")
    print(f"   Avg segments per file: {len(embeddings) / len(unique_files):.1f}")

    # Get label for each file (use most common label)
    file_labels = []
    for file_path in unique_files:
        indices = file_to_indices[file_path]
        file_labels_for_this_file = labels[indices]
        # Most common label
        most_common = Counter(file_labels_for_this_file).most_common(1)[0][0]
        file_labels.append(most_common)

    file_labels = np.array(file_labels)

    # Determine rare species at FILE level
    label_counts = Counter(file_labels)
    sorted_counts = sorted(label_counts.values())
    threshold_idx = int(len(sorted_counts) * 0.2)
    rare_threshold = sorted_counts[threshold_idx] if threshold_idx < len(sorted_counts) else sorted_counts[-1]

    print(f"\n🔍 Rare species analysis:")
    print(f"   Rare threshold: {rare_threshold} files")
    print(f"   Species below threshold: {sum(1 for c in label_counts.values() if c <= rare_threshold)}")

    # Remove species with <2 files from stratified splitting
    valid_files = []
    valid_labels = []

    singleton_files = []
    singleton_labels = []

    for f, label in zip(unique_files, file_labels):

        if label_counts[label] < 2:
            singleton_files.append(f)
            singleton_labels.append(label)
        else:
            valid_files.append(f)
            valid_labels.append(label)

    print(f"\n⚠️  Singleton species files: {len(singleton_files)}")
    print(f"   These will be added to TRAIN only")

    # SPLIT AT FILE LEVEL
    print(f"\n✂️  Splitting at FILE level (prevents data leakage)...")

    files_temp, files_test, labels_temp, labels_test = train_test_split(
        valid_files, valid_labels, test_size=0.15, stratify=valid_labels, random_state=42
    )

    files_train, files_val, labels_train, labels_val = train_test_split(
        files_temp, labels_temp, test_size=0.15, stratify=labels_temp, random_state=42
    )

    # Add singleton species to train set only
    files_train = list(files_train) + singleton_files
    labels_train = list(labels_train) + singleton_labels

    print(f"   Train files: {len(files_train)}")
    print(f"   Val files: {len(files_val)}")
    print(f"   Test files: {len(files_test)}")

    # Get indices for each split
    train_indices = [idx for f in files_train for idx in file_to_indices[f]]
    val_indices = [idx for f in files_val for idx in file_to_indices[f]]
    test_indices = [idx for f in files_test for idx in file_to_indices[f]]

    print(f"\n   Train segments: {len(train_indices)}")
    print(f"   Val segments: {len(val_indices)}")
    print(f"   Test segments: {len(test_indices)}")

    # Verify no overlap
    train_files_set = set(files_train)
    val_files_set = set(files_val)
    test_files_set = set(files_test)

    overlap_train_val = train_files_set & val_files_set
    overlap_train_test = train_files_set & test_files_set
    overlap_val_test = val_files_set & test_files_set

    if overlap_train_val or overlap_train_test or overlap_val_test:
        print(f"\n❌ ERROR: File overlap detected!")
        print(f"   Train-Val overlap: {len(overlap_train_val)}")
        print(f"   Train-Test overlap: {len(overlap_train_test)}")
        print(f"   Val-Test overlap: {len(overlap_val_test)}")
        exit(1)
    else:
        print(f"\n✅ NO file overlap - splits are clean!")

    # Create datasets
    X_train = embeddings[train_indices]
    y_train = labels[train_indices]

    X_val = embeddings[val_indices]
    y_val = labels[val_indices]

    X_test = embeddings[test_indices]
    y_test = labels[test_indices]

    # Determine rare species for each segment
    is_rare_train = np.array([label_counts[label] <= rare_threshold for label in y_train])
    is_rare_val = np.array([label_counts[label] <= rare_threshold for label in y_val])
    is_rare_test = np.array([label_counts[label] <= rare_threshold for label in y_test])

    print(f"\n   Rare segments in train: {is_rare_train.sum()} / {len(is_rare_train)}")
    print(f"   Rare segments in val: {is_rare_val.sum()} / {len(is_rare_val)}")
    print(f"   Rare segments in test: {is_rare_test.sum()} / {len(is_rare_test)}")

    # Save splits for future use
    datasets = {
        'train': {'embeddings': X_train, 'labels': y_train, 'is_rare': is_rare_train},
        'val': {'embeddings': X_val, 'labels': y_val, 'is_rare': is_rare_val},
        'test': {'embeddings': X_test, 'labels': y_test, 'is_rare': is_rare_test},
        'idx_to_species': idx_to_species,
        'species_to_idx': {v: int(k) for k, v in idx_to_species.items()},
        'split_info': {
            'train_files': files_train,
            'val_files': files_val,
            'test_files': files_test,
            'method': 'file_level_stratified',
            'no_leakage': True
        }
    }

    datasets_path = output_dir / 'datasets.npy'
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(datasets_path, datasets)
    print(f"\n💾 Saved splits to: {datasets_path}")

    # Create model
    print(f"\n🏗️  Creating fusion model...")
    print(f"   Architecture: {ARCHITECTURE_TYPE.upper()}")
    print(f"   Input: {BIRDNET_DIM} (probabilities) + {PERCH_DIM} = {total_dim} dims")

    model = DualEmbeddingClassifier(
        n_species=n_species,
        birdnet_dim=BIRDNET_DIM,
        perch_dim=PERCH_DIM,
        hidden_dims=HIDDEN_DIMS,
        dropout=0.3
    )

    print(f"   Hidden layers: {HIDDEN_DIMS}")
    print(f"   Output: {n_species} species")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create data loaders
    train_dataset = BirdEmbeddingDataset(X_train, y_train, is_rare_train)
    val_dataset = BirdEmbeddingDataset(X_val, y_val, is_rare_val)

    # Auto-detect batch size and learning rate if not provided
    if batch_size is None:
        batch_size = 256 if ARCHITECTURE_TYPE == 'constrained' else 512
    if learning_rate is None:
        learning_rate = 0.001 if ARCHITECTURE_TYPE == 'constrained' else 0.0005

    print(f"\n⚙️  Training configuration:")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=8, pin_memory=True
    )

    # Initialize trainer
    trainer = DualEmbeddingTrainer(
        model=model,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        rare_species_weight=1.2,
        use_focal_loss=True
    )

    # Compute class weights
    print(f"\n⚖️  Computing class weights...")
    train_label_counts = Counter(y_train)
    max_samples = max(train_label_counts.values())

    class_weights_dict = {}
    for species_idx in range(n_species):
        count = train_label_counts.get(species_idx, 1)
        weight = np.sqrt(max_samples / count)
        class_weights_dict[species_idx] = weight

    class_weights = torch.FloatTensor([
        class_weights_dict.get(i, 1.0) for i in range(n_species)
    ])

    print(f"   Min weight: {class_weights.min():.2f}")
    print(f"   Max weight: {class_weights.max():.2f}")
    print(f"   Ratio: {class_weights.max()/class_weights.min():.1f}:1")

    # Show highest weighted species
    sorted_weights = sorted(class_weights_dict.items(), key=lambda x: x[1], reverse=True)
    print(f"\n   Top 5 highest weighted species (rarest):")
    for idx, weight in sorted_weights[:5]:
        species = idx_to_species.get(idx, f"Species_{idx}")
        count = train_label_counts.get(idx, 0)
        print(f"      {species:40s} weight={weight:.2f} (n={count})")

    # Backup old model
    model_filename = 'fusion_model_perchv2.pth'
    old_model = output_dir / model_filename

    if old_model.exists():
        backup = output_dir / f"{model_filename.replace('.pth', '_BACKUP.pth')}"
        import shutil
        shutil.copy(old_model, backup)
        print(f"\n💾 Backed up old model to: {backup.name}")

    # Train
    checkpoint_path = output_dir / f"{model_filename.replace('.pth', '_checkpoint.pth')}"

    print(f"\n🚀 Starting training...")
    print(f"   Using BirdNET PROBABILITIES (not logits)")
    print(f"   Temperature scaling: {metadata.get('temperature', 'unknown')}")
    print(f"   Multi-label preserved: {metadata.get('multi_label', False)}")
    print(f"   Device: {trainer.device}")
    print(f"   FILE-LEVEL SPLITS: No data leakage! ✅")

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        class_weights=class_weights,
        checkpoint_path=str(checkpoint_path),
        start_epoch=0
    )

    # Save final model
    final_model_path = output_dir / model_filename
    torch.save(model.state_dict(), final_model_path)

    print(f"\n✅ Training complete!")
    print(f"   Final model: {final_model_path}")

    # Test on held-out test set
    print(f"\n🧪 Testing on held-out test set (NO files seen during training)...")

    # model.eval()
    # if trainer.device == 'cuda':
    #     model = model.cuda()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    test_dataset = BirdEmbeddingDataset(X_test, y_test, is_rare_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    correct = 0
    total = 0

    with torch.no_grad():
        for embeddings_batch, labels_batch, _ in test_loader:
            # if trainer.device == 'cuda':
            #     embeddings_batch = embeddings_batch.cuda()
            #     labels_batch = labels_batch.cuda()
            embeddings_batch = embeddings_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            outputs = model(embeddings_batch)
            _, predicted = torch.max(outputs, 1)
            
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

    test_accuracy = correct / total * 100

    print(f"   Test accuracy: {test_accuracy:.2f}%")
    print(f"   ✅ This is the TRUE accuracy (no data leakage)")

    # Save architecture info
    arch_info = {
        'total_dim': total_dim,
        'birdnet_dim': BIRDNET_DIM,
        'perch_dim': PERCH_DIM,
        'architecture': {
            'type': ARCHITECTURE_TYPE,
            'hidden_dims': HIDDEN_DIMS,
            'birdnet_dim': BIRDNET_DIM,
            'perch_dim': PERCH_DIM,
            'total_dim': total_dim
        },
        'n_species': n_species,
        'test_accuracy': test_accuracy,
        'model_path': str(final_model_path),
        'temperature': metadata.get('temperature', 1.8),
        'uses_probabilities': True,
        'multi_label': True,
        'split_method': 'file_level_stratified',
        'no_data_leakage': True,
        'species_mapping': {
            'idx_to_species': {str(k): v for k, v in idx_to_species.items()},
            'species_to_idx': {v: str(k) for k, v in idx_to_species.items()}
        }
    }

    arch_info_path = output_dir / 'architecture_info.json'
    import json
    with open(arch_info_path, 'w') as f:
        json.dump(arch_info, f, indent=2)
    print(f"\n📋 Architecture info saved to: {arch_info_path}")

    print(f"\n{'='*60}")
    print(f"✅ TRAINING COMPLETE (NO DATA LEAKAGE)!")
    print(f"{'='*60}")

    print(f"\n🎯 Summary:")
    print(f"  • Architecture: {BIRDNET_DIM} (BirdNET probs) + {PERCH_DIM} (PERCH) = {total_dim} dims")
    print(f"  • Type: {ARCHITECTURE_TYPE.upper()}")
    print(f"  • Split method: FILE-LEVEL (no leakage) ✅")
    print(f"  • Train/Val/Test files: {len(files_train)}/{len(files_val)}/{len(files_test)}")
    print(f"  • Train/Val/Test segments: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    print(f"  • Test accuracy: {test_accuracy:.2f}%")
    print(f"  • Model saved: {final_model_path}")

    print(f"\n✨ Key features:")
    print(f"  • File-level splits (same file never in train+test)")
    print(f"  • BirdNET uses probabilities directly")
    print(f"  • Temperature scaling preserved")
    print(f"  • Multi-label semantics maintained")
    print(f"  • True test accuracy on unseen files")
    
    return history, test_accuracy


def main():
    """Main function with argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Train fusion model with file-level splits (no data leakage)"
    )
    
    parser.add_argument(
        '-i', '--input',
        default=None,  # Will use DEFAULT_PATHS
        help='Path to embeddings .npy file'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        default=None,  # Will use DEFAULT_PATHS
        help='Output directory for model and artifacts'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (auto-detected if not specified)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate (auto-detected if not specified)'
    )
    
    args = parser.parse_args()
    
    train_fusion_model(
        embedding_data_path=args.input,
        output_dir=args.output_dir,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )


if __name__ == "__main__":
    main()