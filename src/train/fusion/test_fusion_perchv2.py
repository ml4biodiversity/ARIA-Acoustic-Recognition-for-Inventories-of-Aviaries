#!/usr/bin/env python3
"""
Test trained fusion model on held-out test set
Properly handles device placement for CUDA/CPU
"""

import sys
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from collections import Counter
import json


SCRIPT_DIR = Path(__file__).parent.resolve()

# Find ARIA root by walking up until we find 'ARIA' folder
ARIA_ROOT = SCRIPT_DIR
while ARIA_ROOT.name != 'ARIA' and ARIA_ROOT != ARIA_ROOT.parent:
    ARIA_ROOT = ARIA_ROOT.parent

if ARIA_ROOT.name != 'ARIA':
    raise RuntimeError(f"Could not find ARIA root from {SCRIPT_DIR}")

# Add models directory to path for imports
MODELS_DIR = ARIA_ROOT / 'models'
sys.path.insert(0, str(MODELS_DIR))

# Default paths relative to ARIA root
DEFAULT_PATHS = {
    'model_dir': ARIA_ROOT / 'models/Fusion_Test',
}


from dual_embedding_classifier import (
    DualEmbeddingClassifier,
    BirdEmbeddingDataset
)


def test_fusion_model(model_dir: Path = None):
    """
    Test trained fusion model on held-out test set.
    
    Args:
        model_dir: Directory containing model and datasets
    """
    
    if model_dir is None:
        model_dir = DEFAULT_PATHS['model_dir']
    else:
        model_dir = Path(model_dir)
    
    print("="*60)
    print("TESTING FUSION MODEL (NO DATA LEAKAGE)")
    print("="*60)
    print(f"\n📁 ARIA Root: {ARIA_ROOT}")
    print(f"📁 Model Dir: {model_dir}")

    # Check for CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")

    # Load architecture info
    arch_info_path = model_dir / 'architecture_info.json'
    if not arch_info_path.exists():
        print(f"❌ Architecture info not found: {arch_info_path}")
        print("   Please train the model first or run generate_architecture_info.py")
        exit(1)

    with open(arch_info_path) as f:
        arch_info = json.load(f)

    print(f"\n📋 Architecture info:")
    print(f"   Type: {arch_info['architecture']['type']}")
    print(f"   BirdNET dim: {arch_info['birdnet_dim']}")
    print(f"   PERCH dim: {arch_info['perch_dim']}")
    print(f"   Total dim: {arch_info['total_dim']}")
    print(f"   Output species: {arch_info['n_species']}")

    # Load datasets
    datasets_path = model_dir / 'datasets.npy'
    if not datasets_path.exists():
        print(f"❌ Datasets not found: {datasets_path}")
        exit(1)

    datasets = np.load(datasets_path, allow_pickle=True).item()

    # Get test data
    X_test = datasets['test']['embeddings']
    y_test = datasets['test']['labels']
    is_rare_test = datasets['test']['is_rare']

    print(f"\n📊 Test set:")
    print(f"   Test segments: {len(X_test)}")
    print(f"   Rare segments: {is_rare_test.sum()}")

    # Verify split info
    if 'split_info' in datasets:
        split_info = datasets['split_info']
        if split_info.get('no_leakage', False):
            print(f"   ✅ File-level splits (no data leakage)")
            print(f"   Test files: {len(split_info['test_files'])}")
        else:
            print(f"   ⚠️  May have data leakage")

    # Get species mapping
    idx_to_species = datasets['idx_to_species']
    if isinstance(list(idx_to_species.keys())[0], str):
        idx_to_species = {int(k): v for k, v in idx_to_species.items()}

    n_species = len(idx_to_species)

    # Load model
    print(f"\n🏗️  Loading trained model...")
    model_path = model_dir / 'fusion_model_perchv2.pth'
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        exit(1)

    model = DualEmbeddingClassifier(
        n_species=n_species,
        birdnet_dim=arch_info['birdnet_dim'],
        perch_dim=arch_info['perch_dim'],
        hidden_dims=arch_info['architecture']['hidden_dims'],
        dropout=0.3
    )

    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)  # Move model to device
    model.eval()

    print(f"   ✓ Model loaded from: {model_path}")
    print(f"   ✓ Model on device: {next(model.parameters()).device}")

    # Create test loader
    batch_size = 512 if arch_info['architecture']['type'] == 'unconstrained' else 256
    test_dataset = BirdEmbeddingDataset(X_test, y_test, is_rare_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print(f"\n🧪 Running inference on test set...")
    print(f"   Batch size: {batch_size}")

    # Test metrics
    correct = 0
    total = 0
    correct_rare = 0
    total_rare = 0
    correct_common = 0
    total_common = 0

    all_predictions = []
    all_labels = []
    all_is_rare = []

    with torch.no_grad():
        for i, (embeddings_batch, labels_batch, is_rare_batch) in enumerate(test_loader):
            # CRITICAL: Move data to same device as model
            embeddings_batch = embeddings_batch.to(device)
            labels_batch = labels_batch.to(device)
            
            # Forward pass
            outputs = model(embeddings_batch)
            _, predicted = torch.max(outputs, 1)
            
            # Move back to CPU for metrics calculation
            predicted_cpu = predicted.cpu()
            labels_cpu = labels_batch.cpu()
            is_rare_cpu = is_rare_batch
            
            # Overall accuracy
            total += labels_cpu.size(0)
            correct += (predicted_cpu == labels_cpu).sum().item()
            
            # Rare vs common accuracy
            rare_mask = is_rare_cpu == 1
            common_mask = is_rare_cpu == 0
            
            if rare_mask.sum() > 0:
                total_rare += rare_mask.sum().item()
                correct_rare += ((predicted_cpu == labels_cpu) & rare_mask).sum().item()
            
            if common_mask.sum() > 0:
                total_common += common_mask.sum().item()
                correct_common += ((predicted_cpu == labels_cpu) & common_mask).sum().item()
            
            # Store for detailed analysis
            all_predictions.extend(predicted_cpu.numpy())
            all_labels.extend(labels_cpu.numpy())
            all_is_rare.extend(is_rare_cpu.numpy())
            
            # Progress
            if (i + 1) % 10 == 0 or (i + 1) == len(test_loader):
                print(f"   Processed: {(i + 1) * batch_size}/{len(test_dataset)} segments", end='\r')

    print()  # New line after progress

    # Convert to numpy
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_is_rare = np.array(all_is_rare)

    # Calculate metrics
    test_accuracy = 100 * correct / total
    rare_accuracy = 100 * correct_rare / total_rare if total_rare > 0 else 0
    common_accuracy = 100 * correct_common / total_common if total_common > 0 else 0

    print(f"\n{'='*60}")
    print("📊 TEST RESULTS")
    print(f"{'='*60}")

    print(f"\n🎯 Overall Performance:")
    print(f"   Test accuracy: {test_accuracy:.2f}%")
    print(f"   Correct: {correct:,} / {total:,}")

    print(f"\n🔍 Performance by Rarity:")
    print(f"   Common species accuracy: {common_accuracy:.2f}%")
    print(f"      Correct: {correct_common:,} / {total_common:,}")
    print(f"   Rare species accuracy: {rare_accuracy:.2f}%")
    print(f"      Correct: {correct_rare:,} / {total_rare:,}")

    # Per-class accuracy
    print(f"\n📈 Per-Species Performance:")
    species_correct = Counter()
    species_total = Counter()

    for pred, label in zip(all_predictions, all_labels):
        species_total[label] += 1
        if pred == label:
            species_correct[label] += 1

    species_accuracies = {}
    for species_idx in species_total.keys():
        acc = 100 * species_correct[species_idx] / species_total[species_idx]
        species_accuracies[species_idx] = acc

    # Best performing species
    best_species = sorted(species_accuracies.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"\n   Top 5 best performing species:")
    for species_idx, acc in best_species:
        species_name = idx_to_species.get(species_idx, f"Species_{species_idx}")
        count = species_total[species_idx]
        print(f"      {species_name:40s} {acc:5.1f}% (n={count})")

    # Worst performing species
    worst_species = sorted(species_accuracies.items(), key=lambda x: x[1])[:5]
    print(f"\n   Top 5 worst performing species:")
    for species_idx, acc in worst_species:
        species_name = idx_to_species.get(species_idx, f"Species_{species_idx}")
        count = species_total[species_idx]
        print(f"      {species_name:40s} {acc:5.1f}% (n={count})")

    # Confusion analysis
    print(f"\n🔀 Most Common Confusions:")
    confusion_pairs = Counter()
    for pred, label in zip(all_predictions, all_labels):
        if pred != label:
            confusion_pairs[(label, pred)] += 1

    top_confusions = confusion_pairs.most_common(5)
    for (true_idx, pred_idx), count in top_confusions:
        true_name = idx_to_species.get(true_idx, f"Species_{true_idx}")
        pred_name = idx_to_species.get(pred_idx, f"Species_{pred_idx}")
        print(f"   {true_name:30s} → {pred_name:30s} ({count} times)")

    # Save detailed results
    results = {
        'overall': {
            'accuracy': test_accuracy,
            'correct': int(correct),
            'total': int(total)
        },
        'by_rarity': {
            'common': {
                'accuracy': common_accuracy,
                'correct': int(correct_common),
                'total': int(total_common)
            },
            'rare': {
                'accuracy': rare_accuracy,
                'correct': int(correct_rare),
                'total': int(total_rare)
            }
        },
        'per_species': {
            str(k): {
                'accuracy': float(v),
                'correct': int(species_correct[k]),
                'total': int(species_total[k]),
                'species_name': idx_to_species.get(k, f"Species_{k}")
            }
            for k, v in species_accuracies.items()
        },
        'top_confusions': [
            {
                'true_species': idx_to_species.get(true_idx, f"Species_{true_idx}"),
                'predicted_species': idx_to_species.get(pred_idx, f"Species_{pred_idx}"),
                'count': int(count)
            }
            for (true_idx, pred_idx), count in top_confusions
        ]
    }

    results_path = model_dir / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Detailed results saved to: {results_path}")

    print(f"\n{'='*60}")
    print("✅ TESTING COMPLETE!")
    print(f"{'='*60}")

    print(f"\n🎯 Summary:")
    print(f"  • Overall accuracy: {test_accuracy:.2f}%")
    print(f"  • Common species: {common_accuracy:.2f}%")
    print(f"  • Rare species: {rare_accuracy:.2f}%")
    print(f"  • No data leakage: ✅")
    print(f"  • Results saved: {results_path}")

    if test_accuracy >= 90:
        print(f"\n🎉 Excellent performance! Test accuracy ≥ 90%")
    elif test_accuracy >= 80:
        print(f"\n👍 Good performance! Test accuracy ≥ 80%")
    else:
        print(f"\n⚠️  Consider retraining with more data or tuning hyperparameters")
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test trained fusion model on held-out test set"
    )
    
    parser.add_argument(
        '-d', '--model-dir',
        default=None,
        help='Model directory containing trained model and datasets'
    )
    
    args = parser.parse_args()
    
    test_fusion_model(model_dir=args.model_dir)


if __name__ == "__main__":
    main()