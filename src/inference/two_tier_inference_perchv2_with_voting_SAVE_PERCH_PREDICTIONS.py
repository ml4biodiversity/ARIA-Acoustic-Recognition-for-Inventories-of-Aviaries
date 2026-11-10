#!/usr/bin/env python3
"""
Three-Tier Inference with Hybrid Ensemble Voting - PERCH V2
============================================================

FEATURES:
- Aligned with integrated_embedding_extractor_perchv2.py
- Uses PERCH v2 (built-in weights)
- Hybrid ensemble voting for multi-model predictions
- Returns top-3 species with original confidence values
- Option to use tier-based fallback OR ensemble voting

NEW: EMBEDDING SAVING
- Saves 93-dim PERCH embeddings (before directory filtering)
- Saves K-dim PERCH embeddings (after directory filtering)
- Saves 1536-dim raw PERCH embeddings

NEW: PERCH PREDICTIONS SAVING
- Saves PERCH v2 predictions to separate CSV file
- Includes file name, start/end time, species, confidence
"""

import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import csv
import json
from datetime import datetime

# ============================================================
# ROBUST PATH HANDLING - works from any directory
# ============================================================
SCRIPT_DIR = Path(__file__).parent.resolve()  # src/inference/

# Find ARIA root by walking up until we find 'ARIA' folder
ARIA_ROOT = SCRIPT_DIR
while ARIA_ROOT.name != 'ARIA' and ARIA_ROOT != ARIA_ROOT.parent:
    ARIA_ROOT = ARIA_ROOT.parent

if ARIA_ROOT.name != 'ARIA':
    raise RuntimeError(f"Could not find ARIA root from {SCRIPT_DIR}")

# Add required directories to path for imports
SEGMENT_DIR = ARIA_ROOT / 'src/segment'
TRAIN_FUSION_DIR = ARIA_ROOT / 'src/train/fusion'
INFERENCE_DIR = ARIA_ROOT / 'src/inference'

sys.path.insert(0, str(SEGMENT_DIR))
sys.path.insert(0, str(TRAIN_FUSION_DIR))
sys.path.insert(0, str(INFERENCE_DIR))

# Default paths relative to ARIA root
DEFAULT_PATHS = {
    'system_config': ARIA_ROOT / 'models/Fusion_Test/architecture_info.json',
    'zoo_config': ARIA_ROOT / 'config/zoo_species_config.json',
    'birdnet_model': ARIA_ROOT / 'models/ZooCustom_v1.tflite',
    'birdnet_labels': ARIA_ROOT / 'models/ZooCustom_v1_Labels.txt',
    'perch_mapping': ARIA_ROOT / 'src/fetch/perch_v2_zoo_species_mapping.json',
    'results_dir': ARIA_ROOT / 'results',
    'embeddings_dir': ARIA_ROOT / 'results_embeddings',
}

print(f"📁 ARIA Root: {ARIA_ROOT}")
print(f"📁 Script Dir: {SCRIPT_DIR}")
# ============================================================

from integrated_embedding_extractor_perchv2_with_raw_embeddings import IntegratedEmbeddingExtractor
from dual_embedding_classifier import DualEmbeddingClassifier
from zoo_config_parser import ZooConfigParser


def normalize_species_name(species: str) -> str:
    """
    Normalize species name to common name only
    
    Handles formats:
    - "Turdus merula_Common Blackbird" → "Common Blackbird"
    - "Common Blackbird" → "Common Blackbird"
    
    Args:
        species: Original species name
    
    Returns:
        Normalized common name
    """
    if "_" in species:
        return species.split("_", 1)[1]
    else:
        return species


def hybrid_ensemble_voting(birdnet_preds: List[Tuple[str, float]], 
                          fusion_preds: List[Tuple[str, float]], 
                          perch_preds: List[Tuple[str, float]],
                          model_weights: Dict[str, float] = None,
                          consensus_boost: float = 0.5,
                          top_k: int = 3,
                          verbose: bool = False) -> List[Tuple[str, float, Dict]]:
    """
    Hybrid ensemble voting with species normalization and flexible top-k
    
    FIXES:
    1. Normalizes species names to avoid duplicates
       (e.g., "Common Blackbird" == "Turdus merula_Common Blackbird")
    2. Returns all predictions if ≤ top_k, otherwise returns top_k
    
    Args:
        birdnet_preds: [(species, confidence), ...]
        fusion_preds: [(species, confidence), ...]
        perch_preds: [(species, confidence), ...]
        model_weights: Weight for each model's contribution
        consensus_boost: Extra score when 2+ models agree
        top_k: Maximum number of species to return
        verbose: Print voting details
    
    Returns:
        List of (species, max_confidence, metadata)
    """
    
    # Default weights
    if model_weights is None:
        model_weights = {
            'BirdNET': 1.0,  # High quality, multi-label
            'Fusion': 0.8,   # Trained on zoo data
            'PERCH': 0.6     # Fallback, constrained
        }
    
    # Collect predictions with NORMALIZED names
    species_data = defaultdict(lambda: {
        'voting_score': 0.0,
        'models': [],
        'confidences': {},
        'max_confidence': 0.0,
        'source_of_max': None,
        'original_names': set(),
        'best_original_name': None
    })
    
    # Process BirdNET predictions
    for species, conf in birdnet_preds:
        normalized = normalize_species_name(species)
        
        species_data[normalized]['voting_score'] += conf * model_weights['BirdNET']
        species_data[normalized]['models'].append('BirdNET')
        species_data[normalized]['confidences']['BirdNET'] = conf
        species_data[normalized]['original_names'].add(species)
        
        if conf > species_data[normalized]['max_confidence']:
            species_data[normalized]['max_confidence'] = conf
            species_data[normalized]['source_of_max'] = 'BirdNET'
            species_data[normalized]['best_original_name'] = species
    
    # Process Fusion predictions
    for species, conf in fusion_preds:
        normalized = normalize_species_name(species)
        
        species_data[normalized]['voting_score'] += conf * model_weights['Fusion']
        species_data[normalized]['models'].append('Fusion')
        species_data[normalized]['confidences']['Fusion'] = conf
        species_data[normalized]['original_names'].add(species)
        
        if conf > species_data[normalized]['max_confidence']:
            species_data[normalized]['max_confidence'] = conf
            species_data[normalized]['source_of_max'] = 'Fusion'
            species_data[normalized]['best_original_name'] = species
    
    # Process PERCH predictions
    for species, conf in perch_preds:
        normalized = normalize_species_name(species)
        
        species_data[normalized]['voting_score'] += conf * model_weights['PERCH']
        species_data[normalized]['models'].append('PERCH')
        species_data[normalized]['confidences']['PERCH'] = conf
        species_data[normalized]['original_names'].add(species)
        
        if conf > species_data[normalized]['max_confidence']:
            species_data[normalized]['max_confidence'] = conf
            species_data[normalized]['source_of_max'] = 'PERCH'
            species_data[normalized]['best_original_name'] = species
    
    # Apply consensus boost
    for normalized, data in species_data.items():
        vote_count = len(data['models'])
        
        if vote_count >= 2:
            data['voting_score'] += consensus_boost
            data['is_consensus'] = True
        else:
            data['is_consensus'] = False
        
        data['vote_count'] = vote_count
    
    # Sort by voting score
    ranked = sorted(species_data.items(),
                   key=lambda x: x[1]['voting_score'],
                   reverse=True)
    
    # FLEXIBLE TOP-K: Return all if ≤ top_k, otherwise return top_k
    if len(ranked) <= top_k:
        selected = ranked
    else:
        selected = ranked[:top_k]
    
    # Format output with BEST original name
    result = []
    for normalized, data in selected:
        metadata = {
            'vote_count': data['vote_count'],
            'is_consensus': data['is_consensus'],
            'models': data['models'],
            'confidences': data['confidences'],
            'source_of_max': data['source_of_max'],
            'voting_score': data['voting_score'],
            'normalized_name': normalized,
            'all_name_variants': list(data['original_names'])
        }
        
        # Use the original name from the source with max confidence
        best_name = data['best_original_name']
        
        result.append((best_name, data['max_confidence'], metadata))
    
    # Verbose output
    if verbose:
        print(f"\n  🗳️  HYBRID ENSEMBLE VOTING")
        print(f"     Input predictions:")
        print(f"       BirdNET: {len(birdnet_preds)} species")
        print(f"       Fusion:  {len(fusion_preds)} species")
        print(f"       PERCH:   {len(perch_preds)} species")
        print(f"\n     After deduplication: {len(species_data)} unique species")
        print(f"     Returning: {len(result)} predictions")
        print(f"\n     Voting results:")
        
        for i, (species, conf, meta) in enumerate(result, 1):
            consensus_mark = "✓" if meta['is_consensus'] else " "
            models_str = "+".join(meta['models'])
            
            if len(meta['all_name_variants']) > 1:
                variants_str = f" [merged {len(meta['all_name_variants'])} variants]"
            else:
                variants_str = ""
            
            print(f"       {i}. {species:30s} conf={conf:.3f} [{consensus_mark}] from {meta['source_of_max']} (votes: {models_str}){variants_str}")
    
    return result


class TwoTierInferenceEnhanced:
    """Three-tier bird classification with zoo-constrained PERCH v2 and ensemble voting"""
    
    def __init__(self,
                 system_config_path: str = None,
                 zoo_config_path: str = None,
                 birdnet_model_path: str = None,
                 birdnet_labels_path: str = None,
                 perch_species_mapping_path: str = None,
                 tier1_threshold: float = 0.85,
                 min_confidence: float = 0.30,
                 perch_min_confidence: float = 0.1,
                 top_k: int = 3,
                 temperature: float = 1.8,
                 segment_length: float = 3.0,
                 overlap: float = 0.0,
                 use_ensemble_voting: bool = True,
                 ensemble_weights: Dict[str, float] = None,
                 consensus_boost: float = 0.5,
                 save_embeddings: bool = False,
                 save_perch_predictions: bool = False,
                 perch_predictions_top_k: int = 5,
                 verbose: bool = False):
        
        # Resolve default paths
        if system_config_path is None:
            system_config_path = str(DEFAULT_PATHS['system_config'])
        if zoo_config_path is None:
            zoo_config_path = str(DEFAULT_PATHS['zoo_config'])
        if birdnet_model_path is None:
            birdnet_model_path = str(DEFAULT_PATHS['birdnet_model'])
        if birdnet_labels_path is None:
            birdnet_labels_path = str(DEFAULT_PATHS['birdnet_labels'])
        if perch_species_mapping_path is None:
            perch_species_mapping_path = str(DEFAULT_PATHS['perch_mapping'])
        
        self.tier1_threshold = tier1_threshold
        self.min_confidence = min_confidence
        self.perch_min_confidence = perch_min_confidence
        self.top_k = top_k
        self.verbose = verbose
        self.use_ensemble_voting = use_ensemble_voting
        self.consensus_boost = consensus_boost
        self.save_embeddings = save_embeddings
        self.save_perch_predictions = save_perch_predictions
        self.perch_predictions_top_k = perch_predictions_top_k
        
        # Default ensemble weights
        if ensemble_weights is None:
            self.ensemble_weights = {'BirdNET': 1.0, 'Fusion': 0.8, 'PERCH': 0.6}
        else:
            self.ensemble_weights = ensemble_weights
        
        self.allowed_species_set: Optional[Set[str]] = None
        self.zoo_birds_set: Optional[Set[str]] = None
        self.constraint_active = False
        self.allowed_indices: Optional[List[int]] = None
        
        # ============ Embedding storage ============
        self.collected_embeddings_93dim = []
        self.collected_embeddings_filtered = []
        self.collected_embeddings_1536dim = []
        self.collected_file_paths = []
        self.collected_file_names = []
        self.collected_segment_indices = []
        self.collected_start_times = []
        self.collected_end_times = []
        
        # ============ PERCH predictions storage ============
        self.collected_perch_predictions = []
        # ========================================================
        
        print("="*60)
        print("THREE-TIER INFERENCE - PERCH V2 WITH ENSEMBLE VOTING")
        print("="*60)
        print(f"📁 ARIA Root: {ARIA_ROOT}")
        
        # Load system config
        print(f"\n📋 Loading system config: {system_config_path}")
        with open(system_config_path, 'r') as f:
            self.config = json.load(f)
        
        arch = self.config['architecture']
        self.use_constrained = (arch['type'] == 'constrained')
        self.birdnet_dim = arch['birdnet_dim']
        self.perch_dim = arch['perch_dim']
        
        print(f"✓ Architecture: {arch['type'].upper()}")
        print(f"✓ BirdNET dim: {self.birdnet_dim}")
        print(f"✓ PERCH dim: {self.perch_dim} (constrained: {self.use_constrained})")
        
        self.idx_to_species = {int(k): v for k, v in 
                              self.config['species_mapping']['idx_to_species'].items()}
        self.species_to_idx = {v: int(k) for k, v in self.idx_to_species.items()}
        self.n_species = self.config['n_species']
        
        print(f"✓ Species: {self.n_species}")
        
        # Initialize zoo config
        print(f"\n🦁 Loading zoo configuration: {zoo_config_path}")
        self.zoo_parser = ZooConfigParser(zoo_config_path)
        
        # Initialize embedding extractor
        print(f"\n🔧 Initializing embedding extractor (PERCH v2)...")
        print(f"   BirdNET model: {birdnet_model_path}")
        print(f"   BirdNET labels: {birdnet_labels_path}")
        print(f"   PERCH mapping: {perch_species_mapping_path}")
        
        self.extractor = IntegratedEmbeddingExtractor(
            birdnet_model_path=birdnet_model_path,
            birdnet_labels_path=birdnet_labels_path,
            perch_species_mapping_path=perch_species_mapping_path,
            min_conf=0.01,
            temperature=temperature,
            use_constrained_perch=self.use_constrained,
            segment_length=segment_length,
            overlap=overlap
        )
        
        print(f"✓ BirdNET species: {len(self.extractor.birdnet_species)}")
        print(f"✓ PERCH species (constrained): {len(self.extractor.perch_species_indices) if self.extractor.perch_species_indices else 'N/A'}")
        
        # Initialize fusion model
        print(f"\n🤖 Loading fusion model...")
        
        # Handle model path - could be absolute or relative in config
        model_path_from_config = self.config['model_path']
        if Path(model_path_from_config).is_absolute():
            fusion_model_path = Path(model_path_from_config)
        else:
            # Try relative to ARIA root
            fusion_model_path = ARIA_ROOT / model_path_from_config
            if not fusion_model_path.exists():
                # Try in same directory as config
                fusion_model_path = Path(system_config_path).parent / 'fusion_model_perchv2.pth'
        
        self.fusion_model = DualEmbeddingClassifier(
            n_species=self.n_species,
            birdnet_dim=self.birdnet_dim,
            perch_dim=self.perch_dim,
            hidden_dims=arch['hidden_dims'],
            dropout=0.3
        )
        
        # Load to CPU first to avoid CUDA deserialization errors
        print(f"   Loading weights from: {fusion_model_path}")
        self.fusion_model.load_state_dict(torch.load(fusion_model_path, map_location='cpu'))
        
        # Then move to appropriate device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            self.fusion_model.to(self.device)
            self.fusion_model.eval()
            print(f"✓ Model loaded on {self.device}")
            
            if self.device.type == 'cuda':
                try:
                    print(f"   GPU: {torch.cuda.get_device_name(0)}")
                    print(f"   Available GPUs: {torch.cuda.device_count()}")
                except:
                    print(f"   ⚠️  CUDA device info unavailable (but model on CUDA)")
        except RuntimeError:
            print(f"   ⚠️  CUDA failed, falling back to CPU")
            self.device = torch.device('cpu')
            self.fusion_model.to(self.device)
            self.fusion_model.eval()
            print(f"✓ Model loaded on {self.device}")
            print(f"   Note: TensorFlow (PERCH) will still use GPU")
        
        print(f"\n✅ Inference ready!")
        print(f"   Mode: {'ENSEMBLE VOTING' if self.use_ensemble_voting else 'TIER-BASED FALLBACK'}")
        if self.use_ensemble_voting:
            print(f"   Ensemble weights: BirdNET={self.ensemble_weights['BirdNET']}, Fusion={self.ensemble_weights['Fusion']}, PERCH={self.ensemble_weights['PERCH']}")
            print(f"   Consensus boost: {self.consensus_boost}")
        else:
            print(f"   Tier1 threshold: {self.tier1_threshold}")
            print(f"   Min confidence (T1/T2): {self.min_confidence}")
            print(f"   PERCH min confidence (T3): {self.perch_min_confidence}")
        
        # Print saving status
        if self.save_embeddings:
            print(f"   💾 Embedding saving: ENABLED")
        if self.save_perch_predictions:
            print(f"   📝 PERCH prediction saving: ENABLED (top-{self.perch_predictions_top_k})")
    
    def set_constraints_from_directory(self, directory: str):
        """Set species constraints from directory and build allowed indices"""
        print(f"\n🔒 Setting species constraints...")
        
        species_list = self.zoo_parser.get_species_from_directory(directory)
        self.allowed_species_set = set(species_list)
        self.constraint_active = True
        
        wild_birds = self.zoo_parser.wild_birds
        self.zoo_birds_set = self.allowed_species_set - wild_birds
        
        # Build list of allowed indices in BirdNET species list
        self.allowed_indices = []
        for idx, species in enumerate(self.extractor.birdnet_species):
            if self.is_species_allowed(species):
                self.allowed_indices.append(idx)
        
        print(f"✓ Allowed species: {len(self.allowed_species_set)}")
        print(f"✓ Zoo-specific: {len(self.zoo_birds_set)}")
        print(f"✓ Allowed BirdNET indices: {len(self.allowed_indices)}/{len(self.extractor.birdnet_species)}")
        # PERCH-only extra species that belong to this cage
        self.allowed_extra_idx = [i for i, s in enumerate(getattr(self.extractor, 'perch_extra_species', []))
                                  if self.is_species_allowed(s)]
        print(f"✓ PERCH-only extra species in this cage: "
              f"{[self.extractor.perch_extra_species[i] for i in self.allowed_extra_idx]}")
        
        if self.save_embeddings:
            print(f"✓ Filtered embedding dim: {len(self.allowed_indices)}")
    
    def is_species_allowed(self, species_name: str) -> bool:
        if not self.constraint_active or self.allowed_species_set is None:
            return True

        if "_" in species_name:
            common_name = species_name.split("_", 1)[1]
        else:
            common_name = species_name

        def norm(x):
            return " ".join(
                x.lower()
                .replace("-", " ")
                .replace("'", "")
                .replace("’", "")
                .split()
            )

        allowed_norm = {norm(s) for s in self.allowed_species_set}
        return norm(common_name) in allowed_norm
    
    def get_birdnet_predictions(self, probabilities: np.ndarray) -> List[Tuple[str, float]]:
        """Get BirdNET predictions"""
        predictions = []
        
        for idx, prob in enumerate(probabilities):
            if prob >= self.min_confidence and idx < len(self.extractor.birdnet_species):
                species = self.extractor.birdnet_species[idx]
                if self.is_species_allowed(species):
                    predictions.append((species, float(prob)))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions
    
    def get_fusion_predictions(self, combined_embedding: np.ndarray) -> List[Tuple[str, float]]:
        """Get predictions from fusion model"""
        with torch.no_grad():
            emb_tensor = torch.FloatTensor(combined_embedding).unsqueeze(0).to(self.device)
            logits = self.fusion_model(emb_tensor)[0]
            probs = F.softmax(logits, dim=0).cpu().numpy()
        
        predictions = []
        for idx in range(len(probs)):
            conf = float(probs[idx])
            if conf >= self.min_confidence and idx in self.idx_to_species:
                species = self.idx_to_species[idx]
                if self.is_species_allowed(species):
                    predictions.append((species, conf))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions
    
    def get_perch_predictions(self, segment_data: Dict, use_threshold: bool = True, verbose: bool = False) -> List[Tuple[str, float]]:
        """
        Get PERCH v2 predictions with zoo-specific filtering and SOFTMAX
        """
        predictions = []
        
        try:
            perch_logits = segment_data.get('perch_embedding', np.array([]))
            
            if len(perch_logits) == 0:
                if verbose:
                    print(f"      [PERCH] No embedding data")
                return predictions
            
            if self.allowed_indices is None or len(self.allowed_indices) == 0:
                if verbose:
                    print(f"      [PERCH] No allowed indices set")
                return predictions
            
            # Filter to allowed species
            filtered_logits = perch_logits[self.allowed_indices]
            extra_idx = getattr(self, 'allowed_extra_idx', [])
            extra_all = segment_data.get('perch_extra_logits', None)
            if extra_idx and extra_all is not None and len(extra_all) > 0:
                extra_logits = np.asarray(extra_all)[extra_idx]
            else:
                extra_idx = []
                extra_logits = np.zeros(0, dtype=np.float32)
            n_main = len(filtered_logits)
            filtered_logits = np.concatenate([filtered_logits, extra_logits])
            
            if verbose:
                print(f"      [PERCH v2] Filtered from {len(perch_logits)} to {len(filtered_logits)} species")
            
            # Apply SOFTMAX
            exp_logits = np.exp(filtered_logits - np.max(filtered_logits))
            filtered_probs = exp_logits / np.sum(exp_logits)
            
            if verbose:
                print(f"      [PERCH v2] Prob sum: {np.sum(filtered_probs):.4f}")
            
            # Build predictions
            for i, prob in enumerate(filtered_probs):
                if not use_threshold or prob >= self.perch_min_confidence:
                    if i < n_main:
                        birdnet_idx = self.allowed_indices[i]
                        species = self.extractor.birdnet_species[birdnet_idx]
                    else:
                        species = self.extractor.perch_extra_species[extra_idx[i - n_main]]
                    predictions.append((species, float(prob)))
        
        except Exception as e:
            if verbose:
                print(f"      [PERCH v2] Error: {e}")
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions
    
    def _collect_perch_predictions(self, perch_preds: List[Tuple[str, float]], 
                                file_path: str, file_name: str, 
                                start_time: float, end_time: float,
                                segment_idx: int,
                                min_save_confidence: float = 0.2):
        """
        Collect PERCH v2 predictions for later saving.
        """
        if not self.save_perch_predictions:
            return
        
        # old before inference issue
        # # Filter predictions by minimum confidence threshold BEFORE taking top-k
        # filtered_preds = [(s, c) for s, c in perch_preds if c >= min_save_confidence]
        
        # # Take top-k from filtered predictions
        # top_preds = filtered_preds[:self.perch_predictions_top_k]
        top_preds = perch_preds[:self.perch_predictions_top_k]
        
        for rank, (species, confidence) in enumerate(top_preds, 1):
            # Normalize species name to get common name
            common_name = normalize_species_name(species)
            
            self.collected_perch_predictions.append({
                'file_path': file_path,
                'file_name': file_name,
                'segment_idx': segment_idx,
                'start_time': start_time,
                'end_time': end_time,
                'rank': rank,
                'species_full': species,
                'species_common': common_name,
                'confidence': confidence
            })
    
    def _collect_embeddings(self, segment_data: Dict, file_path: str, file_name: str, segment_idx: int):
        """
        Collect PERCH embeddings for later saving.
        """
        if not self.save_embeddings:
            return
        
        perch_logits = segment_data.get('perch_embedding', np.array([]))
        perch_raw = segment_data.get('perch_raw_embedding', np.array([]))
        
        if len(perch_logits) == 0:
            return
        
        # Store 93-dim logits (before directory filtering)
        self.collected_embeddings_93dim.append(perch_logits.copy())
        
        # Store filtered logits (after directory filtering)
        if self.allowed_indices is not None and len(self.allowed_indices) > 0:
            filtered_logits = perch_logits[self.allowed_indices]
            self.collected_embeddings_filtered.append(filtered_logits.copy())
        else:
            self.collected_embeddings_filtered.append(perch_logits.copy())
        
        # Store 1536-dim raw embeddings
        if len(perch_raw) > 0:
            self.collected_embeddings_1536dim.append(perch_raw.copy())
        else:
            self.collected_embeddings_1536dim.append(np.zeros(1536, dtype=np.float32))
        
        # Store metadata
        self.collected_file_paths.append(file_path)
        self.collected_file_names.append(file_name)
        self.collected_segment_indices.append(segment_idx)
        self.collected_start_times.append(segment_data.get('start_time', 0.0))
        self.collected_end_times.append(segment_data.get('end_time', 0.0))
    
    def predict_segment_with_voting(self, segment_data: Dict, 
                                     file_path: str = "", file_name: str = "", 
                                     segment_idx: int = 0,
                                     verbose: bool = False) -> Dict:
        """
        Predict using ensemble voting
        """
        if verbose:
            print(f"\n  ═══ SEGMENT {segment_data['start_time']:.1f}s - {segment_data['end_time']:.1f}s ═══")
        
        # Get predictions from all models
        birdnet_preds = self.get_birdnet_predictions(segment_data['birdnet_probs'])
        fusion_preds = self.get_fusion_predictions(segment_data['combined'])
        perch_preds = self.get_perch_predictions(segment_data, use_threshold=False, verbose=False)
        
        # Collect PERCH predictions
        self._collect_perch_predictions(
            perch_preds=perch_preds,
            file_path=file_path,
            file_name=file_name,
            start_time=segment_data['start_time'],
            end_time=segment_data['end_time'],
            segment_idx=segment_idx
        )
        
        # Filter perch_preds for voting (apply threshold)
        _extra = set(getattr(self.extractor, 'perch_extra_species', []))
        perch_preds_filtered = [(s, c) for s, c in perch_preds if c >= self.perch_min_confidence and s not in _extra]
        
        # Apply ensemble voting
        voted_predictions = hybrid_ensemble_voting(
            birdnet_preds=birdnet_preds,
            fusion_preds=fusion_preds,
            perch_preds=perch_preds_filtered,
            model_weights=self.ensemble_weights,
            consensus_boost=self.consensus_boost,
            top_k=self.top_k,
            verbose=verbose
        )
        
        # Extract (species, confidence) pairs for compatibility
        predictions = [(sp, conf) for sp, conf, meta in voted_predictions]
        
        return {
            'predictions': predictions,
            'method': 'Ensemble Voting',
            'tier': 'N/A',
            'start_time': segment_data['start_time'],
            'end_time': segment_data['end_time'],
            'voting_metadata': voted_predictions
        }
    
    def predict_segment_with_tiers(self, segment_data: Dict,
                                    file_path: str = "", file_name: str = "",
                                    segment_idx: int = 0,
                                    verbose: bool = False) -> Dict:
        """
        Predict using tier-based fallback (original approach)
        """
        birdnet_probabilities = segment_data['birdnet_probs']
        
        if verbose:
            print(f"\n  ═══ SEGMENT {segment_data['start_time']:.1f}s - {segment_data['end_time']:.1f}s ═══")
        
        # Get PERCH predictions early (for saving, regardless of tier used)
        perch_preds_all = self.get_perch_predictions(segment_data, use_threshold=False, verbose=False)
        
        # Collect PERCH predictions
        self._collect_perch_predictions(
            perch_preds=perch_preds_all,
            file_path=file_path,
            file_name=file_name,
            start_time=segment_data['start_time'],
            end_time=segment_data['end_time'],
            segment_idx=segment_idx
        )
        
        # TIER 1: BIRDNET
        tier1_predictions = self.get_birdnet_predictions(birdnet_probabilities)
        max_birdnet_conf = tier1_predictions[0][1] if tier1_predictions else 0.0
        
        if max_birdnet_conf >= self.tier1_threshold:
            if verbose:
                print(f"  ✓ TIER 1: {tier1_predictions[0][0]} ({max_birdnet_conf:.4f})")
            
            return {
                'predictions': tier1_predictions,
                'tier': 1,
                'method': 'BirdNET',
                'start_time': segment_data['start_time'],
                'end_time': segment_data['end_time']
            }

        # TIER 2: FUSION
        combined_embedding = segment_data['combined']
        tier2_predictions = self.get_fusion_predictions(combined_embedding)
        max_fusion_conf = tier2_predictions[0][1] if tier2_predictions else 0.0
        
        if max_fusion_conf > self.min_confidence:
            if verbose:
                print(f"  ✓ TIER 2: {tier2_predictions[0][0]} ({max_fusion_conf:.4f})")
            
            return {
                'predictions': tier2_predictions,
                'tier': 2,
                'method': 'Fusion',
                'start_time': segment_data['start_time'],
                'end_time': segment_data['end_time']
            }
        
        # TIER 3: PERCH
        _extra = set(getattr(self.extractor, 'perch_extra_species', []))
        tier3_predictions = [(s, c) for s, c in perch_preds_all if c >= self.perch_min_confidence and s not in _extra]
        max_perch_conf = tier3_predictions[0][1] if tier3_predictions else 0.0
        
        if tier3_predictions and max_perch_conf > max_birdnet_conf and max_perch_conf > max_fusion_conf:
            if verbose:
                print(f"  ✓ TIER 3: {tier3_predictions[0][0]} ({max_perch_conf:.4f})")
            
            return {
                'predictions': tier3_predictions,
                'tier': 3,
                'method': 'PERCH v2',
                'start_time': segment_data['start_time'],
                'end_time': segment_data['end_time']
            }
        
        # BEST AVAILABLE
        candidates = []
        if tier1_predictions:
            candidates.append((tier1_predictions, max_birdnet_conf, 1, 'BirdNET'))
        if tier2_predictions:
            candidates.append((tier2_predictions, max_fusion_conf, 2, 'Fusion'))
        if tier3_predictions:
            candidates.append((tier3_predictions, max_perch_conf, 3, 'PERCH v2'))
        
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_preds, best_conf, best_tier, best_method = candidates[0]
            
            return {
                'predictions': best_preds,
                'tier': best_tier,
                'method': f'{best_method} (Best Available)',
                'start_time': segment_data['start_time'],
                'end_time': segment_data['end_time']
            }
        
        # GUARANTEED FALLBACK
        if perch_preds_all:
            return {
                'predictions': perch_preds_all,
                'tier': 3,
                'method': 'PERCH v2 (Guaranteed Fallback)',
                'start_time': segment_data['start_time'],
                'end_time': segment_data['end_time']
            }
        else:
            return {
                'predictions': [],
                'tier': 0,
                'method': 'CRITICAL: All models failed',
                'start_time': segment_data['start_time'],
                'end_time': segment_data['end_time']
            }
    
    def predict_segment(self, segment_data: Dict, 
                        file_path: str = "", file_name: str = "",
                        segment_idx: int = 0,
                        verbose: bool = False) -> Dict:
        """
        Main prediction method - routes to voting or tier-based approach
        """
        if self.use_ensemble_voting:
            return self.predict_segment_with_voting(segment_data, file_path, file_name, segment_idx, verbose)
        else:
            return self.predict_segment_with_tiers(segment_data, file_path, file_name, segment_idx, verbose)
    
    def predict_file(self, audio_path: str, verbose: bool = False) -> List[Dict]:
        """Predict species for an audio file"""
        segment_embeddings = self.extractor.extract_embeddings_from_file(audio_path)
        
        if not segment_embeddings:
            return []
        
        detections = []
        file_name = Path(audio_path).name
        
        for i, segment_data in enumerate(segment_embeddings):
            # Collect embeddings
            self._collect_embeddings(segment_data, audio_path, file_name, i)
            
            # Predict (also collects PERCH predictions internally)
            result = self.predict_segment(
                segment_data, 
                file_path=audio_path, 
                file_name=file_name, 
                segment_idx=i,
                verbose=verbose
            )
            result['segment_index'] = i
            detections.append(result)
        
        return detections
    
    def process_directory(self, directory: str, audio_extensions: List[str] = None) -> Dict:
        """Process all audio files in a directory"""
        if audio_extensions is None:
            audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
        
        directory_path = Path(directory)
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend(directory_path.rglob(f'*{ext}'))
            audio_files.extend(directory_path.rglob(f'*{ext.upper()}'))
        
        audio_files = sorted(list(set(audio_files)))
        
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING")
        print(f"{'='*60}")
        print(f"📁 Directory: {directory}")
        print(f"🎵 Files: {len(audio_files)}")
        if self.save_embeddings:
            print(f"💾 Embedding saving: ENABLED")
        if self.save_perch_predictions:
            print(f"📝 PERCH prediction saving: ENABLED (top-{self.perch_predictions_top_k})")
        print(f"{'='*60}\n")
        
        results = []
        successful = 0
        failed = 0
        
        # Track method distribution
        if self.use_ensemble_voting:
            method_counts = {'Ensemble Voting': 0}
        else:
            tier_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        
        for i, audio_file in enumerate(audio_files, 1):
            try:
                print(f"[{i}/{len(audio_files)}] {audio_file.name:<50}", end=" ", flush=True)
                
                detections = self.predict_file(str(audio_file))
                
                if self.use_ensemble_voting:
                    if detections:
                        method_counts['Ensemble Voting'] += 1
                else:
                    for det in detections:
                        tier_counts[det.get('tier', 0)] += 1
                
                file_result = {
                    'file_path': str(audio_file),
                    'file_name': audio_file.name,
                    'num_detections': len(detections),
                    'detections': detections,
                    'status': 'success'
                }
                
                results.append(file_result)
                
                if detections and detections[0]['predictions']:
                    method = detections[0].get('method', 'Unknown')
                    tier = detections[0].get('tier', 'N/A')
                    top_sp, top_conf = detections[0]['predictions'][0]
                    
                    if self.use_ensemble_voting:
                        print(f"✓ {top_sp}({top_conf*100:.0f}%)")
                    else:
                        print(f"✓ T{tier} | {top_sp}({top_conf*100:.0f}%)")
                else:
                    print(f"✓ No predictions")
                
                successful += 1
                
            except Exception as e:
                print(f"✗ ERROR: {str(e)[:40]}")
                
                results.append({
                    'file_path': str(audio_file),
                    'file_name': audio_file.name,
                    'error': str(e),
                    'status': 'failed'
                })
                failed += 1
        
        summary = {
            'directory': str(directory),
            'timestamp': datetime.now().isoformat(),
            'total_files': len(audio_files),
            'successful_files': successful,
            'failed_files': failed,
            'results': results
        }
        
        if self.use_ensemble_voting:
            summary['method_distribution'] = method_counts
        else:
            summary['tier_distribution'] = tier_counts
        
        if self.save_embeddings:
            summary['collected_embeddings'] = len(self.collected_embeddings_93dim)
        
        if self.save_perch_predictions:
            summary['collected_perch_predictions'] = len(self.collected_perch_predictions)
        
        return summary
    
    def save_perch_predictions_csv(self, output_path: str = None):
        """
        Save collected PERCH v2 predictions to a CSV file.
        """
        if not self.save_perch_predictions:
            print("⚠️  PERCH prediction saving was not enabled!")
            return
        
        if len(self.collected_perch_predictions) == 0:
            print("⚠️  No PERCH predictions collected!")
            return
        
        if output_path is None:
            output_path = str(DEFAULT_PATHS['results_dir'] / 'perch_predictions.csv')
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"SAVING PERCH V2 PREDICTIONS")
        print(f"{'='*60}")
        print(f"📊 Total prediction rows: {len(self.collected_perch_predictions)}")
        print(f"📊 Unique files: {len(set(p['file_name'] for p in self.collected_perch_predictions))}")
        print(f"📊 Unique segments: {len(set((p['file_name'], p['segment_idx']) for p in self.collected_perch_predictions))}")
        
        # Write CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'file_name',
                'file_path', 
                'segment_idx',
                'start_time',
                'end_time',
                'rank',
                'species_full',
                'species_common',
                'confidence'
            ])
            
            # Data rows
            for pred in self.collected_perch_predictions:
                writer.writerow([
                    pred['file_name'],
                    pred['file_path'],
                    pred['segment_idx'],
                    f"{pred['start_time']:.2f}",
                    f"{pred['end_time']:.2f}",
                    pred['rank'],
                    pred['species_full'],
                    pred['species_common'],
                    f"{pred['confidence']:.6f}"
                ])
        
        print(f"\n✅ Saved PERCH predictions: {output_file}")
        print(f"   File size: {output_file.stat().st_size / 1024:.2f} KB")
        
        # Print species distribution
        species_counts = defaultdict(int)
        for pred in self.collected_perch_predictions:
            if pred['rank'] == 1:
                species_counts[pred['species_common']] += 1
        
        print(f"\n   Top-1 species distribution (top 10):")
        for species, count in sorted(species_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"     {species:40s}: {count:5d}")
        
        print(f"{'='*60}\n")
    
    def save_collected_embeddings(self, output_dir: str = None, prefix: str = "perchv2_embeddings"):
        """
        Save collected PERCH embeddings to THREE .npz files.
        """
        if not self.save_embeddings:
            print("⚠️  Embedding saving was not enabled!")
            return
        
        if len(self.collected_embeddings_93dim) == 0:
            print("⚠️  No embeddings collected!")
            return
        
        if output_dir is None:
            output_dir = DEFAULT_PATHS['embeddings_dir']
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert to arrays
        embeddings_93dim = np.array(self.collected_embeddings_93dim, dtype=np.float32)
        embeddings_filtered = np.array(self.collected_embeddings_filtered, dtype=np.float32)
        embeddings_1536dim = np.array(self.collected_embeddings_1536dim, dtype=np.float32)
        
        print(f"\n{'='*60}")
        print(f"SAVING PERCH V2 EMBEDDINGS")
        print(f"{'='*60}")
        print(f"📊 Total segments: {len(self.collected_embeddings_93dim)}")
        print(f"📊 Unique files: {len(set(self.collected_file_paths))}")
        print(f"📊 93-dim logits shape: {embeddings_93dim.shape}")
        print(f"📊 Filtered logits shape: {embeddings_filtered.shape}")
        print(f"📊 1536-dim embeddings shape: {embeddings_1536dim.shape}")
        
        # Common arrays
        file_paths_arr = np.array(self.collected_file_paths, dtype=object)
        file_names_arr = np.array(self.collected_file_names, dtype=object)
        segment_indices_arr = np.array(self.collected_segment_indices, dtype=np.int32)
        start_times_arr = np.array(self.collected_start_times, dtype=np.float32)
        end_times_arr = np.array(self.collected_end_times, dtype=np.float32)
        
        common_metadata = {
            'timestamp': datetime.now().isoformat(),
            'total_embeddings': len(self.collected_embeddings_93dim),
            'total_unique_files': len(set(self.collected_file_paths)),
            'model': 'PERCH v2 (perch_hoplite)',
            'allowed_indices': self.allowed_indices if self.allowed_indices else [],
            'allowed_species': list(self.allowed_species_set) if self.allowed_species_set else []
        }
        
        # Save 93-dim logits
        file_93dim = output_path / f"{prefix}_93dim_logits.npz"
        np.savez_compressed(
            file_93dim,
            embeddings=embeddings_93dim,
            file_paths=file_paths_arr,
            file_names=file_names_arr,
            segment_indices=segment_indices_arr,
            start_times=start_times_arr,
            end_times=end_times_arr,
            metadata={**common_metadata, 'embedding_type': 'constrained_93dim_logits', 'embedding_dim': 93}
        )
        print(f"\n✅ Saved 93-dim logits: {file_93dim}")
        
        # Save filtered logits
        file_filtered = output_path / f"{prefix}_filtered_{embeddings_filtered.shape[1]}dim_logits.npz"
        np.savez_compressed(
            file_filtered,
            embeddings=embeddings_filtered,
            file_paths=file_paths_arr,
            file_names=file_names_arr,
            segment_indices=segment_indices_arr,
            start_times=start_times_arr,
            end_times=end_times_arr,
            metadata={**common_metadata, 'embedding_type': f'directory_filtered_{embeddings_filtered.shape[1]}dim_logits', 'embedding_dim': embeddings_filtered.shape[1]}
        )
        print(f"✅ Saved filtered logits: {file_filtered}")
        
        # Save 1536-dim raw embeddings
        file_1536dim = output_path / f"{prefix}_1536dim_embeddings.npz"
        np.savez_compressed(
            file_1536dim,
            embeddings=embeddings_1536dim,
            file_paths=file_paths_arr,
            file_names=file_names_arr,
            segment_indices=segment_indices_arr,
            start_times=start_times_arr,
            end_times=end_times_arr,
            metadata={**common_metadata, 'embedding_type': 'raw_1536dim_embeddings', 'embedding_dim': 1536}
        )
        print(f"✅ Saved 1536-dim raw embeddings: {file_1536dim}")
        print(f"{'='*60}\n")
    
    def clear_collected_data(self):
        """Clear all collected data (embeddings and predictions)"""
        self.collected_embeddings_93dim = []
        self.collected_embeddings_filtered = []
        self.collected_embeddings_1536dim = []
        self.collected_file_paths = []
        self.collected_file_names = []
        self.collected_segment_indices = []
        self.collected_start_times = []
        self.collected_end_times = []
        self.collected_perch_predictions = []
        
        print("✓ Cleared all collected data (embeddings and predictions)")
    
    def save_results_csv(self, summary: Dict, output_path: str = None):
        """Save results to CSV"""
        if output_path is None:
            output_path = str(DEFAULT_PATHS['results_dir'] / 'inference_results.csv')
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            if self.use_ensemble_voting:
                writer.writerow(['File', 'Status', 'Segment', 'Method', 
                               'Species', 'Confidence', 'Start', 'End'])
            else:
                writer.writerow(['File', 'Status', 'Segment', 'Tier', 'Method', 
                               'Species', 'Confidence', 'Start', 'End'])
            
            for result in summary['results']:
                if result['status'] == 'success' and result['detections']:
                    for seg_idx, detection in enumerate(result['detections']):
                        if detection['predictions']:
                            for species, conf in detection['predictions']:
                                if self.use_ensemble_voting:
                                    writer.writerow([
                                        result['file_name'],
                                        'success',
                                        seg_idx,
                                        detection['method'],
                                        species,
                                        f"{conf:.4f}",
                                        f"{detection['start_time']:.2f}",
                                        f"{detection['end_time']:.2f}"
                                    ])
                                else:
                                    writer.writerow([
                                        result['file_name'],
                                        'success',
                                        seg_idx,
                                        detection.get('tier', 'N/A'),
                                        detection['method'],
                                        species,
                                        f"{conf:.4f}",
                                        f"{detection['start_time']:.2f}",
                                        f"{detection['end_time']:.2f}"
                                    ])
                        else:
                            if self.use_ensemble_voting:
                                writer.writerow([
                                    result['file_name'],
                                    'success',
                                    seg_idx,
                                    detection['method'],
                                    'NO_PREDICTION',
                                    '0.0000',
                                    f"{detection['start_time']:.2f}",
                                    f"{detection['end_time']:.2f}"
                                ])
                            else:
                                writer.writerow([
                                    result['file_name'],
                                    'success',
                                    seg_idx,
                                    detection.get('tier', 0),
                                    detection['method'],
                                    'NO_PREDICTION',
                                    '0.0000',
                                    f"{detection['start_time']:.2f}",
                                    f"{detection['end_time']:.2f}"
                                ])
                else:
                    if self.use_ensemble_voting:
                        writer.writerow([
                            result['file_name'],
                            result['status'],
                            '', '', result.get('error', ''), '', '', ''
                        ])
                    else:
                        writer.writerow([
                            result['file_name'],
                            result['status'],
                            '', '', '', result.get('error', ''), '', '', ''
                        ])
        
        print(f"\n✓ Results saved: {output_file}")
    
    def print_summary(self, summary: Dict):
        """Print summary"""
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Files: {summary['total_files']}")
        print(f"Successful: {summary['successful_files']}")
        print(f"Failed: {summary['failed_files']}")
        
        if self.use_ensemble_voting:
            print(f"\nMethod: Ensemble Voting")
            print(f"Predictions: {summary.get('method_distribution', {}).get('Ensemble Voting', 0)}")
        else:
            print(f"\nTier Distribution:")
            tier_names = ['No Pred', 'BirdNET', 'Fusion', 'PERCH v2']
            for tier, count in sorted(summary.get('tier_distribution', {}).items()):
                print(f"  T{tier} ({tier_names[tier]}): {count}")
        
        if self.save_embeddings:
            print(f"\n💾 Collected embeddings: {summary.get('collected_embeddings', 0)}")
        
        if self.save_perch_predictions:
            print(f"📝 Collected PERCH predictions: {summary.get('collected_perch_predictions', 0)}")
        
        print(f"{'='*60}\n")


def main():
    """Main processing with robust paths"""
    
    print(f"\n📁 Using ARIA Root: {ARIA_ROOT}")
    print(f"📁 Default paths:")
    for name, path in DEFAULT_PATHS.items():
        exists = "✓" if path.exists() else "✗"
        print(f"   {exists} {name}: {path}")
    
    inference = TwoTierInferenceEnhanced(
        # All paths now default to ARIA_ROOT-relative if None
        system_config_path=None,  # Uses DEFAULT_PATHS['system_config']
        zoo_config_path=None,     # Uses DEFAULT_PATHS['zoo_config']
        birdnet_model_path=None,  # Uses DEFAULT_PATHS['birdnet_model']
        birdnet_labels_path=None, # Uses DEFAULT_PATHS['birdnet_labels']
        perch_species_mapping_path=None,  # Uses DEFAULT_PATHS['perch_mapping']
        tier1_threshold=0.70,
        min_confidence=0.30,
        perch_min_confidence=0.05,
        temperature=1.8,
        segment_length=3.0,
        overlap=0.0,
        use_ensemble_voting=True,
        ensemble_weights={'BirdNET': 1.0, 'Fusion': 0.8, 'PERCH': 0.8},
        consensus_boost=0.5,
        save_embeddings=True,
        save_perch_predictions=True,
        perch_predictions_top_k=10,
        verbose=True
    )
    
    # Example: process a directory
    datasets = [
      #   "/home/zoodata/preprocessed/fl_antwerpen_pied_avocets_april2026_data", 
        # "/home/zoodata/preprocessed/fl_antwerpen_savanna1_april2026_data", naja
        # "/home/zoodata/preprocessed/fl_antwerpen_savanna2_april2026_data", naja
      #  "/home/zoodata/preprocessed/fl_antwerpen_village_weavers_april2026_data", 
        # "/home/zoodata/preprocessed/fl_aachen_avocet_may2026_data", ---
        # "/home/zoodata/preprocessed/fl_aachen_balistarling_may2026_data", naja
       #  "/home/zoodata/preprocessed/fl_aachen_flamingos_may2026_data", ---
        # "/home/zoodata/preprocessed/fl_aachen_ibises_may2026_data", naja
         "/home/zoodata/preprocessed/fl_cologne_aviaries_by_sealions_jan2026", 
        # "/home/zoodata/preprocessed/fl_cologne_zoo_aviary4_jan26_data", ---
       #  "/home/zoodata/preprocessed/fl_cologne_zoo_swallows_dec25_data", --
    ]

    from pathlib import Path

    for dataset_path in datasets:
        name = Path(dataset_path).name
        print(f"\n🚀 Processing: {name}")

        inference.clear_collected_data()
        inference.set_constraints_from_directory(dataset_path)

        summary = inference.process_directory(dataset_path)
        inference.print_summary(summary)

        inference.save_results_csv(
            summary,
            str(DEFAULT_PATHS['results_dir'] / f"{name}_ensemble.csv")
        )

        inference.save_perch_predictions_csv(
            str(DEFAULT_PATHS['results_dir'] / f"{name}_PERCH_predictions.csv")
        )


if __name__ == "__main__":
    main()