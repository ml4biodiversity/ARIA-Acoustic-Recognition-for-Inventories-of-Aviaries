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
"""

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

from integrated_embedding_extractor_perchv2 import IntegratedEmbeddingExtractor
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
                 system_config_path: str,
                 zoo_config_path: str = 'zoo_species_config.json',
                 birdnet_model_path: str = 'ZooCustom_v1.tflite',
                 birdnet_labels_path: str = 'ZooCustom_v1_Labels.txt',
                 perch_species_mapping_path: str = 'perch_v2_zoo_species_mapping.json',
                 tier1_threshold: float = 0.85,
                 min_confidence: float = 0.30,
                 perch_min_confidence: float = 0.1,
                 top_k: int = 3,
                 temperature: float = 1.8,
                 segment_length: float = 3.0,
                 overlap: float = 0.0,
                 use_ensemble_voting: bool = True,  # NEW: Toggle voting vs tier-based
                 ensemble_weights: Dict[str, float] = None,  # NEW: Model weights for voting
                 consensus_boost: float = 0.5,  # NEW: Consensus boost
                 verbose: bool = False):
        
        self.tier1_threshold = tier1_threshold
        self.min_confidence = min_confidence
        self.perch_min_confidence = perch_min_confidence
        self.top_k = top_k
        self.verbose = verbose
        self.use_ensemble_voting = use_ensemble_voting
        self.consensus_boost = consensus_boost
        
        # Default ensemble weights
        if ensemble_weights is None:
            self.ensemble_weights = {'BirdNET': 1.0, 'Fusion': 0.8, 'PERCH': 0.6}
        else:
            self.ensemble_weights = ensemble_weights
        
        self.allowed_species_set: Optional[Set[str]] = None
        self.zoo_birds_set: Optional[Set[str]] = None
        self.constraint_active = False
        self.allowed_indices: Optional[List[int]] = None
        
        print("="*60)
        print("THREE-TIER INFERENCE - PERCH V2 WITH ENSEMBLE VOTING")
        print("="*60)
        
        # Load system config
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
        print(f"\n🦁 Loading zoo configuration...")
        self.zoo_parser = ZooConfigParser(zoo_config_path)
        
        # Initialize embedding extractor
        print(f"\n🔧 Initializing embedding extractor (PERCH v2)...")
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
        
        fusion_model_path = Path(self.config['model_path'])
        
        self.fusion_model = DualEmbeddingClassifier(
            n_species=self.n_species,
            birdnet_dim=self.birdnet_dim,
            perch_dim=self.perch_dim,
            hidden_dims=arch['hidden_dims'],
            dropout=0.3
        )
        
        # Load to CPU first to avoid CUDA deserialization errors
        print(f"   Loading weights from: {fusion_model_path.name}")
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
    
    def is_species_allowed(self, species_name: str) -> bool:
        """Check if species is allowed"""
        if not self.constraint_active or self.allowed_species_set is None:
            return True
        
        if "_" in species_name:
            common_name = species_name.split("_", 1)[1]
        else:
            common_name = species_name
        
        return common_name in self.allowed_species_set
    
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
    
    def get_perch_predictions(self, segment_data: Dict, use_threshold: bool = True, verbose: bool = True) -> List[Tuple[str, float]]:
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
                    birdnet_idx = self.allowed_indices[i]
                    species = self.extractor.birdnet_species[birdnet_idx]
                    predictions.append((species, float(prob)))
        
        except Exception as e:
            if verbose:
                print(f"      [PERCH v2] Error: {e}")
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions
    
    def predict_segment_with_voting(self, segment_data: Dict, verbose: bool = False) -> Dict:
        """
        Predict using ensemble voting
        """
        if verbose:
            print(f"\n  ═══ SEGMENT {segment_data['start_time']:.1f}s - {segment_data['end_time']:.1f}s ═══")
        
        # Get predictions from all models
        birdnet_preds = self.get_birdnet_predictions(segment_data['birdnet_probs'])
        fusion_preds = self.get_fusion_predictions(segment_data['combined'])
        perch_preds = self.get_perch_predictions(segment_data, use_threshold=True, verbose=False)
        
        # Apply ensemble voting
        voted_predictions = hybrid_ensemble_voting(
            birdnet_preds=birdnet_preds,
            fusion_preds=fusion_preds,
            perch_preds=perch_preds,
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
            'tier': 'N/A',  # No tier in voting mode
            'start_time': segment_data['start_time'],
            'end_time': segment_data['end_time'],
            'voting_metadata': voted_predictions  # Full metadata
        }
    
    def predict_segment_with_tiers(self, segment_data: Dict, verbose: bool = False) -> Dict:
        """
        Predict using tier-based fallback (original approach)
        """
        birdnet_probabilities = segment_data['birdnet_probs']
        
        if verbose:
            print(f"\n  ═══ SEGMENT {segment_data['start_time']:.1f}s - {segment_data['end_time']:.1f}s ═══")
        
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
        tier3_predictions = self.get_perch_predictions(segment_data, use_threshold=True, verbose=verbose)
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
        tier3_fallback = self.get_perch_predictions(segment_data, use_threshold=False, verbose=verbose)
        
        if tier3_fallback:
            return {
                'predictions': tier3_fallback,
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
    
    def predict_segment(self, segment_data: Dict, verbose: bool = False) -> Dict:
        """
        Main prediction method - routes to voting or tier-based approach
        """
        if self.use_ensemble_voting:
            return self.predict_segment_with_voting(segment_data, verbose)
        else:
            return self.predict_segment_with_tiers(segment_data, verbose)
    
    def predict_file(self, audio_path: str, verbose: bool = False) -> List[Dict]:
        """Predict species for an audio file"""
        segment_embeddings = self.extractor.extract_embeddings_from_file(audio_path)
        
        if not segment_embeddings:
            return []
        
        detections = []
        for i, segment_data in enumerate(segment_embeddings):
            result = self.predict_segment(segment_data, verbose=verbose)
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
        print(f"{'='*60}\n")
        
        results = []
        successful = 0
        failed = 0
        
        # Track method distribution (tier or voting)
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
        
        return summary
    
    def save_results_csv(self, summary: Dict, output_path: str):
        """Save results to CSV"""
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
        
        print(f"{'='*60}\n")


def main():
    """Main processing"""
    
    inference = TwoTierInferenceEnhanced(
        system_config_path='/home/I6405975/fusionstrategy/models/fusion_perchv2/architecture_info.json',
        zoo_config_path='zoo_species_config.json',
        birdnet_model_path='ZooCustom_v1.tflite',
        birdnet_labels_path='ZooCustom_v1_Labels.txt',
        perch_species_mapping_path='perch_v2_zoo_species_mapping.json',
        tier1_threshold=0.70,
        min_confidence=0.30,
        perch_min_confidence=0.2,
        temperature=1.8,
        segment_length=3.0,
        overlap=0.0,
        use_ensemble_voting=True,  # SET TO TRUE FOR ENSEMBLE VOTING
        ensemble_weights={'BirdNET': 1.0, 'Fusion': 0.8, 'PERCH': 0.8},
        consensus_boost=0.5,
        verbose=True
    )
    
    test_dir = "/home/zoodata/preprocessed/fl_zoo_parc_aug25_data/"
    inference.set_constraints_from_directory(test_dir)
    
    summary = inference.process_directory(test_dir)
    inference.print_summary(summary)
    
    output_csv = "/home/I6405975/fusionstrategy/models/fusion_perchv2/results/zoo_parc_aug25_ensemble_voting_NEW.csv"
    inference.save_results_csv(summary, output_csv)


if __name__ == "__main__":
    main()