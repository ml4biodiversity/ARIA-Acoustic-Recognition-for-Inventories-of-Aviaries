import numpy as np
import torch
import librosa
import jax
import jax.numpy as jnp
from typing import List, Tuple, Dict, Optional
import warnings
import json
from pathlib import Path
import pandas as pd
import tensorflow as tf


warnings.filterwarnings('ignore')
print(tf.config.list_physical_devices('GPU'))

# import for birdnetlib
try:
    from birdnetlib import Recording
    from birdnetlib.analyzer import Analyzer
except ImportError:
    raise ImportError("birdnetlib not installed. Install with: pip install birdnetlib")

# import for perch 2.0
import os
os.environ['TFHUB_CACHE_DIR'] = os.path.expanduser('~/.cache/tfhub')
from perch_hoplite.zoo import model_configs


class IntegratedEmbeddingExtractor:
    """
    Extracts embeddings from both BirdNET (using birdnetlib) and PERCH models.
    
    UPDATED VERSION - NOW RETURNS BOTH:
    - perch_embedding: Logits (93-dim constrained OR ~15000-dim full)
    - perch_raw_embedding: Raw 1536-dim embeddings (NEW!)
    
    The raw 1536-dim embeddings are useful for:
    - Acoustic similarity analysis
    - Clustering and visualization
    - Training custom classifiers
    - Few-shot learning
    """
    
    def __init__(self,
                 birdnet_model_path: str = 'ZooCustom_v1.tflite',
                 birdnet_labels_path: str = 'ZooCustom_v1_Labels.txt',
                 perch_species_mapping_path: str = 'perch_v2_zoo_species_mapping.json',
                 min_conf: float = 0.01,
                 temperature: float = 1.8,
                 use_constrained_perch: bool = True,
                 segment_length: float = 3.0,
                 overlap: float = 0.0):
        """
        Initialize the embedding extractor
        
        Args:
            birdnet_model_path: Path to BirdNET TFLite model
            birdnet_labels_path: Path to BirdNET labels file
            perch_species_mapping_path: Path to PERCH species mapping JSON
            min_conf: Minimum confidence threshold for BirdNET
            temperature: Temperature scaling for BirdNET probabilities
            use_constrained_perch: Whether to constrain PERCH to zoo species
            segment_length: Length of audio segments in seconds
            overlap: Overlap between segments in seconds
        """
        self.min_conf = min_conf
        self.temperature = temperature
        self.use_constrained_perch = use_constrained_perch
        self.segment_length = segment_length
        self.overlap = overlap
        
        # perch uses 32khz
        self.perch_sr = 32000
        
        print(f"✓ Configuration:")
        print(f"  Segment length: {segment_length}s")
        print(f"  Overlap: {overlap}s")
        print(f"  Min confidence: {min_conf}")
        print(f"  Temperature scaling: {temperature}")
        print(f"  Constrained PERCH: {use_constrained_perch}")
        
        # initialize birdnet analyzer
        print(f"\n📊 Loading BirdNET with birdnetlib...")
        
        # derive labels path if not provided
        if birdnet_labels_path is None:
            birdnet_labels_path = birdnet_model_path.replace('.tflite', '_Labels.txt')
        
        # check files exist
        if not Path(birdnet_model_path).exists():
            raise FileNotFoundError(f"BirdNET model not found: {birdnet_model_path}")
        if not Path(birdnet_labels_path).exists():
            raise FileNotFoundError(f"BirdNET labels not found: {birdnet_labels_path}")
        
        # initialize birdnet analyzer
        self.analyzer = Analyzer(
            classifier_model_path=str(birdnet_model_path),
            classifier_labels_path=str(birdnet_labels_path)
        )
        
        # load species list from labels
        with open(birdnet_labels_path, 'r') as f:
            self.birdnet_species = [line.strip() for line in f]
        self.n_birdnet_species = len(self.birdnet_species)
        print(f"  ✓ Loaded {self.n_birdnet_species} BirdNET species")
        
        # create species name to index mapping for faster lookup
        self.birdnet_species_to_idx = {species: idx for idx, species in enumerate(self.birdnet_species)}
        
        # load perch 2.0 model (HAS WEIGHTS BUILT-IN!)
        print("\n🦜 Loading PERCH 2.0 model...")
        print("  ℹ️  PERCH 2.0 uses built-in weights - no separate files needed!")
        self.perch_base_model = model_configs.load_model_by_name('perch_v2')
        print(f"  ✓ PERCH 2.0 loaded with pretrained weights")
        
        # load PERCH labels for species name lookup
        self.perch_labels_df = None
        possible_label_paths = [
            'perch_v2_labels.csv',
            'labels.csv',
            Path.home() / '.cache/kagglehub/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2/2/assets/labels.csv',
        ]
        
        for label_path in possible_label_paths:
            label_path = Path(label_path)
            if label_path.exists():
                try:
                    self.perch_labels_df = pd.read_csv(label_path)
                    print(f"  ✓ Loaded PERCH labels from: {label_path}")
                    print(f"  ✓ PERCH label columns: {list(self.perch_labels_df.columns)}")
                    break
                except Exception as e:
                    print(f"  Warning: Could not load {label_path}: {e}")
        
        if self.perch_labels_df is None:
            print(f"  ⚠️  PERCH labels.csv not found - species names will be generic")
        
        # load perch species mapping for constrained mode
        self.perch_species_indices = None
        self.perch_extra_species = []   # PERCH-only species (not BirdNET labels)
        self.perch_extra_indices = []
        self._last_extra_logits = np.zeros(0, dtype=np.float32)
        self.perch_unmapped_positions = []
        if use_constrained_perch:
            if Path(perch_species_mapping_path).exists():
                with open(perch_species_mapping_path, 'r') as f:
                    perch_mapping = json.load(f)
                
                # create mapping from birdnet species to perch indices
                self.perch_species_indices = []
                unmapped = []
                
                for bn_species in self.birdnet_species:
                    found = False
                    
                    # CRITICAL FIX: Try exact match first!
                    if bn_species in perch_mapping:
                        self.perch_species_indices.append(int(perch_mapping[bn_species]))
                        found = True
                    
                    # Fallback: Try after normalizing both sides
                    if not found:
                        clean_bn = bn_species.lower().replace('_', ' ').strip()
                        
                        for perch_name, perch_idx in perch_mapping.items():
                            clean_perch = perch_name.lower().replace('_', ' ').strip()
                            
                            if clean_bn == clean_perch:
                                self.perch_species_indices.append(int(perch_idx))
                                found = True
                                break
                    
                    # Last resort: Substring match
                    if not found:
                        clean_bn = bn_species.lower().replace('_', ' ').strip()
                        
                        for perch_name, perch_idx in perch_mapping.items():
                            clean_perch = perch_name.lower().replace('_', ' ').strip()
                            
                            if clean_bn in clean_perch or clean_perch in clean_bn:
                                self.perch_species_indices.append(int(perch_idx))
                                found = True
                                break
                    
                    # No match - use fallback
                    if not found:
                        unmapped.append(bn_species)
                        self.perch_species_indices.append(int(list(perch_mapping.values())[0]))
                
                self.perch_unmapped_positions = [
                    i for i, idx in enumerate(self.perch_species_indices) if idx == 0
                ]
                print(f"  ✓ Mapped {len(self.birdnet_species) - len(unmapped)}/{len(self.birdnet_species)} species to PERCH")
                # --- PERCH-only extra species: in mapping, index > 0, not a BirdNET label ---
                def _cn(x):
                    x = x.split('_', 1)[1] if '_' in x else x
                    return ' '.join(x.lower().replace('-', ' ').replace("'", '').replace('’', '').split())
                _bn_common = {_cn(s) for s in self.birdnet_species}
                _seen = set()
                for _label, _idx in perch_mapping.items():
                    _c = _cn(_label)
                    if int(_idx) > 0 and _c not in _bn_common and _c not in _seen:
                        _seen.add(_c)
                        self.perch_extra_species.append(_label)
                        self.perch_extra_indices.append(int(_idx))
                print(f"  ✓ PERCH-only extra species available: {len(self.perch_extra_species)}")
                if self.perch_unmapped_positions:
                    print(f"  ℹ️  {len(self.perch_unmapped_positions)} species have no PERCH coverage - forced to near-zero score")
                if unmapped and len(unmapped) <= 5:
                    print(f"  Unmapped species: {unmapped}")
                elif unmapped:
                    print(f"  Unmapped species: {unmapped[:5]}... and {len(unmapped)-5} more")
            else:
                print(f"  Warning: PERCH species mapping not found at {perch_species_mapping_path}")
                self.use_constrained_perch = False
        
        print("\n✅ Models loaded successfully!")
        print(f"   Raw embedding dimension: 1536 (always available)")
    
    def apply_temperature_scaling(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to BirdNET probabilities.
        """
        if self.temperature == 1.0:
            return probabilities
        
        # convert probabilities to logits
        probabilities = np.clip(probabilities, 1e-7, 1 - 1e-7)
        logits = np.log(probabilities / (1 - probabilities))
        
        # scale logits by temperature
        scaled_logits = logits / self.temperature
        
        # convert back to probabilities
        scaled_probs = 1.0 / (1.0 + np.exp(-scaled_logits))
        
        # important: do not normalize to sum to 1 (multi-label!)
        return scaled_probs
    
    def get_perch_species_name(self, perch_idx: int) -> str:
        """
        Get human-readable species name from PERCH global index.
        """
        if self.perch_labels_df is None:
            return f"Species_{perch_idx}"
        
        if perch_idx < 0 or perch_idx >= len(self.perch_labels_df):
            return f"OutOfRange_{perch_idx}"
        
        try:
            row = self.perch_labels_df.iloc[perch_idx]
            
            if 'inat2024_fsd50k' in self.perch_labels_df.columns:
                species = str(row['inat2024_fsd50k'])
                if pd.notna(species) and species.strip():
                    return species
            
            for col in ['scientific_name', 'scientificname', 'species', 'label']:
                if col in self.perch_labels_df.columns and pd.notna(row.get(col)):
                    return str(row[col])
            
            return f"Index_{perch_idx}"
        
        except Exception as e:
            return f"Error_{perch_idx}"
    
    def extract_perch_embedding_from_segment(self, audio_segment: np.ndarray, show_predictions: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract PERCH embedding AND logits for an audio segment.
        
        UPDATED: Now returns BOTH:
        1. Logits (constrained or full, depending on use_constrained_perch)
        2. Raw 1536-dim embeddings (always)
        
        Args:
            audio_segment: Audio array at 32kHz
            show_predictions: Whether to print top predictions
            
        Returns:
            Tuple of (logits, raw_embeddings)
            - logits: (93,) if constrained, (~15000,) if unconstrained
            - raw_embeddings: (1536,) always
        """
        
        # 1) run PERCH model
        outputs = self.perch_base_model.embed(audio_segment[np.newaxis, :])
        
        # 2) Extract raw 1536-dim embeddings (ALWAYS available)
        raw_embeddings = outputs.embeddings
        raw_embeddings = np.array(raw_embeddings)
        
        # Handle possible dimensions: (batch, time, dim) or (batch, dim)
        if raw_embeddings.ndim == 3:
            raw_embeddings = raw_embeddings.mean(axis=1)  # average over time
        if raw_embeddings.ndim == 2:
            raw_embeddings = raw_embeddings[0]  # remove batch dimension
        
        raw_embeddings = np.array(raw_embeddings, dtype=np.float32)
        
        # 3) Extract logits from classification head
        logits = None
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
            
            # handle dict format: {'label': logits}
            if isinstance(logits, dict):
                logits = logits.get('label', list(logits.values())[0])
            
            logits = np.array(logits)
            
            # remove batch dimension if present
            if logits.ndim == 2:
                logits = logits[0]
            
            if self.perch_extra_indices:
                self._last_extra_logits = np.array(logits[self.perch_extra_indices], dtype=np.float32)
            # 4) constrained PERCH: select subset of species
            if self.use_constrained_perch and self.perch_species_indices is not None:
                constrained_logits = logits[self.perch_species_indices]  # shape (93,)
                if self.perch_unmapped_positions:
                    constrained_logits[self.perch_unmapped_positions] = -100.0
                
                if show_predictions:
                    exp_logits = np.exp(constrained_logits - np.max(constrained_logits))
                    probs = exp_logits / np.sum(exp_logits)
                    top_k = min(5, len(constrained_logits))
                    top_indices = np.argsort(constrained_logits)[-top_k:][::-1]
                
                logits = np.array(constrained_logits, dtype=np.float32)
            else:
                logits = np.array(logits, dtype=np.float32)
        else:
            # Fallback: return zeros for logits if not available
            print("  Warning: PERCH outputs have no logits attribute")
            if self.use_constrained_perch and self.perch_species_indices:
                logits = np.zeros(len(self.perch_species_indices), dtype=np.float32)
            else:
                logits = np.zeros(15000, dtype=np.float32)  # approximate
        
        return logits, raw_embeddings
    
    
    def extract_embeddings_from_file(self, audio_path: str) -> List[Dict]:
        """
        Extract embeddings from an audio file.
        
        Creates fixed-length segments and processes with both models.
        
        Returns:
            List of dicts, each containing:
            - birdnet_probs: (93,) BirdNET probabilities
            - perch_embedding: Logits (93-dim or ~15000-dim)
            - perch_raw_embedding: (1536,) Raw embeddings (NEW!)
            - combined: Concatenated [birdnet_probs, perch_embedding]
            - start_time: Segment start in seconds
            - end_time: Segment end in seconds
        """
        # first, let's load the audio for both models
        try:
            # load audio for perch (32khz)
            perch_audio, _ = librosa.load(audio_path, sr=self.perch_sr, mono=True)
            
            # calculate number of segments
            duration = len(perch_audio) / self.perch_sr
            
            if duration < self.segment_length:
                # audio too short, pad it
                min_samples = int(self.segment_length * self.perch_sr)
                perch_audio = np.pad(perch_audio, (0, min_samples - len(perch_audio)), mode='constant')
                duration = self.segment_length
            
            # determine segments
            if self.overlap > 0:
                step = self.segment_length - self.overlap
            else:
                step = self.segment_length
            
            segments = []
            start_time = 0
            while start_time + self.segment_length <= duration + 0.1:  # small tolerance
                end_time = start_time + self.segment_length
                segments.append((start_time, end_time))
                start_time += step
            
            # if no segments (shouldn't happen), create at least one
            if not segments:
                segments = [(0, min(self.segment_length, duration))]
            
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return []
        
        # now analyze with birdnet
        try:
            recording = Recording(
                analyzer=self.analyzer,
                path=str(audio_path),
                min_conf=self.min_conf
            )
            recording.analyze()
        except Exception as e:
            print(f"Error analyzing with BirdNET {audio_path}: {e}")
            # still continue with zero probabilities
            recording = None
        
        # extract embeddings for each segment
        embeddings = []
        
        for segment_idx, (start_time, end_time) in enumerate(segments):
            # extract perch segment
            start_sample = int(start_time * self.perch_sr)
            end_sample = int(end_time * self.perch_sr)
            
            # ensure we have exactly the right length
            expected_samples = int(self.segment_length * self.perch_sr)
            
            if end_sample - start_sample < expected_samples:
                # pad if needed
                perch_segment = perch_audio[start_sample:end_sample]
                perch_segment = np.pad(
                    perch_segment,
                    (0, expected_samples - len(perch_segment)),
                    mode='constant'
                )
            else:
                perch_segment = perch_audio[start_sample:start_sample + expected_samples]
            
            # extract perch embedding AND raw embedding
            try:
                self._last_extra_logits = np.full(len(self.perch_extra_indices), -100.0, dtype=np.float32)
                show_preds = (segment_idx == 0)  # Only show for first segment
                perch_logits, perch_raw_embedding = self.extract_perch_embedding_from_segment(
                    perch_segment, 
                    show_predictions=show_preds
                )
            except Exception as e:
                print(f"Error extracting PERCH embedding: {e}")
                # use zeros as fallback
                if self.use_constrained_perch and self.perch_species_indices:
                    perch_logits = np.zeros(len(self.perch_species_indices), dtype=np.float32)
                else:
                    perch_logits = np.zeros(15000, dtype=np.float32)
                perch_raw_embedding = np.zeros(1536, dtype=np.float32)
            
            # create birdnet probability vector for this segment
            birdnet_probs = np.zeros(self.n_birdnet_species, dtype=np.float32)
            
            # fill in probabilities from birdnet detections
            if recording is not None and hasattr(recording, 'detections'):
                for detection in recording.detections:
                    det_start = detection.get('start_time', 0)
                    det_end = detection.get('end_time', 0)
                    
                    if det_start < end_time and det_end > start_time:
                        scientific = detection.get('scientific_name', '')
                        common = detection.get('common_name', '')
                        
                        scientific_norm = scientific.replace('_', ' ').strip()
                        common_norm = common.replace('_', ' ').strip()
                        
                        possible_names = [
                            f"{scientific_norm}_{common_norm}",
                            f"{scientific}_{common}",
                            f"{scientific_norm} {common_norm}",
                            scientific_norm,
                            common_norm,
                            scientific,
                            common
                        ]
                        
                        for species_name in possible_names:
                            if species_name in self.birdnet_species_to_idx:
                                idx = self.birdnet_species_to_idx[species_name]
                                confidence = detection.get('confidence', 0)
                                birdnet_probs[idx] = max(birdnet_probs[idx], confidence)
                                break  
                            
            # apply temperature scaling to birdnet probabilities
            birdnet_probs = self.apply_temperature_scaling(birdnet_probs)
            
            # combine embeddings (using logits, not raw embeddings, for backward compatibility)
            combined = np.concatenate([birdnet_probs, perch_logits])
            
            embeddings.append({
                'birdnet_probs': birdnet_probs,
                'perch_embedding': perch_logits,           # Logits (93-dim or ~15000-dim)
                'perch_raw_embedding': perch_raw_embedding, # NEW: Raw 1536-dim embeddings
                'perch_extra_logits': self._last_extra_logits.copy(),  # PERCH-only species
                'combined': combined,
                'start_time': start_time,
                'end_time': end_time
            })
        
        return embeddings


# Example usage
if __name__ == "__main__":
    print("\n" + "="*60)
    print("PERCH 2.0 Embedding Extractor - WITH RAW EMBEDDINGS")
    print("="*60)
    
    print("\nℹ️  This version returns BOTH:")
    print("   - perch_embedding: Logits (93-dim constrained or ~15000-dim full)")
    print("   - perch_raw_embedding: Raw 1536-dim embeddings (NEW!)")
    
    extractor = IntegratedEmbeddingExtractor(
        birdnet_model_path='ZooCustom_v1.tflite',
        birdnet_labels_path='ZooCustom_v1_Labels.txt',
        perch_species_mapping_path='perch_v2_zoo_species_mapping.json',
        temperature=1.8,
        use_constrained_perch=True
    )
    
    print("\n✅ Extractor initialized successfully!")
    print("   Ready to extract embeddings from audio files")
    
    # Test if a file exists
    test_files = ['test.wav', 'example.wav']
    for test_file in test_files:
        if Path(test_file).exists():
            print(f"\n🎵 Testing on: {test_file}")
            
            embeddings = extractor.extract_embeddings_from_file(test_file)
            
            print(f"\n📊 Extracted {len(embeddings)} segments")
            if embeddings:
                first = embeddings[0]
                print(f"\n  First segment details:")
                print(f"    Time: {first['start_time']:.1f}s - {first['end_time']:.1f}s")
                print(f"    BirdNET probs shape: {first['birdnet_probs'].shape}")
                print(f"    PERCH logits shape: {first['perch_embedding'].shape}")
                print(f"    PERCH raw embedding shape: {first['perch_raw_embedding'].shape}")  # NEW!
                print(f"    Combined shape: {first['combined'].shape}")
            break
    else:
        print("\n   ℹ️  No test audio found")
        print("   To test, provide a .wav file as 'test.wav' or 'example.wav'")
