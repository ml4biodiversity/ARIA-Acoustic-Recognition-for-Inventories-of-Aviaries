# map_zoo_species
"""
Map zoo species to PERCH 2.0 indices - WORKING VERSION
Handles actual species matching with robust fuzzy logic
"""

import json
import pandas as pd
from pathlib import Path
from difflib import SequenceMatcher

print("="*60)
print("ZOO → PERCH 2.0 MAPPER - WORKING VERSION")
print("="*60)

# Step 1: Load BirdNET species
print(f"\n1️⃣  Loading your BirdNET species...")

SCRIPT_DIR = Path(__file__).resolve().parent
ARIA_ROOT = SCRIPT_DIR.parent.parent

birdnet_file = ARIA_ROOT / "config" / "WholeBirdsList.txt"

#birdnet_file = "zoostuff/checkpoints/ARIA/config/WholeBirdsList.txt"

if not Path(birdnet_file).exists():
    print(f"   ❌ File not found: {birdnet_file}")
    alt = input("   Enter path to your BirdNET labels file: ").strip()
    if alt:
        birdnet_file = alt

if not Path(birdnet_file).exists():
    print(f"   ❌ ERROR: {birdnet_file} not found!")
    exit(1)

with open(birdnet_file, 'r') as f:
    birdnet_species = [line.strip() for line in f if line.strip()]

print(f"   ✅ Loaded {len(birdnet_species)} species")
print(f"   First 5:")
for sp in birdnet_species[:5]:
    print(f"      {sp}")

# Step 2: Load PERCH mapping
print(f"\n2️⃣  Loading PERCH 2.0 species mapping...")

mapping_file = SCRIPT_DIR / "perch_v2_species_mapping.json"

#mapping_file = "perch_v2_species_mapping.json"

if not Path(mapping_file).exists():
    print(f"   ❌ ERROR: {mapping_file} not found!")
    print(f"   Run: python explore_perch_species_WORKING.py first")
    exit(1)

with open(mapping_file, 'r') as f:
    perch_species_to_idx = json.load(f)

print(f"   ✅ Loaded {len(perch_species_to_idx)} PERCH species")

# Create lowercase lookup
perch_lower = {k.lower(): (k, v) for k, v in perch_species_to_idx.items()}
print(f"   ✅ Created lowercase lookup")

# Step 3: Matching function
def find_best_match(birdnet_name):
    """Find best PERCH match for a BirdNET species"""
    
    # Extract scientific name from BirdNET format: "Scientific_Common"
    if '_' in birdnet_name:
        scientific = birdnet_name.split('_')[0].strip()
    else:
        scientific = birdnet_name.strip()
    
    scientific_lower = scientific.lower()
    
    # Method 1: Exact match (case-insensitive)
    if scientific_lower in perch_lower:
        orig_key, idx = perch_lower[scientific_lower]
        return {
            'match': True,
            'type': 'exact',
            'perch_species': orig_key,
            'perch_idx': idx,
            'confidence': 1.0
        }
    
    # Method 2: Try removing subspecies (last word)
    if ' ' in scientific:
        parts = scientific.split()
        if len(parts) > 2:  # Has subspecies
            genus_species = ' '.join(parts[:2]).lower()
            if genus_species in perch_lower:
                orig_key, idx = perch_lower[genus_species]
                return {
                    'match': True,
                    'type': 'subspecies_removed',
                    'perch_species': orig_key,
                    'perch_idx': idx,
                    'confidence': 0.95
                }
    
    # Method 3: Fuzzy matching (sequence similarity)
    best_match = None
    best_ratio = 0.0
    
    for perch_lower_name, (perch_orig, perch_idx) in perch_lower.items():
        ratio = SequenceMatcher(None, scientific_lower, perch_lower_name).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = (perch_orig, perch_idx)
    
    if best_match and best_ratio >= 0.85:
        return {
            'match': True,
            'type': 'fuzzy',
            'perch_species': best_match[0],
            'perch_idx': best_match[1],
            'confidence': best_ratio
        }
    
    # No match found
    return {
        'match': False,
        'type': 'unmapped',
        'perch_species': None,
        'perch_idx': None,
        'confidence': best_ratio
    }

# Step 4: Map all species
print(f"\n3️⃣  Mapping species...")

results = {
    'exact': [],
    'fuzzy': [],
    'unmapped': []
}

birdnet_to_perch = {}

for bn_species in birdnet_species:
    match_result = find_best_match(bn_species)
    
    if match_result['match']:
        birdnet_to_perch[bn_species] = match_result['perch_idx']
        
        entry = {
            'birdnet': bn_species,
            'perch_species': match_result['perch_species'],
            'perch_idx': match_result['perch_idx'],
            'confidence': match_result['confidence']
        }
        
        if match_result['type'] == 'exact':
            results['exact'].append(entry)
        else:
            results['fuzzy'].append(entry)
    else:
        results['unmapped'].append(bn_species)

# Print results
print(f"\n   📊 Mapping Results:")
print(f"      ✅ Exact matches: {len(results['exact'])}")
print(f"      ⚠️  Fuzzy matches: {len(results['fuzzy'])}")
print(f"      ❌ Unmapped: {len(results['unmapped'])}")

# Show samples
if results['exact']:
    print(f"\n   ✅ Sample exact matches:")
    for entry in results['exact'][:5]:
        print(f"      {entry['birdnet']:50s} → [{entry['perch_idx']:5d}] {entry['perch_species']}")

if results['fuzzy']:
    print(f"\n   ⚠️  Sample fuzzy matches (REVIEW THESE):")
    for entry in results['fuzzy'][:10]:
        print(f"      {entry['birdnet']:50s}")
        print(f"         → [{entry['perch_idx']:5d}] {entry['perch_species']} (confidence: {entry['confidence']:.2f})")

if results['unmapped']:
    print(f"\n   ❌ Unmapped species:")
    for sp in results['unmapped'][:10]:
        print(f"      {sp}")
    if len(results['unmapped']) > 10:
        print(f"      ... and {len(results['unmapped'])-10} more")

# Step 5: Save mapping
print(f"\n4️⃣  Saving mapping...")

# Create final mapping (use 0 for unmapped as placeholder)
final_mapping = {}
for bn_species in birdnet_species:
    if bn_species in birdnet_to_perch:
        final_mapping[bn_species] = int(birdnet_to_perch[bn_species])
    else:
        final_mapping[bn_species] = 0  # Placeholder - needs manual fix

#output_file = "perch_v2_zoo_species_mapping.json"
output_file = SCRIPT_DIR / "perch_v2_zoo_species_mapping.json"

with open(output_file, 'w') as f:
    json.dump(final_mapping, f, indent=2)
print(f"   ✅ Saved: {output_file}")

# Save detailed report
report = {
    'summary': {
        'total': len(birdnet_species),
        'exact': len(results['exact']),
        'fuzzy': len(results['fuzzy']),
        'unmapped': len(results['unmapped']),
        'success_rate': (len(results['exact']) + len(results['fuzzy'])) / len(birdnet_species) * 100
    },
    'exact_matches': results['exact'],
    'fuzzy_matches': results['fuzzy'],
    'unmapped_species': results['unmapped']
}

with open('perch_v2_mapping_report.json', 'w') as f:
    json.dump(report, f, indent=2)
print(f"   ✅ Saved: perch_v2_mapping_report.json")

# Save unmapped list
if results['unmapped']:
    with open('unmapped_species.txt', 'w') as f:
        for sp in results['unmapped']:
            f.write(f"{sp}\n")
    print(f"   ✅ Saved: unmapped_species.txt")

print(f"\n{'='*60}")
print("✅ MAPPING COMPLETE!")
print("="*60)

mapped_count = len(results['exact']) + len(results['fuzzy'])
print(f"\n📊 Final Summary:")
print(f"   Total species: {len(birdnet_species)}")
print(f"   Successfully mapped: {mapped_count} ({mapped_count/len(birdnet_species)*100:.1f}%)")
print(f"   • Exact matches: {len(results['exact'])}")
print(f"   • Fuzzy matches: {len(results['fuzzy'])} ⚠️  REVIEW RECOMMENDED")
print(f"   • Unmapped: {len(results['unmapped'])}")

if results['unmapped']:
    print(f"\n⚠️  WARNING: {len(results['unmapped'])} species unmapped!")
    print(f"   These are set to index 0 (placeholder)")
    print(f"   Review unmapped_species.txt and fix manually")

print(f"\n📁 Files created:")
print(f"   ✅ {output_file} ← USE THIS IN PIPELINE")
print(f"   ✅ perch_v2_mapping_report.json")
if results['unmapped']:
    print(f"   ⚠️  unmapped_species.txt")

print(f"\n💡 Next steps:")
if results['fuzzy']:
    print(f"   1. Review fuzzy matches in report (confidence < 1.0)")
if results['unmapped']:
    print(f"   2. Fix unmapped species:")
    print(f"      python search_perch_species.py '<species_name>'")
print(f"   3. Integrate into pipeline:")
print(f"      extractor = IntegratedEmbeddingExtractor(")
print(f"          perch_species_mapping_path='{output_file}',")
print(f"          ...)")
print()
