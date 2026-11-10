#!/usr/bin/env python3
"""
Explore PERCH 2.0 species mappings - WORKING VERSION
Properly handles the inat2024_fsd50k column structure
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import os

print("="*60)
print("PERCH 2.0 SPECIES MAPPER - WORKING VERSION")
print("="*60)

# Step 1: Load labels CSV
print(f"\n1️⃣  Loading PERCH 2.0 species list...")

# Find the file
labels_file = None
locations = [
    "perch_v2_labels.csv",
    Path.home() / ".cache/kagglehub/models/google/bird-vocalization-classifier/tensorFlow2/perch_v2/2/assets/labels.csv"
]

for loc in locations:
    if Path(loc).exists():
        labels_file = str(loc)
        print(f"   📂 Found: {loc}")
        break

if not labels_file:
    print("   ❌ ERROR: labels.csv not found!")
    print("   Run: python locate_perch_species_files.py first")
    exit(1)

# Load CSV
df = pd.read_csv(labels_file)
print(f"   ✅ Loaded: {len(df)} rows")
print(f"   📊 Columns: {list(df.columns)}")

# Get the column name (should be 'inat2024_fsd50k')
species_column = df.columns[0]
print(f"   📝 Species column: '{species_column}'")

# Show first 10
print(f"\n   👀 First 10 species:")
for i in range(min(10, len(df))):
    species = df.iloc[i][species_column]
    print(f"      [{i:5d}] {species}")

# Step 2: Create mappings
print(f"\n2️⃣  Creating species mappings...")

species_to_idx = {}
idx_to_species = {}

for idx in range(len(df)):
    species = str(df.iloc[idx][species_column]).strip()
    if species and species != 'nan':  # Skip empty/NaN values
        species_to_idx[species] = int(idx)
        idx_to_species[int(idx)] = species

print(f"   ✅ Created mappings for {len(species_to_idx)} species")

# Verify with some indices
print(f"\n   🔍 Verification - checking indices [0, 1, 6154, 6920]:")
for idx in [0, 1, 6154, 6920]:
    if idx < len(df):
        species = df.iloc[idx][species_column]
        print(f"      [{idx:5d}] → {species}")

# Step 3: Save mappings
print(f"\n3️⃣  Saving mappings to JSON...")

# Save species -> index
with open('perch_v2_species_mapping.json', 'w') as f:
    json.dump(species_to_idx, f, indent=2)
print(f"   ✅ Saved: perch_v2_species_mapping.json")

# Save index -> species (as strings for JSON compatibility)
with open('perch_v2_idx_to_species.json', 'w') as f:
    json.dump({str(k): v for k, v in idx_to_species.items()}, f, indent=2)
print(f"   ✅ Saved: perch_v2_idx_to_species.json")

# Save DataFrame
df.to_pickle('perch_v2_labels.pkl')
print(f"   ✅ Saved: perch_v2_labels.pkl")

# Step 4: Create search tool
print(f"\n4️⃣  Creating search tool...")

search_script = '''#!/usr/bin/env python3
"""Search PERCH 2.0 species"""
import json
import sys

# Load mapping
with open('perch_v2_idx_to_species.json', 'r') as f:
    idx_to_species = json.load(f)

if len(sys.argv) < 2:
    print("Usage: python search_perch_species.py <search_term>")
    print("Example: python search_perch_species.py vulture")
    sys.exit(1)

query = " ".join(sys.argv[1:]).lower()
print(f"Searching for: '{query}'")
print("="*80)

matches = []
for idx, species in idx_to_species.items():
    if query in species.lower():
        matches.append((int(idx), species))

matches.sort(key=lambda x: x[0])  # Sort by index

if matches:
    print(f"Found {len(matches)} matches:")
    for idx, species in matches:
        print(f"  [{idx:5d}] {species}")
else:
    print(f"No matches found for '{query}'")

print("="*80)
'''

with open('search_perch_species.py', 'w') as f:
    f.write(search_script)
os.chmod('search_perch_species.py', 0o755)
print(f"   ✅ Created: search_perch_species.py")

# Step 5: Test searches
print(f"\n5️⃣  Testing searches for your zoo species...")

test_terms = ['vulture', 'neophron', 'gyps', 'flamingo', 'phoenicopterus']
for term in test_terms:
    matches = [species for species in species_to_idx.keys() if term.lower() in species.lower()]
    print(f"\n   🔍 '{term}': {len(matches)} matches")
    for species in matches[:3]:
        idx = species_to_idx[species]
        print(f"      [{idx:5d}] {species}")
    if len(matches) > 3:
        print(f"      ... and {len(matches)-3} more")

print(f"\n{'='*60}")
print("✅ MAPPING COMPLETE!")
print("="*60)

print(f"\n📊 Summary:")
print(f"   Total species: {len(species_to_idx)}")
print(f"   Index range: 0 to {len(df)-1}")

print(f"\n📁 Files created:")
print(f"   ✅ perch_v2_species_mapping.json")
print(f"   ✅ perch_v2_idx_to_species.json")
print(f"   ✅ perch_v2_labels.pkl")
print(f"   ✅ search_perch_species.py")

print(f"\n💡 Next step:")
print(f"   Run: python map_zoo_species_WORKING.py")
print()
