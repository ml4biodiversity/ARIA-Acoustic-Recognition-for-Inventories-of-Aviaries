#!/usr/bin/env python3
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
