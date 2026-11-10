#!/usr/bin/env python3
"""
Accept a candidate PERCH mapping from perch_mapping_review.json and write
it into perch_v2_zoo_species_mapping.json (the file the real pipeline uses).

Usage:
  python3 accept_perch_candidate.py "Scientific name_Common Name" <index>

Example:
  python3 accept_perch_candidate.py "Otidiphaps aruensis_White-naped Pheasant-Pigeon" 9409

The index must be one of the candidates lookup_new_perch_mappings.py found for
that species (check perch_mapping_review.json) - this script does not verify
the index against PERCH's list, it just writes what you tell it to.

If you decide none of the candidates are actually the right species (they just
sound alike), do nothing - leave the species out of perch_v2_zoo_species_mapping.json.
"""
import json
import sys
from pathlib import Path

FETCH_DIR = Path(__file__).resolve().parent.parent / "src" / "fetch"
MAPPING_FILE = FETCH_DIR / "perch_v2_zoo_species_mapping.json"
REVIEW_FILE = Path(__file__).resolve().parent.parent / "config" / "perch_mapping_review.json"


def main():
    if len(sys.argv) != 3:
        print('Usage: python3 accept_perch_candidate.py "Scientific name_Common Name" <index>')
        sys.exit(1)

    target = sys.argv[1]
    try:
        index = int(sys.argv[2])
    except ValueError:
        print(f"[ERR] Index must be a whole number, got: {sys.argv[2]}")
        sys.exit(1)

    mapping = {}
    if MAPPING_FILE.exists():
        with open(MAPPING_FILE, "r", encoding="utf-8") as f:
            mapping = json.load(f)

    mapping[target] = index
    with open(MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"Added: {target} -> {index}  (in {MAPPING_FILE.name})")

    if REVIEW_FILE.exists():
        with open(REVIEW_FILE, "r", encoding="utf-8") as f:
            review = json.load(f)
        if target in review:
            del review[target]
            with open(REVIEW_FILE, "w", encoding="utf-8") as f:
                json.dump(review, f, indent=2, ensure_ascii=False)
            print(f"Removed {target} from {REVIEW_FILE.name} (resolved)")


if __name__ == "__main__":
    main()
