#!/usr/bin/env python3
"""
Usage:
  python3 lookup_new_perch_mappings.py "Scientific name_Common Name" [...]

EXACT match             -> written to perch_v2_zoo_species_mapping.json
Candidates found         -> written as 0 (safe default) AND saved to
                             perch_mapping_review.json for you to check;
                             accept_perch_candidate.py can overwrite the 0 later
No match / no candidates -> written as 0 (sentinel meaning "no PERCH coverage")

Every species gets a real entry after running this - never left entirely
absent from the mapping file, so it can never fall into the "missing key"
fallback bug.
"""
import json
import sys
from pathlib import Path

ARIA = Path(__file__).resolve().parent
FETCH_DIR = ARIA / "src" / "fetch"
PERCH_MASTER = FETCH_DIR / "perch_v2_species_mapping.json"
MAPPING_FILE = FETCH_DIR / "perch_v2_zoo_species_mapping.json"
REVIEW_FILE = ARIA / "perch_mapping_review.json"


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 lookup_new_perch_mappings.py "Scientific name_Common Name" [...]')
        sys.exit(1)

    targets = sys.argv[1:]
    perch = json.load(open(PERCH_MASTER, encoding="utf-8"))
    mapping = json.load(open(MAPPING_FILE, encoding="utf-8")) if MAPPING_FILE.exists() else {}
    review = json.load(open(REVIEW_FILE, encoding="utf-8")) if REVIEW_FILE.exists() else {}

    added, zeroed, needs_review = [], [], []

    for target in targets:
        if "_" not in target:
            print(f"  SKIP (expected 'Scientific name_Common Name'): {target}")
            continue
        sci = target.split("_", 1)[0].strip()
        genus = sci.split(" ", 1)[0].strip()

        print("\n" + "=" * 60)
        print(target)

        exact = next(((k, v) for k, v in perch.items() if sci.lower() == k.lower()), None)
        if exact:
            k, v = exact
            print(f"  EXACT: {v}  {k}")
            mapping[target] = v
            added.append((target, v))
            review.pop(target, None)
            continue

        candidates = [{"perch_name": k, "index": v, "match_type": "substring"}
                      for k, v in perch.items()
                      if sci.lower() in k.lower() or k.lower() in sci.lower()]
        if not candidates:
            candidates = [{"perch_name": k, "index": v, "match_type": "genus_only"}
                          for k, v in perch.items()
                          if genus.lower() == k.split(" ", 1)[0].strip().lower()]

        # Either way (candidates or truly nothing) - write 0 now so the
        # species is never left entirely absent from the mapping file.
        mapping[target] = 0
        zeroed.append(target)

        if candidates:
            print("  no exact match - candidates found (set to 0 for now):")
            for c in candidates:
                print(f"    [{c['match_type']}] {c['index']}  {c['perch_name']}")
            review[target] = {"status": "needs_review", "candidates": candidates}
            needs_review.append(target)
        else:
            print("  NOTHING FOUND - set to 0 (no PERCH coverage)")
            review.pop(target, None)

    if added or zeroed:
        json.dump(mapping, open(MAPPING_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        if added:
            print(f"\nAuto-added {len(added)} exact match(es):")
            for t, v in added:
                print(f"    {t}: {v}")
        if zeroed:
            print(f"\nSet {len(zeroed)} species to 0:")
            for t in zeroed:
                print(f"    {t}")

    if needs_review or review:
        json.dump(review, open(REVIEW_FILE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    if needs_review:
        print(f"\n{len(needs_review)} species have candidates to review - saved to {REVIEW_FILE}")
        print("They're set to 0 for now (safe). To accept a candidate instead:")
        print('    python3 accept_perch_candidate.py "<species label>" <index>')


if __name__ == "__main__":
    main()