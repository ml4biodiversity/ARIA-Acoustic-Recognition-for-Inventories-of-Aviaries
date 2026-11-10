# python src/fetch/locate_perch_species_files.py
"""
Locate and copy PERCH 2.0 species mapping files to working directory
"""

import os
import shutil
from pathlib import Path

print("="*60)
print("PERCH 2.0 SPECIES FILES LOCATOR")
print("="*60)

# Find the kagglehub cache directory
cache_base = Path.home() / '.cache' / 'kagglehub' / 'models' / 'google' / 'bird-vocalization-classifier'

print(f"\n🔍 Searching for PERCH 2.0 files...")
print(f"   Base cache: {cache_base}")

# Find the most recent version
if cache_base.exists():
    # Look for tensorFlow2/perch_v2 directories
    perch_dirs = list(cache_base.glob("tensorFlow2/perch_v2*/**/assets"))
    
    if perch_dirs:
        # Get the latest version
        assets_dir = sorted(perch_dirs)[-1]
        
        print(f"\n✅ Found PERCH 2.0 assets directory:")
        print(f"   {assets_dir}")
        
        # List all files in assets
        print(f"\n📁 Files in assets directory:")
        asset_files = list(assets_dir.glob("*"))
        for f in sorted(asset_files):
            size = f.stat().st_size
            size_str = f"{size:,} bytes" if size < 1024*1024 else f"{size/(1024*1024):.2f} MB"
            print(f"   - {f.name} ({size_str})")
        
        # Key files we need
        labels_file = assets_dir / "labels.csv"
        ebird_file = assets_dir / "perch_v2_ebird_classes.csv"
        
        # Check if files exist
        print(f"\n📊 Key species mapping files:")
        if labels_file.exists():
            print(f"   ✅ labels.csv: {labels_file}")
            print(f"      Size: {labels_file.stat().st_size:,} bytes")
        else:
            print(f"   ❌ labels.csv not found")
        
        if ebird_file.exists():
            print(f"   ✅ perch_v2_ebird_classes.csv: {ebird_file}")
            print(f"      Size: {ebird_file.stat().st_size:,} bytes")
        else:
            print(f"   ❌ perch_v2_ebird_classes.csv not found")
        
        # Copy files to current directory
        print(f"\n💾 Copying files to current directory...")
        
        output_dir = Path.cwd()
        copied_files = []
        
        if labels_file.exists():
            dest = output_dir / "perch_v2_labels.csv"
            shutil.copy(labels_file, dest)
            copied_files.append(dest)
            print(f"   ✅ Copied: {dest.name}")
        
        if ebird_file.exists():
            dest = output_dir / "perch_v2_ebird_classes.csv"
            shutil.copy(ebird_file, dest)
            copied_files.append(dest)
            print(f"   ✅ Copied: {dest.name}")
        
        # Create a text file with paths
        info_file = output_dir / "perch_v2_file_locations.txt"
        with open(info_file, 'w') as f:
            f.write("PERCH 2.0 Species Mapping Files\n")
            f.write("="*60 + "\n\n")
            f.write(f"Cache Location:\n{assets_dir}\n\n")
            f.write("Files:\n")
            if labels_file.exists():
                f.write(f"- labels.csv: {labels_file}\n")
            if ebird_file.exists():
                f.write(f"- perch_v2_ebird_classes.csv: {ebird_file}\n")
            f.write(f"\nCopied to working directory:\n")
            for cf in copied_files:
                f.write(f"- {cf.name}\n")
        
        print(f"\n📝 File locations saved to: {info_file.name}")
        
        print(f"\n{'='*60}")
        print("✅ SUCCESS!")
        print("="*60)
        print(f"\n📊 Summary:")
        print(f"   Original location: {assets_dir}")
        print(f"   Copied {len(copied_files)} files to: {output_dir}")
        print(f"\n💡 Next steps:")
        print(f"   1. Examine species mappings: python explore_perch_species.py")
        print(f"   2. Map your zoo species to PERCH indices")
        print(f"   3. Update your species mapping JSON files")
        
    else:
        print(f"\n❌ PERCH 2.0 assets not found!")
        print(f"   The model may not have been downloaded yet.")
        print(f"   Run: python test_perch_v2.py")

else:
    print(f"\n❌ Cache directory not found: {cache_base}")
    print(f"   The model has not been downloaded yet.")
    print(f"   Run: python test_perch_v2.py")

print()
