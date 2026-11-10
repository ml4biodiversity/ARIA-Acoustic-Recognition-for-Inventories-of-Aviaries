#!/usr/bin/env python3
"""
Parse zoo configuration and extract species constraints
"""

import json
import re
import sys
from pathlib import Path
from typing import List, Tuple, Optional

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from run_inference import parse_folder_name as _shared_parse_folder_name, get_cage_birds as _shared_get_cage_birds


class ZooConfigParser:
    """
    Parser for zoo configuration that extracts species constraints
    based on directory name
    """
    
    def __init__(self, config_path: str = 'zoo_species_config.json'):
        """
        Initialize parser
        
        Args:
            config_path: Path to zoo_config.json
        """
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        # Load configuration
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Extract wild birds (always included)
        self.wild_birds = set(self.config.get('wild_birds', []))
        
        print(f"✓ Loaded zoo config from: {config_path}")
        print(f"  Wild birds: {len(self.wild_birds)} species")
        print(f"  Zoos: {len(self.config['zoos'])} configured")
    
    def parse_directory_name(self, directory: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse directory name to extract zoo name and cage identifier.
        Delegates to the shared, config-driven parse_folder_name() from
        run_inference.py so both scripts stay consistent.
        """
        dir_name = Path(directory).name
        zoo_name, cage_id = _shared_parse_folder_name(dir_name, self.config)
        return zoo_name, cage_id

    def get_species_list(self, 
                        zoo_name: Optional[str] = None,
                        cage_id: Optional[str] = None) -> List[str]:
        """
        Get constrained species list for a zoo/cage
        
        Args:
            zoo_name: Name of zoo (e.g., 'zoo_parc', 'beekse_bergen')
            cage_id: Cage identifier (e.g., 'aug25', '20250404')
        
        Returns:
            List of species names (wild birds + zoo-specific birds)
        """
        # Start with wild birds (always included)
        species_set = self.wild_birds.copy()
        
        if zoo_name is None:
            # No zoo specified, return only wild birds
            return sorted(list(species_set))
        
        cage_birds = _shared_get_cage_birds(self.config, zoo_name, cage_id)
        if not cage_birds:
            print(f"⚠️  Zoo '{zoo_name}' not found in config or no birds configured, using only wild birds")
            return sorted(list(species_set))

        species_set.update(cage_birds)
        print(f"📋 Using species for: {zoo_name}/{cage_id or 'all'}")
        print(f"   Cage/zoo birds: {len(cage_birds)}")
        
        return sorted(list(species_set))
    
    def get_species_from_directory(self, directory: str) -> List[str]:
        """
        Get species list directly from directory path
        
        Args:
            directory: Directory path
        
        Returns:
            List of constrained species names
        """
        zoo_name, cage_id = self.parse_directory_name(directory)
        
        print(f"\n🔍 Parsing directory: {Path(directory).name}")
        print(f"   Detected zoo: {zoo_name or 'Unknown'}")
        print(f"   Detected cage: {cage_id or 'All'}")
        
        species_list = self.get_species_list(zoo_name, cage_id)
        
        print(f"   Total species: {len(species_list)} (wild: {len(self.wild_birds)}, zoo: {len(species_list) - len(self.wild_birds)})")
        
        return species_list


# Test function
if __name__ == "__main__":
    parser = ZooConfigParser()
    
    # Test cases
    test_dirs = [
        "/home/zoodata/preprocessed/fl_zoo_parc_aug25_data/",
        "/home/zoodata/preprocessed/fl_beekse_bergen20250404/",
        "/home/zoodata/preprocessed/fl_gaia_zoo_taiga_data/",
        "/home/zoodata/preprocessed/fl_zoo_eindhoven20250308/"
    ]
    
    print("\n" + "="*60)
    print("TESTING ZOO CONFIG PARSER")
    print("="*60)
    
    for test_dir in test_dirs:
        species = parser.get_species_from_directory(test_dir)
        print(f"\n   Species list ({len(species)}):")
        for sp in species[:5]:
            print(f"     - {sp}")
        if len(species) > 5:
            print(f"     ... and {len(species) - 5} more")