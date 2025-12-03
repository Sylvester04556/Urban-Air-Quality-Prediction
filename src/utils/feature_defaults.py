"""
Feature defaults loader for production deployment.
Loads location-specific features and global fallback from JSON files.
"""

import os
import sys
from typing import Dict, List, Optional
from pathlib import Path

from src.logger import logging
from src.exception import CustomException
from src.utils.common import load_json


class FeatureDefaults:
    """
    Manages feature defaults for production predictions.
    Loads location-specific features and global fallback.
    """
    
    def __init__(
        self,
        location_lookup_path: str = 'artifacts/feature_engineering/location_features_lookup.json',
        medians_path: str = 'artifacts/feature_engineering/feature_medians.json'
    ):
        """
        Initialize FeatureDefaults loader.
        
        Args:
            location_lookup_path: Path to location features lookup JSON
            medians_path: Path to feature medians JSON
        """
        self.location_lookup_path = location_lookup_path
        self.medians_path = medians_path
        
        # Storage
        self._location_lookup = None
        self._medians = None
        
        # Load data
        self._load_data()
        
        logging.info("FeatureDefaults initialized successfully")
    
    def _load_data(self):
        """Load lookup files from disk"""
        try:
            # Load location lookup
            if os.path.exists(self.location_lookup_path):
                self._location_lookup = load_json(self.location_lookup_path)
                logging.info(f"Loaded location lookup with {len(self._location_lookup['locations'])} location(s)")
            else:
                logging.warning(f"Location lookup file not found: {self.location_lookup_path}")
                self._location_lookup = {'locations': {}, 'global_fallback': {}}
            
            # Load medians
            if os.path.exists(self.medians_path):
                self._medians = load_json(self.medians_path)
                logging.info(f"Loaded feature medians for {self._medians['metadata']['n_features_total']} features")
            else:
                logging.warning(f"Feature medians file not found: {self.medians_path}")
                self._medians = {'all_features': {}}
                
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_location_features(self, location_id: str) -> Dict[str, float]:
        """
        Get historical features for a specific location.
        
        Args:
            location_id: Location identifier (e.g., 'global', 'NYC-001')
            
        Returns:
            dict: Historical features for the location
        """
        try:
            locations = self._location_lookup.get('locations', {})
            
            # Check if this location exists
            if location_id in locations:
                features = locations[location_id]['features']
                n_samples = locations[location_id]['n_samples']
                logging.info(f"Retrieved features for location '{location_id}' ({n_samples} samples)")
                return features
            else:
                # Location not found - use global fallback
                logging.warning(f"Location '{location_id}' not found. Using global fallback.")
                return self.get_global_fallback()
                
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_global_fallback(self) -> Dict[str, float]:
        """
        Get global fallback features (median across all locations).
        
        Returns:
            dict: Global fallback features
        """
        try:
            global_fallback = self._location_lookup.get('global_fallback', {})
            
            if not global_fallback:
                logging.warning("Global fallback empty, using all feature medians")
                global_fallback = self._medians.get('all_features', {})
            
            logging.info(f"Retrieved global fallback with {len(global_fallback)} features")
            return global_fallback
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_available_locations(self) -> List[str]:
        """
        Get list of available location IDs.
        
        Returns:
            list: Available location IDs
        """
        try:
            locations = list(self._location_lookup.get('locations', {}).keys())
            logging.info(f"Retrieved {len(locations)} available location(s)")
            return locations
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_location_info(self, location_id: str) -> Dict:
        """
        Get information about a location.
        
        Args:
            location_id: Location identifier
            
        Returns:
            dict: Location metadata (n_samples, last_seen, etc.)
        """
        try:
            locations = self._location_lookup.get('locations', {})
            
            if location_id in locations:
                info = {
                    'id': location_id,
                    'n_samples': locations[location_id]['n_samples'],
                    'last_seen': locations[location_id]['last_seen'],
                    'n_features': len(locations[location_id]['features'])
                }
                logging.info(f"Retrieved info for location '{location_id}'")
                return info
            else:
                logging.warning(f"Location '{location_id}' not found")
                return None
                
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_time_series_feature_list(self) -> List[str]:
        """
        Get list of time-series features.
        
        Returns:
            list: Time-series feature names
        """
        try:
            feature_list = self._location_lookup.get('metadata', {}).get('feature_list', [])
            logging.info(f"Retrieved {len(feature_list)} time-series features")
            return feature_list
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def has_location_data(self) -> bool:
        """
        Check if we have multiple locations or just global.
        
        Returns:
            bool: True if multiple locations exist
        """
        try:
            locations = self._location_lookup.get('locations', {})
            has_multiple = len(locations) > 1 or (len(locations) == 1 and 'global' not in locations)
            return has_multiple
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_metadata(self) -> Dict:
        """
        Get metadata about the feature defaults.
        
        Returns:
            dict: Metadata information
        """
        try:
            return {
                'location_lookup': self._location_lookup.get('metadata', {}),
                'medians': self._medians.get('metadata', {})
            }
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test feature defaults
    logging.info("="*70)
    logging.info("TESTING FEATURE DEFAULTS LOADER")
    logging.info("="*70)
    
    # Initialize
    feature_defaults = FeatureDefaults()
    
    # Get available locations
    locations = feature_defaults.get_available_locations()
    print(f"\n Available locations: {locations}")
    
    # Get features for first location
    if len(locations) > 0:
        first_location = locations[0]
        print(f"\n Testing location: {first_location}")
        
        # Get location info
        info = feature_defaults.get_location_info(first_location)
        print(f"   Location info: {info}")
        
        # Get features
        features = feature_defaults.get_location_features(first_location)
        print(f"   Number of features: {len(features)}")
        print(f"   Sample features:")
        for i, (feature, value) in enumerate(list(features.items())[:5], 1):
            print(f"      {i}. {feature}: {value:.4f}")
    
    # Get global fallback
    print(f"\n Testing global fallback:")
    global_features = feature_defaults.get_global_fallback()
    print(f"   Number of features: {len(global_features)}")
    print(f"   Sample features:")
    for i, (feature, value) in enumerate(list(global_features.items())[:5], 1):
        print(f"      {i}. {feature}: {value:.4f}")
    
    # Get time-series features list
    ts_features = feature_defaults.get_time_series_feature_list()
    print(f"\n Time-series features: {len(ts_features)}")
    print(f"   {ts_features}")
    
    # Get metadata
    metadata = feature_defaults.get_metadata()
    print(f"\n Metadata:")
    print(f"   Locations: {metadata['location_lookup'].get('n_locations', 0)}")
    print(f"   Time-series features: {metadata['location_lookup'].get('n_time_series_features', 0)}")
    print(f"   Total features: {metadata['medians'].get('n_features_total', 0)}")
    
    logging.info("\n Feature defaults test completed!")