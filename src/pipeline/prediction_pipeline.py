"""
Prediction pipeline - orchestrates the entire prediction flow.
Combines feature engineering, feature defaults, and model prediction.
This is the main interface for making predictions in production.
"""

import sys
from typing import Dict, Any

from src.logger import logging
from src.exception import CustomException
from src.utils.feature_defaults import FeatureDefaults
from src.components.feature_engineer import FeatureEngineer
from src.components.model_predictor import ModelPredictor


class PredictionPipeline:
    """
    End-to-end prediction pipeline.
    Takes raw user input and returns PM2.5 prediction.
    """
    
    def __init__(self):
        """Initialize all components"""
        try:
            logging.info("Initializing PredictionPipeline...")
            
            # Initialize components
            self.feature_defaults = FeatureDefaults()
            self.feature_engineer = FeatureEngineer()
            self.model_predictor = ModelPredictor()
            
            logging.info("PredictionPipeline initialized successfully")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict(
        self, 
        user_input: Dict[str, Any], 
        location_id: str = 'global'
    ) -> Dict[str, Any]:
        """
        Make PM2.5 prediction from raw user input.
        
        Args:
            user_input: Dictionary with raw user data
                {
                    'date': '2024-12-03',
                    'temperature_2m_above_ground': 25.0,
                    'relative_humidity_2m_above_ground': 60.0,
                    'specific_humidity_2m_above_ground': 0.012,
                    'precipitable_water_entire_atmosphere': 25.0,
                    'u_component_of_wind_10m_above_ground': 3.0,
                    'v_component_of_wind_10m_above_ground': 4.0,
                    'L3_NO2_NO2_column_number_density': 50.0,
                    'L3_CO_CO_column_number_density': 800.0,
                    'L3_SO2_SO2_column_number_density': 20.0,
                    'L3_HCHO_tropospheric_HCHO_column_number_density': 10.0,
                    'L3_O3_O3_column_number_density': 300.0
                }
            location_id: Location identifier (default: 'global')
        
        Returns:
            dict: Prediction results
                {
                    'pm25_predicted': 68.5,
                    'air_quality_category': 'Moderate',
                    'confidence': 'HIGH',
                    'location_id': 'global',
                    'model_used': 'XGBoost (Optuna)',
                    'message': 'Using historical data from global location'
                }
        """
        try:
            logging.info("="*70)
            logging.info("STARTING PREDICTION PIPELINE")
            logging.info("="*70)
            logging.info(f"Location: {location_id}")
            
            # ========================================
            # STEP 1: GET HISTORICAL FEATURES
            # ========================================
            logging.info("\n[STEP 1/4] Loading historical features...")
            
            historical_features = self.feature_defaults.get_location_features(location_id)
            location_info = self.feature_defaults.get_location_info(location_id)
            
            # Determine confidence based on location
            available_locations = self.feature_defaults.get_available_locations()
            
            if location_id in available_locations:
                confidence = 'HIGH'
                n_samples = location_info['n_samples'] if location_info else 0
                message = f"Using historical data from {location_id} ({n_samples} training samples)"
            else:
                confidence = 'MEDIUM'
                message = f"New location - using global fallback (based on {len(available_locations)} training location(s))"
            
            logging.info(f"   Historical features: {len(historical_features)}")
            logging.info(f"   Confidence: {confidence}")
            logging.info(f"   {message}")
            
            # ========================================
            # STEP 2: ENGINEER CURRENT FEATURES
            # ========================================
            logging.info("\n[STEP 2/4] Engineering features from user input...")
            
            current_features = self.feature_engineer.process_user_input(user_input)
            
            logging.info(f"   Current features: {len(current_features)}")
            
            # ========================================
            # STEP 3: COMBINE ALL FEATURES
            # ========================================
            logging.info("\n[STEP 3/4] Combining historical and current features...")
            
            # Merge historical and current features
            all_features = {
                **historical_features,  # Historical (rolling, lags, place stats)
                **current_features      # Current (temporal, wind, pollutants, ratios)
            }
            
            logging.info(f"   Total features: {len(all_features)}")
            
            # Validate we have all required features
            required_features = set(self.model_predictor.feature_names)
            provided_features = set(all_features.keys())
            missing_features = required_features - provided_features
            
            if missing_features:
                logging.warning(f"   Missing {len(missing_features)} features - filling with 0")
                for feature in missing_features:
                    all_features[feature] = 0.0
            
            # ========================================
            # STEP 4: MAKE PREDICTION
            # ========================================
            logging.info("\n[STEP 4/4] Making prediction...")
            
            prediction_result = self.model_predictor.predict_with_category(all_features)
            
            # ========================================
            # ASSEMBLE FINAL RESULT
            # ========================================
            final_result = {
                'pm25_predicted': prediction_result['pm25_predicted'],
                'air_quality_category': prediction_result['air_quality_category'],
                'confidence': confidence,
                'location_id': location_id,
                'model_used': prediction_result['model_used'],
                'message': message,
                'pm25_log_scale': prediction_result['pm25_log_scale'],
                'n_features_used': len(all_features),
                'n_historical_features': len(historical_features),
                'n_current_features': len(current_features)
            }
            
            logging.info("\n" + "="*70)
            logging.info("PREDICTION PIPELINE COMPLETE")
            logging.info("="*70)
            logging.info(f"Result: {final_result['pm25_predicted']} μg/m³ ({final_result['air_quality_category']})")
            logging.info(f"Confidence: {final_result['confidence']}")
            
            return final_result
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_available_locations(self):
        """
        Get list of available locations.
        
        Returns:
            list: Available location IDs
        """
        try:
            return self.feature_defaults.get_available_locations()
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """
        Get information about the pipeline components.
        
        Returns:
            dict: Pipeline information
        """
        try:
            return {
                'model_info': self.model_predictor.get_model_info(),
                'available_locations': self.get_available_locations(),
                'n_time_series_features': len(self.feature_defaults.get_time_series_feature_list()),
                'pipeline_version': '1.0.0'
            }
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test prediction pipeline
    logging.info("="*70)
    logging.info("TESTING PREDICTION PIPELINE")
    logging.info("="*70)
    
    # Initialize pipeline
    print("\n Initializing prediction pipeline...")
    pipeline = PredictionPipeline()
    
    # Get pipeline info
    print("\n Pipeline Information:")
    info = pipeline.get_pipeline_info()
    print(f"   Model: {info['model_info']['model_name']}")
    print(f"   Model Type: {info['model_info']['model_type']}")
    print(f"   Total Features: {info['model_info']['n_features']}")
    print(f"   Available Locations: {info['available_locations']}")
    print(f"   Time-Series Features: {info['n_time_series_features']}")
    
    # Sample user input (realistic values)
    sample_input = {
        'date': '2024-12-03',
        'temperature_2m_above_ground': 15.5,  # °C (winter)
        'relative_humidity_2m_above_ground': 65.0,  # %
        'specific_humidity_2m_above_ground': 0.008,
        'precipitable_water_entire_atmosphere': 20.0,
        'u_component_of_wind_10m_above_ground': 2.5,  # m/s
        'v_component_of_wind_10m_above_ground': 3.0,  # m/s
        'L3_NO2_NO2_column_number_density': 45.0,
        'L3_CO_CO_column_number_density': 750.0,
        'L3_SO2_SO2_column_number_density': 15.0,
        'L3_HCHO_tropospheric_HCHO_column_number_density': 8.0,
        'L3_O3_O3_column_number_density': 280.0
    }
    
    print("\n Sample User Input:")
    print(f"   Date: {sample_input['date']}")
    print(f"   Temperature: {sample_input['temperature_2m_above_ground']}°C")
    print(f"   Humidity: {sample_input['relative_humidity_2m_above_ground']}%")
    print(f"   Wind: u={sample_input['u_component_of_wind_10m_above_ground']}, v={sample_input['v_component_of_wind_10m_above_ground']}")
    print(f"   NO2: {sample_input['L3_NO2_NO2_column_number_density']}")
    print(f"   CO: {sample_input['L3_CO_CO_column_number_density']}")
    
    # Make prediction
    print("\n Making prediction...")
    print("="*70)
    
    result = pipeline.predict(
        user_input=sample_input,
        location_id='global'
    )
    
    print("\n" + "="*70)
    print("🎯 PREDICTION RESULTS")
    print("="*70)
    print(f"\n   PM2.5 Prediction: {result['pm25_predicted']} μg/m³")
    print(f"   Air Quality: {result['air_quality_category']}")
    print(f"   Confidence: {result['confidence']}")
    print(f"   Location: {result['location_id']}")
    print(f"   Model: {result['model_used']}")
    print(f"\n   ℹ  {result['message']}")
    print(f"\n    Features Used:")
    print(f"      Total: {result['n_features_used']}")
    print(f"      Historical: {result['n_historical_features']}")
    print(f"      Current: {result['n_current_features']}")
    
    # Test with different conditions
    print("\n" + "="*70)
    print(" TESTING DIFFERENT CONDITIONS")
    print("="*70)
    
    # Test 1: High pollution
    print("\n High Pollution Scenario:")
    high_pollution = sample_input.copy()
    high_pollution['L3_NO2_NO2_column_number_density'] = 120.0
    high_pollution['L3_CO_CO_column_number_density'] = 2000.0
    
    result1 = pipeline.predict(high_pollution, 'global')
    print(f"   PM2.5: {result1['pm25_predicted']} μg/m³ ({result1['air_quality_category']})")
    
    # Test 2: Low wind (poor dispersion)
    print("\nLow Wind Scenario:")
    low_wind = sample_input.copy()
    low_wind['u_component_of_wind_10m_above_ground'] = 0.5
    low_wind['v_component_of_wind_10m_above_ground'] = 0.3
    
    result2 = pipeline.predict(low_wind, 'global')
    print(f"   PM2.5: {result2['pm25_predicted']} μg/m³ ({result2['air_quality_category']})")
    
    # Test 3: Summer (high temperature)
    print("\n3 Summer Scenario:")
    summer = sample_input.copy()
    summer['date'] = '2024-07-15'
    summer['temperature_2m_above_ground'] = 32.0
    summer['relative_humidity_2m_above_ground'] = 45.0
    
    result3 = pipeline.predict(summer, 'global')
    print(f"   PM2.5: {result3['pm25_predicted']} μg/m³ ({result3['air_quality_category']})")
    
    print("\n" + "="*70)
    logging.info(" Prediction pipeline test completed!")