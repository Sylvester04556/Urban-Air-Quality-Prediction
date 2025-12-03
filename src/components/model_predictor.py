"""
Model predictor component.
Loads trained model, scaler, and feature names.
Makes predictions on new data.
"""

import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from src.logger import logging
from src.exception import CustomException
from src.utils.common import load_pickle, load_text_file


class ModelPredictor:
    """
    Loads trained model and makes predictions.
    """
    
    def __init__(
        self,
        model_path: str = None,
        scaler_path: str = 'artifacts/scalers/standard_scaler.pkl',
        feature_names_path: str = 'artifacts/models/feature_names.txt'
    ):
        """
        Initialize model predictor.
        
        Args:
            model_path: Path to saved model (will auto-detect if None)
            scaler_path: Path to saved scaler
            feature_names_path: Path to feature names file
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.feature_names_path = feature_names_path
        
        # Storage
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_name = None
        self.needs_scaling = False
        
        # Load everything
        self._load_artifacts()
        
        logging.info("ModelPredictor initialized successfully")
    
    def _find_model_file(self) -> str:
        """
        Auto-detect the best model file.
        
        Returns:
            str: Path to model file
        """
        try:
            models_dir = Path('artifacts/models')
            
            if not models_dir.exists():
                raise FileNotFoundError(f"Models directory not found: {models_dir}")
            
            # Find all model pickle files
            model_files = list(models_dir.glob('best_model_*.pkl'))
            
            if len(model_files) == 0:
                raise FileNotFoundError("No model files found in artifacts/models/")
            
            # Use the first one (should only be one best model)
            model_path = str(model_files[0])
            
            # Extract model name from filename
            model_name = model_files[0].stem.replace('best_model_', '').replace('_', ' ')
            
            logging.info(f"Auto-detected model: {model_name}")
            return model_path, model_name
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _load_artifacts(self):
        """Load model, scaler, and feature names"""
        try:
            # 1. Load model
            if self.model_path is None:
                self.model_path, self.model_name = self._find_model_file()
            
            self.model = load_pickle(self.model_path)
            logging.info(f"Model loaded: {self.model_name}")
            
            # 2. Determine if scaling is needed
            model_type = type(self.model).__name__
            linear_models = ['LinearRegression', 'Ridge', 'Lasso', 'ElasticNet']
            self.needs_scaling = any(m in model_type for m in linear_models)
            
            logging.info(f"Model type: {model_type}, needs_scaling: {self.needs_scaling}")
            
            # 3. Load scaler (if needed)
            if self.needs_scaling:
                self.scaler = load_pickle(self.scaler_path)
                logging.info("Scaler loaded")
            else:
                logging.info("Scaler not needed for tree-based model")
            
            # 4. Load feature names
            self.feature_names = load_text_file(self.feature_names_path)
            logging.info(f"Loaded {len(self.feature_names)} feature names")
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def prepare_features(self, features: Dict[str, float]) -> pd.DataFrame:
        """
        Prepare features for prediction.
        
        Args:
            features: Dictionary with all feature values
            
        Returns:
            pd.DataFrame: Features in correct order, ready for model
        """
        try:
            # Check if we have all required features
            missing_features = set(self.feature_names) - set(features.keys())
            
            if missing_features:
                logging.warning(f"Missing {len(missing_features)} features. These should be filled by caller.")
                # Fill missing with 0
                for feature in missing_features:
                    features[feature] = 0.0
            
            # Create DataFrame with features in correct order
            df = pd.DataFrame([features], columns=self.feature_names)
            
            logging.info(f"Prepared features: {df.shape}")
            return df
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict(self, features: Dict[str, float]) -> Tuple[float, float]:
        """
        Make prediction on new data.
        
        Args:
            features: Dictionary with all feature values (60 features)
            
        Returns:
            tuple: (prediction_log_scale, prediction_original_scale)
        """
        try:
            logging.info("Starting prediction...")
            
            # 1. Prepare features
            df = self.prepare_features(features)
            
            # 2. Scale if needed
            if self.needs_scaling:
                df_scaled = self.scaler.transform(df)
                logging.info("Features scaled")
            else:
                df_scaled = df.values
                logging.info("Features used without scaling")
            
            # 3. Predict (log scale)
            prediction_log = self.model.predict(df_scaled)[0]
            
            # 4. Convert to original scale (reverse log1p)
            prediction_original = np.expm1(prediction_log)
            
            logging.info(f"Prediction: log={prediction_log:.4f}, original={prediction_original:.2f} μg/m³")
            
            return float(prediction_log), float(prediction_original)
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_air_quality_category(self, pm25: float) -> str:
        """
        Get air quality category based on PM2.5 value.
        Based on EPA Air Quality Index.
        
        Args:
            pm25: PM2.5 concentration (μg/m³)
            
        Returns:
            str: Air quality category
        """
        try:
            if pm25 <= 12:
                return "Good"
            elif pm25 <= 35:
                return "Moderate"
            elif pm25 <= 55:
                return "Unhealthy for Sensitive Groups"
            elif pm25 <= 150:
                return "Unhealthy"
            elif pm25 <= 250:
                return "Very Unhealthy"
            else:
                return "Hazardous"
                
        except Exception as e:
            raise CustomException(e, sys)
    
    def predict_with_category(self, features: Dict[str, float]) -> Dict:
        """
        Make prediction and return with air quality category.
        
        Args:
            features: Dictionary with all feature values
            
        Returns:
            dict: Prediction results with category
        """
        try:
            # Make prediction
            pred_log, pred_original = self.predict(features)
            
            # Get category
            category = self.get_air_quality_category(pred_original)
            
            result = {
                'pm25_predicted': round(pred_original, 2),
                'pm25_log_scale': round(pred_log, 4),
                'air_quality_category': category,
                'model_used': self.model_name or type(self.model).__name__
            }
            
            logging.info(f"Prediction complete: {pred_original:.2f} μg/m³ ({category})")
            
            return result
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        try:
            info = {
                'model_name': self.model_name or 'Unknown',
                'model_type': type(self.model).__name__,
                'n_features': len(self.feature_names),
                'needs_scaling': self.needs_scaling,
                'model_path': self.model_path
            }
            
            return info
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test model predictor
    logging.info("="*70)
    logging.info("TESTING MODEL PREDICTOR")
    logging.info("="*70)
    
    # Initialize predictor
    print("\n Initializing model predictor...")
    predictor = ModelPredictor()
    
    # Get model info
    print("\n Model Information:")
    info = predictor.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")
    
    # Create sample features (normally from FeatureEngineer + FeatureDefaults)
    print("\n Creating sample features...")
    
    # For testing, create dummy features (all zeros except a few)
    sample_features = {name: 0.0 for name in predictor.feature_names}
    
    # Set some reasonable values (these would come from feature engineering)
    sample_features.update({
        'temperature_2m_above_ground': 25.0,
        'relative_humidity_2m_above_ground': 60.0,
        'wind_speed': 5.0,
        'month_sin': 0.0,
        'month_cos': 1.0,
        'total_pollutant_load': 880.0,
    })
    
    print(f"   Created {len(sample_features)} features")
    
    # Make prediction
    print("\n Making prediction...")
    result = predictor.predict_with_category(sample_features)
    
    print("\n Prediction Results:")
    print(f"   PM2.5: {result['pm25_predicted']} μg/m³")
    print(f"   Air Quality: {result['air_quality_category']}")
    print(f"   Model: {result['model_used']}")
    print(f"   Log Scale: {result['pm25_log_scale']}")
    
    # Test air quality categories
    print("\nAir Quality Categories Test:")
    test_values = [8, 20, 45, 80, 180, 300]
    for val in test_values:
        category = predictor.get_air_quality_category(val)
        print(f"   PM2.5 = {val:3d} μg/m³ → {category}")
    
    logging.info("\n Model predictor test completed!")