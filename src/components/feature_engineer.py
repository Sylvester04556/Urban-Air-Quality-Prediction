"""
Feature engineering for raw user input.
Calculates features that can be derived from current data:
- Temporal features (from date)
- Wind features (from u/v components)
- Pollutant interactions
- Atmospheric ratios
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any
import sys

from src.logger import logging
from src.exception import CustomException


class FeatureEngineer:
    """
    Engineers features from raw user input for prediction.
    """
    
    def __init__(self):
        """Initialize feature engineer"""
        logging.info("FeatureEngineer initialized")
    
    def engineer_temporal_features(self, date_input: str) -> Dict[str, Any]:
        """
        Create temporal features from date.
        
        Args:
            date_input: Date string (e.g., '2024-12-03') or datetime object
            
        Returns:
            dict: Temporal features
        """
        try:
            # Convert to datetime if string
            if isinstance(date_input, str):
                date = pd.to_datetime(date_input)
            else:
                date = date_input
            
            # Extract basic temporal features
            features = {
                'year': date.year,
                'month': date.month,
                'day': date.day,
                'dayofweek': date.dayofweek,
                'dayofyear': date.dayofyear,
                'week': date.isocalendar()[1],
                'quarter': date.quarter,
                'is_weekend': 1 if date.dayofweek >= 5 else 0,
            }
            
            # Season (Northern Hemisphere)
            month = date.month
            if month in [12, 1, 2]:
                features['season'] = 'Winter'
            elif month in [3, 4, 5]:
                features['season'] = 'Spring'
            elif month in [6, 7, 8]:
                features['season'] = 'Summer'
            else:
                features['season'] = 'Fall'
            
            # Cyclical encoding
            features['month_sin'] = np.sin(2 * np.pi * date.month / 12)
            features['month_cos'] = np.cos(2 * np.pi * date.month / 12)
            features['dayofweek_sin'] = np.sin(2 * np.pi * date.dayofweek / 7)
            features['dayofweek_cos'] = np.cos(2 * np.pi * date.dayofweek / 7)
            features['day_sin'] = np.sin(2 * np.pi * date.day / 31)
            features['day_cos'] = np.cos(2 * np.pi * date.day / 31)
            
            logging.info("Temporal features engineered successfully")
            return features
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def engineer_wind_features(self, u_component: float, v_component: float) -> Dict[str, float]:
        """
        Calculate wind features from u and v components.
        
        Args:
            u_component: East-West wind component (m/s)
            v_component: North-South wind component (m/s)
            
        Returns:
            dict: Wind features
        """
        try:
            # Wind speed (magnitude)
            wind_speed = np.sqrt(u_component**2 + v_component**2)
            
            # Wind direction (radians)
            wind_direction_rad = np.arctan2(v_component, u_component)
            
            # Wind direction (degrees, 0-360)
            wind_direction_deg = np.degrees(wind_direction_rad) % 360
            
            # Wind direction category
            if 337.5 <= wind_direction_deg or wind_direction_deg < 22.5:
                direction_category = 'N'
            elif 22.5 <= wind_direction_deg < 67.5:
                direction_category = 'NE'
            elif 67.5 <= wind_direction_deg < 112.5:
                direction_category = 'E'
            elif 112.5 <= wind_direction_deg < 157.5:
                direction_category = 'SE'
            elif 157.5 <= wind_direction_deg < 202.5:
                direction_category = 'S'
            elif 202.5 <= wind_direction_deg < 247.5:
                direction_category = 'SW'
            elif 247.5 <= wind_direction_deg < 292.5:
                direction_category = 'W'
            else:
                direction_category = 'NW'
            
            features = {
                'wind_speed': float(wind_speed),
                'wind_direction': float(wind_direction_rad),
                'wind_direction_deg': float(wind_direction_deg),
                'wind_direction_category': direction_category,
                'u_component_of_wind_10m_above_ground': float(u_component),
                'v_component_of_wind_10m_above_ground': float(v_component)
            }
            
            logging.info(f"Wind features engineered: speed={wind_speed:.2f} m/s, direction={direction_category}")
            return features
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def engineer_pollutant_features(self, pollutants: Dict[str, float]) -> Dict[str, float]:
        """
        Create pollutant interaction features.
        
        Args:
            pollutants: dict with pollutant concentrations
                {
                    'L3_NO2_NO2_column_number_density': value,
                    'L3_CO_CO_column_number_density': value,
                    'L3_SO2_SO2_column_number_density': value,
                    'L3_HCHO_tropospheric_HCHO_column_number_density': value,
                    'L3_O3_O3_column_number_density': value
                }
            
        Returns:
            dict: Pollutant interaction features
        """
        try:
            # Extract pollutants (use 0 if not provided)
            NO2 = pollutants.get('L3_NO2_NO2_column_number_density', 0)
            CO = pollutants.get('L3_CO_CO_column_number_density', 0)
            SO2 = pollutants.get('L3_SO2_SO2_column_number_density', 0)
            HCHO = pollutants.get('L3_HCHO_tropospheric_HCHO_column_number_density', 0)
            O3 = pollutants.get('L3_O3_O3_column_number_density', 0)
            
            # Calculate interactions
            features = {
                # Raw pollutants
                'L3_NO2_NO2_column_number_density': float(NO2),
                'L3_CO_CO_column_number_density': float(CO),
                'L3_SO2_SO2_column_number_density': float(SO2),
                'L3_HCHO_tropospheric_HCHO_column_number_density': float(HCHO),
                'L3_O3_O3_column_number_density': float(O3),
                
                # Aggregations
                'total_pollutant_load': float(NO2 + CO + SO2 + HCHO),
                'avg_pollutant_concentration': float((NO2 + CO + SO2 + HCHO) / 4),
                
                # Interactions
                'CO_NO2_interaction': float(CO * NO2),
                'NO2_SO2_interaction': float(NO2 * SO2),
                
                # Simple AQI proxy (weighted sum)
                'AQI_proxy': float(0.4 * NO2 + 0.3 * CO + 0.2 * SO2 + 0.1 * HCHO)
            }
            
            logging.info(f"Pollutant features engineered: total_load={features['total_pollutant_load']:.2f}")
            return features
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def engineer_weather_ratios(
        self, 
        temperature: float, 
        humidity: float, 
        pressure: float,
        wind_speed: float,
        total_pollutants: float
    ) -> Dict[str, float]:
        """
        Create weather and atmospheric ratio features.
        
        Args:
            temperature: Temperature (°C)
            humidity: Relative humidity (%)
            pressure: Atmospheric pressure
            wind_speed: Wind speed (m/s)
            total_pollutants: Total pollutant load
            
        Returns:
            dict: Ratio features
        """
        try:
            features = {
                # Temperature-humidity interactions
                'temp_humidity_ratio': float(temperature / (humidity + 1)),
                'temp_humidity_interaction': float(temperature * humidity),
                
                # Heat index (simplified)
                'heat_index': float(temperature + 0.5 * humidity),
                
                # Atmospheric stability
                'temp_pressure_ratio': float(temperature / (pressure + 1)),
                
                # Pollutant dispersion potential
                'pollutant_per_windspeed': float(total_pollutants / (wind_speed + 0.1)),
                
                # Humidity category
                'humidity_high': 1 if humidity > 70 else 0,
                'humidity_low': 1 if humidity < 30 else 0,
            }
            
            logging.info("Weather ratio features engineered successfully")
            return features
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def process_user_input(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process complete user input and generate all calculable features.
        
        Args:
            user_input: Dictionary with user-provided data
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
        
        Returns:
            dict: All engineered features
        """
        try:
            logging.info("Starting feature engineering for user input")
            
            all_features = {}
            
            # 1. Temporal features
            if 'date' in user_input:
                temporal = self.engineer_temporal_features(user_input['date'])
                all_features.update(temporal)
            
            # 2. Wind features
            if 'u_component_of_wind_10m_above_ground' in user_input and \
               'v_component_of_wind_10m_above_ground' in user_input:
                wind = self.engineer_wind_features(
                    user_input['u_component_of_wind_10m_above_ground'],
                    user_input['v_component_of_wind_10m_above_ground']
                )
                all_features.update(wind)
            
            # 3. Add raw weather features
            weather_features = [
                'temperature_2m_above_ground',
                'relative_humidity_2m_above_ground',
                'specific_humidity_2m_above_ground',
                'precipitable_water_entire_atmosphere'
            ]
            for feature in weather_features:
                if feature in user_input:
                    all_features[feature] = float(user_input[feature])
            
            # 4. Pollutant features
            pollutants = {
                k: v for k, v in user_input.items() 
                if k.startswith('L3_')
            }
            if pollutants:
                pollutant_features = self.engineer_pollutant_features(pollutants)
                all_features.update(pollutant_features)
            
            # 5. Weather ratios
            if all(k in all_features for k in ['temperature_2m_above_ground', 'relative_humidity_2m_above_ground']):
                ratios = self.engineer_weather_ratios(
                    temperature=all_features['temperature_2m_above_ground'],
                    humidity=all_features['relative_humidity_2m_above_ground'],
                    pressure=user_input.get('pressure', 1013.25),  # Default sea level pressure
                    wind_speed=all_features.get('wind_speed', 0),
                    total_pollutants=all_features.get('total_pollutant_load', 0)
                )
                all_features.update(ratios)
            
            logging.info(f"Feature engineering complete. Generated {len(all_features)} features")
            return all_features
            
        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    # Test feature engineer
    logging.info("="*70)
    logging.info("TESTING FEATURE ENGINEER")
    logging.info("="*70)
    
    # Initialize
    engineer = FeatureEngineer()
    
    # Sample user input
    sample_input = {
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
    
    print("\n Sample user input:")
    for key, value in sample_input.items():
        print(f"   {key}: {value}")
    
    # Process input
    print("\n Processing user input...")
    engineered_features = engineer.process_user_input(sample_input)
    
    print(f"\ Engineered {len(engineered_features)} features")
    
    # Display by category
    print("\n Temporal Features:")
    temporal_keys = ['year', 'month', 'day', 'season', 'month_sin', 'month_cos']
    for key in temporal_keys:
        if key in engineered_features:
            print(f"   {key}: {engineered_features[key]}")
    
    print("\n Wind Features:")
    wind_keys = ['wind_speed', 'wind_direction_deg', 'wind_direction_category']
    for key in wind_keys:
        if key in engineered_features:
            print(f"   {key}: {engineered_features[key]}")
    
    print("\n Pollutant Features:")
    pollutant_keys = ['total_pollutant_load', 'avg_pollutant_concentration', 'AQI_proxy']
    for key in pollutant_keys:
        if key in engineered_features:
            print(f"   {key}: {engineered_features[key]:.2f}")
    
    print("\n Weather Ratios:")
    ratio_keys = ['temp_humidity_ratio', 'heat_index', 'pollutant_per_windspeed']
    for key in ratio_keys:
        if key in engineered_features:
            print(f"   {key}: {engineered_features[key]:.2f}")
    
    logging.info("\n Feature engineer test completed!")