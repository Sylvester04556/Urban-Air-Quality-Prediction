
import os
import sys
import pickle
import json
import yaml
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List

from src.logger import logging
from src.exception import CustomException


def load_pickle(file_path: str) -> Any:
    """
    Load a pickle file.
    
    Args:
        file_path: Path to the pickle file
        
    Returns:
        The unpickled object
        
    Raises:
        CustomException: If file not found or loading fails
    """
    try:
        logging.info(f"Loading pickle file from: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        
        logging.info(f"Successfully loaded pickle file: {file_path}")
        return obj
        
    except Exception as e:
        raise CustomException(e, sys)


def save_pickle(obj: Any, file_path: str) -> None:
    """
    Save an object to a pickle file.
    
    Args:
        obj: Object to save
        file_path: Path where to save the pickle file
        
    Raises:
        CustomException: If saving fails
    """
    try:
        logging.info(f"Saving pickle file to: {file_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        
        logging.info(f"Successfully saved pickle file: {file_path}")
        
    except Exception as e:
        raise CustomException(e, sys)


def load_json(file_path: str) -> Dict:
    """
    Load a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        dict: Parsed JSON data
        
    Raises:
        CustomException: If file not found or loading fails
    """
    try:
        logging.info(f"Loading JSON file from: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        logging.info(f"Successfully loaded JSON file: {file_path}")
        return data
        
    except Exception as e:
        raise CustomException(e, sys)


def save_json(data: Dict, file_path: str, indent: int = 4) -> None:
    """
    Save data to a JSON file.
    
    Args:
        data: Dictionary to save
        file_path: Path where to save the JSON file
        indent: Indentation for pretty printing
        
    Raises:
        CustomException: If saving fails
    """
    try:
        logging.info(f"Saving JSON file to: {file_path}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=indent)
        
        logging.info(f"Successfully saved JSON file: {file_path}")
        
    except Exception as e:
        raise CustomException(e, sys)


def load_text_file(file_path: str) -> List[str]:
    """
    Load a text file and return lines as list.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        list: List of lines from the file
        
    Raises:
        CustomException: If file not found or loading fails
    """
    try:
        logging.info(f"Loading text file from: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
        
        logging.info(f"Successfully loaded {len(lines)} lines from: {file_path}")
        return lines
        
    except Exception as e:
        raise CustomException(e, sys)


def ensure_dir(directory: str) -> None:
    """
    Create directory if it doesn't exist.
    
    Args:
        directory: Path to directory
    """
    try:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Directory ensured: {directory}")
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == "__main__":
    # Test utilities
    logging.info("Testing common utilities...")
    
    # Test save and load JSON
    test_data = {"test": "data", "number": 123}
    test_file = "test_output/test.json"
    
    save_json(test_data, test_file)
    loaded_data = load_json(test_file)
    
    print(f"Original: {test_data}")
    print(f"Loaded: {loaded_data}")
    print(f"Match: {test_data == loaded_data}")
    
    logging.info("Common utilities test complete!")