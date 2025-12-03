

import sys
from src.logger import logging


def error_message_detail(error, error_detail: sys):
    """
    Generate detailed error message with file name and line number.
    
    Args:
        error: The error object
        error_detail: sys module to extract exception info
        
    Returns:
        str: Formatted error message
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    
    error_message = f"Error occurred in script: [{file_name}] at line [{line_number}]: {str(error)}"
    
    return error_message


class CustomException(Exception):
    """
    Custom exception class that logs detailed error information.
    
    Usage:
        try:
            # some code
        except Exception as e:
            raise CustomException(e, sys)
    """
    
    def __init__(self, error_message, error_detail: sys):
        """
        Initialize custom exception.
        
        Args:
            error_message: The original error message
            error_detail: sys module to extract exception info
        """
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
        
        # Log the error
        logging.error(self.error_message)
    
    def __str__(self):
        """Return the detailed error message"""
        return self.error_message


if __name__ == "__main__":
    # Test custom exception
    try:
        logging.info("Testing custom exception...")
        
        # Simulate an error
        result = 1 / 0
        
    except Exception as e:
        logging.info("Exception caught, raising CustomException...")
        raise CustomException(e, sys)