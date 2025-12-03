

import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Create log file with timestamp
LOG_FILE = f"{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter(
    "[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s"
)
console_handler.setFormatter(console_formatter)

# Get root logger and add console handler
logger = logging.getLogger()
logger.addHandler(console_handler)

if __name__ == "__main__":
    # Test logging
    logging.info("Logger initialized successfully!")
    logging.warning("This is a warning message")
    logging.error("This is an error message")
    logging.info(f"Logs are being saved to: {LOG_FILE_PATH}")