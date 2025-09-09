import logging
import os
from datetime import datetime

def setup_logger():
    logger = logging.getLogger('RecyclingPrediction')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to prevent duplicates
    logger.handlers = []
    
    # Create log directory
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    
    # Use timestamped log file
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = os.path.join(log_dir, f'pipeline_{timestamp}.log')
    
    # Create file handler with UTF-8 encoding
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logger initialized with log file: {log_file}")
    return logger
