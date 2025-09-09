import logging
import os

def setup_logger():
    logger = logging.getLogger('RecyclingPrediction')
    logger.setLevel(logging.INFO)
    
    # Create file handler
    os.makedirs('logs', exist_ok=True)
    fh = logging.FileHandler('logs/pipeline.log')
    fh.setLevel(logging.INFO)
    
    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add handlers to logger
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger