import logging
from pathlib import Path

def setup_data_prep_logger(log_file="logs/data_prep.log"):
    """Configure and return a logger"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("DataPrepPipeline")
    return logger

# Create a global logger instance
logger = setup_data_prep_logger()