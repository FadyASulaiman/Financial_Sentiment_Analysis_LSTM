# utils/logger.py
import logging

def setup_production_logger(log_file="logs/production.log"):
    """This logger tracks all the pipleines combined along with predictions for the production model."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("PROD")
    return logger

# Create a global logger instance
logger = setup_production_logger()