import logging
import os

def setup_feature_engineering_pipeline_logging(log_file: str = "feature_engineering.log", level=logging.INFO):
    """Setup logging configuration.
    
    Args:
        log_file: Path to log file
        level: Logging level
    """
    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# Create a global logger instance
logger = setup_feature_engineering_pipeline_logging()