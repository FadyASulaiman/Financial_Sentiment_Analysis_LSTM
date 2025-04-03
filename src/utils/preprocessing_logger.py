# utils/logger.py
"""Logging utilities for the preprocessing pipeline."""

import logging

def setup_preprocessing_logger(log_file="logs/finance_preprocessing.log"):
    """Configure and return a logger."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("FinancePreprocessing")
    return logger

# Create a global logger instance
logger = setup_preprocessing_logger()