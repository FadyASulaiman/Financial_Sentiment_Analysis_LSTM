# utils/logger.py
"""Logging utilities for the preprocessing pipeline."""

import logging

def setup_logger(log_file="finance_preprocessing.log"):
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