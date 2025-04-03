import logging
from pathlib import Path

def setup_eda_logger(log_file="logs/finance_sentiment_eda.log"):
    """Configure and return a logger"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger("FinanceSentimentEDA")
    return logger

# Create a global logger instance
logger = setup_eda_logger()