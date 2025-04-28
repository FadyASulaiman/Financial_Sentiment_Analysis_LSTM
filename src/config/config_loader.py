import os
import yaml
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file or use default.
    """

    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                logger.info(f"Loaded configuration from {config_path}")
                return config
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {str(e)}")

    # If config import fails, use hardcoded default config
    logger.warning("Using hardcoded default configuration")
    return get_hardcoded_default_config()

def get_hardcoded_default_config() -> Dict[str, Any]:

    return {
        "version": "1.0.0",
        "random_seed": 42,
        "features": {
            "n_gram_range": [1, 3],
            "min_df": 5,
            "max_df": 0.9,
            "tfidf_max_features": 5000,
            "entity_extraction": {
                "use_spacy": True,
                "use_regex": True,
                "use_gazetteer": True
            },
            "financial_indicators": {
                "window_size": 5
            }
        },
        "output": {
            "file_name": "feature_engineered_data.csv"
        }
    }