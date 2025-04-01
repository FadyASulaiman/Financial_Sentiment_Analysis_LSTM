import os
import yaml
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Default configuration file path
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file or use default.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Dict containing configuration
    """
    try:
        if os.path.exists(DEFAULT_CONFIG_PATH):
            with open(DEFAULT_CONFIG_PATH, 'r') as file:
                config = yaml.safe_load(file)
                logger.info(f"Loaded default configuration from {DEFAULT_CONFIG_PATH}")
                return config
    except Exception as e:
        logger.error(f"Error loading default configuration: {str(e)}")
    
    # If all else fails, use hardcoded default config
    logger.warning("Using hardcoded default configuration")
    return get_hardcoded_default_config()

def get_hardcoded_default_config() -> Dict[str, Any]:
    """
    Return hardcoded default configuration.
    """
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