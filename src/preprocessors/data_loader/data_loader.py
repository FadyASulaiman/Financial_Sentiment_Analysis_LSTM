import os
import pandas as pd
from utils.logger import logger

class DataLoader:
    """Class for loading and preprocessing the dataset"""
    
    def __init__(self, file_path=None, data=None):
        """
        Initialize with either a file path to CSV or a pandas DataFrame
        
        Args:
            file_path (str, optional): Path to CSV file
            data (pd.DataFrame, optional): Pandas DataFrame
        """
        self.file_path = file_path
        self.data = data
        
    def load_data(self):
        """Load data from CSV file or use provided DataFrame"""
        try:
            if self.file_path:
                if os.path.exists(self.file_path):
                    self.data = pd.read_csv(self.file_path)
                    logger.info(f"Successfully loaded data from {self.file_path}")
                else:
                    raise FileNotFoundError(f"File not found: {self.file_path}")
            elif self.data is None:
                raise ValueError("No data provided. Please provide either a file path or DataFrame.")
            
            # Basic validation of expected columns
            if 'Sentence' not in self.data.columns or 'Sentiment' not in self.data.columns:
                raise ValueError("Dataset must contain 'Sentence' and 'Sentiment' columns")
                
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
            
    def preprocess_data(self):
        """Preprocess the loaded data"""
        try:
            if self.data is None:
                self.load_data()
                
            # Check for missing values
            missing_values = self.data.isnull().sum()
            logger.info(f"Missing values in dataset:\n{missing_values}")
            
            # Drop rows with missing values
            original_len = len(self.data)
            self.data = self.data.dropna()
            dropped_rows = original_len - len(self.data)
            if dropped_rows > 0:
                logger.info(f"Dropped {dropped_rows} rows with missing values")
            
            # Add length of sentences as a feature
            self.data['sentence_length'] = self.data['Sentence'].apply(len)
            self.data['word_count'] = self.data['Sentence'].apply(lambda x: len(str(x).split()))
            
            # Ensure sentiment values are standardized (lowercase)
            self.data['Sentiment'] = self.data['Sentiment'].str.lower()
            
            return self.data
        except Exception as e:
            logger.error(f"Error preprocessing data: {str(e)}")
            raise