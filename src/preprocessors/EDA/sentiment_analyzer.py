import pandas as pd
from src.utils.loggers.eda_logger import logger

class SentimentAnalyzer:
    """Class for analyzing sentiment distributions and patterns"""
    
    def __init__(self, data):
        """
        Initialize with a pandas DataFrame
        
        Args:
            data (pd.DataFrame): Pandas DataFrame with sentiment data
        """
        self.data = data
        
    def sentiment_distribution(self):
        """Analyze the distribution of sentiments"""
        try:
            sent_counts = self.data['Sentiment'].value_counts()
            sent_percentages = self.data['Sentiment'].value_counts(normalize=True) * 100
            
            result = pd.DataFrame({
                'Count': sent_counts,
                'Percentage': sent_percentages
            })
            
            logger.info(f"Sentiment distribution:\n{result}")
            return result
        except Exception as e:
            logger.error(f"Error analyzing sentiment distribution: {str(e)}")
            raise
    
    def text_length_by_sentiment(self):
        """Analyze text length patterns by sentiment"""
        try:
            length_stats = self.data.groupby('Sentiment')['sentence_length'].agg([
                'count', 'mean', 'median', 'min', 'max', 'std'
            ]).round(2)
            
            logger.info(f"Text length statistics by sentiment:\n{length_stats}")
            return length_stats
        except Exception as e:
            logger.error(f"Error analyzing text length by sentiment: {str(e)}")
            raise
    
    def word_count_by_sentiment(self):
        """Analyze word count patterns by sentiment"""
        try:
            word_stats = self.data.groupby('Sentiment')['word_count'].agg([
                'count', 'mean', 'median', 'min', 'max', 'std'
            ]).round(2)
            
            logger.info(f"Word count statistics by sentiment:\n{word_stats}")
            return word_stats
        except Exception as e:
            logger.error(f"Error analyzing word count by sentiment: {str(e)}")
            raise