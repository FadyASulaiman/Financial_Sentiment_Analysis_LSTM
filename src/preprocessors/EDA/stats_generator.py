import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.eda_logger import logger

class StatsGenerator:
    """Class for generating numerical statistics about the dataset"""

    def __init__(self, data, output_dir):
        """
        Initialize with a pandas DataFrame and output directory
        
        Args:
            data (pd.DataFrame): Pandas DataFrame with processed data
            output_dir (Path): Directory to save outputs
        """
        self.data = data
        self.output_dir = output_dir

    def generate_basic_stats(self):
        """Generate basic statistics about the dataset"""
        try:
            # Dataset size
            total_samples = len(self.data)

            # Sentiment distribution
            sentiment_counts = self.data['Sentiment'].value_counts()
            sentiment_percentages = self.data['Sentiment'].value_counts(normalize=True) * 100

            # Text length statistics
            length_stats = self.data['sentence_length'].describe()

            # Word count statistics
            word_stats = self.data['word_count'].describe()

            # Create summary DataFrame
            summary = pd.DataFrame({
                'Metric': ['Total samples', 
                           *[f'{s} samples' for s in sentiment_counts.index],
                           *[f'{s} percentage' for s in sentiment_percentages.index],
                           'Avg text length', 'Min text length', 'Max text length',
                           'Avg word count', 'Min word count', 'Max word count'],
                'Value': [total_samples, 
                         *sentiment_counts.values,
                         *[f'{p:.2f}%' for p in sentiment_percentages.values],
                         f"{length_stats['mean']:.2f}", int(length_stats['min']), int(length_stats['max']),
                         f"{word_stats['mean']:.2f}", int(word_stats['min']), int(word_stats['max'])]
            })

            # Save to file
            output_path = self.output_dir / 'basic_stats.csv'
            summary.to_csv(output_path, index=False)

            logger.info(f"Basic statistics saved to {output_path}")
            return summary
        except Exception as e:
            logger.error(f"Error generating basic statistics: {str(e)}")
            raise

    def generate_detailed_stats(self):
        """Generate more detailed statistics by sentiment"""
        try:
            # Group by sentiment
            grouped = self.data.groupby('Sentiment')

            # Text length statistics by sentiment
            length_stats = grouped['sentence_length'].describe()

            # Word count statistics by sentiment
            word_stats = grouped['word_count'].describe()

            # Save to files
            length_path = self.output_dir / 'text_length_stats.csv'
            word_path = self.output_dir / 'word_count_stats.csv'

            length_stats.to_csv(length_path)
            word_stats.to_csv(word_path)

            logger.info(f"Detailed statistics saved to {length_path} and {word_path}")
            return length_stats, word_stats
        except Exception as e:
            logger.error(f"Error generating detailed statistics: {str(e)}")
            raise

    def calculate_correlation_matrix(self):
        """Calculate correlation matrix for numerical features"""
        try:
            # Select numerical columns
            numerical_data = self.data.select_dtypes(include=[np.number])
            
            # Add sentiment as dummy variables
            sentiment_dummies = pd.get_dummies(self.data['Sentiment'], prefix='sentiment')
            numerical_data = pd.concat([numerical_data, sentiment_dummies], axis=1)
            
            # Calculate correlation matrix
            corr_matrix = numerical_data.corr()
            
            # Save to file
            output_path = self.output_dir / 'correlation_matrix.csv'
            corr_matrix.to_csv(output_path)
            
            # Also create a heatmap
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                        mask=mask, linewidths=0.5)
            
            plt.title('Correlation Matrix of Numerical Features')
            plt.tight_layout()
            
            heatmap_path = self.output_dir / 'correlation_heatmap.png'
            plt.savefig(heatmap_path)
            plt.close()
            
            logger.info(f"Correlation matrix saved to {output_path} and {heatmap_path}")
            return corr_matrix
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {str(e)}")
            raise