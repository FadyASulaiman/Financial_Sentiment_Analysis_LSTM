import logging
from typing import List, Optional, Union
import json
import ast
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from pathlib import Path

class FiqaDataLoading:
    # Class constants
    SENTIMENT_BINS = [-1.0001, -0.33, 0.33, 1.0001]
    SENTIMENT_LABELS = ['negative', 'neutral', 'positive']
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None
        self.logger = logging.getLogger(__name__)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def load_and_preprocess(self) -> pd.DataFrame:
        """Load JSON data and convert to structured DataFrame"""
        try:
            with open(self.filepath, encoding='utf-8') as f:
                raw_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

        records = []
        for entry_id, content in raw_data.items():
            try:
                for info in content['info']:
                    record = {
                        'id': entry_id,
                        'sentence': content['sentence'],
                        'snippets': info['snippets'],
                        'target': info['target'],
                        'sentiment_score': float(info['sentiment_score']),
                        'aspects': self.safe_literal_eval(info['aspects'])
                    }
                    records.append(record)
            except KeyError as e:
                self.logger.warning(f"Missing key in entry {entry_id}: {str(e)}")
            except ValueError as e:
                self.logger.warning(f"Invalid value in entry {entry_id}: {str(e)}")

        self.df = pd.DataFrame(records)
        self._enhance_data()
        return self.df

    def safe_literal_eval(self, x: Union[str, List]) -> List:
        if x is None:
            return []
        if isinstance(x, list):
            return x
        try:
            return ast.literal_eval(x.replace("'", "\""))
        except (ValueError, SyntaxError) as e:
            self.logger.warning(f"Error parsing aspect: {str(e)}")
            return [str(x)]

    def _enhance_data(self):
        """Create additional features and clean data"""
        # Handle NaN values in sentiment_score
        self.df['sentiment_score'] = pd.to_numeric(self.df['sentiment_score'], errors='coerce')
        
        # Fill NaN values with 0 (neutral sentiment)
        self.df['sentiment_score'].fillna(0, inplace=True)
        
        # Sentiment classification
        self.df['sentiment_class'] = pd.cut(
            self.df['sentiment_score'], 
            bins=self.SENTIMENT_BINS,
            labels=self.SENTIMENT_LABELS,
            right=True,
            include_lowest=True
        )

        # Aspect hierarchy processing
        self.df['primary_aspect'] = self.df['aspects'].apply(
            lambda x: self._extract_aspect(x, 0))
        self.df['secondary_aspect'] = self.df['aspects'].apply(
            lambda x: self._extract_aspect(x, 1))
        
        self.df['snippet_text'] = self.df['snippets'].apply(
            lambda x: ' '.join(self.safe_literal_eval(x)) if pd.notna(x) else '')

    @staticmethod
    def _extract_aspect(aspects: List, level: int) -> Optional[str]:
        if not aspects or not isinstance(aspects, list):
            return None
        try:
            parts = aspects[0].split('/')
            return parts[level].strip() if level < len(parts) else None
        except (IndexError, AttributeError):
            return None


class FiqaEDA:
    # Class constants
    DEFAULT_FIGSIZE = (12, 6)
    WORDCLOUD_FIGSIZE = (18, 6)
    STOPWORDS = set(["a", "as", "and", "by", "on", "of", "to", "the", "in", 
                     "for", "with", "Â", "from", "at", "its", "their", "has"])
    
    def __init__(self, df: pd.DataFrame, output_dir: Optional[str] = None):
        self.df = df.copy()  # Create a copy to prevent modifications to original
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Create output directory if it doesn't exist
        if self.output_dir:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Validate input DataFrame
        required_columns = ['sentiment_score', 'sentiment_class', 'primary_aspect',
                          'secondary_aspect', 'snippet_text', 'target']
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

    def save_plot(self, fig, name: str):
        """Save plot to specified output directory"""
        if self.output_dir:
            try:
                output_path = Path(self.output_dir) / f"{name}.png"
                fig.savefig(output_path, bbox_inches='tight', dpi=300)
                self.logger.info(f"Plot saved to {output_path}")
            except Exception as e:
                self.logger.error(f"Error saving plot {name}: {str(e)}")

    def _validate_data(self) -> dict:
        """Perform data quality checks and return validation results"""
        validation_results = {
            'total_entries': len(self.df),
            'missing_values': self.df.isnull().sum().to_dict(),
            'invalid_scores': len(self.df[
                (self.df['sentiment_score'] < -1) | 
                (self.df['sentiment_score'] > 1)
            ]),
            'empty_snippets': len(self.df[self.df['snippet_text'] == '']),
            'empty_sentences': len(self.df[self.df['sentence'] == '']),
            'unique_targets': self.df['target'].nunique(),
            'unique_primary_aspects': self.df['primary_aspect'].nunique()
        }
        
        print("=== Data Validation Report ===")
        print(f"Total entries: {validation_results['total_entries']}")
        print("\nMissing values:")
        for col, count in validation_results['missing_values'].items():
            print(f"{col}: {count}")
        print(f"\nInvalid sentiment scores: {validation_results['invalid_scores']}")
        print(f"Empty snippets: {validation_results['empty_snippets']}")
        print(f"Empty sentences: {validation_results['empty_sentences']}")
        print(f"Unique targets: {validation_results['unique_targets']}")
        print(f"Unique primary aspects: {validation_results['unique_primary_aspects']}")
        
        return validation_results

    def analyze_sentiment_distribution(self):
        """Generate sentiment distribution visualizations"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.DEFAULT_FIGSIZE)
            
            # Sentiment score distribution
            sns.histplot(data=self.df, x='sentiment_score', bins=20, kde=True, ax=ax1)
            ax1.set_title('Sentiment Score Distribution')
            ax1.set_xlabel('Sentiment Score')
            ax1.set_ylabel('Count')
            
            # Sentiment class distribution
            sentiment_counts = self.df['sentiment_class'].value_counts()
            sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, ax=ax2)
            ax2.set_title('Sentiment Class Distribution')
            ax2.set_xlabel('Sentiment Class')
            ax2.set_ylabel('Count')
            
            plt.tight_layout()
            self.save_plot(fig, 'sentiment_distribution')
            return fig
            
        except Exception as e:
            self.logger.error(f"Error in sentiment distribution analysis: {str(e)}")
            raise

    def analyze_aspects(self):
        """Generate aspect analysis visualizations"""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.DEFAULT_FIGSIZE)
            
            # Primary aspects
            primary_counts = self.df['primary_aspect'].value_counts()
            sns.barplot(y=primary_counts.index, x=primary_counts.values, ax=ax1)
            ax1.set_title('Primary Aspect Distribution')
            ax1.set_xlabel('Count')
            
            # Secondary aspects (top 15)
            secondary_counts = self.df['secondary_aspect'].value_counts().head(15)
            sns.barplot(y=secondary_counts.index, x=secondary_counts.values, ax=ax2)
            ax2.set_title('Top 15 Secondary Aspects')
            ax2.set_xlabel('Count')
            
            plt.tight_layout()
            self.save_plot(fig, 'aspect_distribution')
            return fig
            
        except Exception as e:
            self.logger.error(f"Error in aspect analysis: {str(e)}")
            raise

    def generate_word_clouds(self):
        """Generate sentiment-specific word clouds"""
        try:
            fig, axes = plt.subplots(1, 3, figsize=self.WORDCLOUD_FIGSIZE)
            
            sentiment_colors = {
                'positive': 'Greens',
                'neutral': 'viridis',
                'negative': 'Reds'
            }
            
            for i, sentiment in enumerate(['positive', 'neutral', 'negative']):
                text = ' '.join(self.df[
                    self.df['sentiment_class'] == sentiment]['snippet_text']
                )
                
                if not text.strip():  # Check if text is empty
                    self.logger.warning(f"No text found for {sentiment} sentiment")
                    continue
                    
                wc = WordCloud(
                    width=1200, 
                    height=800,
                    background_color='white',
                    stopwords=self.STOPWORDS,
                    colormap=sentiment_colors[sentiment],
                    max_words=200,
                    random_state=42
                ).generate(text)

                axes[i].imshow(wc)
                axes[i].set_title(f'{sentiment.capitalize()} Sentiment Terms')
                axes[i].axis('off')

            plt.tight_layout()
            self.save_plot(fig, 'sentiment_wordclouds')
            return fig
            
        except Exception as e:
            self.logger.error(f"Error generating word clouds: {str(e)}")
            raise

    def analyze_targets(self):
        """Analyze target companies/organizations"""
        try:
            fig, ax = plt.subplots(figsize=self.DEFAULT_FIGSIZE)
            
            target_counts = self.df['target'].value_counts().head(10)
            sns.barplot(y=target_counts.index, x=target_counts.values)
            ax.set_title('Top 10 Frequently Mentioned Targets')
            ax.set_xlabel('Count')
            
            plt.tight_layout()
            self.save_plot(fig, 'target_distribution')
            return fig
            
        except Exception as e:
            self.logger.error(f"Error in target analysis: {str(e)}")
            raise

    def run_full_analysis(self):
        """Execute complete EDA pipeline"""
        try:
            validation_results = self._validate_data()
            
            print("\n=== Basic Statistics ===")
            print(self.df.describe(include='all'))
            
            figures = {
                'sentiment': self.analyze_sentiment_distribution(),
                'aspects': self.analyze_aspects(),
                'wordclouds': self.generate_word_clouds(),
                'targets': self.analyze_targets()
            }
            
            return validation_results, figures
            
        except Exception as e:
            self.logger.error(f"Error in full analysis: {str(e)}")
            raise

if __name__ == "__main__":
    loader = FiqaDataLoading("data/raw/FiQA_ABSA_task1/task1_headline_ABSA_train.json")
    df = loader.load_and_preprocess()

    analyzer = FiqaEDA(df, output_dir="analysis_output")
    validation_results, figures = analyzer.run_full_analysis()
