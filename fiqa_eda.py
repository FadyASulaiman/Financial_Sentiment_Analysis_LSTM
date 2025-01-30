import json
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from matplotlib import rcParams

# Configuration
rcParams.update({'figure.autolayout': True})
# plt.style.use('seaborn-whitegrid')

class FiqaEDA:
    def __init__(self, filepath):
        self.filepath = filepath
        self.df = None
        self.aspect_hierarchy = []
        
    def _load_and_preprocess(self):
        """Load JSON data and convert to structured DataFrame"""
        with open(self.filepath) as f:
            raw_data = json.load(f)
        
        records = []
        for entry_id, content in raw_data.items():
            base_entry = {
                'id': entry_id,
                'sentence': content['sentence']
            }
            for info in content['info']:
                record = base_entry.copy()
                record.update({
                    'snippets': ast.literal_eval(info['snippets']),
                    'target': info['target'],
                    'sentiment_score': float(info['sentiment_score']),
                    'aspects': ast.literal_eval(info['aspects'])
                })
                records.append(record)
                
        self.df = pd.DataFrame(records)
        self._enhance_data()

    def _enhance_data(self):
        """Create additional features and clean data"""
        # Sentiment classification
        bins = [-1, -0.33, 0.33, 1]
        labels = ['negative', 'neutral', 'positive']
        self.df['sentiment_class'] = pd.cut(self.df['sentiment_score'], 
                                          bins=bins, labels=labels)
        
        # Aspect hierarchy processing
        self.df['primary_aspect'] = self.df['aspects'].apply(
            lambda x: x[0].split('/')[0] if len(x) > 0 else 'unknown'
        )
        self.df['secondary_aspect'] = self.df['aspects'].apply(
            lambda x: x[0].split('/')[1] if len(x[0].split('/')) > 1 else 'none'
        )
        
        # Text processing
        self.df['snippet_text'] = self.df['snippets'].str.join(' ')

    def _validate_data(self):
        """Data quality checks"""
        print("=== Data Validation Report ===")
        print(f"Total entries: {len(self.df)}")
        print("\nMissing values:")
        print(self.df.isnull().sum())
        
        # Check for invalid sentiment scores
        invalid_scores = self.df[~self.df['sentiment_score'].between(-1, 1)]
        print(f"\nInvalid sentiment scores: {len(invalid_scores)}")

    def analyze_sentiment_distribution(self):
        """Generate sentiment visualizations"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        sns.histplot(self.df['sentiment_score'], bins=20, kde=True)
        plt.title('Sentiment Score Distribution')
        
        plt.subplot(1, 2, 2)
        self.df['sentiment_class'].value_counts().plot(kind='bar')
        plt.title('Sentiment Class Distribution')
        
        plt.tight_layout()
        plt.show()

    def analyze_aspects(self):
        """Aspect category analysis"""
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        
        # Primary aspects
        primary_counts = self.df['primary_aspect'].value_counts()
        sns.barplot(y=primary_counts.index, x=primary_counts.values, ax=ax[0])
        ax[0].set_title('Primary Aspect Distribution')
        
        # Secondary aspects (top 15)
        secondary_counts = self.df['secondary_aspect'].value_counts().head(15)
        sns.barplot(y=secondary_counts.index, x=secondary_counts.values, ax=ax[1])
        ax[1].set_title('Top 15 Secondary Aspects')
        
        plt.tight_layout()
        plt.show()

    def generate_word_clouds(self):
        """Generate sentiment-specific word clouds"""
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, sentiment in enumerate(['positive', 'neutral', 'negative']):
            text = ' '.join(self.df[self.df['sentiment_class'] == sentiment]['snippet_text'])
            wc = WordCloud(
                width=400, 
                height=400, 
                background_color='white',
                colormap='viridis' if sentiment == 'neutral' else 
                        'Greens' if sentiment == 'positive' else 'Reds'
            ).generate(text)
            
            ax[i].imshow(wc)
            ax[i].set_title(f'{sentiment.capitalize()} Sentiment Terms')
            ax[i].axis('off')
        
        plt.show()

    def analyze_targets(self):
        """Company/organization analysis"""
        plt.figure(figsize=(10, 6))
        
        target_counts = self.df['target'].value_counts().head(10)
        sns.barplot(y=target_counts.index, x=target_counts.values)
        plt.title('Top 10 Frequently Mentioned Targets')
        plt.show()

    def run_full_analysis(self):
        """Execute complete EDA pipeline"""
        self._load_and_preprocess()
        self._validate_data()
        
        print("\n=== Basic Statistics ===")
        print(self.df.describe(include='all'))
        
        self.analyze_sentiment_distribution()
        self.analyze_aspects()
        self.generate_word_clouds()
        self.analyze_targets()

# Usage
if __name__ == "__main__":
    analyzer = FiqaEDA("data/FiQA_ABSA_task1/task1_headline_ABSA_train.json")  # Replace with actual path
    analyzer.run_full_analysis()