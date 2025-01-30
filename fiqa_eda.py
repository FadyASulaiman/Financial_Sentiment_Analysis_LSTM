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

    def _load_and_preprocess(self):
        """Load JSON data and convert to structured DataFrame"""
        with open(self.filepath, encoding='utf-8') as f:  # Add encoding for potential unicode chars
            raw_data = json.load(f)

        records = []
        for entry_id, content in raw_data.items():
            for info in content['info']:
                record = {
                    'id': entry_id,
                    'sentence': content['sentence'],
                    'snippets': info['snippets'],  # Keep original string representation for later processing
                    'target': info['target'],
                    'sentiment_score': float(info['sentiment_score']),
                    'aspects': ast.literal_eval(info['aspects']) # Evaluate aspects here
                }
                records.append(record)

        self.df = pd.DataFrame(records)
        self._enhance_data()

    def _enhance_data(self):
        """Create additional features and clean data"""

        # Sentiment classification (using cut is generally better for defined boundaries)
        bins = [-1, -0.33, 0.33, 1]
        labels = ['negative', 'neutral', 'positive']
        self.df['sentiment_class'] = pd.cut(self.df['sentiment_score'], bins=bins, labels=labels, right=True, include_lowest=True)

        # Aspect hierarchy processing (handle potential errors more robustly)
        def extract_aspect(aspects, level):
            try:
                if aspects and isinstance(aspects[0], str): # Check if aspects is not empty and contains strings
                    return aspects[0].split('/')[level] if len(aspects[0].split('/')) > level else None
                else:
                    return None
            except (IndexError, TypeError):  # Catch potential errors if the format is unexpected
                return None


        self.df['primary_aspect'] = self.df['aspects'].apply(lambda x: extract_aspect(x, 0))
        self.df['secondary_aspect'] = self.df['aspects'].apply(lambda x: extract_aspect(x, 1))

        def safe_literal_eval(x):
            try:
                if isinstance(x, str):
                    return ast.literal_eval(x)
                else:
                    return []
            except (SyntaxError, ValueError):  # Handle SyntaxError specifically
                # Handle cases where literal_eval fails, e.g., unescaped characters
                if isinstance(x, str):
                    return [x] # Return the string as a list if eval fails
                else:
                    return []


        self.df['snippet_text'] = self.df['snippets'].apply(lambda x: ' '.join(safe_literal_eval(x)))


    def _validate_data(self):
        """Data quality checks"""
        print("=== Data Validation Report ===")
        print(f"Total entries: {len(self.df)}")
        print("\nMissing values:")
        print(self.df.isnull().sum())

        # Check for invalid sentiment scores (outside -1 to 1 range)
        invalid_scores = self.df[(self.df['sentiment_score'] < -1) | (self.df['sentiment_score'] > 1)]
        print(f"\nInvalid sentiment scores (outside -1 to 1): {len(invalid_scores)}")

        # Check for empty snippets or sentences
        print(f"\nEmpty snippets: {len(self.df[self.df['snippet_text'] == ''])}")
        print(f"Empty sentences: {len(self.df[self.df['sentence'] == ''])}")



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
        """Generate sentiment-specific word clouds with stopword removal"""
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        stopwords = set(["a"])  # Use default stopwords and add custom ones

        for i, sentiment in enumerate(['positive', 'neutral', 'negative']):
            text = ' '.join(self.df[self.df['sentiment_class'] == sentiment]['snippet_text'])
            wc = WordCloud(width=800, height=400, # Increased size for better readability
                          background_color='white', stopwords=stopwords,
                          colormap='viridis' if sentiment == 'neutral' else
                          'Greens' if sentiment == 'positive' else 'Reds',
                          max_words=200).generate(text) # Limit to top 200 words

            ax[i].imshow(wc)
            ax[i].set_title(f'{sentiment.capitalize()} Sentiment Terms')
            ax[i].axis('off')

        plt.tight_layout() # Ensure layout doesn't overlap
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



# class FiqaEDA:
#     def __init__(self, filepath):
#         self.filepath = filepath
#         self.df = None
#         self.aspect_hierarchy = []
        
#     def _load_and_preprocess(self):
#         """Load JSON data and convert to structured DataFrame"""
#         with open(self.filepath) as f:
#             raw_data = json.load(f)
        
#         records = []
#         for entry_id, content in raw_data.items():
#             base_entry = {
#                 'id': entry_id,
#                 'sentence': content['sentence']
#             }
#             for info in content['info']:
#                 record = base_entry.copy()
#                 record.update({
#                     'snippets': ast.literal_eval(info['snippets']),
#                     'target': info['target'],
#                     'sentiment_score': float(info['sentiment_score']),
#                     'aspects': ast.literal_eval(info['aspects'])
#                 })
#                 records.append(record)
                
#         self.df = pd.DataFrame(records)
#         self._enhance_data()

#     def _enhance_data(self):
#         """Create additional features and clean data"""
#         # Sentiment classification
#         bins = [-1, -0.33, 0.33, 1]
#         labels = ['negative', 'neutral', 'positive']
#         self.df['sentiment_class'] = pd.cut(self.df['sentiment_score'], 
#                                           bins=bins, labels=labels)
        
#         # Aspect hierarchy processing
#         self.df['primary_aspect'] = self.df['aspects'].apply(
#             lambda x: x[0].split('/')[0] if len(x) > 0 else 'unknown'
#         )
#         self.df['secondary_aspect'] = self.df['aspects'].apply(
#             lambda x: x[0].split('/')[1] if len(x[0].split('/')) > 1 else 'none'
#         )
        
#         # Text processing
#         self.df['snippet_text'] = self.df['snippets'].str.join(' ')

#     def _validate_data(self):
#         """Data quality checks"""
#         print("=== Data Validation Report ===")
#         print(f"Total entries: {len(self.df)}")
#         print("\nMissing values:")
#         print(self.df.isnull().sum())
        
#         # Check for invalid sentiment scores
#         invalid_scores = self.df[~self.df['sentiment_score'].between(-1, 1)]
#         print(f"\nInvalid sentiment scores: {len(invalid_scores)}")

#     def analyze_sentiment_distribution(self):
#         """Generate sentiment visualizations"""
#         plt.figure(figsize=(12, 5))
        
#         plt.subplot(1, 2, 1)
#         sns.histplot(self.df['sentiment_score'], bins=20, kde=True)
#         plt.title('Sentiment Score Distribution')
        
#         plt.subplot(1, 2, 2)
#         self.df['sentiment_class'].value_counts().plot(kind='bar')
#         plt.title('Sentiment Class Distribution')
        
#         plt.tight_layout()
#         plt.show()

#     def analyze_aspects(self):
#         """Aspect category analysis"""
#         fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        
#         # Primary aspects
#         primary_counts = self.df['primary_aspect'].value_counts()
#         sns.barplot(y=primary_counts.index, x=primary_counts.values, ax=ax[0])
#         ax[0].set_title('Primary Aspect Distribution')
        
#         # Secondary aspects (top 15)
#         secondary_counts = self.df['secondary_aspect'].value_counts().head(15)
#         sns.barplot(y=secondary_counts.index, x=secondary_counts.values, ax=ax[1])
#         ax[1].set_title('Top 15 Secondary Aspects')
        
#         plt.tight_layout()
#         plt.show()

#     def generate_word_clouds(self):
#         """Generate sentiment-specific word clouds"""
#         fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        
#         for i, sentiment in enumerate(['positive', 'neutral', 'negative']):
#             text = ' '.join(self.df[self.df['sentiment_class'] == sentiment]['snippet_text'])
#             wc = WordCloud(
#                 width=400, 
#                 height=400, 
#                 background_color='white',
#                 colormap='viridis' if sentiment == 'neutral' else 
#                         'Greens' if sentiment == 'positive' else 'Reds'
#             ).generate(text)
            
#             ax[i].imshow(wc)
#             ax[i].set_title(f'{sentiment.capitalize()} Sentiment Terms')
#             ax[i].axis('off')
        
#         plt.show()

#     def analyze_targets(self):
#         """Company/organization analysis"""
#         plt.figure(figsize=(10, 6))
        
#         target_counts = self.df['target'].value_counts().head(10)
#         sns.barplot(y=target_counts.index, x=target_counts.values)
#         plt.title('Top 10 Frequently Mentioned Targets')
#         plt.show()

#     def run_full_analysis(self):
#         """Execute complete EDA pipeline"""
#         self._load_and_preprocess()
#         self._validate_data()
        
#         print("\n=== Basic Statistics ===")
#         print(self.df.describe(include='all'))
        
#         self.analyze_sentiment_distribution()
#         self.analyze_aspects()
#         self.generate_word_clouds()
#         self.analyze_targets()

# Usage
if __name__ == "__main__":
    analyzer = FiqaEDA("data/FiQA_ABSA_task1/task1_headline_ABSA_train.json")  # Replace with actual path
    analyzer.run_full_analysis()