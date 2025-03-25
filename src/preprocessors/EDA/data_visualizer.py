import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
from src.utils.logger import logger

class DataVisualizer:
    """Class for visualizing insights from the dataset"""
    
    def __init__(self, data, output_dir):
        """
        Initialize with a pandas DataFrame and output directory
        
        Args:
            data (pd.DataFrame): Pandas DataFrame with processed data
            output_dir (Path): Directory to save visualization outputs
        """
        self.data = data
        self.output_dir = output_dir
        self.sentiment_colors = {
            'positive': 'green',
            'neutral': 'blue',
            'negative': 'red'
        }
        
        # Set visualization style
        sns.set(style="whitegrid")
        plt.rcParams.update({'font.size': 12})
    
    def plot_sentiment_distribution(self):
        """Plot distribution of sentiment classes"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Count plot
            ax = sns.countplot(
                x='Sentiment', 
                data=self.data,
                palette=self.sentiment_colors,
                order=self.data['Sentiment'].value_counts().index
            )
            
            # Add percentage labels
            total = len(self.data)
            for p in ax.patches:
                percentage = f'{100 * p.get_height() / total:.1f}%'
                ax.annotate(
                    f'{int(p.get_height())}\n({percentage})',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center',
                    va='center',
                    xytext=(0, 10),
                    textcoords='offset points'
                )
            
            plt.title('Distribution of Sentiment Classes')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')
            plt.tight_layout()
            
            # Save the figure
            output_path = self.output_dir / 'sentiment_distribution.png'
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Sentiment distribution plot saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error plotting sentiment distribution: {str(e)}")
            raise
    
    def plot_wordclouds(self, text_processor):
        """Generate and save wordclouds for each sentiment class"""
        try:
            # Get unique sentiment values
            sentiments = self.data['Sentiment'].unique()
            
            wordcloud_paths = {}
            
            for sentiment in sentiments:
                # Filter data for the sentiment
                filtered_data = self.data[self.data['Sentiment'] == sentiment]
                
                if len(filtered_data) == 0:
                    continue
                
                # Combine all text for the sentiment
                all_text = ' '.join(filtered_data['cleaned_text'])
                
                # Set color based on sentiment
                color_func = lambda *args, **kwargs: self.sentiment_colors.get(sentiment, 'gray')
                
                # Generate wordcloud
                wordcloud = WordCloud(
                    width=800, 
                    height=400,
                    background_color='white',
                    stopwords=STOPWORDS,
                    max_words=100,
                    collocations=False,
                    color_func=color_func
                ).generate(all_text)
                
                # Plot
                plt.figure(figsize=(12, 8))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                plt.title(f'Word Cloud for {sentiment.capitalize()} Sentiment')
                plt.tight_layout()
                
                # Save the figure
                output_path = self.output_dir / f'wordcloud_{sentiment}.png'
                plt.savefig(output_path)
                plt.close()
                
                wordcloud_paths[sentiment] = output_path
                logger.info(f"Wordcloud for {sentiment} sentiment saved to {output_path}")
            
            return wordcloud_paths
        except Exception as e:
            logger.error(f"Error generating wordclouds: {str(e)}")
            raise
    
    def plot_common_words(self, text_processor, top_n=15):
        """Plot most common words for each sentiment"""
        try:
            # Get unique sentiment values
            sentiments = self.data['Sentiment'].unique()
            
            common_words_paths = {}
            
            for sentiment in sentiments:
                # Get most common words for the sentiment
                top_words = text_processor.get_most_common_words(sentiment, top_n)
                
                if not top_words:
                    continue
                
                # Extract words and counts
                words, counts = zip(*top_words)
                
                # Create DataFrame for plotting
                word_df = pd.DataFrame({'Word': words, 'Count': counts})
                
                # Plot
                plt.figure(figsize=(12, 8))
                ax = sns.barplot(x='Count', y='Word', data=word_df, 
                                color=self.sentiment_colors.get(sentiment, 'gray'))
                
                # Add count labels
                for i, v in enumerate(counts):
                    ax.text(v + 0.1, i, str(v), va='center')
                
                plt.title(f'Most Common Words in {sentiment.capitalize()} Sentiment')
                plt.xlabel('Count')
                plt.ylabel('Word')
                plt.tight_layout()
                
                # Save the figure
                output_path = self.output_dir / f'common_words_{sentiment}.png'
                plt.savefig(output_path)
                plt.close()
                
                common_words_paths[sentiment] = output_path
                logger.info(f"Common words plot for {sentiment} sentiment saved to {output_path}")
            
            return common_words_paths
        except Exception as e:
            logger.error(f"Error plotting common words: {str(e)}")
            raise
    
    def plot_text_length_distribution(self):
        """Plot distribution of text lengths by sentiment"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Boxplot
            ax = sns.boxplot(x='Sentiment', y='sentence_length', data=self.data, 
                            palette=self.sentiment_colors)
            
            # Add swarm plot for individual data points
            sns.swarmplot(x='Sentiment', y='sentence_length', data=self.data, 
                          alpha=0.5, size=2, color='black')
            
            plt.title('Distribution of Text Lengths by Sentiment')
            plt.xlabel('Sentiment')
            plt.ylabel('Text Length (characters)')
            plt.tight_layout()
            
            # Save the figure
            output_path = self.output_dir / 'text_length_distribution.png'
            plt.savefig(output_path)
            plt.close()
            
            # Also create a violin plot
            plt.figure(figsize=(12, 8))
            sns.violinplot(x='Sentiment', y='sentence_length', data=self.data, 
                          palette=self.sentiment_colors)
            
            plt.title('Violin Plot of Text Lengths by Sentiment')
            plt.xlabel('Sentiment')
            plt.ylabel('Text Length (characters)')
            plt.tight_layout()
            
            violin_path = self.output_dir / 'text_length_violin.png'
            plt.savefig(violin_path)
            plt.close()
            
            logger.info(f"Text length distribution plots saved to {output_path} and {violin_path}")
            return output_path, violin_path
        except Exception as e:
            logger.error(f"Error plotting text length distribution: {str(e)}")
            raise
    
    def plot_word_count_distribution(self):
        """Plot distribution of word counts by sentiment"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Boxplot
            ax = sns.boxplot(x='Sentiment', y='word_count', data=self.data, 
                            palette=self.sentiment_colors)
            
            # Add swarm plot for individual data points
            sns.swarmplot(x='Sentiment', y='word_count', data=self.data, 
                          alpha=0.5, size=2, color='black')
            
            plt.title('Distribution of Word Counts by Sentiment')
            plt.xlabel('Sentiment')
            plt.ylabel('Word Count')
            plt.tight_layout()
            
            # Save the figure
            output_path = self.output_dir / 'word_count_distribution.png'
            plt.savefig(output_path)
            plt.close()
            
            # Also create a histogram
            plt.figure(figsize=(12, 8))
            for sentiment in self.data['Sentiment'].unique():
                subset = self.data[self.data['Sentiment'] == sentiment]
                sns.histplot(subset['word_count'], label=sentiment, 
                            color=self.sentiment_colors.get(sentiment, 'gray'),
                            kde=True, alpha=0.6)
            
            plt.title('Histogram of Word Counts by Sentiment')
            plt.xlabel('Word Count')
            plt.ylabel('Frequency')
            plt.legend()
            plt.tight_layout()
            
            hist_path = self.output_dir / 'word_count_histogram.png'
            plt.savefig(hist_path)
            plt.close()
            
            logger.info(f"Word count distribution plots saved to {output_path} and {hist_path}")
            return output_path, hist_path
        except Exception as e:
            logger.error(f"Error plotting word count distribution: {str(e)}")
            raise
    
    def plot_sentiment_heatmap(self, text_processor, n_words=20):
        """Create a heatmap showing word frequency by sentiment"""
        try:
            # Get unique sentiments
            sentiments = self.data['Sentiment'].unique()
            
            # Get top words for each sentiment
            all_top_words = set()
            sentiment_top_words = {}
            
            for sentiment in sentiments:
                top_words = text_processor.get_most_common_words(sentiment, n_words)
                if top_words:
                    words, _ = zip(*top_words)
                    sentiment_top_words[sentiment] = dict(top_words)
                    all_top_words.update(words)
            
            # Create DataFrame for heatmap
            heatmap_data = []
            for word in all_top_words:
                row = {'Word': word}
                for sentiment in sentiments:
                    row[sentiment] = sentiment_top_words.get(sentiment, {}).get(word, 0)
                heatmap_data.append(row)
            
            # Convert to DataFrame and pivot
            heatmap_df = pd.DataFrame(heatmap_data)
            pivot_df = heatmap_df.set_index('Word')
            
            # Sort by total frequency
            pivot_df['total'] = pivot_df.sum(axis=1)
            pivot_df = pivot_df.sort_values('total', ascending=False).drop('total', axis=1)
            
            # Select top words
            pivot_df = pivot_df.head(n_words)
            
            # Plot
            plt.figure(figsize=(14, 10))
            mask = pivot_df.isnull()
            sns.heatmap(pivot_df, annot=True, fmt="d", cmap="YlGnBu", 
                        mask=mask, linewidths=0.5)
            
            plt.title('Word Frequency Heatmap by Sentiment')
            plt.tight_layout()
            
            # Save the figure
            output_path = self.output_dir / 'sentiment_word_heatmap.png'
            plt.savefig(output_path)
            plt.close()
            
            logger.info(f"Sentiment heatmap saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error plotting sentiment heatmap: {str(e)}")
            raise