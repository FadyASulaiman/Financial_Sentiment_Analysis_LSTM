import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from src.utils.logger import logger

class InsightExtractor:
    """Class for extracting meaningful insights from the data"""
    
    def __init__(self, data, text_processor, output_dir):
        """
        Initialize with processed data and output directory
        
        Args:
            data (pd.DataFrame): Processed data
            text_processor (TextProcessor): Text processor object
            output_dir (Path): Directory to save outputs
        """
        self.data = data
        self.text_processor = text_processor
        self.output_dir = output_dir
    
    def extract_distinctive_features(self, n_features=20):
        """Extract distinctive words/features for each sentiment class"""
        try:
            # Get unique sentiments
            sentiments = self.data['Sentiment'].unique()
            
            insights = []
            
            for sentiment in sentiments:
                # Get data for this sentiment
                sentiment_data = self.data[self.data['Sentiment'] == sentiment]
                
                # Get data for other sentiments
                other_data = self.data[self.data['Sentiment'] != sentiment]
                
                # Get common words for this sentiment
                sentiment_words = self.text_processor.get_most_common_words(sentiment, n_features*2)
                sentiment_word_dict = dict(sentiment_words)
                
                # Get common words for other sentiments
                other_words = []
                for other_sent in self.data['Sentiment'].unique():
                    if other_sent != sentiment:
                        other_words.extend(self.text_processor.get_most_common_words(other_sent, n_features))
                other_word_dict = dict(other_words)
                
                # Find distinctive words
                distinctive = []
                for word, count in sentiment_words:
                    # Calculate ratio of occurrence in this sentiment vs others
                    other_count = other_word_dict.get(word, 0)
                    
                    # Avoid division by zero
                    if other_count == 0:
                        ratio = count  # Infinite ratio, but just use count for simplicity
                    else:
                        # Normalize by number of samples
                        ratio = (count / len(sentiment_data)) / (other_count / len(other_data))
                    
                    distinctive.append((word, count, ratio))
                
                # Sort by ratio
                distinctive.sort(key=lambda x: x[2], reverse=True)
                
                # Take top N
                top_distinctive = distinctive[:n_features]
                
                insights.append({
                    'sentiment': sentiment,
                    'distinctive_words': top_distinctive
                })
            
            # Create DataFrame with insights
            insights_df = pd.DataFrame(columns=['Sentiment', 'Word', 'Count', 'Distinctiveness'])
            
            for insight in insights:
                sentiment = insight['sentiment']
                for word, count, ratio in insight['distinctive_words']:
                    insights_df = pd.concat([
                        insights_df,
                        pd.DataFrame({
                            'Sentiment': [sentiment],
                            'Word': [word],
                            'Count': [count],
                            'Distinctiveness': [ratio]
                        })
                    ], ignore_index=True)
            
            # Save to file
            output_path = self.output_dir / 'distinctive_features.csv'
            insights_df.to_csv(output_path, index=False)
            
            logger.info(f"Distinctive features saved to {output_path}")
            
            # Create visualizations
            plt.figure(figsize=(15, 10))
            
            for i, sentiment in enumerate(sentiments):
                sentiment_insights = insights_df[insights_df['Sentiment'] == sentiment]
                
                # Plot
                plt.subplot(len(sentiments), 1, i+1)
                bars = plt.barh(sentiment_insights['Word'], sentiment_insights['Distinctiveness'])
                
                # Color bars by count
                counts = sentiment_insights['Count']
                norm = plt.Normalize(counts.min(), counts.max())
                sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
                sm.set_array([])
                
                for j, bar in enumerate(bars):
                    bar.set_color(plt.cm.viridis(norm(counts.iloc[j])))
                
                plt.title(f'Most Distinctive Words for {sentiment.capitalize()} Sentiment')
                plt.xlabel('Distinctiveness Ratio')
                plt.tight_layout()
            
            plt.colorbar(sm, ax=plt.gca(), label='Word Count')
            plt.tight_layout()
            
            # Save the figure
            plot_path = self.output_dir / 'distinctive_features.png'
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Distinctive features plot saved to {plot_path}")
            
            return insights_df
        except Exception as e:
            logger.error(f"Error extracting distinctive features: {str(e)}")
            raise
    
    def extract_summary_insights(self):
        """Extract and summarize key insights from the data"""
        try:
            insights = []
            
            # 1. Overall sentiment distribution
            sentiment_counts = self.data['Sentiment'].value_counts()
            dominant_sentiment = sentiment_counts.idxmax()
            dominant_percentage = (sentiment_counts.max() / len(self.data)) * 100
            
            insights.append(f"Dominant sentiment: {dominant_sentiment.capitalize()} "
                           f"({dominant_percentage:.1f}% of samples)")
            
            # 2. Text length patterns
            by_sentiment = self.data.groupby('Sentiment')
            avg_lengths = by_sentiment['sentence_length'].mean()
            longest_sentiment = avg_lengths.idxmax()
            shortest_sentiment = avg_lengths.idxmin()
            
            insights.append(f"Longest text on average: {longest_sentiment.capitalize()} "
                           f"({avg_lengths[longest_sentiment]:.1f} characters)")
            insights.append(f"Shortest text on average: {shortest_sentiment.capitalize()} "
                           f"({avg_lengths[shortest_sentiment]:.1f} characters)")
            
            # 3. Word count patterns
            avg_words = by_sentiment['word_count'].mean()
            most_words = avg_words.idxmax()
            fewest_words = avg_words.idxmin()
            
            insights.append(f"Most words on average: {most_words.capitalize()} "
                           f"({avg_words[most_words]:.1f} words)")
            insights.append(f"Fewest words on average: {fewest_words.capitalize()} "
                           f"({avg_words[fewest_words]:.1f} words)")
            
            # 4. Most common words overall
            all_words = ' '.join(self.data['cleaned_text']).split()
            word_counts = Counter(all_words)
            top_words = word_counts.most_common(5)
            
            insights.append(f"Most common words overall: {', '.join([w for w, _ in top_words])}")
            
            # 5. Most distinctive words by sentiment
            sentiments = self.data['Sentiment'].unique()
            for sentiment in sentiments:
                # Get distinctive words for this sentiment
                sentiment_words = self.text_processor.get_most_common_words(sentiment, 20)
                
                # Get words for other sentiments
                other_words = []
                for other_sent in sentiments:
                    if other_sent != sentiment:
                        other_words.extend(self.text_processor.get_most_common_words(other_sent, 20))
                
                other_word_dict = dict(other_words)
                sentiment_word_dict = dict(sentiment_words)
                
                # Find words unique to this sentiment
                unique_words = []
                for word, count in sentiment_words:
                    if word not in other_word_dict:
                        unique_words.append(word)
                
                if unique_words:
                    insights.append(f"Words unique to {sentiment.capitalize()} sentiment: "
                                   f"{', '.join(unique_words[:5])}")
            
            # Write insights to file
            output_path = self.output_dir / 'key_insights.txt'
            with open(output_path, 'w') as f:
                for i, insight in enumerate(insights, 1):
                    f.write(f"{i}. {insight}\n")
            
            logger.info(f"Key insights saved to {output_path}")
            return insights
        except Exception as e:
            logger.error(f"Error extracting summary insights: {str(e)}")
            raise