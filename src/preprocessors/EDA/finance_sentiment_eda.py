from pathlib import Path
import pandas as pd
import mlflow

from src.preprocessors.data_loader.data_loader import DataLoader
from src.preprocessors.data_preprocessors.text_preprocessor import TextProcessor
from src.preprocessors.EDA.sentiment_analyzer import SentimentAnalyzer
from src.preprocessors.EDA.data_visualizer import DataVisualizer
from src.preprocessors.EDA.stats_generator import StatsGenerator
from src.preprocessors.EDA.insight_extractor import InsightExtractor
from src.utils.eda_logger import logger

class FinanceSentimentEDA:
    """Main class to orchestrate the EDA process"""
    
    def __init__(self, file_path=None, data=None, output_dir="Analysis output"):
        """
        Initialize the EDA process
        
        Args:
            file_path (str, optional): Path to CSV file
            data (pd.DataFrame, optional): Pandas DataFrame
            output_dir (str): Directory to save outputs
        """
        self.file_path = file_path
        self.data = data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize MLflow
        mlflow.set_tracking_uri('file:./mlruns')
        mlflow.set_experiment('finance_sentiment_eda')
        
        logger.info(f"EDA initialized, outputs will be saved to {self.output_dir}")
    
    def run_eda(self):
        """Run the complete EDA pipeline"""
        try:
            with mlflow.start_run() as run:
                run_id = run.info.run_id
                logger.info(f"Started MLflow run with ID: {run_id}")
                
                # Log parameters
                mlflow.log_param("file_path", self.file_path)
                mlflow.log_param("output_dir", str(self.output_dir))
                
                # Step 1: Load and preprocess data
                logger.info("Step 1: Loading and preprocessing data")
                loader = DataLoader(file_path=self.file_path, data=self.data)
                self.data = loader.load_data()
                self.data = loader.preprocess_data()
                
                # Log dataset info
                mlflow.log_param("dataset_size", len(self.data))
                mlflow.log_param("sentiment_classes", list(self.data['Sentiment'].unique()))
                
                # Step 2: Process text
                logger.info("Step 2: Processing text")
                text_processor = TextProcessor(self.data)
                self.data = text_processor.process_text()
                
                # Step 3: Analyze sentiment distributions
                logger.info("Step 3: Analyzing sentiment distributions")
                sentiment_analyzer = SentimentAnalyzer(self.data)
                sentiment_dist = sentiment_analyzer.sentiment_distribution()
                text_length_stats = sentiment_analyzer.text_length_by_sentiment()
                word_count_stats = sentiment_analyzer.word_count_by_sentiment()
                
                # Log sentiment distribution
                for sentiment, count in sentiment_dist['Count'].items():
                    mlflow.log_metric(f"count_{sentiment}", count)
                    mlflow.log_metric(f"percentage_{sentiment}", sentiment_dist['Percentage'][sentiment])
                
                # Step 4: Generate visualizations
                logger.info("Step 4: Generating visualizations")
                visualizer = DataVisualizer(self.data, self.output_dir)
                
                # Sentiment distribution plot
                dist_plot_path = visualizer.plot_sentiment_distribution()
                mlflow.log_artifact(dist_plot_path)
                
                # Wordclouds
                wordcloud_paths = visualizer.plot_wordclouds(text_processor)
                for sentiment, path in wordcloud_paths.items():
                    mlflow.log_artifact(path)
                
                # Common words
                common_words_paths = visualizer.plot_common_words(text_processor)
                for sentiment, path in common_words_paths.items():
                    mlflow.log_artifact(path)
                
                # Text length distribution
                length_plot_paths = visualizer.plot_text_length_distribution()
                for path in length_plot_paths:
                    mlflow.log_artifact(path)
                
                # Word count distribution
                word_count_paths = visualizer.plot_word_count_distribution()
                for path in word_count_paths:
                    mlflow.log_artifact(path)
                
                # Sentiment heatmap
                heatmap_path = visualizer.plot_sentiment_heatmap(text_processor)
                mlflow.log_artifact(heatmap_path)
                
                # Step 5: Generate statistics
                logger.info("Step 5: Generating statistics")
                stats_generator = StatsGenerator(self.data, self.output_dir)
                
                # Basic stats
                basic_stats = stats_generator.generate_basic_stats()
                mlflow.log_artifact(self.output_dir / 'basic_stats.csv')
                
                # Detailed stats
                detailed_stats = stats_generator.generate_detailed_stats()
                mlflow.log_artifact(self.output_dir / 'text_length_stats.csv')
                mlflow.log_artifact(self.output_dir / 'word_count_stats.csv')
                
                # Correlation matrix
                corr_matrix = stats_generator.calculate_correlation_matrix()
                mlflow.log_artifact(self.output_dir / 'correlation_matrix.csv')
                mlflow.log_artifact(self.output_dir / 'correlation_heatmap.png')
                
                # Step 6: Extract insights
                logger.info("Step 6: Extracting insights")
                insight_extractor = InsightExtractor(self.data, text_processor, self.output_dir)
                
                # Distinctive features
                distinctive_features = insight_extractor.extract_distinctive_features()
                mlflow.log_artifact(self.output_dir / 'distinctive_features.csv')
                mlflow.log_artifact(self.output_dir / 'distinctive_features.png')
                
                # Summary insights
                key_insights = insight_extractor.extract_summary_insights()
                mlflow.log_artifact(self.output_dir / 'key_insights.txt')
                
                # Step 7: Create summary report
                logger.info("Step 7: Creating summary report")
                self._create_summary_report()
                mlflow.log_artifact(self.output_dir / 'summary_report.html')
                
                logger.info(f"EDA completed successfully. Results saved to {self.output_dir}")
                
                return self.data
        except Exception as e:
            logger.error(f"Error in EDA pipeline: {str(e)}")
            raise
    
    def _create_summary_report(self):
        """Create an HTML summary report of all EDA findings"""
        try:
            report_path = self.output_dir / 'summary_report.html'
            
            # Load insights
            insights_path = self.output_dir / 'key_insights.txt'
            with open(insights_path, 'r') as f:
                insights = f.read()
            
            # Create HTML report
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Finance Sentiment EDA - Summary Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .section {{ margin-bottom: 30px; }}
                    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                    pre {{ background-color: #f5f5f5; padding: 10px; overflow: auto; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .insights {{ background-color: #f9f9f9; padding: 15px; border-left: 4px solid #2c3e50; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Finance Sentiment EDA - Summary Report</h1>
                    <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <div class="section">
                        <h2>Key Insights</h2>
                        <div class="insights">
                            <pre>{insights}</pre>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Sentiment Distribution</h2>
                        <img src="sentiment_distribution.png" alt="Sentiment Distribution">
                    </div>
                    
                    <div class="section">
                        <h2>Word Clouds</h2>
                        <div style="display: flex; flex-wrap: wrap; gap: 20px;">
            """
            
            # Add wordclouds
            for sentiment in self.data['Sentiment'].unique():
                cloud_path = f"wordcloud_{sentiment}.png"
                if (self.output_dir / cloud_path).exists():
                    html += f"""
                            <div>
                                <h3>{sentiment.capitalize()} Sentiment</h3>
                                <img src="{cloud_path}" alt="Word Cloud for {sentiment} Sentiment">
                            </div>
                    """
            
            html += """
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Most Common Words by Sentiment</h2>
                        <div style="display: flex; flex-wrap: wrap; gap: 20px;">
            """
            
            # Add common words plots
            for sentiment in self.data['Sentiment'].unique():
                words_path = f"common_words_{sentiment}.png"
                if (self.output_dir / words_path).exists():
                    html += f"""
                            <div>
                                <h3>{sentiment.capitalize()} Sentiment</h3>
                                <img src="{words_path}" alt="Common Words for {sentiment} Sentiment">
                            </div>
                    """
            
            html += """
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Text Length Distribution</h2>
                        <div>
                            <img src="text_length_distribution.png" alt="Text Length Distribution">
                        </div>
                        <div>
                            <img src="text_length_violin.png" alt="Text Length Violin Plot">
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Word Count Distribution</h2>
                        <div>
                            <img src="word_count_distribution.png" alt="Word Count Distribution">
                        </div>
                        <div>
                            <img src="word_count_histogram.png" alt="Word Count Histogram">
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Distinctive Features</h2>
                        <div>
                            <img src="distinctive_features.png" alt="Distinctive Features">
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>Correlation Analysis</h2>
                        <div>
                            <img src="correlation_heatmap.png" alt="Correlation Heatmap">
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Write to file
            with open(report_path, 'w') as f:
                f.write(html)
            
            logger.info(f"Summary report saved to {report_path}")
            return report_path
        except Exception as e:
            logger.error(f"Error creating summary report: {str(e)}")
            raise