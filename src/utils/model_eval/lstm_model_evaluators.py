import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report
)
import mlflow

from src.utils.loggers.model_training_and_eval_logger import logger


class SentimentModelEvaluator:
    """Class to evaluate sentiment model performance"""
    
    def __init__(self, model, output_dir='evaluation'):
        """Initialize evaluator"""
        self.model = model
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        self.class_names = ['Negative', 'Neutral', 'Positive']


    def evaluate(self, X_test, y_test, run_id=None):
        """Evaluate model and generate reports
        
        Args:
            X_test: Test data
            y_test: Test labels
            run_id: MLflow run ID
        """
        logger.info("Starting model evaluation")
        
        # Get predictions - for multi-class
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Classification report
        report = classification_report(y_test, y_pred, 
                                      target_names=self.class_names,
                                      output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        logger.info(f"Classification report:\n{report_df}")
        
        # Generate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Create visualizations
        self._plot_confusion_matrix(cm, run_id)
        
        # Log results to MLflow
        if run_id:
            with mlflow.start_run(run_id=run_id):
                # Log metrics
                mlflow.log_metrics({
                    "test_accuracy": report_df.loc['accuracy', 'f1-score'],
                    "test_macro_f1": report_df.loc['macro avg', 'f1-score'],
                    "test_weighted_f1": report_df.loc['weighted avg', 'f1-score'],
                    "test_negative_f1": report_df.loc['Negative', 'f1-score'],
                    "test_neutral_f1": report_df.loc['Neutral', 'f1-score'],
                    "test_positive_f1": report_df.loc['Positive', 'f1-score']
                })
                
                # Log figures
                mlflow.log_artifact(f"{self.output_dir}/confusion_matrix.png")
                
                # Log detailed report
                report_path = f"{self.output_dir}/classification_report.csv"
                report_df.to_csv(report_path)
                mlflow.log_artifact(report_path)
        
        logger.info("Evaluation completed")
        return report_df
    
    def _plot_confusion_matrix(self, cm, run_id=None):
        """Plot confusion matrix
        
        Args:
            cm: Confusion matrix
            run_id: MLflow run ID
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/confusion_matrix.png")
        plt.close()