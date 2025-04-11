from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import joblib
import warnings
warnings.filterwarnings('ignore')


class BaselineSVMModel:

    # 1. Data Loading Module
    def load_data(self,file_path):
        """Load the CSV file into a pandas DataFrame."""
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded successfully with shape: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
        
    # Modified Model Building Module with SVM
    def build_model(self, df):
        """Build and train a sentiment analysis model using SVM."""
        # Split the data
        X = df['Sentence']  # Just use the text column
        y = df['Sentiment']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Create the full pipeline with classifier
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', LinearSVC(random_state=42))
        ])

        # Train the model
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred = pipeline.predict(X_test)

        # Evaluate the model
        print("\nModel Evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.tight_layout()


        return pipeline, X_test, y_test

    # 5. Model Tuning Module
    def tune_model(self, pipeline, X_train, y_train, X_test, y_test):
        """Tune model hyperparameters using GridSearchCV."""
        param_grid = {
            'tfidf__max_features': [1000, 5000],
            'tfidf__ngram_range': [(1, 1), (1, 2)],
            'classifier__C': [0.1, 1, 10]
        }
        
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1, scoring='f1_weighted')
        grid_search.fit(X_train, y_train)
        
        
        print("\nBest parameters:", grid_search.best_params_)
        
        # Evaluate best model
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        print("\nTuned Model Evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return best_model


    # 6. Model Saving Module
    def save_model(self, model, filename='finance_sentiment_baseline_SVM_model.pkl'):
        """Save the trained model to disk."""
        try:
            joblib.dump(model, "data/" + filename)
            print(f"\nModel saved successfully as {filename}")
        except Exception as e:
            print(f"Error saving model: {e}")


    # 7. Run Pipeline
    def run_sentiment_analysis_pipeline(self, file_path):
        """Run the complete sentiment analysis pipeline."""
        # Load data
        df = self.load_data(file_path)
        if df is None:
            return
        
        
        # Handle Company column - Convert to binary feature (has_company)
        df['has_company'] = df['Company'].apply(lambda x: 0 if x == 'None' else 1)
        
        # Build initial model
        X = df['Sentence']
        y = df['Sentiment']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        pipeline, _, _ = self.build_model(df)
        
        # Tune model
        best_model = self.tune_model(pipeline, X_train, y_train, X_test, y_test)
        
        # Save model
        self.save_model(best_model)
        
        return best_model

    # 8. Example of prediction function
    def predict_sentiment(self, model, new_data):
        """Predict sentiment for new finance headlines."""
        # Ensure new_data has the required columns
        required_cols = ['Sentence', 'Event', 'Sector', 'Company']
        for col in required_cols:
            if col not in new_data.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Add has_company feature
        new_data['has_company'] = new_data['Company'].apply(lambda x: 0 if x == 'None' else 1)
        
        # Make predictions
        predictions = model.predict(new_data)
        probabilities = model.predict_proba(new_data)
        
        # Add predictions to the dataframe
        result = new_data.copy()
        result['predicted_sentiment'] = predictions
        
        for i, class_name in enumerate(model.classes_):
            result[f'prob_{class_name}'] = probabilities[:, i]
        
        return result


if __name__ == "__main__":
    b = BaselineSVMModel()

    file_path = "data/feature_engineered_data.csv"
    model = b.run_sentiment_analysis_pipeline(file_path)