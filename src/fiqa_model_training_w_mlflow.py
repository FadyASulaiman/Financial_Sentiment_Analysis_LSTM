import mlflow
import mlflow.sklearn
import mlflow.keras
import mlflow.pyfunc
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

import joblib
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from transformers import TFAutoModelForSequenceClassification

from skopt import BayesSearchCV

from keras.models import Model
from keras.layers import Dense, LSTM, Conv1D, GlobalMaxPooling1D, Input, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

class BaseModel(ABC):
    """Abstract base class for all models"""

    def __init__(self, random_state=42):
        self.model = None
        self.random_state = random_state
        self.best_params_ = None

    @abstractmethod
    def build_model(self, **kwargs):
        pass
    
    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None):
        pass
    
    def predict(self, X):
        return self.model.predict(X)
    
    def save(self, path):
        joblib.dump(self.model, path)
        
    def load(self, path):
        self.model = joblib.load(path)
    
    @abstractmethod
    def log_model(self, run_id):
        pass

class TraditionalMLModel(BaseModel):
    """Handles traditional machine learning models"""
    
    MODEL_TYPES = {
        'svm': SVC,
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier
    }
    
    def __init__(self, model_type='random_forest', param_space=None, **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.param_space = param_space or self._default_param_space()
        self.scoring = 'f1_weighted'
        
    def _default_param_space(self):
        return {
            'n_estimators': (100, 1000),
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.3)
        }
    
    def build_model(self, **kwargs):
        return self.MODEL_TYPES[self.model_type](
            random_state=self.random_state,
            **kwargs
        )
    
    def log_model(self, run_id):
        mlflow.sklearn.log_model(
            sk_model=self.model,
            artifact_path="model",
            registered_model_name=f"{self.model_type}_FiQA",
            signature=infer_signature(self.X_train, self.model.predict(self.X_train))
        )
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.X_train = X_train  # Store for signature inference
        opt = BayesSearchCV(
            estimator=self.build_model(),
            search_spaces=self.param_space,
            n_iter=30,
            cv=3,
            scoring=self.scoring,
            random_state=self.random_state
        )
        
        with mlflow.start_run(nested=True):
            mlflow.log_params(self.param_space)
            opt.fit(X_train, y_train)
            
            self.model = opt.best_estimator_
            self.best_params_ = opt.best_params_
            
            mlflow.log_params(opt.best_params_)
            mlflow.log_metric("best_cv_score", opt.best_score_)
            
        return opt.best_score_

class DeepLearningModel(BaseModel):
    """Handles deep learning architectures"""
    
    def __init__(self, model_type='lstm', input_shape=None, **kwargs):
        super().__init__(**kwargs)
        self.model_type = model_type
        self.input_shape = input_shape
        self.history = None
        
    def build_model(self, **params):
        if self.model_type == 'lstm':
            return self._build_lstm(**params)
        elif self.model_type == 'cnn':
            return self._build_cnn(**params)
        elif self.model_type == 'transformer':
            return self._build_transformer()
            
    def _build_lstm(self, units=64, dropout=0.2):
        inputs = Input(shape=self.input_shape)
        x = LSTM(units, return_sequences=True)(inputs)
        x = Dropout(dropout)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(3, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def _build_cnn(self, filters=64, kernel_size=3):
        inputs = Input(shape=self.input_shape)
        x = Conv1D(filters, kernel_size, activation='relu')(inputs)
        x = GlobalMaxPooling1D()(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(3, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        model.compile(
            optimizer=Adam(0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def _build_transformer(self):
        model = TFAutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=3
        )
        model.compile(
            optimizer=Adam(3e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def log_model(self, run_id):
        mlflow.keras.log_model(
            keras_model=self.model,
            artifact_path="model",
            registered_model_name=f"{self.model_type}_FiQA",
            signature=infer_signature(self.X_train, self.model.predict(self.X_train))
        )
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.X_train = X_train  # Store for signature inference
        self.model = self.build_model()
        
        mlflow.tensorflow.autolog(
            log_input_examples=True,
            log_model_signatures=True,
            registered_model_name=f"{self.model_type}_FiQA"
        )
        
        callbacks = [
            EarlyStopping(patience=3, restore_best_weights=True),
            ModelCheckpoint('best_model.h5', save_best_only=True),
            mlflow.keras.MLflowCallback(
                run_id=mlflow.active_run().info.run_id
            )
        ]
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        return max(self.history.history['val_accuracy'])

class ModelTrainer:
    # ... (previous ModelTrainer implementation) ...
    
    def __init__(self, features, targets, test_size=0.2):
        self.features = features
        self.targets = targets
        self.test_size = test_size
        self.models = {}
        self.results = {}
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            features, targets,
            test_size=test_size,
            stratify=targets,
            random_state=42
        )
        # MLFlow setup
        self.client = MlflowClient()
        self._setup_mlflow()
        
    def _setup_mlflow(self):
        mlflow.set_tracking_uri("http://localhost:5000")  # Update with your URI
        mlflow.set_experiment("FiQA_Sentiment_Analysis")
        
    def _log_artifacts(self):
        # Log important dataset characteristics
        class_distribution = pd.value_counts(self.targets).to_dict()
        mlflow.log_dict(class_distribution, "class_distribution.json")
        
        # Log feature distribution plots
        plt.figure(figsize=(10, 6))
        sns.histplot(self.features.mean(axis=1))
        plt.title("Feature Value Distribution")
        mlflow.log_figure(plt.gcf(), "feature_distribution.png")
        plt.close()


    def add_model(self, name, model):
        self.models[name] = model
        
    def _evaluate(self, y_true, y_pred):
        print(classification_report(y_true, y_pred))
        return {
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'accuracy': np.mean(y_true == y_pred)
        }
    
    def run_cross_validation(self, model, cv=5):
        skf = StratifiedKFold(cv)
        scores = []
        
        for train_idx, val_idx in skf.split(self.X_train, self.y_train):
            X_fold_train = self.X_train[train_idx]
            y_fold_train = self.y_train[train_idx]
            X_val = self.X_train[val_idx]
            y_val = self.y_train[val_idx]
            
            model.train(X_fold_train, y_fold_train, X_val, y_val)
            preds = model.predict(X_val)
            scores.append(self._evaluate(y_val, preds))
            
        return np.mean(scores)
    
        
    def train_all(self, use_cross_validation=True):
        with mlflow.start_run(run_name="Full_Training_Run"):
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_param("num_features", self.features.shape[1])
            self._log_artifacts()
            
            for name, model in self.models.items():
                with mlflow.start_run(run_name=name, nested=True):
                    mlflow.set_tag("model_type", name)
                    
                    print(f"\n=== Training {name} ===")
                    if use_cross_validation:
                        cv_score = self.run_cross_validation(model)
                        mlflow.log_metric("cv_score", cv_score)
                        print(f"CV Score: {cv_score}")
                        
                    # Final training on full data
                    final_score = model.train(self.X_train, self.y_train)
                    test_preds = model.predict(self.X_test)
                    test_metrics = self._evaluate(self.y_test, test_preds)
                    
                    # Log metrics
                    mlflow.log_metrics({
                        f"test_{k}": v for k, v in test_metrics.items()
                    })
                    mlflow.log_metric("final_score", final_score)
                    
                    # Log model
                    model.log_model(mlflow.active_run().info.run_id)
                    
                    self.results[name] = {
                        'cv_score': cv_score if use_cross_validation else None,
                        'test_metrics': test_metrics,
                        'model': model
                    }

    def save_best_model(self, metric='f1_weighted'):
        best_name = max(self.results, key=lambda k: self.results[k]['test_metrics'][metric])
        best_model = self.results[best_name]['model']
        
        # Register best model in MLflow Model Registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        registered_model = mlflow.register_model(
            model_uri,
            f"Best_{best_name}_FiQA"
        )
        
        # Transition model to Production stage
        self.client.transition_model_version_stage(
            name=registered_model.name,
            version=registered_model.version,
            stage="Production"
        )
        
        print(f"Registered best model: {registered_model.name}")

# Example Usage
if __name__ == "__main__":
    # Initialize MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Assuming features and targets are prepared
    # features = ...
    # targets = ...
    
    trainer = ModelTrainer(features, targets)
    
    # Add models
    trainer.add_model('svm', TraditionalMLModel(
        model_type='svm',
        param_space={
            'C': (1e-3, 1e3, 'log-uniform'),
            'gamma': (1e-4, 1e-1, 'log-uniform')
        }
    ))
    
    trainer.add_model('text_cnn', DeepLearningModel(
        model_type='cnn',
        input_shape=(300,)
    ))
    
    # Execute training
    trainer.train_all()
    trainer.save_best_model()