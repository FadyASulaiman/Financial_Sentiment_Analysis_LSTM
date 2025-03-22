import pandas as pd
from pipelines.lstm_pipeline import LSTMPipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import yaml
import pickle

# WIP - Entry point to the whole pipeline

# Load configuration
with open('config/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Load dataset
df = pd.read_csv('data/fiqa_dataset.csv')
X = df['text'].tolist()
y = df['sentiment'].tolist()

# Initialize and fit the pipeline now that everything is in place for the model training
pipeline = LSTMPipeline(config)
pipeline.fit(X, y)
X_sequences = pipeline.transform(X)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_sequences, y_categorical, test_size=0.2, random_state=42)

# Train test split is amazing and this jeboard is more amazing now that I have the white one in th 
# Build the LSTM model
model = Sequential([
    Embedding(input_dim=config['tokenizer']['num_words'],
              output_dim=config['model']['embedding_dim'],
              input_length=config['tokenizer']['max_length']),
    LSTM(config['model']['lstm_units'], return_sequences=False),
    Dropout(config['model']['dropout_rate']),
    Dense(config['model']['dense_units'], activation='relu'),
    Dropout(config['model']['dropout_rate']),
    Dense(config['model']['output_units'], activation=config['model']['activation'])
])

# Compile the model, ctaegoricak crossentropy and accuracy as a metric
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Display the model architecture
model.summary()

# Define early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Save the model and pipeline
model.save('models/lstm_sentiment_model.h5')
with open('pipelines/lstm_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)


# structuring this project all over again
"""
What are the pieces that need to be built for this project to succeed:

1. Data acquisition: The papers mentioned that they had data over 10,000 examples, this amount is supposed to be publicly accessible.
2. Data cleaning & Preprocessing: The neccessary cleaning for the data would be:
    - Removing punctution marks (!, ?, ., :, ;, ,)
    - Removing HTML tags (<>) (unlikely to happen in this corpus)
    - Remove links (https://)
    - Remove special chars ((, ), +, -, =, \, |, {, }, [, ])
    - lowercasing
    - Stop word removal
    - Handle Currency symbols (£$) maybe with a token like <CUR>
    - lemmatization & stemming
    - tokenization
3. Data augmentation: possible if we need more training data for the model to perform better
4. Feature engineering: feature engineering should not be hard for this dataset

5. model structure:
    - We are using an LSTM model
    - Embeddigns layer
    - Input layer
    - LSTM layer
    - Dropout layer (for regularization & to prevent overfitting)
    - LSTM layer
    - Output layer (with ReLU activation)
    - loss function: Categorical crossentropy
    - metric: accuracy
    - Optimizer: Adam

6. Hyperparameters used

7. Epochs: 3 to 10 

8. Pipelining:
    - For generalization, we must create a pipeline where the test data would also go through
    - including:
        - preprocessing
        - Transformers (Via pickle)

9. Train/Test/CV split: 70%/15%/15%
10. Deployment:
    - Cloud provider: GCP
    - Containerization: yes, docker.
    - Container deployment: Google Cloud run
    - Versioning: Artifact registry
    - database: CloudSQL
    - frontend: Gradio, through a very cheap web-host.
    - personal website: No.
    - CI/CD: No.

11. Experiment Tracking and lifecycle: MLflow
12. Showcasing: Canva, through a video demo
13. Writeup: Throughout the project we have to document everything to make the writeup easier.
"""