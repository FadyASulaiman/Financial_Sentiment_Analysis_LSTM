Proposed project structure:

**1. Data acquisition:** The papers mentioned that they had data over 10,000 examples, this amount of data is supposed to be publicly accessible.

**2. Data cleaning & Preprocessing:** The neccessary cleaning for the data would be:
    - Removing punctution marks (!, ?, ., :, ;, ,)
    - Removing HTML tags (<>) (unlikely to happen in this corpus)
    - Remove links (https://)
    - Remove special chars ((, ), +, -, =, \, |, {, }, [, ], @)
    - Remove extra whitespace
    - Remove space between digits and decimal points
    - Remove space before or after a comma in large numbers
    - Replace percentages
    - Replace months, dates, years, and hours
    - lowercasing
    - Stop word removal
    - Replace stock tickers (e.g., $AAPL) with a special token 'STOCK'
    - Handle Currency symbols (£, $) maybe with a token like <CUR>
    - lemmatization
    - tokenization

**3. Data augmentation:** possible if we ended up needing more training data for the model to perform better, let's put synonym replacement and back translation into consideration.

**4. Feature engineering:** feature engineering should not be hard for this dataset. note: we might need to use SMOTE for oversampling the minority class (negative sentiment)

**5. model structure:**
    - We are using an LSTM model
    - Embeddigns layer
    - Input layer
    - LSTM layer
    - Dropout layer (for regularization & to prevent overfitting)
    - LSTM layer (experiment with a single or double LSTM layer structure - Prone to overfitting)
    - Output layer (with softmax activation (Optimal for multi-class classification))
    - loss function: Categorical crossentropy
    - metrics: accuracy, f1, precision, recall & confusion matrix
    - Optimizer: Adam

**6. Hyperparameters:**
    - Embedding dimension (if not pre-trained)
    - LSTM hidden size
    - Number of LSTM layers (1 vs 2) 
    - Dropout rate
    - Learning rate for Adam
    - Batch size

**7. Epochs:** 3 to 10 (Could increase if model underfits)
    - With Early stopping

**8. Pipelining:**
    - For generalization, we must create a pipeline where the test data would also go through
    - including:
        - preprocessing
        - Transformers (Via pickle)

**9. Train/Test/CV split:** 70%/15%/15% (Also Experiment with 80/10/10). Make sure to use stratification.

**10. Deployment:**
    - Cloud provider: GCP
    - Containerization: yes, docker.
    - Container deployment: Google Cloud run
    - Versioning: Artifact registry
    - database: CloudSQL
    - frontend: Gradio, deployed through a web-host.
    - personal website: No.
    - CI/CD: No.

**11. Experiment Tracking and lifecycle:** MLflow
    - Make sure to include robust loging for hyperparameters, metrics and confusion matrices

**12. Showcasing:** Canva, through a video demo
**13. Writeup:** Throughout the project we have to document everything to make the writeup easier.

