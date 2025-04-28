# FinSentiment: Advanced Bi-LSTM Financial News Sentiment Analysis

[Financial Sentiment Analysis](https://img.shields.io/badge/NLP-Sentiment%20Analysis-blue)
[TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-orange)
[Accuracy](https://img.shields.io/badge/Accuracy-80.2%25-brightgreen)

An advanced deep learning model that accurately classifies financial news headlines into positive, neutral, or negative sentiment using a Bidirectional LSTM architecture enhanced with Hugging Face embeddings and synthetic data generation.

## 📊 Performance Highlights

The model achieves **80.1% accuracy** on financial sentiment classification with strong performance across all sentiment classes:

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Negative | 0.73 | 0.91 | 0.81 |
| Neutral | 0.86 | 0.65 | 0.74 |
| Positive | 0.84 | 0.83 | 0.83 |
| **Macro Avg** | **0.81** | **0.80** | **0.79** |

## 🔍 Project Overview

This project addresses the challenge of sentiment analysis in financial news, which is fundamentally different from general sentiment analysis due to domain-specific language and implications. By accurately classifying financial headlines, this model can provide valuable insights for financial decision-making and market trend analysis.

### Key Features:

- **Bidirectional LSTM Architecture**: Captures contextual information from both directions in text
- **Context-Aware Embeddings**: Leverages Hugging Face pre-trained embeddings
- **Synthetic Data Generation**: Uses transformer models to create balanced, high-quality training data
- **Production-Ready Deployment**: Fully containerized solution deployed on Google Cloud Platform
- **Interactive UI**: Elegant frontend interface for real-time sentiment predictions

## 🛠️ Technical Implementation

- **Deep Learning Architecture**: Bidirectional LSTM with dropout layers to prevent overfitting
- **Data Augmentation**: SMOTE and transformer-based synthetic data generation
- **NLP Pipeline**: Comprehensive text preprocessing including HTML cleaning, lemmatization, and contextual filtering
- **Tech Stack**: TensorFlow, scikit-learn, SpaCy, NLTK, FastAPI, React

## 📋 Dataset

This project uses the [Financial Sentiment Analysis dataset from Kaggle](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis) enhanced with synthetic samples generated using transformer models to improve class balance and model generalization.

## 🚀 Getting Started

### Prerequisites

- Python 3.11
- Docker (optional, for containerized deployment)
- Anaconda (For python environment)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/finsentiment.git
cd finsentiment

# Create virtual environment
conda create -n env_name python=3.11
conda activate env_name

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Download SpaCy model
python -m spacy download en_core_web_sm
```

### Dependencies

For production:
```
tensorflow>=2.16.0
scikit-learn>=1.6.1
numpy>=1.26.4
pandas>=2.2.3
nltk>=3.9.1
beautifulsoup4>=4.12.3
spacy>=3.7.2
scikit-optimize>=0.10.2
fastapi>=0.115.12
uvicorn>=0.34.2
tqdm>=4.66.5
pydantic>=2.10.3
python-dateutil>=2.9.0
```

Other dependencies used during development:
```
mlflow
ipykernel
seaborn
matplotlib
gensim
transformers
sentence-transformers
```

## 💻 Usage


### Local Inference

```python
from finsentiment.model import SentimentAnalyzer

# Initialize the analyzer
analyzer = SentimentAnalyzer()

# Analyze a single headline
result = analyzer.predict("Tech stocks surge after positive earnings report")
print(f"Sentiment: {result['sentiment']}, Confidence: {result['confidence']:.2f}")
```

### Web Interface

A user-friendly interface is available at [https://finsentiment.yourdomain.com](https://finsentiment.yourdomain.com) where you can:
- Enter financial news headlines
- Get instant sentiment predictions
- View confidence scores and visualizations
- Explore model performance highlights


## 📈 Development Process

This project was developed over a two-month period. Development phases include:

1. **Ideation & Data Collection**: Research on financial sentiment analysis challenges & previous work
2. **Exploratory Data Analysis**: Understanding data distribution and characteristics
3. **Data Cleaning**: Removing HTML, special chars, URLs, irrelevant numbers
4. **Synthetic Data Generation**: Creating balanced training data using transformers
5. **Feature Engineering**: Extracting meaningful features from financial text
6. **Data Preparation**: Lemmatization, tokenization, sequence padding
7. **Model Training**: Developing and training the Bi-LSTM architecture
8. **Evaluation & Tuning**: Hyperparameter optimization and model refinement
9. **Deployment**: Containerization, GCP setup, CI/CD pipeline, React frontend
10. **Testing & Debugging**: End-to-end testing of the complete solution

## 🌟 Future Improvements

- Implement explainable AI features to highlight influential phrases
- Add real-time market data correlation to validate sentiment predictions
- Expand language support for international financial news

## 📞 Contact

For questions, feedback, or collaboration opportunities:

- LinkedIn: [LinkedIn Profile](https://linkedin.com/in/yourusername)
- Email: Sulaiman.a.fady@gmail.com


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. I'd be more than happy if you use, modify, and distribute this code as long as appropriate credit is given for the original work.

---

**Note**: This project was developed as a portfolio piece demonstrating advanced NLP, deep learning, and full-stack deployment capabilities relevant to ML Engineering and Data Science roles.