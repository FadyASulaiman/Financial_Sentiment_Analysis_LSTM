
# Example usage of the production inference pipeline with a single string
from .sentiment_analyzer import analyze_sentiment

input_text = "Company XYZ reported better than expected earnings, with quarterly revenue up 15% year-over-year."

input_text = "Net sales increased by 25% from last year"

input_text = "$AAPL stock fell after the release of the newest iphone"

input_text = "The development of the new factory for Tesla is going steady"

# Get prediction for the single string
result = analyze_sentiment(input_text)
print(f"Predicted sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.4f}")

input_text = "$AAPL stock fell after the release of the newest iphone"

# Get prediction for the single string
result = analyze_sentiment(input_text)
print(f"Predicted sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']:.4f}")