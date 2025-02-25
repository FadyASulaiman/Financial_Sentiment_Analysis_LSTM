I started with the intent of using a random forest classifier but then - after preparing the feature engineering pipleine - resorted to using an LSTM model:
    why:

how did this change my feature engineering pipeline?
LSTM neural networks are designed to work best with sequential data, giving them the ability to utilize representations that preserve the contextual information ingerent to text and natural language, which is different from the requirements of a traditional ML model that often rely on fixed-size, numeric feature vectors. 

As a result, I resorted to removing the Term Frequency-Inverse Document Frequency (TF-IDF) & Bag of Words (BoW) Transformers.

I also removed sentiment lexicons and statistical feature transformers as they are not directly compatible with LSTM inputs.

Another change is that I resorted to integrate the embedding in the model itself which is made possible with the keras library.