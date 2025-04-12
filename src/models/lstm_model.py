import tensorflow as tf
from keras.models import Sequential # type: ignore
from keras.layers import ( # type: ignore
    Embedding, LSTM, Bidirectional, Dense, Dropout, 
    GlobalMaxPooling1D, BatchNormalization
)
from src.models.model_base.model_base import ModelBase

from src.utils.loggers.model_training_and_eval_logger import logger

class LSTMSentimentModel(ModelBase):
    """LSTM model for sentiment analysis"""
    
    def __init__(self, 
                 vocab_size,
                 max_sequence_length,
                 embedding_dim=100,
                 lstm_units=128,
                 bidirectional=True,
                 dropout_rate=0.3,
                 recurrent_dropout=0.3,
                 num_classes=3,
                 embedding_matrix=None,
                 model_name='lstm_sentiment',
                 log_dir='logs'):
        """Initialize LSTM sentiment model"""
        super().__init__(model_name, log_dir)
        
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.bidirectional = bidirectional
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.num_classes = num_classes
        self.embedding_matrix = embedding_matrix
        
        logger.info(f"Initializing {model_name} with vocab_size={vocab_size}, "
                   f"sequence_length={max_sequence_length}, embedding_dim={embedding_dim}")
        
        # Build the model
        self.build()
    

    def build(self):
        """Build the LSTM model architecture"""
        logger.info("Building LSTM model architecture")
        
        model = Sequential()
        
        # Add embedding layer
        if self.embedding_matrix is not None:
            logger.info("Using pre-trained embeddings")
            model.add(Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                weights=[self.embedding_matrix],
                input_length=self.max_sequence_length,
                trainable=False,  # Freeze pre-trained embeddings
                mask_zero=False   # Disable masking to avoid issues with pooling
            ))
        else:
            logger.info("Using trainable embeddings")
            model.add(Embedding(
                input_dim=self.vocab_size,
                output_dim=self.embedding_dim,
                input_length=self.max_sequence_length,
                mask_zero=False  # Disable masking to avoid issues with pooling
            ))
        
        # Add LSTM layer(s)
        if self.bidirectional:
            logger.info(f"Adding Bidirectional LSTM with {self.lstm_units} units")
            model.add(Bidirectional(LSTM(
                self.lstm_units,
                dropout=self.dropout_rate,
                recurrent_dropout=self.recurrent_dropout,
                return_sequences=True
            )))
        else:
            logger.info(f"Adding LSTM with {self.lstm_units} units")
            model.add(LSTM(
                self.lstm_units,
                dropout=self.dropout_rate,
                recurrent_dropout=self.recurrent_dropout,
                return_sequences=True
            ))
        
        # Add global pooling to reduce sequence dimension
        model.add(GlobalMaxPooling1D())
        
        # Add regularization
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Add dense layer
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        
        # Output layer - for 3-class classification
        logger.info(f"Building multi-class classification output with {self.num_classes} classes")
        model.add(Dense(self.num_classes, activation='softmax'))
        
        self.model = model
        logger.info("LSTM model architecture built successfully")
        
        return self
    
    def compile(self, learning_rate=0.001, metrics=None):
        """Compile the model with appropriate loss function"""
        # Use only metrics compatible with sparse (integer) labels
        metrics = ['accuracy']
        
        logger.info(f"Using sparse compatible metrics: {metrics}")
        
        # Use sparse categorical crossentropy for integer labels
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=metrics
        )
        
        logger.info("Model compiled successfully")
        return self