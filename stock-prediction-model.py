import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import joblib
from typing import Tuple, Optional

class StockPredictor:
    def __init__(self):
        self.sequence_length = 60
        self.model = self._build_model()
        self.scaler = MinMaxScaler()
        
    def _build_model(self) -> Sequential:
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(self.sequence_length, 11)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=25, activation='relu'),
            Dense(units=1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error'
        )
        return model
    
    def prepare_data(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare data for LSTM model"""
        features = [
            'Close', 'Volume', 'RSI', 'MACD', 'MACD_Signal',
            'BB_upper', 'BB_lower', 'EMA_9', 'SMA_30', 'ADX', 'OBV'
        ]
        
        # Scale the features
        scaled_data = self.scaler.fit_transform(data[features])
        
        # Create sequences
        X = []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
        
        return np.array(X)
    
    def predict(self, data: pd.DataFrame, horizon: str, 
                sentiment_score: Optional[float] = None) -> Tuple[np.ndarray, float]:
        """Generate predictions and confidence scores"""
        # Prepare data
        X = self.prepare_data(data)
        
        # Generate base predictions
        predictions = []
        last_sequence = X[-1]
        
        # Convert horizon to number of days
        n_days = int(horizon[:-1])
        
        # Generate predictions for each day in the horizon
        for _ in range(n_days):
            # Make prediction
            pred = self.model.predict(last_sequence.reshape(1, self.sequence_length, 11), verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence for next prediction
            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1] = pred
        
        # Adjust predictions based on sentiment if available
        if sentiment_score is not None:
            sentiment_adjustment = 0.01 * sentiment_score  # 1% adjustment per sentiment point
            predictions = [p * (1 + sentiment_adjustment) for p in predictions]
        
        # Calculate confidence score based on prediction variance
        confidence = 1.0 / (1.0 + np.std(predictions))
        
        # Inverse transform predictions to original scale
        predictions = self.scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        )
        
        return predictions, confidence
    
    def train(self, train_data: pd.DataFrame, validation_data: pd.DataFrame,
              epochs: int = 50, batch_size: int = 32) -> dict:
        """Train the model on historical data"""
        # Prepare training data
        X_train = self.prepare_data(train_data)
        y_train = train_data['Close'].values[self.sequence_length:]
        
        # Prepare validation data
        X_val = self.prepare_data(validation_data)
        y_val = validation_data['Close'].values[self.sequence_length:]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        return history.history
    
    def save_model(self, path: str):
        """Save model and scaler"""
        self.model.save(f"{path}/lstm_model")
        joblib.dump(self.scaler, f"{path}/scaler.pkl")
    
    def load_model(self, path: str):
        """Load saved model and scaler"""
        self.model = tf.keras.models.load_model(f"{path}/lstm_model")
        self.scaler = joblib.load(f"{path}/scaler.pkl")