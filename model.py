import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def train_and_predict():
    # Download Tesla stock data
    df = yf.download('TSLA', start='2015-01-01', end='2025-01-01')
    data = df[['Close']].values

    # Scale data
    scaler = MinMaxScaler(feature_range=(0,1))
    data_scaled = scaler.fit_transform(data)

    # Prepare sequences
    seq_len = 60
    X, y = [], []
    for i in range(seq_len, len(data_scaled)):
        X.append(data_scaled[i-seq_len:i])
        y.append(data_scaled[i])
    X, y = np.array(X), np.array(y)

    # Train-test split
    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build LSTM model
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(seq_len,1)),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

    # Predict
    preds = model.predict(X_test)
    preds = scaler.inverse_transform(preds)
    real = scaler.inverse_transform(y_test)

    return real.flatten().tolist(), preds.flatten().tolist()
