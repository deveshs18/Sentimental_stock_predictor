import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_and_preprocess_stock_data(file_path, sequence_length):
    """
    Loads stock data from a CSV file, preprocesses it for LSTM, and splits it into training and testing sets.

    Args:
        file_path (str): Path to the CSV file.
        sequence_length (int): Length of the input sequences.

    Returns:
        tuple: X_train, y_train, X_test, y_test, scaler
    """
    # Load data, using the first row as header and skipping the second row.
    df = pd.read_csv(file_path, header=0, skiprows=[1])

    # Select 'Close' price and scale
    close_prices = df['Close'].values.reshape(-1, 1)

    # Check for NaNs in close_prices before scaling
    if np.isnan(close_prices).any():
        print("NaNs found in 'Close' prices before scaling. Check CSV data.")
        # Handle NaNs, e.g., by forward fill, backward fill, or dropping rows
        # For now, let's try forward fill as an example
        df['Close'] = df['Close'].ffill()
        close_prices = df['Close'].values.reshape(-1, 1)
        if np.isnan(close_prices).any():
            print("NaNs still present after ffill. Dropping rows with NaNs.")
            df.dropna(subset=['Close'], inplace=True)
            close_prices = df['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(close_prices)

    if np.isnan(scaled_prices).any() or np.isinf(scaled_prices).any():
        print("NaNs or Infs found in scaled_prices. Problem with scaling or source data.")
        # This would be a critical issue to resolve.
        # For debugging, one might print parts of close_prices and scaled_prices.

    # Create sequences
    X, y = [], []
    for i in range(len(scaled_prices) - sequence_length):
        X.append(scaled_prices[i:i + sequence_length, 0])
        y.append(scaled_prices[i + sequence_length, 0])

    X, y = np.array(X), np.array(y)

    if np.isnan(X).any() or np.isinf(X).any():
        print("NaNs or Infs found in X sequences.")
    if np.isnan(y).any() or np.isinf(y).any():
        print("NaNs or Infs found in y labels.")

    # Split data (80% train, 20% test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    return X_train, y_train, X_test, y_test, scaler

if __name__ == "__main__":
    # Define parameters
    sample_file_path = 'data/historical_prices/AAPL.csv'
    seq_length = 60

    # Load and preprocess data
    X_train, y_train, X_test, y_test, scaler = load_and_preprocess_stock_data(sample_file_path, seq_length)

    # Print shapes to verify
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
