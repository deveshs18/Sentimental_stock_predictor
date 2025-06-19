import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import math
import os
import pandas as pd # Added for predict_next_day
import joblib
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Assuming preprocess_lstm_data.py is in the same directory or accessible in PYTHONPATH
from preprocess_lstm_data import load_and_preprocess_stock_data

def build_and_train_lstm_model(X_train, y_train, X_test, y_test, scaler_obj, sequence_length, input_shape_dims, stock_ticker_symbol, epochs=10): # Added epochs
    """
    Builds, trains, evaluates, and saves an LSTM model.

    Args:
        X_train (np.array): Training input sequences.
        y_train (np.array): Training output values.
        X_test (np.array): Testing input sequences.
        y_test (np.array): Testing output values (scaled).
        scaler_obj (sklearn.preprocessing.MinMaxScaler): Scaler used for data.
        sequence_length (int): The length of the input sequences.
        input_shape_dims (tuple): The shape of the input data for LSTM (sequence_length, num_features).
        stock_ticker_symbol (str): The stock ticker symbol (e.g., 'AAPL').

    Returns:
        tuple: (trained_model, y_test_actual, predictions_actual)
    """
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape_dims),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)  # To predict the single next 'Close' price
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    # Note: 50 epochs can be time-consuming. For this subtask, using 10 epochs.
    # For actual training, 50 or more epochs with early stopping would be more appropriate.
    print("Training model...")
    # Using epochs=epochs for configurable training duration
    model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_test, y_test), verbose=1)

    # Evaluate the model
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss}")

    # Make predictions
    predictions_scaled = model.predict(X_test)

    # Inverse transform predictions and y_test
    predictions_actual = scaler_obj.inverse_transform(predictions_scaled)
    y_test_actual = scaler_obj.inverse_transform(y_test.reshape(-1, 1))

    # Calculate RMSE
    rmse = math.sqrt(mean_squared_error(y_test_actual, predictions_actual))
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')

    # Save the model
    model_save_path = f'models/lstm_{stock_ticker_symbol.lower()}_model.h5'
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Save the scaler
    scaler_save_path = f'models/lstm_{stock_ticker_symbol.lower()}_scaler.joblib'
    joblib.dump(scaler_obj, scaler_save_path)
    print(f"Scaler saved to {scaler_save_path}")

    return model, y_test_actual, predictions_actual, rmse

def plot_predictions(y_test_actual_scale, predictions_actual_scale, stock_ticker_symbol):
    """
    Plots actual vs. predicted stock prices.

    Args:
        y_test_actual_scale (np.array): Array of actual stock prices.
        predictions_actual_scale (np.array): Array of predicted stock prices.
        stock_ticker_symbol (str): The stock ticker symbol (e.g., 'AAPL').
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual_scale, color='blue', label='Actual Prices')
    plt.plot(predictions_actual_scale, color='red', label='Predicted Prices')
    plt.title(f"{stock_ticker_symbol} Stock Price Prediction")
    plt.xlabel("Time (days in test set)")
    plt.ylabel("Price")
    plt.legend()

    # Create output directory if it doesn't exist
    if not os.path.exists('output'):
        os.makedirs('output')

    plot_save_path = f'output/lstm_predictions_{stock_ticker_symbol.lower()}.png'
    plt.savefig(plot_save_path)
    print(f"Plot saved to {plot_save_path}")
    # plt.show() # Would display plot if in interactive environment

def predict_next_day(stock_ticker: str, sequence_length: int = 60) -> float | None:
    """
    Predicts the next day's closing price for a given stock.

    Args:
        stock_ticker (str): The stock ticker symbol (e.g., 'AAPL').
        sequence_length (int): The length of the input sequences used for training.

    Returns:
        float | None: The predicted next day's closing price, or None if an error occurs.
    """
    model_path = f'models/lstm_{stock_ticker.lower()}_model.h5'
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None

    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    scaler_path = f'models/lstm_{stock_ticker.lower()}_scaler.joblib'
    if not os.path.exists(scaler_path):
        print(f"Error: Scaler file not found at {scaler_path}")
        return None
    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        print(f"Error loading scaler: {e}")
        return None

    data_file_path = f'data/historical_prices/{stock_ticker}.csv'
    if not os.path.exists(data_file_path):
        print(f"Error: Data file not found at {data_file_path}") # Keep this check for the data file
        return None

    # Load the raw data again to get the most recent sequence
    try:
        df = pd.read_csv(data_file_path, header=0, skiprows=[1])
        if 'Close' not in df.columns:
            print(f"Error: 'Close' column not found in {data_file_path}")
            return None

        # Ensure 'Close' is numeric and handle NaNs
        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
        df['Close'] = df['Close'].ffill() # Forward fill first
        df.dropna(subset=['Close'], inplace=True) # Then drop if any NaNs remain at the beginning

        if len(df) < sequence_length:
            print(f"Error: Not enough data in {data_file_path} to form a sequence of length {sequence_length}. Data length: {len(df)}")
            return None

        close_prices = df['Close'].values
        last_sequence_raw = close_prices[-sequence_length:]

        # Scale this sequence
        # Note: Using the scaler fitted on the original training set.
        last_sequence_scaled = scaler.transform(last_sequence_raw.reshape(-1, 1))

        # Reshape for model input
        last_sequence_reshaped = np.reshape(last_sequence_scaled, (1, sequence_length, 1))

        # Make prediction
        predicted_scaled = model.predict(last_sequence_reshaped)

        # Inverse transform the prediction
        predicted_price = scaler.inverse_transform(predicted_scaled)

        return predicted_price[0][0]

    except Exception as e:
        print(f"Error during prediction for {stock_ticker}: {e}")
        return None

def get_or_train_lstm_for_stock(
    ticker: str,
    sequence_length: int = 60,
    data_dir: str = 'data/historical_prices/',
    models_dir: str = 'models/',
    lstm_predictions_csv: str = 'data/lstm_daily_predictions.csv',
    training_epochs: int = 10  # Default epochs for on-demand training
) -> float | None:
    """
    Gets a prediction from an existing LSTM model or trains a new one if not found,
    then updates the shared predictions CSV.
    """
    model_path = os.path.join(models_dir, f'lstm_{ticker.lower()}_model.h5')
    scaler_path = os.path.join(models_dir, f'lstm_{ticker.lower()}_scaler.joblib')
    data_file_path = os.path.join(data_dir, f'{ticker.upper()}.csv') # Ticker is upper in historical_prices

    if not os.path.exists(data_file_path):
        print(f"Error: Historical data file not found at {data_file_path} for ticker {ticker}.")
        # Log this instead of just printing for production
        logging.error(f"Historical data file not found at {data_file_path} for ticker {ticker}.")
        return None

    predicted_price = None

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print(f"Found existing model and scaler for {ticker}. Attempting prediction.")
        # Log this
        logging.info(f"Using existing LSTM model and scaler for {ticker}.")
        predicted_price = predict_next_day(ticker, sequence_length) # predict_next_day uses paths relative to its own constants/structure
                                                                 # We need to ensure predict_next_day can find models in models_dir and data in data_dir
                                                                 # predict_next_day currently hardcodes paths like 'models/lstm_{stock_ticker.lower()}_model.h5'
                                                                 # This is compatible if models_dir is 'models/'
    else:
        print(f"Model and/or scaler not found for {ticker}. Training new model.")
        logging.info(f"Attempting to train a new LSTM model for {ticker}.")
        try:
            X_train_orig, y_train_orig, X_test_orig, y_test_orig_scaled, scaler_obj_for_training = \
                load_and_preprocess_stock_data(data_file_path, sequence_length)

            if X_train_orig.size == 0 or X_test_orig.size == 0:
                print(f"Warning: Not enough data to train model for {ticker} after preprocessing from {data_file_path}. Skipping training.")
                logging.warning(f"Insufficient data for {ticker} from {data_file_path} for LSTM training.")
                return None

            # Reshape X_train and X_test to be 3D for LSTM
            X_train = np.reshape(X_train_orig, (X_train_orig.shape[0], X_train_orig.shape[1], 1))
            X_test = np.reshape(X_test_orig, (X_test_orig.shape[0], X_test_orig.shape[1], 1))

            print(f"Data preprocessed for {ticker}. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
            logging.info(f"Data preprocessed for {ticker}. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

            # Build, train, evaluate and save the model
            # Note: build_and_train_lstm_model saves the scaler_obj_for_training and model
            # Pass training_epochs to the function
            trained_model, _, _, _ = build_and_train_lstm_model(
                X_train, y_train_orig, X_test, y_test_orig_scaled, scaler_obj_for_training,
                sequence_length, (X_train.shape[1], 1), ticker, epochs=training_epochs # Pass epochs
            )
            print(f"Successfully trained and saved model for {ticker}.")
            logging.info(f"Successfully trained LSTM model for {ticker} with {training_epochs} epochs.")

            # Predict next day's price using the newly trained model
            predicted_price = predict_next_day(ticker, sequence_length)

        except Exception as e:
            print(f"An error occurred during training or prediction for {ticker}: {e}")
            logging.error(f"Error training LSTM model for {ticker}: {e}", exc_info=True)
            return None

    if predicted_price is not None:
        print(f"Predicted next day's closing price for {ticker}: {predicted_price:.2f}")
        logging.info(f"LSTM Predicted price for {ticker}: {predicted_price:.2f}")
        try:
            predictions_df = pd.DataFrame(columns=['stock_ticker', 'predicted_close_price', 'prediction_timestamp'])
            if os.path.exists(lstm_predictions_csv):
                try:
                    predictions_df = pd.read_csv(lstm_predictions_csv)
                except pd.errors.EmptyDataError:
                    print(f"Warning: {lstm_predictions_csv} was empty. Initializing new DataFrame.")
                    logging.warning(f"{lstm_predictions_csv} was empty. Initializing new DataFrame.")
                except Exception as e:
                    print(f"Error reading {lstm_predictions_csv}: {e}. Initializing new DataFrame.")
                    logging.error(f"Error reading {lstm_predictions_csv}: {e}. Initializing new DataFrame.")

            # Ensure correct dtypes, especially if file was empty or just created
            predictions_df['stock_ticker'] = predictions_df['stock_ticker'].astype(str)
            predictions_df['predicted_close_price'] = pd.to_numeric(predictions_df['predicted_close_price'], errors='coerce')
            # 'prediction_timestamp' can remain object/string or be converted to datetime if needed for other ops

            # Remove existing rows for the current ticker
            predictions_df = predictions_df[predictions_df['stock_ticker'] != ticker].copy() # Use .copy() to avoid SettingWithCopyWarning

            # Create new prediction entry
            new_prediction_data = {
                'stock_ticker': ticker,
                'predicted_close_price': predicted_price,
                'prediction_timestamp': datetime.now().isoformat()
            }
            new_prediction_df = pd.DataFrame([new_prediction_data])

            # Concatenate old and new predictions
            predictions_df = pd.concat([predictions_df, new_prediction_df], ignore_index=True)

            # Save updated predictions
            predictions_csv_dir = os.path.dirname(lstm_predictions_csv)
            if predictions_csv_dir and not os.path.exists(predictions_csv_dir): # Check if dirname is not empty string
                os.makedirs(predictions_csv_dir, exist_ok=True)

            predictions_df.to_csv(lstm_predictions_csv, index=False)
            print(f"Updated LSTM predictions saved to {lstm_predictions_csv} for ticker {ticker}.")
            logging.info(f"Updated {lstm_predictions_csv} with prediction for {ticker}.")

        except Exception as e:
            print(f"Error updating {lstm_predictions_csv} for {ticker}: {e}")
            logging.error(f"Error updating {lstm_predictions_csv} for {ticker}: {e}", exc_info=True)
            # Do not return None here, as prediction was successful, only CSV update failed.
            # The function should still return the predicted_price.

    return predicted_price


if __name__ == "__main__":

    target_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'ENPH'] # Expanded list


    all_predictions_data = []
    sequence_length = 60  # Define sequence length

    for stock_ticker in target_stocks:
        print(f"\nProcessing stock: {stock_ticker}")
        data_file_path = f'data/historical_prices/{stock_ticker}.csv'
        model_path = f'models/lstm_{stock_ticker.lower()}_model.h5'
        scaler_path = f'models/lstm_{stock_ticker.lower()}_scaler.joblib'

        if not os.path.exists(data_file_path):
            print(f"Warning: Data file not found for {stock_ticker} at {data_file_path}. Skipping.")
            continue

        predicted_price = None
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            print(f"Found existing model and scaler for {stock_ticker}.")
            # predict_next_day loads the model and scaler internally by path
            predicted_price = predict_next_day(stock_ticker=stock_ticker, sequence_length=sequence_length)
        else:
            print(f"Model and/or scaler not found for {stock_ticker}. Training new model.")
            try:
                # Load and preprocess data
                X_train_orig, y_train_orig, X_test_orig, y_test_orig_scaled, scaler_obj_for_training = \
                    load_and_preprocess_stock_data(data_file_path, sequence_length)

                if X_train_orig.size == 0 or X_test_orig.size == 0 :
                    print(f"Warning: Not enough data to train model for {stock_ticker} after preprocessing. Skipping.")
                    continue

                # Reshape X_train and X_test to be 3D
                X_train = np.reshape(X_train_orig, (X_train_orig.shape[0], X_train_orig.shape[1], 1))
                X_test = np.reshape(X_test_orig, (X_test_orig.shape[0], X_test_orig.shape[1], 1))

                print(f"X_train reshaped: {X_train.shape}")
                print(f"X_test reshaped: {X_test.shape}")

                # Build, train, evaluate and save the model
                # Note: build_and_train_lstm_model saves the scaler_obj_for_training
                trained_model, y_test_actual, predictions_actual, rmse_value = build_and_train_lstm_model(
                    X_train, y_train_orig, X_test, y_test_orig_scaled, scaler_obj_for_training,
                    sequence_length, (X_train.shape[1], 1), stock_ticker
                )

                # Plot predictions only if training occurred
                plot_predictions(y_test_actual, predictions_actual, stock_ticker)
                print(f"Successfully trained and saved model/plot for {stock_ticker}.")

                # Predict next day's price using the newly trained model
                predicted_price = predict_next_day(stock_ticker=stock_ticker, sequence_length=sequence_length)

            except FileNotFoundError:
                print(f"Error: Data file {data_file_path} confirmed present but load_and_preprocess_stock_data failed to find it. Skipping {stock_ticker}.")
                continue
            except Exception as e:
                print(f"An error occurred during training or initial prediction for {stock_ticker}: {e}. Skipping.")
                continue

        if predicted_price is not None:
            print(f"Predicted next day's closing price for {stock_ticker}: {predicted_price:.2f}")
            all_predictions_data.append({
                'stock_ticker': stock_ticker,
                'predicted_close_price': predicted_price,
                'prediction_timestamp': datetime.now().isoformat()
            })
        else:
            print(f"Could not generate prediction for {stock_ticker}.")

    if all_predictions_data:
        predictions_df = pd.DataFrame(all_predictions_data)
        output_csv_path = 'data/lstm_daily_predictions.csv'
        # Ensure data directory exists
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        predictions_df.to_csv(output_csv_path, index=False)
        print(f"\nLSTM predictions saved to {output_csv_path}")
    else:
        print("\nNo predictions were generated to save.")
