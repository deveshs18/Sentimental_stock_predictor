import datetime
import logging
import os
import sys

import altair as alt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Constants
SEQ_LEN = 100
# Target company for stock prediction
TARGET_COMPANY = os.environ.get("TARGET_COMPANY", "GOOGL")
# Number of future days to predict
FUTURE_DAYS = int(os.environ.get("FUTURE_DAYS", "30"))
# LSTM model path
# LSTM_MODEL_PATH = os.environ.get("LSTM_MODEL_PATH", "models/lstm_model.h5")


def load_and_preprocess_data(target_company: str) -> pd.DataFrame:
    """Loads and preprocesses stock data for the target company."""
    logger.info("Loading and preprocessing data for %s", target_company)
    # Construct the CSV file path based on the target company
    csv_file_path = f"data/historical_prices/{target_company.upper()}.csv" # Ensure company ticker is uppercase like in filenames
    try:
        # yfinance data typically doesn't have extra rows to skip
        data = pd.read_csv(csv_file_path)
        logger.info("Successfully loaded data from %s", csv_file_path)
    except FileNotFoundError:
        logger.error("CSV file not found: %s", csv_file_path)
        # You might want to raise an exception here or handle it differently
        raise
    except Exception as e:
        logger.error(f"Error reading CSV file {csv_file_path}: {e}", exc_info=True)
        raise

    # Convert 'Date' column to datetime objects
    if "Date" not in data.columns:
        logger.error("No 'Date' column found in the data.")
        raise ValueError("No 'Date' column found in the data.")
    data["Date"] = pd.to_datetime(data["Date"])

    # Sort data by date and set as index
    data.sort_values("Date", inplace=True)
    data.set_index("Date", inplace=True)

    # Check for 'Close' column after loading
    if "Close" not in data.columns:
        logger.error(f"No 'Close' price column found in the data. Available columns: {data.columns.tolist()}")
        raise ValueError("No 'Close' price column found in the data.")

    logger.info("Data loaded and preprocessed successfully with 'Close' column present.")
    return data


def prepare_llm_context_data(
    data: pd.DataFrame, seq_len: int = SEQ_LEN, lstm_model_path: str # Add new argument
) -> tuple[np.ndarray, MinMaxScaler, pd.DataFrame]:
    """Prepares data for the LLM context and LSTM model."""
    logger.info("Preparing data for LLM context and LSTM model.")

    # Ensure data is sorted by date for correct scaling and LSTM processing
    data = data.sort_index(ascending=True)

    # Scale the 'Close' price data
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Reshape for scaler: expects 2D array [n_samples, n_features]
    scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))
    logger.debug("Data scaled successfully.")

    # Prepare sequences for LSTM if the model file exists
    if os.path.exists(lstm_model_path): # Use the passed argument
        logger.info("LSTM model found at %s. Preparing sequences for LSTM.", lstm_model_path)
        # Create sequences for LSTM input
        x_data = []
        for i in range(seq_len, len(scaled_data)):
            x_data.append(scaled_data[i - seq_len : i, 0])
        x_data = np.array(x_data)
        # Reshape for LSTM: [samples, time steps, features]
        x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))
        logger.debug("LSTM sequences prepared with shape: %s", x_data.shape)
    else:
        logger.warning(
            "LSTM model not found at %s. Skipping LSTM sequence preparation.",
            lstm_model_path, # Use the passed argument
        )
        # Return empty array if no LSTM model to prevent errors downstream
        x_data = np.array([])

    # Select the last 'seq_len' days for the LLM context, ensuring it's from the original, unscaled data for clarity
    # and directly relevant to current market conditions.
    llm_context_data = data.tail(seq_len)
    logger.info(
        "LLM context data prepared using the last %d days of historical data.", seq_len
    )

    return x_data, scaler, llm_context_data


def generate_dynamic_prompt(
    data: pd.DataFrame, company_name: str, future_days: int
) -> str:
    """Generates a dynamic prompt for the LLM based on recent stock data, including LSTM predictions."""
    logger.info(
        "Generating dynamic prompt for %s covering %d future days.",
        company_name,
        future_days,
    )

    prompt_lines = [
        f"Company: {company_name}",
        "Recent Stock Data with LSTM Predictions:",
    ]
    # Prepare data with 'prev_close' for trend calculation
    data["prev_close"] = data["Close"].shift(1)

    for date, row in data.iterrows():
        line = f"On {date.strftime('%Y-%m-%d')}: Close Price was ${row['Close']:.2f}"
        # Check if LSTM predicted price is available and not NaN
        if "predicted_close_price" in row and pd.notna(row["predicted_close_price"]):
            line += f", LSTM Next Day Close Prediction: ${row['predicted_close_price']:.2f}"
            # Calculate percentage change if 'prev_close' is available
            if pd.notna(row.get("prev_close")):
                pct_change = ((row["predicted_close_price"] - row["prev_close"]) / row["prev_close"]) * 100
                direction = "📈" if pct_change > 0 else "📉"
                line += f" ({direction}{abs(pct_change):.2f}%)"
        else:
            line += ", LSTM Prediction: Not available"
        prompt_lines.append(line)

    # Summary and call to action for LLM
    last_date = data.index[-1].strftime("%Y-%m-%d")
    last_close_price = data["Close"].iloc[-1]
    prompt_lines.append(
        f"\nSummary: Last actual close on {last_date} was ${last_close_price:.2f}."
    )
    prompt_lines.append(
        f"Please provide a stock price prediction for {company_name} for the next {future_days} days, "
        "considering the historical data and LSTM insights provided above. Offer a brief analysis, "
        "an estimated price range, and a general market sentiment (e.g., bullish, bearish, neutral)."
    )

    final_prompt = "\n".join(prompt_lines)
    logger.debug("Generated LLM prompt: %s", final_prompt)
    return final_prompt


def get_llm_prediction(prompt: str) -> str:
    """Gets a prediction from the LLM."""
    # Placeholder for actual LLM API call
    # In a real scenario, this would involve an API request to an LLM
    logger.info("Sending prompt to LLM: %s", prompt)
    # Simulate LLM response
    return (
        "Based on the recent upward trend and current market conditions, the"
        " outlook for the next 30 days is cautiously optimistic (Bullish)."
        " Expected price range: $155-$165."
    )


def predict_future_prices_with_lstm(
    model_path: str, x_data: np.ndarray, scaler: MinMaxScaler, future_days: int
) -> np.ndarray:
    """Predicts future stock prices using the LSTM model."""
    logger.info("Loading LSTM model from %s", model_path)
    model = tf.keras.models.load_model(model_path)
    logger.info("LSTM model loaded successfully.")

    # Use the last sequence from x_data as the starting point for prediction
    last_sequence = x_data[-1:]  # Keep the 3D shape

    future_predictions = []
    current_sequence = last_sequence

    for _ in range(future_days):
        # Predict the next point
        next_pred_scaled = model.predict(current_sequence)
        future_predictions.append(next_pred_scaled[0, 0])  # Store the scaled prediction

        # Update the sequence for the next prediction
        # Reshape next_pred_scaled to be [1, 1, 1] to match LSTM input dimensions for a single feature
        new_element_reshaped = next_pred_scaled.reshape(1, 1, 1)
        current_sequence = np.append(current_sequence[:, 1:, :], new_element_reshaped, axis=1)


    # Inverse transform the scaled predictions to actual prices
    predicted_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    logger.info("Future prices predicted with LSTM: %s", predicted_prices.flatten())
    return predicted_prices.flatten()


def plot_predictions(
    historical_data: pd.DataFrame,
    predicted_prices: np.ndarray,
    company_name: str,
    future_days: int,
) -> None:
    """Plots historical and predicted stock prices."""
    logger.info("Plotting predictions for %s", company_name)
    last_date = historical_data.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1), periods=future_days
    )

    # Create DataFrames for plotting
    historical_df = historical_data["Close"].reset_index()
    historical_df.columns = ["Date", "Price"]
    historical_df["Type"] = "Historical"

    predicted_df = pd.DataFrame(
        {"Date": future_dates, "Price": predicted_prices}
    )
    predicted_df["Type"] = "Predicted"

    combined_df = pd.concat([historical_df, predicted_df])

    chart = (
        alt.Chart(combined_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("Date:T", title="Date"),
            y=alt.Y("Price:Q", title="Stock Price (USD)"),
            color="Type:N",
            tooltip=["Date", "Price", "Type"],
        )
        .properties(
            title=f"{company_name} Stock Price: Historical and Predicted",
            width=800,
            height=400,
        )
    )
    chart.save(f"{company_name}_stock_prediction.html")
    logger.info(
        "Prediction plot saved to %s_stock_prediction.html", company_name
    )


def main():
    """Main function to run the stock prediction process."""
    logger.info("Starting stock prediction process for %s", TARGET_COMPANY)

    # Load and preprocess data
    data = load_and_preprocess_data(TARGET_COMPANY)

    # Dynamically define LSTM model path
    lstm_model_path = f"models/lstm_{TARGET_COMPANY.lower()}_model.h5"
    # scaler_path = f"models/lstm_{TARGET_COMPANY.lower()}_scaler.joblib" # Scaler path might be needed later or in other functions

    # Prepare data for LLM and LSTM
    x_lstm_data, scaler, llm_context_data = prepare_llm_context_data(
        data, SEQ_LEN, lstm_model_path # Pass the dynamic path
    )

    # Generate dynamic prompt for LLM
    prompt = generate_dynamic_prompt(
        llm_context_data, TARGET_COMPANY, FUTURE_DAYS
    )
    logger.info("Generated LLM prompt: %s", prompt)

    # Get prediction from LLM (simulated)
    llm_prediction_text = get_llm_prediction(prompt)
    logger.info("LLM Prediction: %s", llm_prediction_text)

    # Predict future prices with LSTM if data is available
    if x_lstm_data.size > 0:
        predicted_prices_lstm = predict_future_prices_with_lstm(
            lstm_model_path, x_lstm_data, scaler, FUTURE_DAYS
        )
        logger.info(
            "LSTM Predicted Prices for the next %d days: %s",
            FUTURE_DAYS,
            predicted_prices_lstm,
        )

        # Plot historical and predicted prices
        plot_predictions(data, predicted_prices_lstm, TARGET_COMPANY, FUTURE_DAYS)
    else:
        logger.warning(
            "Skipping LSTM prediction and plotting due to missing LSTM model or"
            " insufficient data."
        )

    logger.info("Stock prediction process finished.")



    # Ensure queried companies are included even if they weren't in the top N
    if not queried_companies_data_df.empty:
        # Get the official names from the mapping for queried companies
        queried_official_names = set(queried_companies_data_df['company'])
        
        # Find any queried companies that might be missing from the top N
        missing_queried = []
        for company in queried_official_names:
            if company not in top_n_by_score_df['company'].values:
                missing_queried.append(company)
        
        # If we have missing queried companies, add them to the context
        if missing_queried:
            logger.info(f"Adding missing queried companies to final context: {missing_queried}")
            # Get the company data for missing queried companies
            missing_data = company_df[company_df['company'].isin(missing_queried)].copy()
            
            # Add LSTM predictions if available
            if 'Ticker' in missing_data.columns and not lstm_df.empty:
                missing_data = pd.merge(
                    missing_data,
                    lstm_df[['stock_ticker', 'predicted_close_price']],
                    left_on='Ticker',
                    right_on='stock_ticker',
                    how='left'
                ).drop(columns=['stock_ticker'])
            
            # Add sector information
            missing_data['theme'] = missing_data['company'].map(theme_map)
            missing_data['macro_sentiment_score'] = missing_data['theme'].map(macro_map).fillna(0)
            
            # Add to queried companies data
            queried_companies_data_df = pd.concat([queried_companies_data_df, missing_data])
    
    # Combine batch top companies and queried companies data
    final_context_df = pd.concat([top_n_by_score_df, queried_companies_data_df])
    
    # Remove duplicates, keeping the first occurrence (prioritize queried companies data)
    final_context_df = final_context_df.drop_duplicates(subset=['company'], keep='first').reset_index(drop=True)
    
    logger.info(f"Final LLM context will include {len(final_context_df)} unique companies (Top N by score + Queried).")
    
    # Log the final list of companies being returned
    logger.info(f"Companies in final context: {final_context_df['company'].tolist()}")
    
    if final_context_df.empty:
        logger.warning("LLM context is empty: No top companies met criteria and no queried companies identified/found in data.")
    else:
        # Ensure adjusted_growth_score is numeric before sorting
        final_context_df['adjusted_growth_score'] = pd.to_numeric(final_context_df['adjusted_growth_score'], errors='coerce')
        
        # Sort the final list by adjusted_growth_score for consistent presentation in the prompt, if desired
        final_context_df = final_context_df.sort_values(by="adjusted_growth_score", ascending=False).reset_index(drop=True)
        logger.debug(f"Final companies for LLM context (sorted by score):\n{final_context_df.to_string()}")

    market_sentiment_str = get_qualitative_market_sentiment()
    logger.info(f"Overall market sentiment for LLM context: {market_sentiment_str}")

    return final_context_df, market_sentiment_str

# === Main script execution starts here ===

if __name__ == "__main__":
    main()
