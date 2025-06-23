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
    
    # Define possible file paths to check
    possible_paths = [
        f"data/historical_prices/{target_company}.csv",
        f"data/{target_company}_data.csv"
    ]
    
    data = None
    for csv_file_path in possible_paths:
        if os.path.exists(csv_file_path):
            try:
                # First, read just the first few rows to understand the structure
                with open(csv_file_path, 'r') as f:
                    first_line = f.readline().strip()
                    second_line = f.readline().strip()
                    
                # Check if second line contains ticker information
                if second_line.startswith('Ticker'):
                    # If second line is ticker info, skip it and use first line as header
                    data = pd.read_csv(csv_file_path, header=0, skiprows=[1])
                else:
                    # Otherwise, just read normally
                    data = pd.read_csv(csv_file_path)
                    
                logger.info(f"Successfully loaded data from {csv_file_path}")
                break
            except Exception as e:
                logger.warning(f"Error reading {csv_file_path}: {e}")
                continue
    
    if data is None or data.empty:
        error_msg = f"Could not find or read data file for {target_company}. Tried paths: {', '.join(possible_paths)}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Log the first few rows and column names for debugging
    logger.debug("Raw data columns: %s", data.columns.tolist())
    logger.debug("First few rows of raw data:\n%s", data.head())
    
    # Drop any empty rows or columns
    data = data.dropna(how='all')
    data = data.dropna(axis=1, how='all')
    
    if data.empty:
        raise ValueError("No data remaining after dropping empty rows and columns")
    
    # The first column should be the date column
    date_col = data.columns[0]
    
    # Drop any rows where the date is the column name (e.g., 'Date')
    data = data[data[date_col] != date_col]
    
    # Convert the date column to datetime
    try:
        # Try parsing with dayfirst=True for European format dates
        data[date_col] = pd.to_datetime(data[date_col], dayfirst=False, errors='coerce')
        
        # Drop rows where date parsing failed
        data = data.dropna(subset=[date_col])
        
        if data.empty:
            raise ValueError("No valid dates found in the date column")
            
    except Exception as e:
        logger.error(f"Error parsing dates: {e}")
        raise
    
    # Set the index to the date column
    data = data.set_index(date_col)
    
    # Convert all columns to numeric, coercing errors to NaN
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Drop any rows with all NaN values
    data = data.dropna(how='all')
    
    if data.empty:
        raise ValueError("No data remaining after converting to numeric and dropping NaN values")
    
    # Standardize column names to lowercase for consistency
    data.columns = [str(col).strip().lower() for col in data.columns]
    logger.debug("Processed data columns: %s", data.columns.tolist())
    
    # Ensure the data is sorted by date
    data = data.sort_index()
    
    return data

# Helper function for formatting values in the prompt
def format_value_for_prompt(value):
    if isinstance(value, str):
        return value  # Return string indicators like "N/A (No Batch Data)" directly
    if pd.notna(value):
        try:
            return f"{float(value):.2f}" # Ensure it's a float before formatting
        except ValueError:
            return str(value) # If it can't be float (e.g. already a different string), return as is
    return "N/A" # For pd.NA or None

def generate_dynamic_prompt(user_query, top_companies_df, overall_market_sentiment_string):
    """
    Generates a dynamic prompt for the LLM based on user query, company data, and overall market sentiment.
    """
    prompt_lines = [
        "You are a stock analyst assistant. Analyze the provided company data in the context of the user's query and the stated overall market sentiment."
    ]
    prompt_lines.append(f"\nUser Query: \"{user_query}\"\n")
    prompt_lines.append(f"Overall Market Sentiment: {overall_market_sentiment_string}\n")
    
    instructions = """Your primary role is to answer the user's query by providing a detailed stock and sector analysis for 'tomorrow'. 
    Focus on the following key aspects:
    1. Overall Market Sentiment: Summarize the general market mood based on the provided sentiment data.
    2. Sector Analysis: Highlight which sectors are showing strength or weakness.
    3. Company Analysis: For companies mentioned in the query or showing significant movement, provide:
       - Sentiment analysis (positive/negative/neutral)
       - Growth score and its implications
       - Macro sentiment impact on the sector
       - LSTM next day close price prediction (if available)
    4. Actionable Insights: Provide clear, data-driven recommendations based on the analysis.
    
    Always consider the context of the user's query and provide specific, relevant information.
    If data is missing or unclear, acknowledge it and explain any limitations in the analysis."""

    prompt_lines.append("Company Data Context:")
    if top_companies_df.empty:
        prompt_lines.append("No specific company data is available from the latest pipeline run.")
    else:
        for _, row in top_companies_df.iterrows():
            macro_sentiment_val = row.get('macro_sentiment_score')
            macro_sentiment_str = f"{macro_sentiment_val:.2f}" if pd.notna(macro_sentiment_val) else "N/A"


            line = (
                f"- {row['company']}: Positive Sentiment={format_value_for_prompt(row.get('positive'))}, Neutral Sentiment={format_value_for_prompt(row.get('neutral'))}, "
                f"Negative Sentiment={format_value_for_prompt(row.get('negative'))}, GrowthScore={format_value_for_prompt(row.get('growth_score'))}, "
                f"MacroSentiment={macro_sentiment_str}, Sector={row['theme']}" # macro_sentiment_str and theme are usually fine

            )
            # Initialize company_info with the base line
            company_info = line
            
            # Check if 'predicted_close_price' column exists and the value is not NaN or a placeholder string
            if 'predicted_close_price' in row and pd.notna(row['predicted_close_price']):
                company_info += f"\n  - 🚀 LSTM Next Day Close Prediction: ${row['predicted_close_price']:.2f}"
                # Add some interpretation of the prediction
                if 'current_price' in row and pd.notna(row['current_price']):
                    current_price = row['current_price']
                    predicted_price = row['predicted_close_price']
                    pct_change = ((predicted_price - current_price) / current_price) * 100
                    direction = "↑" if pct_change >= 0 else "↓"
                    company_info += f" ({direction}{abs(pct_change):.2f}%)"
            else:
                company_info += "\n  - LSTM Prediction: Not available"
            prompt_lines.append(company_info)

    prompt_lines.append("\n---") # Separator before instructions


    prompt_lines.append("\nInstructions:\n" + instructions)
    return "\n".join(prompt_lines)

def get_openai_response(prompt_text):
    """
    Calls the OpenAI API with the given prompt and returns the response.
    """
    logger.info("\n📡 Querying OpenAI...\n")
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Or "gpt-4" if you have access
            messages=[
                {"role": "system", "content": "You are a stock analyst assistant."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.7,
            max_tokens=600
        )
        result_text = response.choices[0].message.content
        logger.info(f"🔮 OpenAI Response:\n{result_text}")
        return result_text
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}", exc_info=True)
        # Depending on desired behavior, could re-raise, return None, or a specific error message
        raise  # Re-raise for the main block to handle for now

def prepare_llm_context_data(user_query, top_n=25): # user_query parameter added
    """
    Loads, processes, and filters data to prepare the context for the LLM.
    Ensures companies mentioned in user_query are included.
    Returns a tuple: (DataFrame of top N companies + queried companies, qualitative market sentiment string).
    """
    logger.info(f"Preparing LLM context for query: '{user_query}' with top_n={top_n}") # Updated log
    # Construct paths relative to this script's location (scripts/)
    # Construct paths relative to this script's location (scripts/)
    # Data files are expected to be in ../data/
    current_dir = os.path.dirname(__file__)
    data_files = {
        "company_df": os.path.join(current_dir, "..", "data", "company_sentiment_normalized.csv"),
        "growth_df": os.path.join(current_dir, "..", "data", "predict_growth.csv"),
        "macro_df": os.path.join(current_dir, "..", "data", "macro_sentiment.csv"),
        "mapping_df": os.path.join(current_dir, "..", "data", "nasdaq_top_companies.csv"),
    }
    lstm_predictions_df_path = os.path.join(current_dir, '..', 'data', 'lstm_daily_predictions.csv')
    dfs = {}

    try:
        for df_name, path in data_files.items():
            logger.debug(f"Loading {path}...")
            dfs[df_name] = pd.read_csv(path)

        try:
            logger.debug(f"Loading {lstm_predictions_df_path}...")
            dfs["lstm_df"] = pd.read_csv(lstm_predictions_df_path)
        except FileNotFoundError:
            logger.warning(f"LSTM predictions file not found at {lstm_predictions_df_path}. Creating empty DataFrame.")
            dfs["lstm_df"] = pd.DataFrame(columns=['stock_ticker', 'predicted_close_price'])

        logger.info("All data files loaded successfully (or handled if missing).")
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}. Please ensure all prerequisite scripts have run.")
        raise  # Re-raise to be caught by the caller or main script
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading data: {e}", exc_info=True)

def prepare_llm_context_data(
    data: pd.DataFrame, seq_len: int = SEQ_LEN, lstm_model_path: str = ""  # Default empty string if not provided
) -> tuple[np.ndarray, MinMaxScaler, pd.DataFrame]:
    """Prepares data for the LLM context and LSTM model."""
    logger.info("Preparing data for LLM context and LSTM model.")
    
    # Log available columns for debugging
    logger.debug("Available columns in data: %s", data.columns.tolist())

    # Ensure data is sorted by date for correct scaling and LSTM processing
    data = data.sort_index(ascending=True)
    
    # Find the close price column (case insensitive)
    close_col = next((col for col in data.columns if 'close' in col.lower()), None)
    if close_col is None:
        logger.error("No 'Close' price column found in the data. Available columns: %s", data.columns.tolist())
        raise ValueError("No 'Close' price column found in the data")
        
    logger.info("Using column '%s' for closing prices", close_col)

    # Scale the 'Close' price data
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Reshape for scaler: expects 2D array [n_samples, n_features]
    scaled_data = scaler.fit_transform(data[close_col].values.reshape(-1, 1))
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
    data["prev_close"] = data["close"].shift(1)

    for date, row in data.iterrows():
        line = f"On {date.strftime('%Y-%m-%d')}: Close Price was ${row['close']:.2f}"
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
    last_close_price = data["close"].iloc[-1]
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
    historical_df = historical_data["close"].reset_index()
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


# Dynamically define LSTM model path
lstm_model_path = f"models/lstm_{TARGET_COMPANY.lower()}_model.h5"
# scaler_path = f"models/lstm_{TARGET_COMPANY.lower()}_scaler.joblib" # Scaler path might be needed later or in other functions

def main():
    """Main function to run the stock prediction process."""
    logger.info("Starting stock prediction process for %s", TARGET_COMPANY)

    # Load and preprocess data
    data = load_and_preprocess_data(TARGET_COMPANY)
    
    # Check if data loading was successful
    if data.empty:
        logger.error("Failed to load stock data. Please ensure the data file exists at 'data/%s_data.csv'", TARGET_COMPANY)
        return

    # Prepare data for LLM and LSTM
    x_lstm_data, scaler, llm_context_data = prepare_llm_context_data(
        data, SEQ_LEN, lstm_model_path  # Pass the dynamic path
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

# === Main script execution starts here ===

if __name__ == "__main__":
    main()
