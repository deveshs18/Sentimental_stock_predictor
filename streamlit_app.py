import streamlit as st
import sys
import os
import pandas as pd # Added import

# Add scripts directory to Python path to import stock_predictor
scripts_dir = os.path.join(os.path.dirname(__file__), 'scripts')
if scripts_dir not in sys.path:
    sys.path.append(scripts_dir)

# Now import the functions from stock_predictor
# We expect stock_predictor.py to have:
# - prepare_llm_context_data()
# - generate_dynamic_prompt(user_query, top_companies_df)
# - get_openai_response(prompt_text)
# - setup_logging() (if you have a common logging setup function, otherwise configure basic logging here)
try:
    from stock_predictor import prepare_llm_context_data, generate_dynamic_prompt, get_openai_response
    # Attempt to import setup_logging if it exists, otherwise skip
    try:
        from stock_predictor import logger # Use the logger configured in stock_predictor
    except ImportError:
        # Fallback basic logging if stock_predictor.logger is not available
        import logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        logger.info("Using fallback basic logging for Streamlit app.")

except ImportError as e:
    st.error(f"Error importing functions from stock_predictor: {e}. Ensure stock_predictor.py is in the 'scripts' directory and all its dependencies are met.")
    st.stop() # Stop the app if we can't import necessary functions

def get_stock_performance_data():
    """
    Reads predict_growth.csv and returns top/bottom N stocks.
    """
    predict_growth_file = os.path.join(os.path.dirname(__file__), 'data', 'predict_growth.csv')
    try:
        df = pd.read_csv(predict_growth_file)
        if 'growth_score' not in df.columns:
            logger.error(f"'growth_score' column not found in {predict_growth_file}. Cannot determine top/bottom stocks.")
            return pd.DataFrame(), pd.DataFrame() # Return empty DataFrames

        df_sorted = df.sort_values(by="growth_score", ascending=False)
        top_stocks_df = df_sorted.head(2)[['company', 'growth_score']]
        # For bottom stocks, ensure we don't re-select top stocks if there are few entries
        bottom_stocks_df = df_sorted.tail(2)[['company', 'growth_score']]

        # Handle case with very few companies (e.g. less than 4)
        if len(df) < 4 and not top_stocks_df.equals(bottom_stocks_df): # if they are different already, no issue
             # if less than 4, top and bottom might overlap or be the same
             # if top_stocks_df has 2, bottom_stocks_df will take the same if only 2 companies
             # if only 3 companies, tail(2) will include one from head(2)
             # This logic is okay for small N like 2. If N was larger, more complex logic might be needed.
             pass

        logger.info(f"Successfully loaded and processed stock performance data from {predict_growth_file}")
        return top_stocks_df, bottom_stocks_df
    except FileNotFoundError:
        logger.error(f"Stock performance data file not found: {predict_growth_file}. Please run the data pipeline.")
        return pd.DataFrame(), pd.DataFrame() # Return empty DataFrames
    except Exception as e:
        logger.error(f"Error reading or processing stock performance data: {e}", exc_info=True)
        return pd.DataFrame(), pd.DataFrame()

def get_overall_market_sentiment():
    """
    Reads market_sentiment from predict_growth.csv and returns a qualitative description.
    """
    predict_growth_file = os.path.join(os.path.dirname(__file__), 'data', 'predict_growth.csv')
    try:
        df = pd.read_csv(predict_growth_file)
        if 'market_sentiment' not in df.columns:
            logger.warning(f"'market_sentiment' column not found in {predict_growth_file}.")
            return "Data not available"
        if df.empty:
            logger.warning(f"{predict_growth_file} is empty.")
            return "Data not available"

        # Assuming market_sentiment is the same for all rows, take from the first row.
        market_sentiment_value = df['market_sentiment'].iloc[0]

        if market_sentiment_value > 0.1:
            qualitative_sentiment = "Positive"
        elif market_sentiment_value < -0.1:
            qualitative_sentiment = "Negative"
        else:
            qualitative_sentiment = "Neutral"
        logger.info(f"Overall market sentiment: {qualitative_sentiment} (Raw: {market_sentiment_value})")
        return qualitative_sentiment
    except FileNotFoundError:
        logger.error(f"Market sentiment data file not found: {predict_growth_file}. Please run the data pipeline.")
        return "Data not available"
    except Exception as e:
        logger.error(f"Error reading or processing market sentiment data: {e}", exc_info=True)
        return "Data not available"

def get_top_news(num_headlines=5):
    """
    Reads merged_sentiment_input.csv and returns top N recent news headlines.
    """
    merged_input_file = os.path.join(os.path.dirname(__file__), 'data', 'merged_sentiment_input.csv')
    try:
        df = pd.read_csv(merged_input_file)

        # Filter for news
        news_df = df[df['source'].str.lower() == 'news']

        if news_df.empty:
            logger.info(f"No news articles found in {merged_input_file} after filtering for 'news' source.")
            return []

        # Check for required columns
        if 'timestamp' not in news_df.columns or 'headline' not in news_df.columns:
            logger.error(f"'timestamp' or 'headline' column not found in news data from {merged_input_file}.")
            return []

        # Convert timestamp and sort
        news_df['timestamp'] = pd.to_datetime(news_df['timestamp'], errors='coerce')
        news_df = news_df.dropna(subset=['timestamp']) # Remove rows where timestamp conversion failed
        news_df_sorted = news_df.sort_values(by="timestamp", ascending=False)

        top_headlines = news_df_sorted['headline'].head(num_headlines).tolist()

        logger.info(f"Successfully loaded top {len(top_headlines)} news headlines from {merged_input_file}")
        return top_headlines

    except FileNotFoundError:
        logger.error(f"News data file not found: {merged_input_file}. Please run the data pipeline.")
        return []
    except Exception as e:
        logger.error(f"Error reading or processing news data: {e}", exc_info=True)
        return []

# Main Dashboard Area
st.title("📈 Stock Market Dashboard")

# Fetch data for the dashboard
top_news_headlines = get_top_news()
top_stocks, bottom_stocks = get_stock_performance_data()
market_sentiment_summary = get_overall_market_sentiment()

# Display Overall Market Sentiment
st.subheader("Overall Market Sentiment")
if market_sentiment_summary == "Positive":
    st.metric(label="Market Sentiment", value=market_sentiment_summary, delta="Upward Trend", delta_color="green")
elif market_sentiment_summary == "Negative":
    st.metric(label="Market Sentiment", value=market_sentiment_summary, delta="Downward Trend", delta_color="red")
elif market_sentiment_summary == "Neutral":
    st.metric(label="Market Sentiment", value=market_sentiment_summary, delta="Neutral", delta_color="off")
else: # Data not available
    st.write(market_sentiment_summary)

st.markdown("---") #Separator

# Display Top 5 News Headlines
st.subheader("Top 5 Recent News Headlines")
if top_news_headlines:
    for i, headline in enumerate(top_news_headlines):
        st.markdown(f"- {headline}")
else:
    st.write("No recent news headlines available.")

st.markdown("---") #Separator

# Display Stock Performance
col1, col2 = st.columns(2)

with col1:
    st.subheader("Top 2 Growing Stocks")
    if not top_stocks.empty:
        st.dataframe(top_stocks.style.format({"growth_score": "{:.2f}"}))
    else:
        st.write("No data available for top growing stocks.")

with col2:
    st.subheader("Bottom 2 Lowest Growth Stocks")
    if not bottom_stocks.empty:
        st.dataframe(bottom_stocks.style.format({"growth_score": "{:.2f}"}))
    else:
        st.write("No data available for bottom/lowest growth stocks.")


# Sidebar for Chat Interface
with st.sidebar:
    st.title("🤖 Chat Analyst")
    st.caption("Ask me about stock market trends, company sentiment, or potential growth!")

    # Initialize chat history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What insights are you looking for today?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant's turn
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Analyzing the markets for you... 📈")

            llm_response_content = None # Initialize to ensure it's defined
            try:
                # 1. Get the latest company context data
                logger.info("Streamlit App: Calling prepare_llm_context_data()")
                top_companies_df, market_sentiment_str = prepare_llm_context_data(user_query=prompt) # Pass user query

                if top_companies_df.empty:
                    logger.warning("Streamlit App: No company context data returned from prepare_llm_context_data().")
                    # This warning is for logs; the prompt will inform the user via LLM

                # 2. Generate the dynamic prompt
                logger.info(f"Streamlit App: Generating dynamic prompt for user query: {prompt}")
                dynamic_prompt_text = generate_dynamic_prompt(prompt, top_companies_df, market_sentiment_str) # Pass market_sentiment_str
                logger.debug(f"Streamlit App: Generated prompt: {dynamic_prompt_text}")

                # 3. Get response from OpenAI
                logger.info("Streamlit App: Calling get_openai_response()")

                with st.spinner("Consulting with the AI analyst..."):
                    llm_response_content = get_openai_response(dynamic_prompt_text)

                message_placeholder.markdown(llm_response_content)
                logger.info("Streamlit App: Successfully received and displayed LLM response.")

            except FileNotFoundError as fnf_error:
                logger.error(f"Streamlit App: Critical data file not found: {fnf_error}")
                error_msg = f"Error: A required data file is missing. Please ensure the data pipeline has been run successfully. Details: {fnf_error}"
                message_placeholder.error(error_msg)
                llm_response_content = error_msg # Save error to history
            except Exception as e:
                logger.error(f"Streamlit App: An unexpected error occurred: {e}", exc_info=True)
                error_msg = f"An unexpected error occurred while processing your request: {e}"
                message_placeholder.error(error_msg)
                llm_response_content = error_msg # Save error to history

        # Add assistant response (or error message) to chat history
        if llm_response_content:
            st.session_state.messages.append({"role": "assistant", "content": llm_response_content})

    # Data freshness note
    st.info(
        "**Note:** The analysis is based on data processed by the backend pipeline. "
        "Ensure `scripts/run_all.py` has been run recently for the most up-to-date insights."
    )
