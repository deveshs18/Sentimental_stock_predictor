import streamlit as st
import sys
import os
import pandas as pd

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from app.llm_integration.stock_predictor import prepare_llm_context_data, generate_dynamic_prompt, get_openai_response, logger

def get_stock_performance_data():
    """
    Reads predict_growth.csv and returns top/bottom N stocks including their top headlines.
    """
    predict_growth_file = os.path.join(project_root, 'data', 'predict_growth.csv')
    news_sentiment_file = os.path.join(project_root, 'data', 'merged_sentiment_input.csv')

    try:
        df = pd.read_csv(predict_growth_file)
        if 'growth_score' not in df.columns:
            logger.error(f"'growth_score' column not found in {predict_growth_file}. Cannot determine top/bottom stocks.")
            return pd.DataFrame(), pd.DataFrame()

        news_df = pd.read_csv(news_sentiment_file)
        news_df['company_upper'] = news_df['company'].str.upper()

        df['company_upper'] = df['company'].str.upper()
        df_merged = pd.merge(df, news_df.groupby('company_upper')['headline'].apply(lambda x: x.head(3).tolist()).rename('headlines'), on='company_upper', how='left')

        df_sorted = df_merged.sort_values(by="growth_score", ascending=False)
        top_stocks_df = df_sorted.head(2)[['company', 'growth_score', 'headlines']]
        bottom_stocks_df = df_sorted.tail(2)[['company', 'growth_score', 'headlines']]

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
    predict_growth_file = os.path.join(project_root, 'data', 'predict_growth.csv')
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
    merged_input_file = os.path.join(project_root, 'data', 'merged_sentiment_input.csv')
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
        news_df.loc[:, 'timestamp'] = pd.to_datetime(news_df['timestamp'], errors='coerce')
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
    st.metric(label="Market Sentiment", value="Not Available", delta="No Data", delta_color="off")

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
        st.dataframe(top_stocks[['company', 'growth_score']].style.format({"growth_score": "{:.2f}"}))
        for index, row in top_stocks.iterrows():
            if isinstance(row['headlines'], list):
                st.write(f"**Recent Headlines for {row['company']}:**")
                for headline in row['headlines']:
                    st.markdown(f"- {headline}")
    else:
        st.write("No data available for top growing stocks.")

with col2:
    st.subheader("Bottom 2 Lowest Growth Stocks")
    if not bottom_stocks.empty:
        st.dataframe(bottom_stocks[['company', 'growth_score']].style.format({"growth_score": "{:.2f}"}))
        for index, row in bottom_stocks.iterrows():
            if isinstance(row['headlines'], list):
                st.write(f"**Recent Headlines for {row['company']}:**")
                for headline in row['headlines']:
                    st.markdown(f"- {headline}")
    else:
        st.write("No data available for bottom/lowest growth stocks.")


# Sidebar for Chat Interface
with st.sidebar:
    st.title("🤖 Chat Analyst")
    st.caption("Ask me about stock market trends, company sentiment, or potential growth!")

    # Define advanced prompts first
    ADVANCED_PROMPTS = [
        "Type your own query below...", # Default option
        "Analyze the short-term price movement of [TICKER] using daily historical prices and recent news headlines.",
        "Given 2 years of daily closing prices, identify support and resistance levels for [TICKER].",
        "Perform technical analysis on [TICKER] using 50-day and 200-day moving averages.",
        "Based on news sentiment and recent price trends, is [TICKER] likely to be bullish or bearish tomorrow?",
        "Assess the impact of recent macroeconomic news (like GDP or inflation data) on the S&P 500.",
        "Does positive news sentiment for [TICKER] correlate with upward price trends historically?",
        "When did [TICKER] last experience a “golden cross” and what happened after?",
        "Analyze the volume spikes in [TICKER] and correlate them with news or earnings events.",
        "What does the put/call ratio for [TICKER] indicate about investor sentiment this week?",
        "Assess whether [TICKER] is overbought or oversold using RSI and recent headlines.",
        "Identify insider buying or selling activity for [TICKER] and predict short-term impact.",
        "Summarize the top news stories influencing the semiconductor sector this month.",
        "Compare the performance of growth vs value stocks in the last six months.",
        "Identify stocks with positive momentum and strong recent news sentiment.",
        "How does news about AI and automation impact the robotics and chip sectors?"
    ]

    # Initialize session state variables
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'selected_advanced_prompt_template' not in st.session_state:
        st.session_state.selected_advanced_prompt_template = ADVANCED_PROMPTS[0]
    if 'advanced_query_ticker' not in st.session_state:
        st.session_state.advanced_query_ticker = ""

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    # Consolidate prompt determination logic here
    user_input_from_chat = st.chat_input("What insights are you looking for today?")

    final_user_query = None
    query_source = None # To track if it's from advanced or chat_input

    if st.session_state.selected_advanced_prompt_template != ADVANCED_PROMPTS[0]:
        # User has selected an advanced prompt (and not the default "Type your own...")
        template = st.session_state.selected_advanced_prompt_template
        if "[TICKER]" in template:
            ticker = st.session_state.advanced_query_ticker.strip().upper()
            if ticker:
                final_user_query = template.replace("[TICKER]", ticker)
                query_source = "advanced_with_ticker"
            else:
                st.warning("Please enter a ticker for the selected advanced query or choose a different query.")
                # Potentially clear the selection or handle error more gracefully
                # For now, this will prevent submission by not setting final_user_query
        else:
            final_user_query = template
            query_source = "advanced_no_ticker"

        # Clear the advanced prompt selection so it doesn't persist for the next turn unless re-selected
        # And clear ticker, but it might be better to clear it only on successful submission or new selection
        # For now, let's clear it here to simplify state management for this iteration.
        # Consider moving this clearing to after successful processing if retaining the ticker is desired.
        # st.session_state.selected_advanced_prompt_template = ADVANCED_PROMPTS[0]
        # st.session_state.advanced_query_ticker = ""
        # Note: Clearing here makes the UI jump. Better to clear after processing or let users change it.
        # For this step, we will assume submission implies "use this and then reset for next time"

    elif user_input_from_chat:
        # User typed in the chat input and no advanced prompt (other than default) is selected
        final_user_query = user_input_from_chat
        query_source = "chat_input"

    if final_user_query:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": final_user_query})
        # Display user message
        with st.chat_message("user"):
            st.markdown(final_user_query)

        # Assistant's turn
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Analyzing the markets for you... 📈")

            llm_response_content = None # Initialize to ensure it's defined
            try:
                # 1. Get the latest company context data
                logger.info("Streamlit App: Calling prepare_llm_context_data()")
                # Correctly unpack the three values returned by prepare_llm_context_data
                top_companies_df, market_sentiment_str, queried_companies_normalized = prepare_llm_context_data(user_query=final_user_query)

                if top_companies_df.empty:
                    logger.warning("Streamlit App: No company context data returned from prepare_llm_context_data().")
                    # This warning is for logs; the prompt will inform the user via LLM

                # 2. Generate the dynamic prompt
                logger.info(f"Streamlit App: Generating dynamic prompt for user query: {final_user_query}")
                # Pass all necessary arguments, including the new queried_companies_normalized
                dynamic_prompt_text = generate_dynamic_prompt(final_user_query, top_companies_df, market_sentiment_str, queried_companies_normalized)
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

    st.markdown("---") # Separator

    # Advanced Prompt Selection
    st.subheader("Advanced Query Helper")

    ADVANCED_PROMPTS = [
        "Type your own query below...", # Default option
        "Analyze the short-term price movement of [TICKER] using daily historical prices and recent news headlines.",
        "Given 2 years of daily closing prices, identify support and resistance levels for [TICKER].",
        "Perform technical analysis on [TICKER] using 50-day and 200-day moving averages.",
        "Based on news sentiment and recent price trends, is [TICKER] likely to be bullish or bearish tomorrow?",
        "Assess the impact of recent macroeconomic news (like GDP or inflation data) on the S&P 500.",
        "Does positive news sentiment for [TICKER] correlate with upward price trends historically?",
        "When did [TICKER] last experience a “golden cross” and what happened after?",
        "Analyze the volume spikes in [TICKER] and correlate them with news or earnings events.",
        "What does the put/call ratio for [TICKER] indicate about investor sentiment this week?",
        "Assess whether [TICKER] is overbought or oversold using RSI and recent headlines.",
        "Identify insider buying or selling activity for [TICKER] and predict short-term impact.",
        "Summarize the top news stories influencing the semiconductor sector this month.",
        "Compare the performance of growth vs value stocks in the last six months.",
        "Identify stocks with positive momentum and strong recent news sentiment.",
        "How does news about AI and automation impact the robotics and chip sectors?"
    ]

    if 'selected_advanced_prompt_template' not in st.session_state:
        st.session_state.selected_advanced_prompt_template = ADVANCED_PROMPTS[0]
    if 'advanced_query_ticker' not in st.session_state:
        st.session_state.advanced_query_ticker = ""

    selected_prompt_template = st.selectbox(
        "Or, choose an advanced query:",
        options=ADVANCED_PROMPTS,
        key="selected_advanced_prompt_template_widget", # Use a different key for widget if needed for on_change
        index=ADVANCED_PROMPTS.index(st.session_state.selected_advanced_prompt_template) # Ensure state is reflected
    )
    st.session_state.selected_advanced_prompt_template = selected_prompt_template


    # Ticker input specifically for advanced prompts
    # This input is always present but only relevant if a prompt with [TICKER] is chosen.
    # We will handle the logic of using it when an advanced prompt is submitted.
    if "[TICKER]" in st.session_state.selected_advanced_prompt_template:
        st.text_input(
            "Enter Ticker for selected query (e.g., AAPL, MSFT):",
            key="advanced_query_ticker_input" # Widget key
        )
        # Sync widget state to session_state.advanced_query_ticker
        st.session_state.advanced_query_ticker = st.session_state.advanced_query_ticker_input
    else:
        # If the selected prompt does not need a ticker, clear any previous ticker
        st.session_state.advanced_query_ticker = ""
        # Optionally, disable or hide the ticker input if not needed by the current prompt
        # For now, let's just clear it and rely on the user to ignore it.
        # A more advanced UI could hide/show this input dynamically.
        st.caption(" (No Ticker needed for this query)")

    # The main chat input is still primary. The advanced prompt selection is an alternative.
    # Logic for how to prioritize will be in the chat input handling section.
