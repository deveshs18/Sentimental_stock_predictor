
import streamlit as st
import sys
import os
import pandas as pd

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from app.llm_integration.stock_predictor import prepare_llm_context_data, generate_dynamic_prompt, get_openai_response, logger

@st.cache_data
def load_data():
    """
    Loads all necessary data from the processed files and caches it.
    """
    predict_growth_file = os.path.join(project_root, 'data', 'processed', 'predict_growth.csv')
    news_file = os.path.join(project_root, 'data', 'processed', 'sentiment_analysis.csv')

    data = {
        'growth_df': pd.DataFrame(),
        'news_df': pd.DataFrame(),
        'error': None
    }

    try:
        if os.path.exists(predict_growth_file):
            data['growth_df'] = pd.read_csv(predict_growth_file)
            logger.info(f"Successfully loaded {predict_growth_file}")
        else:
            data['error'] = f"File not found: {predict_growth_file}. Please run the data pipeline."
            logger.error(data['error'])
            return data

        if os.path.exists(news_file):
            data['news_df'] = pd.read_csv(news_file)
            logger.info(f"Successfully loaded {news_file}")
        else:
            # This is not a fatal error for the whole dashboard, but some parts will be affected.
            logger.warning(f"News file not found: {news_file}. Headlines will be missing.")

    except Exception as e:
        data['error'] = f"Error loading data: {e}"
        logger.error(data['error'], exc_info=True)
    
    return data

# --- Main App Logic ---

st.title("📈 Sentimental Stock Predictor")

# Load the data once
data_load_state = st.info("Loading data...")
data = load_data()
data_load_state.empty()

if data['error']:
    st.error(data['error'])
else:
    growth_df = data['growth_df']
    news_df = data['news_df']

    # --- Dashboard --- 

    # 1. Overall Market Sentiment
    st.subheader("Overall Market Sentiment")
    market_sentiment_summary = "Not Available"
    if 'market_sentiment' in growth_df.columns and not growth_df.empty:
        market_sentiment_value = growth_df['market_sentiment'].iloc[0]
        if market_sentiment_value > 0.1:
            market_sentiment_summary = "Positive"
            st.metric(label="Market Sentiment", value="Positive", delta="Upward Trend", delta_color="green")
        elif market_sentiment_value < -0.1:
            market_sentiment_summary = "Negative"
            st.metric(label="Market Sentiment", value="Negative", delta="Downward Trend", delta_color="red")
        else:
            market_sentiment_summary = "Neutral"
            st.metric(label="Market Sentiment", value="Neutral", delta="Neutral", delta_color="off")
    else:
        st.error("Market sentiment data is not available. Please check the pipeline output.")

    st.markdown("---")

    # 2. Top News Headlines
    st.subheader("Top 5 Recent News Headlines")
    if not news_df.empty and 'headline' in news_df.columns:
        news_df['timestamp'] = pd.to_datetime(news_df['timestamp'], errors='coerce', utc=True)
        top_headlines = news_df.sort_values(by='timestamp', ascending=False).head(5)['headline'].tolist()
        for headline in top_headlines:
            st.markdown(f"- {headline}")
    else:
        st.warning("No recent news headlines available.")

    st.markdown("---")

    # 3. Stock Performance
    col1, col2 = st.columns(2)
    if not growth_df.empty and 'growth_score' in growth_df.columns:
        df_sorted = growth_df.sort_values(by="growth_score", ascending=False)
        
        # Merge headlines
        if not news_df.empty:
            # Create a common merge key
            df_sorted['company_upper'] = df_sorted['company'].str.upper()
            news_df['company_upper'] = news_df['company'].str.upper()
            headlines = news_df.groupby('company_upper')['headline'].apply(lambda x: x.head(3).tolist()).rename('headlines')
            df_sorted = pd.merge(df_sorted, headlines, on='company_upper', how='left')

        top_stocks = df_sorted.head(2)
        bottom_stocks = df_sorted.tail(2)

        with col1:
            st.subheader("Top 2 Growing Stocks")
            st.dataframe(top_stocks[['company', 'growth_score']].style.format({"growth_score": "{:.2f}"}))
            for _, row in top_stocks.iterrows():
                if 'headlines' in row and isinstance(row['headlines'], list):
                    st.write(f"**Recent Headlines for {row['company']}:**")
                    for headline in row['headlines']:
                        st.markdown(f"- {headline}")
        
        with col2:
            st.subheader("Bottom 2 Lowest Growth Stocks")
            st.dataframe(bottom_stocks[['company', 'growth_score']].style.format({"growth_score": "{:.2f}"}))
            for _, row in bottom_stocks.iterrows():
                if 'headlines' in row and isinstance(row['headlines'], list):
                    st.write(f"**Recent Headlines for {row['company']}:**")
                    for headline in row['headlines']:
                        st.markdown(f"- {headline}")
    else:
        with col1:
            st.error("No data available for top growing stocks.")
        with col2:
            st.error("No data available for bottom/lowest growth stocks.")

    # --- Sidebar Chat --- 
    with st.sidebar:
        st.title("🤖 Chat Analyst")
        st.caption("Ask me about stock market trends, company sentiment, or potential growth!")

        if 'messages' not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("What insights are you looking for today?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Analyzing... 📈")
                
                try:
                    top_companies_df, market_sentiment_str, queried_companies_normalized = prepare_llm_context_data(user_query=prompt)
                    dynamic_prompt_text = generate_dynamic_prompt(prompt, top_companies_df, market_sentiment_str, queried_companies_normalized)
                    llm_response = get_openai_response(dynamic_prompt_text)
                    message_placeholder.markdown(llm_response)
                    st.session_state.messages.append({"role": "assistant", "content": llm_response})
                except Exception as e:
                    error_msg = f"An error occurred: {e}"
                    logger.error(error_msg, exc_info=True)
                    message_placeholder.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

        st.info("Note: The analysis is based on data processed by the backend pipeline. Ensure `scripts/run_all.py` has been run recently for the most up-to-date insights.")
