
import streamlit as st
import pandas as pd
import sys
import os
from app.llm_integration.stock_predictor import prepare_llm_context_data, generate_dynamic_prompt, get_openai_response

# Add the project root to the Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

st.set_page_config(layout="wide")

st.title("📈 Sentimental Stock Predictor")

st.sidebar.header("Controls")
user_query = st.sidebar.text_input("Ask about a stock (e.g., 'What is the prediction for MSFT?')", "which stock will grow tomorrow")

if st.sidebar.button("Get Prediction"):
    if not user_query:
        st.warning("Please enter a query.")
    else:
        with st.spinner("Analyzing data and generating prediction..."):
            try:
                # 1. Prepare the context data for the LLM
                context_df, market_sentiment, queried_companies = prepare_llm_context_data(user_query)

                # 2. Generate a dynamic prompt
                prompt = generate_dynamic_prompt(user_query, context_df, market_sentiment, queried_companies)

                # 3. Get the response from the LLM
                llm_response = get_openai_response(prompt)

                # Display the results
                st.subheader("🤖 AI Analyst Recommendation")
                st.markdown(llm_response)

                with st.expander("See the data used for this prediction"):
                    st.dataframe(context_df)

            except Exception as e:
                st.error(f"An error occurred: {e}")

