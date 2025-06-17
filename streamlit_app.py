import streamlit as st
import sys
import os

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

st.title("🤖 Sentimental Stock Analyst Chat")
st.caption("Ask me about stock market trends, company sentiment, or potential growth!")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What insights are you looking for today? (e.g., 'Tell me about Apple stock', 'Which tech stocks show positive sentiment?')"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant's turn
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Analyzing the markets for you... 📈")

        try:
            # 1. Get the latest company context data
            # Consider adding caching here for performance (e.g., @st.cache_data)
            # For now, it fetches fresh data on each query.
            logger.info("Streamlit App: Calling prepare_llm_context_data()")
            top_companies_df = prepare_llm_context_data()

            if top_companies_df.empty:
                logger.warning("Streamlit App: No company context data returned from prepare_llm_context_data().")
                st.warning("Could not retrieve detailed company data at this moment. The analysis might be more general.")

            # 2. Generate the dynamic prompt
            logger.info(f"Streamlit App: Generating dynamic prompt for user query: {prompt}")
            dynamic_prompt_text = generate_dynamic_prompt(prompt, top_companies_df)
            logger.debug(f"Streamlit App: Generated prompt: {dynamic_prompt_text}")

            # 3. Get response from OpenAI
            logger.info("Streamlit App: Calling get_openai_response()")

            # Display thinking animation while waiting for OpenAI
            with st.spinner("Consulting with the AI analyst..."):
                llm_response = get_openai_response(dynamic_prompt_text)

            message_placeholder.markdown(llm_response)
            logger.info("Streamlit App: Successfully received and displayed LLM response.")

        except FileNotFoundError as fnf_error:
            logger.error(f"Streamlit App: Critical data file not found: {fnf_error}")
            message_placeholder.error(f"Error: A required data file is missing. Please ensure the data pipeline has been run successfully. Details: {fnf_error}")
        except Exception as e:
            logger.error(f"Streamlit App: An unexpected error occurred: {e}", exc_info=True)
            message_placeholder.error(f"An unexpected error occurred while processing your request: {e}")

    # Add assistant response to chat history
    if 'llm_response' in locals(): # Check if llm_response was defined
        st.session_state.messages.append({"role": "assistant", "content": llm_response})
    else: # Handle cases where llm_response might not be set due to an error before its assignment
        # The error message would have already been displayed by message_placeholder.error
        # We can add a generic error message to history if needed, or leave it
        # For now, let's assume the error displayed is sufficient and don't add to history here.
        pass

# Add a sidebar note about data freshness
st.sidebar.info(
    "**Note:** The analysis is based on data processed by the backend pipeline. "
    "Ensure `scripts/run_all.py` has been run recently for the most up-to-date insights."
)
