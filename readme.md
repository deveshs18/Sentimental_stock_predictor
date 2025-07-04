# Sentimental Stock Predictor

## Overview

This project is a Python-based pipeline that analyzes financial news and Reddit discussions to predict potential stock growth. It uses sentiment analysis, keyword weighting, time decay factors, and overall market sentiment to calculate growth scores for various companies. Finally, it leverages an OpenAI GPT model to interpret these scores and provide a qualitative prediction for the top 3 companies likely to show growth.

## Project Architecture

The project consists of a series of Python scripts orchestrated by `scripts/run_all.py` for data processing, and a Streamlit application (`streamlit_app.py`) for user interaction. The typical workflow is as follows:

1.  **Data Pipeline (`scripts/run_all.py`):**
    *   **Data Collection:**
        *   News articles are fetched from NewsAPI.
    *   Reddit posts are fetched (presumably related to finance/stocks).
    *   **Data Processing:**
        *   Collected data is merged.
        *   Entities (like company names) are extracted.
        *   Company names are normalized for consistency.
    *   **Sentiment Analysis:**
        *   Sentiment (positive, negative, neutral) is determined for the textual data, primarily using the FinBERT model (`yiyanghkust/finbert-tone`).
        *   Keyword analysis and weighting are performed.
        *   Company-specific and broader macro/market sentiment scores are generated.
    *   **Growth Prediction:**
        *   `scripts/predict_growth.py`: Calculates a quantitative `growth_score` for companies. This score incorporates:
            *   Sentiment of related news/discussions.
            *   Relevance of predefined financial keywords (from `data/edw_keywords.csv`).
            *   Time decay (recent news has more weight).
            *   Overall market sentiment (derived from VIX and S&P 500 data via `scripts/market_sentiment.py`).
        *   The results are saved in `data/predict_growth.csv` and other intermediate files.
2.  **LLM Interaction & UI (`streamlit_app.py` and `scripts/stock_predictor.py`):**
    *   `scripts/stock_predictor.py`: Contains core logic for:
        *   Preparing context data by loading and processing outputs from the data pipeline.
        *   Generating dynamic prompts based on user queries and the prepared context.
        *   Querying the OpenAI API with these prompts.
    *   `streamlit_app.py`: Provides an interactive web application with two main components:
        *   **Main Dashboard:** Displays key information such as Overall Market Sentiment, Top 5 Recent News Headlines, Top 2 Growing Stocks, and Bottom 2 Lowest Growth Stocks. This data is sourced from the backend pipeline.
        *   **Chat Sidebar:** Allows users to ask specific stock-related questions. The app uses `stock_predictor.py` functions to prepare context from the pipeline's data, generate a relevant prompt including the user's query, and fetch a response from an LLM.
    *   If `scripts/stock_predictor.py` is run directly, it can output a test prediction to `output/gpt_prediction_dynamic.txt`. However, the primary interaction is intended through the Streamlit UI.

## Directory Structure

-   `data/`: Contains input CSVs (keywords, company lists), intermediate data files generated by the pipeline, and final outputs like `predict_growth.csv`.
-   `logs/`: Contains log files generated by the scripts (e.g., `predict_growth.log`, `stock_predictor.log`).
-   `output/`: May contain test predictions from direct script runs (e.g., `gpt_prediction_dynamic.txt`).
-   `scripts/`: Contains all the Python scripts for the data pipeline and LLM interaction logic.
-   `utils/`: Contains utility Python modules.
-   `streamlit_app.py`: The main file for the Streamlit web user interface.

## Key Scripts

-   `scripts/run_all.py`: Main orchestrator to run the entire data processing pipeline.
-   `scripts/fetch_news.py`: Fetches news from NewsAPI.
-   `scripts/fetch_reddit.py`: Fetches data from Reddit.
-   `scripts/sentiment_analysis.py`: Performs sentiment analysis using FinBERT.
-   `scripts/predict_growth.py`: Calculates quantitative growth scores.
-   `scripts/stock_predictor.py`: Provides core functions to prepare data for LLM, generate dynamic prompts, and query the OpenAI API. Used by `streamlit_app.py`.
-   `scripts/market_sentiment.py`: Fetches market indicators (VIX, S&P 500) to assess overall market sentiment.
-   `streamlit_app.py`: The Streamlit application for interactive chat with the stock analyst AI.

## Setup and Configuration

1.  **Clone the repository.**
2.  **Install dependencies:**
    ```bash
    pip install pandas requests python-dotenv torch transformers textblob nltk thefuzz openai streamlit
    ```
    *(Note: A `requirements.txt` file would be beneficial here.)*
3.  **NLTK Data:**
    The `predict_growth.py` script attempts to download necessary NLTK data (`punkt`, `averaged_perceptron_tagger`, `wordnet`). If this fails due to environment restrictions, you may need to download them manually in a Python interpreter:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    ```
4.  **API Keys:**
    Create a `.env` file in the root directory of the project and add your API keys:
    ```env
    NEWS_API_KEY=your_newsapi_key_here
    OPENAI_API_KEY=your_openai_api_key_here
    ```
    Replace `your_newsapi_key_here` and `your_openai_api_key_here` with your actual API keys.

## How to Run

**1. Run the Data Pipeline:**

To process financial news and generate the underlying data for analysis, execute:

```bash
python scripts/run_all.py
```
This step needs to be run regularly to keep the data fresh. The output from this pipeline (e.g., `data/predict_growth.csv`, `data/company_sentiment_normalized.csv`, etc.) is used by the Streamlit application.

**2. Launch the Interactive Chat UI:**

After the data pipeline has been run, you can start the Streamlit application:
```bash
streamlit run streamlit_app.py
```
This will open a web interface in your browser where you can ask stock-related questions.

Individual scripts in the `scripts/` directory can also be run if their prerequisite data files exist, but the primary way to interact with the LLM is now through the Streamlit UI. Running `scripts/stock_predictor.py` directly might save a sample prediction to `output/gpt_prediction_dynamic.txt`.

## Interactive Chat UI (Streamlit)

The project features a comprehensive Streamlit application (`streamlit_app.py`) that serves as the main user interface. It provides:

-   **A Main Dashboard Area:**
    -   Displays an overview of the current market situation derived from the data pipeline.
    -   Key components include:
        -   Overall Market Sentiment (Positive, Negative, Neutral).
        -   Top 5 most recent news headlines.
        -   Top 2 stocks predicted for growth.
        -   Bottom 2 stocks with the lowest growth scores or potential decline.
-   **An Interactive Chat Sidebar:**
    -   Allows users to ask detailed, natural language questions about stock trends, specific company sentiment, or other financial topics.
    -   The chat leverages the backend data (prepared by `scripts/stock_predictor.py`) and an OpenAI GPT model to provide contextual answers.
    -   Maintains a history of the conversation.

**To run the Streamlit UI:**

1.  Ensure the data pipeline has been run at least once (see "How to Run" above).
2.  Execute the command:
    ```bash
    streamlit run streamlit_app.py
    ```
3.  Open the URL provided by Streamlit in your web browser.

**Important:** For the Streamlit app to provide relevant and up-to-date insights, the data pipeline (`scripts/run_all.py`) must be executed periodically to refresh its data sources.

## Areas for Improvement / Future Work

-   **Add `requirements.txt`:** For easier dependency management.
-   **Robust Error Handling:** Enhance error checking and resilience in all scripts.
-   **Configuration Management:** Centralize file paths and parameters.
-   **Backtesting:** Implement a framework to test the historical accuracy of `predict_growth.py` scores and potentially the LLM's suggestions.
-   **Expand Data Sources:** Incorporate more financial news sources, social media, or financial statements.
-   **Advanced Sentiment Models:** Explore fine-tuning sentiment models or using more domain-specific ones.
-   **Direct Prediction Model:** Consider developing a direct quantitative prediction model as an alternative or supplement to the LLM-based interpretation.
-   **Unit and Integration Tests:** Develop a test suite to ensure reliability.
-   **Code Refinement:** Address potential issues like the `TextBlob` fallback in `predict_growth.py` if FinBERT sentiment is intended.
-   **Detailed Logging:** Improve logging for better traceability and debugging.
