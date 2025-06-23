import logging
import os
import sys
import pandas as pd
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Ensure OPENAI_API_KEY is loaded from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY environment variable not found.")

# Helper function for formatting values in the prompt
def format_value_for_prompt(value):
    if isinstance(value, str):
        return value
    if pd.notna(value):
        try:
            return f"{float(value):.2f}"
        except ValueError:
            return str(value)
    return "N/A"

def prepare_llm_context_data(user_query, top_n=25):
    """
    Loads, processes, and filters data from various CSVs to prepare the context for the LLM.
    This version is used by the Streamlit application. LSTM components have been removed.
    Returns a tuple: (DataFrame of top N companies + queried companies, qualitative market sentiment string).
    """
    logger.info(f"Preparing LLM context for query (LSTM removed): '{user_query}' with top_n={top_n}")
    current_dir = os.path.dirname(__file__)
    
    company_sentiment_path = os.path.join(current_dir, "..", "data", "company_sentiment_normalized.csv")
    predict_growth_path = os.path.join(current_dir, "..", "data", "predict_growth.csv")
    macro_sentiment_path = os.path.join(current_dir, "..", "data", "macro_sentiment.csv")
    # nasdaq_companies_path = os.path.join(current_dir, "..", "data", "nasdaq_top_companies.csv") # Currently unused

    try:
        company_df = pd.read_csv(company_sentiment_path)
        growth_df = pd.read_csv(predict_growth_path)
        macro_df = pd.read_csv(macro_sentiment_path)
        # mapping_df = pd.read_csv(nasdaq_companies_path) # Not used in current merge

        # Normalize ticker columns for merging
        company_df['company'] = company_df['company'].str.upper()
        growth_df['company'] = growth_df['company'].str.upper()
        
        # Merge company sentiment with growth data
        # Assumed: growth_df has 'company', 'growth_score', 'current_price', 'market_sentiment', 'theme'
        # Assumed: company_df has 'company', 'positive', 'neutral', 'negative'
        merged_df = pd.merge(growth_df, company_df, on="company", how="left")

        # LSTM loading and merging logic REMOVED

        # Merge with macro sentiment
        # Assumed: macro_df has 'theme', 'macro_sentiment_score'
        if 'theme' in merged_df.columns and not macro_df.empty:
             merged_df = pd.merge(merged_df, macro_df[['theme', 'macro_sentiment_score']], on="theme", how="left")
        else:
            merged_df['macro_sentiment_score'] = pd.NA # Ensure column exists

        # Determine overall market sentiment
        overall_market_sentiment_value = growth_df['market_sentiment'].iloc[0] if 'market_sentiment' in growth_df.columns and not growth_df.empty else 0
        if overall_market_sentiment_value > 0.1:
            overall_market_sentiment_string = "Positive"
        elif overall_market_sentiment_value < -0.1:
            overall_market_sentiment_string = "Negative"
        else:
            overall_market_sentiment_string = "Neutral"

        final_df = merged_df.sort_values(by="growth_score", ascending=False).head(top_n)
        logger.info(f"Successfully prepared LLM context data (LSTM removed). Shape: {final_df.shape}")
        
        if 'current_price' not in final_df.columns:
            logger.warning("'current_price' column is missing from the merged DataFrame.")
            final_df['current_price'] = pd.NA

        # Ensure 'predicted_close_price' column (related to LSTM) is NOT present
        if 'predicted_close_price' in final_df.columns:
            final_df.drop(columns=['predicted_close_price'], inplace=True, errors='ignore')

        return final_df, overall_market_sentiment_string

    except FileNotFoundError as e:
        logger.error(f"Error loading data for LLM context: {e}. Please ensure all prerequisite scripts have run.")
        return pd.DataFrame(), "Neutral (Data Missing)"
    except Exception as e:
        logger.error(f"An unexpected error occurred while preparing LLM context data: {e}", exc_info=True)
        return pd.DataFrame(), "Neutral (Error)"


def generate_dynamic_prompt(user_query, top_companies_df, overall_market_sentiment_string):
    """
    Generates a dynamic prompt for the LLM based on user query, company data, and overall market sentiment.
    LSTM components have been removed from this version.
    """
    prompt_lines = [
        "You are a stock analyst assistant. Analyze the provided company data in the context of the user's query and the stated overall market sentiment."
    ]
    prompt_lines.append(f"\nUser Query: \"{user_query}\"\n")
    prompt_lines.append(f"Overall Market Sentiment: {overall_market_sentiment_string}\n")
    
    instructions = """Your primary role is to answer the user's query by providing a detailed stock and sector analysis for 'tomorrow'. 
    Focus on the following key aspects:
    1. Overall Market Sentiment: Summarize the general market mood based on the provided sentiment data.
    2. Sector Analysis: Highlight which sectors are showing strength or weakness (e.g. based on 'theme' and 'macro_sentiment_score').
    3. Company Analysis: For companies mentioned in the query or showing significant movement (present in the data), provide:
       - Sentiment analysis (positive/negative/neutral from 'positive', 'neutral', 'negative' columns)
       - Growth score and its implications (from 'growth_score' column)
       - Current price (from 'current_price' column)
       - Macro sentiment impact on the sector (from 'macro_sentiment_score' for the company's 'theme')
    4. Actionable Insights: Provide clear, data-driven recommendations based on the analysis.
    
    Do NOT mention LSTM predictions as they are not part of this analysis.
    Always consider the context of the user's query and provide specific, relevant information.
    If data is missing or unclear, acknowledge it and explain any limitations in the analysis."""

    prompt_lines.append("\nCompany Data Context:")
    if top_companies_df.empty:
        prompt_lines.append("No specific company data is available from the latest pipeline run or an error occurred loading it.")
    else:
        for _, row in top_companies_df.iterrows():
            company_name = row.get('company', 'N/A')
            macro_sentiment_val = row.get('macro_sentiment_score')
            macro_sentiment_str = f"{macro_sentiment_val:.2f}" if pd.notna(macro_sentiment_val) else "N/A"
            theme_str = row.get('theme', 'N/A')
            current_price_str = format_value_for_prompt(row.get('current_price'))

            line = (
                f"- {company_name}: GrowthScore={format_value_for_prompt(row.get('growth_score'))}, "
                f"CurrentPrice=${current_price_str}, " # Added CurrentPrice
                f"PositiveSentiment={format_value_for_prompt(row.get('positive'))}, NeutralSentiment={format_value_for_prompt(row.get('neutral'))}, "
                f"NegativeSentiment={format_value_for_prompt(row.get('negative'))}, "
                f"Sector={theme_str}, MacroSectorSentiment={macro_sentiment_str}"
            )
            
            # LSTM Prediction part COMPLETELY REMOVED
            prompt_lines.append(line)

    prompt_lines.append("\n---\nInstructions:\n" + instructions)
    return "\n".join(prompt_lines)


def get_openai_response(prompt_text):
    """
    Calls the OpenAI API with the given prompt and returns the response.
    """
    if not OPENAI_API_KEY:
        logger.error("OpenAI API key is not configured. Cannot query OpenAI.")
        return "Error: OpenAI API key not configured. Please set the OPENAI_API_KEY environment variable."

    logger.info("\n📡 Querying OpenAI (LSTM removed from prompt)...\n")
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a stock analyst assistant. You do not have access to LSTM model predictions."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.7,
            max_tokens=700 # Adjusted tokens, as prompt might be slightly shorter
        )
        result_text = response.choices[0].message.content
        logger.info(f"🔮 OpenAI Response received.")
        logger.debug(f"OpenAI Response content:\n{result_text}")
        return result_text
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}", exc_info=True)
        return f"Error communicating with OpenAI: {e}"

# Test execution block removed as it's not essential for the primary functionality.
# The main focus is on the functions used by Streamlit.
# To test, one would typically run the Streamlit app itself after these changes.
# And ensure data files (excluding LSTM related ones) are present.
```
