import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('logs/llm_integration.log', mode='w')
file_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(file_handler)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter(log_format))
logger.addHandler(stream_handler)

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def generate_prompt(company_data):
    """
    Generates a dynamic prompt for the LLM based on the fused data.
    """
    prompt_lines = [
        "You are a stock analyst assistant. Analyze the provided company data and provide a direct, actionable stock recommendation."
    ]

    for index, row in company_data.iterrows():
        prompt_lines.append(f"\nCompany: {row['company']}")
        prompt_lines.append(f"Sentiment Score: {row['normalized_sentiment']:.2f}")
        prompt_lines.append(f"ML Prediction: {row['ml_prediction']}")
        prompt_lines.append(f"Final Score: {row['final_score']:.2f}")

    prompt_lines.append("\nInstructions:")
    prompt_lines.append("1. Provide a direct, actionable stock recommendation (e.g., 'buy', 'sell', 'hold').")
    prompt_lines.append("2. Justify your recommendation with both the sentiment and ML signals.")
    prompt_lines.append("3. Acknowledge any data limitations (e.g., missing news, stale prices).")
    prompt_lines.append("4. Your response must be in plain English, concise, and ready for the end user.")

    return "\n".join(prompt_lines)

def get_openai_response(prompt_text):
    """
    Calls the OpenAI API with the given prompt and returns the response.
    """
    if not OPENAI_API_KEY:
        logger.error("OpenAI API key is not configured. Cannot query OpenAI.")
        return "Error: OpenAI API key not configured. Please set the OPENAI_API_KEY environment variable."

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a stock analyst assistant."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.7,
            max_tokens=700
        )
        result_text = response.choices[0].message.content
        logger.info("OpenAI Response received.")
        return result_text
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}", exc_info=True)
        return f"Error communicating with OpenAI: {e}"

def main():
    """
    Main function to generate a prompt and get a response from the LLM.
    """
    try:
        fused_df = pd.read_csv("data/fused_scores.csv")
    except FileNotFoundError:
        logger.error("fused_scores.csv not found. Please run the context fusion pipeline first.")
        return

    top_companies = fused_df.head(3)
    prompt = generate_prompt(top_companies)
    response = get_openai_response(prompt)

    # Save the response
    with open("output/gpt_prediction.txt", "w") as f:
        f.write(response)

    logger.info("Successfully saved LLM response.")

if __name__ == "__main__":
    main()
