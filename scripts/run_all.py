import subprocess
import sys
import os

SCRIPTS = [
    "fetch_news.py",
    "fetch_reddit.py",
    "merge_sources.py",
    "extract_entities.py",
    "normalize_companies.py",
    "sentiment_analysis.py",
    "cms_keyword_counter.py",
    "edw_keyword_weighter.py",
    "predict_growth.py",
    "generate_company_sentiment.py",
    "macro_sentiment_analysis.py",
    "fetch_historical_prices.py",
    # "lstm_stock_predictor.py", # Removed as LSTM components are being deleted
    "stock_predictor.py"
]

def run_script(script):
    print(f"\n🚀 Running {script} ...")
    result = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), script)])
    if result.returncode != 0:
        print(f"❌ Error running {script}. Stopping pipeline.")
        sys.exit(result.returncode)
    print(f"✅ {script} completed.")

def main():
    print("===== Sentimental Stock Predictor Pipeline Runner =====")
    for script in SCRIPTS:
        run_script(script)
    print("\n🎉 All steps finished! Check output/gpt_prediction.txt for the final result.")

if __name__ == "__main__":
    main()
