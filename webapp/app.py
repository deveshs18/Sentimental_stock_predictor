# Initial Flask app setup
from flask import Flask, jsonify, render_template, request
import pandas as pd
import os
import json

app = Flask(__name__)

@app.route('/api/market_sentiment')
def api_market_sentiment():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'macro_sentiment.csv')
    try:
        df = pd.read_csv(csv_path)
        # Assuming 'date' and 'sentiment_score' are relevant columns
        # Adjust column names as per the actual CSV
        if 'date' in df.columns and 'sentiment_score' in df.columns:
            labels = df['date'].tolist()
            data = df['sentiment_score'].tolist()
            return jsonify(labels=labels, data=data)
        else:
            return jsonify(error="Relevant columns (e.g., 'date', 'sentiment_score') not found in CSV"), 500
    except FileNotFoundError:
        return jsonify(error="Data file not found"), 404
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/api/stock_data')
def api_stock_data():
    csv_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'company_sentiment_normalized.csv')
    try:
        df = pd.read_csv(csv_path)
        companies_query = request.args.get('companies')

        # Ensure required columns exist
        required_cols = ['date', 'company_name', 'sentiment_score'] # Add 'stock_price' if available
        if not all(col in df.columns for col in required_cols):
            return jsonify(error="Required columns (date, company_name, sentiment_score) not found in CSV"), 500

        if companies_query:
            selected_companies = [company.strip() for company in companies_query.split(',')]
            df = df[df['company_name'].isin(selected_companies)]
        else:
            # Optional: return data for all companies or a predefined top N
            # For now, returning all if no specific companies are requested
            pass

        if df.empty and companies_query:
            return jsonify(error=f"No data found for specified companies: {companies_query}"), 404

        output_data = []
        for company_name, group in df.groupby('company_name'):
            # Ensure date is sorted if not already
            group = group.sort_values(by='date')
            company_data = {
                "name": company_name,
                "dates": group['date'].tolist(),
                "sentiment_scores": group['sentiment_score'].tolist(),
                # "stock_prices": group['stock_price'].tolist() # Uncomment if stock_price column exists
            }
            # if 'stock_price' in group.columns: # Add stock prices if column exists
            #     company_data["stock_prices"] = group['stock_price'].tolist()
            output_data.append(company_data)

        return jsonify(output_data)

    except FileNotFoundError:
        return jsonify(error="Company sentiment data file not found"), 404
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/api/news')
def api_news():
    news_file_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'company_news.json')
    try:
        with open(news_file_path, 'r') as f:
            news_data = json.load(f)
        # Here you could add processing, e.g. selecting specific fields, filtering, etc.
        # For now, returning all articles.
        # Example: Filter for specific keys if needed
        # processed_articles = []
        # for article in news_data:
        #     processed_articles.append({
        #         "title": article.get("title"),
        #         "snippet": article.get("snippet"),
        #         "source": article.get("source"),
        #         "date": article.get("date"), # Ensure date format is consistent if used for sorting
        #         "url": article.get("url"),
        #         "company_name": article.get("company_name") # Assuming this field exists
        #     })
        return jsonify(news_data)
    except FileNotFoundError:
        return jsonify(error="News data file not found"), 404
    except json.JSONDecodeError:
        return jsonify(error="Invalid JSON format in news data file"), 500
    except Exception as e:
        return jsonify(error=str(e)), 500

@app.route('/')
def dashboard():
    return render_template("dashboard.html")

@app.route('/news')
def news_page():
    return render_template("news.html")

if __name__ == '__main__':
    app.run(debug=True)
