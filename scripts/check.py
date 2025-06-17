import pandas as pd

growth = pd.read_csv("data/predict_growth.csv")
sentiment = pd.read_csv("data/company_sentiment_normalized.csv")

growth_companies = set(growth["company"].unique())
sentiment_companies = set(sentiment["company"].unique())

print("Intersection:", growth_companies & sentiment_companies)
print("Growth only:", growth_companies - sentiment_companies)
print("Sentiment only:", sentiment_companies - growth_companies)
