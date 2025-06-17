import pandas as pd
import spacy
from tqdm import tqdm
import os

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Load sentiment data
input_path = "data/merged_sentiment_input.csv"
output_path = "data/merged_sentiment_output_with_companies.csv"

df = pd.read_csv(input_path)

# Extract organization names (companies) using NER
companies = []

print("🔍 Extracting companies using spaCy NER...")
for headline in tqdm(df["headline"].fillna("")):
    doc = nlp(str(headline))
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    # Pick the first ORG as the primary company (or '' if none)
    companies.append(orgs[0] if orgs else "")

df["company"] = companies

# Save new file
os.makedirs("data", exist_ok=True)
df.to_csv(output_path, index=False)
print(f"✅ Saved with companies to → {output_path}")
