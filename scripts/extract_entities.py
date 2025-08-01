import pandas as pd
import spacy
from tqdm import tqdm
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.normalization_utils import normalize_company_name

# Load SpaCy English model
nlp = spacy.load("en_core_web_sm")

# Load sentiment data
input_path = "data/merged_sentiment_input.csv"
output_path = "data/merged_sentiment_output_with_companies.csv"

df = pd.read_csv(input_path)

# Extract organization names (companies) using NER and normalize
companies = []

print("🔍 Extracting and normalizing companies using spaCy NER...")
for headline in tqdm(df["headline"].fillna("")):
    doc = nlp(str(headline))
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    # Normalize the first ORG entity to a ticker/canonical name
    if orgs:
        normalized = normalize_company_name(orgs[0])
        companies.append(normalized if normalized else "")
    else:
        companies.append("")

df["company"] = companies

# Save new file
os.makedirs("data", exist_ok=True)
df.to_csv(output_path, index=False)
print(f"✅ Saved with companies to → {output_path}")
