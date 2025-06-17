import pandas as pd
import re
import numpy as np
from financial_keywords import extract_financial_keywords


class CountMinSketch:
    def __init__(self, width=500, depth=5):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width))
        self.hash_seeds = [17, 31, 43, 59, 73]

    def _hash(self, word, seed):
        return (hash(word) ^ seed) % self.width

    def add(self, word):
        for i in range(self.depth):
            index = self._hash(word, self.hash_seeds[i])
            self.table[i][index] += 1

    def estimate(self, word):
        return min(self.table[i][self._hash(word, self.hash_seeds[i])] for i in range(self.depth))

# Load data
df = pd.read_csv("data/merged_sentiment_output.csv")

# Extract top financial keywords
keywords_to_use = extract_financial_keywords(df["headline"], top_n=100)

# Initialize CMS
cms = CountMinSketch()

# Feed data into CMS
for _, row in df.iterrows():
    text = str(row.get("headline", "")) + " " + str(row.get("source", ""))
    words = re.findall(r'\w+', text.lower())
    for word in words:
        if word in keywords_to_use:
            cms.add(word)

# Estimate and print top keywords
estimates = [(word, cms.estimate(word)) for word in keywords_to_use]
top_keywords = sorted(estimates, key=lambda x: x[1], reverse=True)

print("📊 Top Keywords by CMS Count:")
for word, count in top_keywords[:15]:
    print(f"🔹 {word}: {int(count)}")
# Save CMS results
cms_df = pd.DataFrame(top_keywords, columns=["keyword", "count"])
cms_df.to_csv("data/cms_keywords.csv", index=False)
print("✅ CMS keywords saved → data/cms_keywords.csv")
