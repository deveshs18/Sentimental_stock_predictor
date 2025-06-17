import nltk
from nltk.corpus import stopwords
import re
from collections import Counter

nltk.download('stopwords')

def extract_financial_keywords(text_series, top_n=100):
    stop_words = set(stopwords.words('english'))
    all_words = []

    for text in text_series.dropna():
        tokens = re.findall(r'\w+', text.lower())
        filtered = [t for t in tokens if t not in stop_words and len(t) > 3]
        all_words.extend(filtered)

    # Get most common meaningful words
    counter = Counter(all_words)
    return set(word for word, count in counter.most_common(top_n))
