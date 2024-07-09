import numpy as np
import pandas as pd
import re


def detect_price_mentions(comment):
    regex_patterns = [
        r'[rR]\$\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?',# Brazilian Real
        r"\$\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?",  # US Dollar and similar formats
        r'ars\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?', # Argentine Peso
        r'clp\s*\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?', # Chilean Peso
        r'\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\s*usd', # US Dollar
        r'\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\s*[rR]\$', # Brazilian Real
        r'\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\s*ars', # Argentine Peso
        r'\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?\s*clp', # Chilean Peso
    ]
    
    combined_pattern = '|'.join(regex_patterns)
    
    matches = re.findall(combined_pattern, comment)
    
    def to_float(value):
        return float(value.replace(',', '.'))

    if len(matches) > 0:
        #get only values without currency
        matches = list(map(to_float, [re.findall(r'\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?', match)[0] for match in matches]))

        return matches
    else:
        return None

if __name__ == "__main__":
    comment = input("Enter a comment: ").strip().lower()
    price_mentions = detect_price_mentions(comment)
    if price_mentions:
        print(f"Price mentions found: {price_mentions}")
    else:
        print("No price mentions found.")