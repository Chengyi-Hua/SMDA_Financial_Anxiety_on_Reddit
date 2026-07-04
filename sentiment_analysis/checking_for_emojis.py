import pandas as pd
import re
from pathlib import Path

DATA_DIR = Path(r"SMDA_Financial_Anxiety_on_Reddit\data")

files = [
    "finanzen_2020_final.csv",
    "finanzen_2025_final_with_emo.csv",
    "personalfinance_2020_final_with_emo.csv",
    "personalfinance_2025_final.csv",
]

emoji_pattern = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "]+",
    flags=re.UNICODE
)

for file in files:
    df = pd.read_csv(DATA_DIR / file)
    texts = df["text_for_sentiment"].fillna("").astype(str)
    emoji_count = texts.apply(lambda x: len(emoji_pattern.findall(x))).sum()
    posts_with_emoji = texts.apply(lambda x: bool(emoji_pattern.search(x))).sum()
    
    print(file)
    print("rows:", len(df))
    print("posts with emoji:", posts_with_emoji)
    print("emoji matches:", emoji_count)
    print()