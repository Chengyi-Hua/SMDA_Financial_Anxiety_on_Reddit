import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Asset directory
ASSET_DIR = os.getenv('ASSET_DIR')
KEYWORD_DIR = os.getenv('KEYWORD_DIR')

# CSV file paths
CSV_EN_2020 = os.getenv('CSV_EN_2020')
CSV_EN_2025 = os.getenv('CSV_EN_2025')
CSV_DE_2020 = os.getenv('CSV_DE_2020')
CSV_DE_2025 = os.getenv('CSV_DE_2025')
CSV_POST_COUNT = os.getenv('CSV_POST_COUNT')
CSV_FDI = os.getenv('CSV_FDI')

#JSON file paths
JSON_DISTRESS_EN = os.getenv('JSON_DISTRESS_EN')
JSON_DISTRESS_DE = os.getenv('JSON_DISTRESS_DE')

# Reddit API parameters for post count fetching
SUBREDDITS = ["personalfinance", "finanzen"]
YEARS = [2020, 2021, 2022, 2023, 2024, 2025]
BASE_URL = "https://arctic-shift.photon-reddit.com/api/posts/search"

# SpaCy models
EN_MODEL = os.getenv('EN_MODEL')
DE_MODEL = os.getenv('DE_MODEL')
NER_DICT = {
  "DATE": "TIME",
  "TIME": "TIME",
  "CARDINAL": "CARDINAL",
  "ORG": "ORG",
  "MONEY": "MONEY",
  "LOC": "GPE",
  "GPE": "GPE",
  "PER": "PERSON",
  "PERSON": "PERSON",
  "ORDINAL": "ORDINAL",
}
# NER_DICT_DE = {
#   "TIME": "TIME",
#   "CARDINAL": "CARDINAL",
#   "ORG": "ORG",
#   "MONEY": "MONEY",
#   "LOC": "GPE",
#   "PER": "PERSON",
#   "ORDINAL": "ORDINAL"
# }

# Analysis parameters
SAMPLE_SIZE = int(os.getenv('SAMPLE_SIZE'))
