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

# Analysis parameters
SAMPLE_SIZE = int(os.getenv('SAMPLE_SIZE'))

# LLM key
HUGGING_FACE_API_KEY = os.getenv('HUGGING_FACE_API_KEY')

# HPC GPU optimization parameters
RANDOM_STATE = 42
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME')
QWEN_MODEL_ID = os.getenv('QWEN_MODEL_ID')
TARGET_TOPIC_MIN = 10
TARGET_TOPIC_MAX = 15

# HPC GPU optimization parameters (can be overridden via env)
GPU_DEVICE = int(os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
BATCH_SIZE_EMBEDDING = 128
UMAP_N_JOBS = -1
EMBEDDING_SHOW_PROGRESS = True

# Lightweight German stopword list to avoid a spaCy dependency.
GERMAN_STOP_WORDS = {
  "aber", "als", "am", "an", "auch", "auf", "aus", "bei", "bin", "bis",
  "bist", "da", "dadurch", "daher", "darum", "das", "daß", "dass", "dein",
  "deine", "dem", "den", "der", "des", "dessen", "deshalb", "die", "dies",
  "dieser", "dieses", "doch", "dort", "du", "durch", "ein", "eine", "einem",
  "einen", "einer", "eines", "er", "es", "euer", "eure", "für", "hat", "hatte",
  "haben", "heute", "hier", "hinter", "ich", "ihr", "ihre", "im", "in", "ist",
  "ja", "jede", "jedem", "jeden", "jeder", "jedes", "jener", "jenes", "jetzt",
  "kann", "kannst", "können", "könnt", "man", "mehr", "mit", "muss", "musst",
  "müssen", "müsst", "nach", "nicht", "nichts", "noch", "nun", "nur", "ob", "oder",
  "ohne", "sehr", "sein", "seine", "sich", "sie", "sind", "so", "solche", "solcher",
  "sondern", "stark", "sowie", "um", "und", "uns", "unser", "unter", "vom", "von",
  "vor", "wann", "war", "waren", "warst", "was", "weg", "weil", "weiter", "welche",
  "welchem", "welchen", "welcher", "welches", "wenn", "wer", "werde", "werden",
  "wie", "wieder", "wir", "wird", "wirst", "wo", "zu", "zum", "zur", "über",
}