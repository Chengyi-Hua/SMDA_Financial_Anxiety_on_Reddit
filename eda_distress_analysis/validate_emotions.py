#!/usr/bin/env python3
"""
validate_emotions.py
====================
Reads each asset/distress_*_with_scores.csv file, runs the appropriate
emotion model on `text_for_sentiment`, and checks whether the model's
predictions agree with the LIWC-based distress labels:

  distress_score_Anxiety  -> model label: nervousness
  distress_score_Anger    -> model label: anger
  distress_score_Sad      -> model label: sadness

Models used:
  English files -> Amirhossein75/multi-label-emotion-classification-reddit-comments-roberta
                   (RoBERTa-base, GoEmotions 28-class, English Reddit)
  German files  -> ChrisLalk/German-Emotions
                   (XLM-RoBERTa-base, GoEmotions 28-class, German translation)

Both models use the same 28 GoEmotions labels — no special patching needed for
the German model (its config.json already has real emotion names).
"""

import glob
import os
import sys
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(__file__))
from load_model import load_model, predict, _extract_labels

# ── Model repos ───────────────────────────────────────────────────────────────
REPO_ENGLISH = "Amirhossein75/multi-label-emotion-classification-reddit-comments-roberta"
REPO_GERMAN  = "ChrisLalk/German-Emotions"

# ── GoEmotions 28-class label list (fallback if config only has LABEL_N) ──────
# Alphabetical order matching the standard GoEmotions dataset.
GO_EMOTIONS_LABELS = [
    "admiration",    # 0
    "amusement",     # 1
    "anger",         # 2
    "annoyance",     # 3
    "approval",      # 4
    "caring",        # 5
    "confusion",     # 6
    "curiosity",     # 7
    "desire",        # 8
    "disappointment",# 9
    "disapproval",   # 10
    "disgust",       # 11
    "embarrassment", # 12
    "excitement",    # 13
    "fear",          # 14
    "gratitude",     # 15
    "grief",         # 16
    "joy",           # 17
    "love",          # 18
    "nervousness",   # 19
    "optimism",      # 20
    "pride",         # 21
    "realization",   # 22
    "relief",        # 23
    "remorse",       # 24
    "sadness",       # 25
    "surprise",      # 26
    "neutral",       # 27
]

# ── Parameters ────────────────────────────────────────────────────────────────
SAMPLE_SIZE  = 200   # rows per file (None = all flagged rows)
THRESHOLD    = 0.05   # emotion model confidence threshold
BATCH_SIZE   = 32    # texts per forward pass
RANDOM_SEED  = 42
ASSET_DIR    = os.path.join(os.path.dirname(__file__), "asset")

# Mapping: distress column -> GoEmotions label
DISTRESS_TO_EMOTION = {
    "distress_score_Anxiety": "nervousness",
    "distress_score_Anger":   "anger",
    "distress_score_Sad":     "sadness",
}

# ── Lazy model cache (avoid reloading same model between files) ───────────────
_model_cache: dict = {}   # repo_id -> (tokenizer, model, labels, device)


def get_model(repo_id: str):
    """Load model once; return cached instance on subsequent calls."""
    if repo_id not in _model_cache:
        print(f"\n  Loading model: {repo_id} …")
        tok, mdl, lbls, dev = load_model(repo_id=repo_id)

        # Patch generic LABEL_N names with GoEmotions names if needed
        if lbls and all(l.startswith("LABEL_") for l in lbls):
            print(f"  Detected generic labels → applying GoEmotions mapping.")
            mdl.config.id2label = {str(i): n for i, n in enumerate(GO_EMOTIONS_LABELS)}
            mdl.config.label2id = {n: i for i, n in enumerate(GO_EMOTIONS_LABELS)}
            lbls = GO_EMOTIONS_LABELS

        print(f"  Model ready on {dev}. Labels: {lbls[:5]} … {lbls[-3:]}")
        _model_cache[repo_id] = (tok, mdl, lbls, dev)

    return _model_cache[repo_id]


def detect_language(filename: str) -> str:
    """Infer language from filename: 'german' or 'english'."""
    name = os.path.basename(filename).lower()
    if "german" in name:
        return "german"
    return "english"


def choose_repo(language: str) -> str:
    return REPO_GERMAN if language == "german" else REPO_ENGLISH


# ── Prediction helpers ────────────────────────────────────────────────────────

def run_predictions(texts: list[str], tokenizer, model, device: str) -> list[dict]:
    """Batch-predict all texts; return list of result dicts."""
    results = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        results.extend(predict(batch, tokenizer, model, device=device, threshold=THRESHOLD))
    return results


def is_predicted(result: dict, emotion_label: str) -> bool:
    return any(r["label"].lower() == emotion_label.lower()
               for r in result["labels_over_threshold"])


def get_score(result: dict, emotion_label: str) -> float:
    for r in result["predictions"]:
        if r["label"].lower() == emotion_label.lower():
            return r["score"]
    return 0.0


# ── Per-file validation ───────────────────────────────────────────────────────

def validate_file(csv_path: str):
    file_stem = os.path.splitext(os.path.basename(csv_path))[0]
    language  = detect_language(csv_path)
    repo_id   = choose_repo(language)

    print(f"\n{'='*65}")
    print(f"File    : {os.path.basename(csv_path)}")
    print(f"Language: {language}  |  Repo: {repo_id}")

    tokenizer, model, labels, device = get_model(repo_id)

    df = pd.read_csv(csv_path, low_memory=False)

    # Keep only rows with at least one distress flag > 0
    distress_cols = list(DISTRESS_TO_EMOTION.keys())
    flag_mask  = (df[distress_cols] > 0.05).any(axis=1)
    df_flagged = df[flag_mask].copy()
    print(f"  Total rows      : {len(df):,}")
    print(f"  Rows with flags : {len(df_flagged):,}")

    # Sample
    # if SAMPLE_SIZE and len(df_flagged) > SAMPLE_SIZE:
    #     df_sample = df_flagged.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED)
    # else:
    df_sample = df_flagged.copy()
    print(f"  Sample used     : {len(df_sample):,}")

    # Drop empty texts
    df_sample = df_sample.dropna(subset=["text_for_sentiment"]).copy()
    df_sample = df_sample[df_sample["text_for_sentiment"].str.strip() != ""].copy()
    texts = df_sample["text_for_sentiment"].tolist()
    print(f"  Valid texts     : {len(texts):,}")

    print("  Running predictions …")
    results = run_predictions(texts, tokenizer, model, device)
    df_sample = df_sample.reset_index(drop=True)

    # Attach model scores / predictions
    for dim, emo in DISTRESS_TO_EMOTION.items():
        df_sample[f"model_{emo}_score"]     = [get_score(r, emo)     for r in results]
        df_sample[f"model_{emo}_predicted"] = [is_predicted(r, emo)  for r in results]

    # ── F1 table (model = ground truth, LIWC flag = predicted) ──────────────
    # For each emotion dimension:
    #   y_true = model_predicted   (1 if model detects the emotion)
    #   y_pred = LIWC flag         (1 if LIWC keyword score > 0)
    # Binary F1 is computed for the positive class (emotion present).
    # Macro F1 = unweighted average across all 3 dimensions.
    print(f"\n  {'Dimension':<25} {'Flag=1':>7} {'Model=1':>7} {'TP':>5}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}")
    print(f"  {'-'*70}")
    summary_rows = []
    f1_scores = []
    for dim, emo in DISTRESS_TO_EMOTION.items():
        y_true = df_sample[f"model_{emo}_predicted"].astype(int).values
        y_pred = (df_sample[dim] > 0).astype(int).values

        n_model = int(y_true.sum())
        n_flag  = int(y_pred.sum())
        tp      = int((y_true & y_pred).sum())

        f1   = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        f1_scores.append(f1)

        print(f"  {dim:<25} {n_flag:>7} {n_model:>7} {tp:>5}  {prec:>6.3f}  {rec:>6.3f}  {f1:>6.3f}")
        summary_rows.append({
            "file":          file_stem,
            "language":      language,
            "model":         repo_id,
            "dimension":     dim,
            "emotion_label": emo,
            "n_liwc_flagged":    n_flag,
            "n_model_predicted": n_model,
            "tp":            tp,
            "precision":     round(prec, 4),
            "recall":        round(rec, 4),
            "f1":            round(f1, 4),
        })

    macro_f1 = sum(f1_scores) / len(f1_scores)
    print(f"  {'─'*70}")
    print(f"  {'Macro F1 (3 dims)':<25} {'':>7} {'':>7} {'':>5}  {'':>6}  {'':>6}  {macro_f1:>6.3f}")

    # Save detailed CSV
    out_path  = os.path.join(ASSET_DIR, f"validation_results_{file_stem}.csv")
    keep_cols = (
        ["created_utc", "id", "text_for_sentiment"]
        + distress_cols
        + [f"model_{e}_score"     for e in DISTRESS_TO_EMOTION.values()]
        + [f"model_{e}_predicted" for e in DISTRESS_TO_EMOTION.values()]
    )
    df_out = df_sample[[c for c in keep_cols if c in df_sample.columns]]
    df_out.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")

    return summary_rows


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    csv_files = sorted(glob.glob(os.path.join(ASSET_DIR, "distress_*_with_scores.csv")))
    if not csv_files:
        print("No distress_*_with_scores.csv files found in asset/")
        sys.exit(1)

    all_summary = []
    for csv_path in csv_files:
        rows = validate_file(csv_path)
        all_summary.extend(rows)

    summary_df   = pd.DataFrame(all_summary)
    summary_path = os.path.join(ASSET_DIR, "validation_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("OVERALL SUMMARY")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved → {summary_path}")
