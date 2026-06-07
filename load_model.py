#!/usr/bin/env python3
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEFAULT_REPO = "Amirhossein75/multi-label-emotion-classification-reddit-comments-roberta"

def _extract_labels(config):
    id2label = getattr(config, "id2label", None)
    if not id2label:
        return None
    try:
        keys_sorted = sorted(id2label.keys(), key=lambda k: int(k))
    except Exception:
        keys_sorted = sorted(id2label.keys())
    return [id2label[k] for k in keys_sorted]

def load_model(repo_id: str = DEFAULT_REPO, device: str | None = None):
    """Load tokenizer and model from Hugging Face.

    Returns: (tokenizer, model, labels, device)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForSequenceClassification.from_pretrained(repo_id)
    model.to(device)
    labels = _extract_labels(model.config)
    return tokenizer, model, labels, device

def predict(texts, tokenizer, model, device: str | None = None, threshold: float = 0.5):
    """Run multi-label prediction. Returns list of results per input.

    Each result is a dict with `predictions` (all labels with scores sorted desc)
    and `labels_over_threshold` (labels with score >= threshold).
    """
    if isinstance(texts, str):
        texts = [texts]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    enc = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**enc)
        logits = outputs.logits
        probs = torch.sigmoid(logits)
    probs = probs.cpu()
    labels = _extract_labels(model.config)
    results = []
    for row in probs:
        row_res = []
        for i, p in enumerate(row.tolist()):
            label = labels[i] if labels and i < len(labels) else str(i)
            row_res.append({"label": label, "score": float(p)})
        row_res_sorted = sorted(row_res, key=lambda x: x["score"], reverse=True)
        results.append({
            "predictions": row_res_sorted,
            "labels_over_threshold": [r for r in row_res_sorted if r["score"] >= threshold],
        })
    return results


if __name__ == "__main__":
    tokenizer, model, labels, device = load_model()
    sample = "I feel anxious about money and bills this month."
    res = predict(sample, tokenizer, model, device=device, threshold=0.3)
    print(f"Input: {sample}\n")
    for r in res:
        print("Top predictions:")
        for p in r["predictions"][:8]:
            print(f"  {p['label']}: {p['score']:.4f}")
        print("\nLabels above threshold:")
        for p in r["labels_over_threshold"]:
            print(f"  {p['label']}: {p['score']:.4f}")
