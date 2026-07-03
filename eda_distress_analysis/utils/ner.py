import json

import requests
import spacy
from spacy.language import Language
from spacy.tokens import Span
from spacy.util import filter_spans


def _build_duckling_http_component(
    url: str,
    locale: str,
    dimensions: list[str],
    timeout_seconds: float = 3.0,
):
    """Create a spaCy component that enriches doc.ents using Duckling HTTP API."""
    label_map = {
        "percentage": "PERCENT",
        "ordinal": "ORDINAL",
        "amount-of-money": "MONEY",
        "number": "CARDINAL",
        "quantity": "QUANTITY",
        "time": "TIME",
        "duration": "DURATION",
        "distance": "DISTANCE",
        "temperature": "TEMPERATURE",
        "volume": "VOLUME",
    }
    warned = {"value": False}

    def _duckling_http_component(doc):
        if not doc.text or not doc.text.strip():
            return doc

        try:
            response = requests.post(
                url,
                data={
                    "text": doc.text,
                    "locale": locale,
                    "dims": json.dumps(dimensions),
                },
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            matches = response.json()
        except Exception as exc:
            # Only print once to avoid noisy notebook output when server is down.
            if not warned["value"]:
                print(f"Duckling HTTP unavailable; skipping enrichment ({exc})")
                warned["value"] = True
            return doc

        duckling_spans = []
        for match in matches:
            try:
                start = int(match.get("start"))
                end = int(match.get("end"))
                dim = (match.get("dim") or "").strip().lower()
                if not dim:
                    continue
                label = label_map.get(dim, dim.upper().replace("-", "_"))
                span = doc.char_span(start, end, label=label, alignment_mode="expand")
                if span is not None:
                    duckling_spans.append(span)
            except Exception:
                continue

        if duckling_spans:
            combined = list(doc.ents) + duckling_spans
            doc.ents = tuple(filter_spans(combined))

        return doc

    return _duckling_http_component


@Language.factory(
    "duckling_http",
    default_config={
        "url": "http://localhost:8000/parse",
        "locale": "de_DE",
        "dimensions": [
            "time",
            "number",
            "ordinal",
            "amount-of-money",
            "quantity",
            "percentage",
            "duration",
            "distance",
            "temperature",
            "volume",
        ],
        "timeout_seconds": 3.0,
    },
)
def create_duckling_http_component(nlp, name, url, locale, dimensions, timeout_seconds):
    return _build_duckling_http_component(
        url=url,
        locale=locale,
        dimensions=dimensions,
        timeout_seconds=timeout_seconds,
    )


def attach_duckling_http(
    nlp,
    url: str = "http://localhost:8000/parse",
    locale: str = "de_DE",
    dimensions: list[str] | None = None,
):
    """Attach Duckling HTTP enrichment component to a spaCy pipeline."""
    dims = dimensions or [
        "time",
        "number",
        "ordinal",
        "amount-of-money",
        "quantity",
        "percentage",
        "duration",
        "distance",
        "temperature",
        "volume",
    ]

    if "duckling_http" in nlp.pipe_names:
        nlp.remove_pipe("duckling_http")

    nlp.add_pipe(
        "duckling_http",
        config={
            "url": url,
            "locale": locale,
            "dimensions": dims,
            "timeout_seconds": 3.0,
        },
        last=True,
    )
    return nlp


def build_nlp_models(en_model: str, de_model: str):
    """Load English and German spaCy models and attach Duckling HTTP enrichment to German."""
    en_nlp = spacy.load(en_model)
    de_nlp = spacy.load(de_model)

    try:
        attach_duckling_http(de_nlp, url="http://localhost:8000/parse", locale="de_DE")
        print("Duckling HTTP attached to German model at http://localhost:8000/parse")
    except Exception as exc:
        print(f"Failed to attach Duckling HTTP component; continuing with spaCy only ({exc})")

    return en_nlp, de_nlp
