"""File ingestion: PDF, CSV and images.

Extracts text (and, for CSV, structured rows) from uploaded health reports and
hands the text to the rule-based extractor. Heavy/optional dependencies
(pdfplumber, pytesseract/Pillow) are imported lazily so the app still runs when
they are absent — the UI simply reports that the format is unavailable.
"""

from __future__ import annotations

import hashlib
import io
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .extractor import extract_observations
from .metrics import REGISTRY, normalize_unit


def file_hash(data: bytes) -> str:
    """Stable content hash used for duplicate-upload detection."""
    return hashlib.sha256(data).hexdigest()[:16]


def parse_file(filename: str, data: bytes, person: str = "self",
               timestamp: Optional[str] = None,
               provider=None) -> Tuple[List[Dict], str]:
    """Dispatch on file type. Returns (observations, extracted_text).

    For text-bearing formats (PDF/image/text), if an LLM `provider` is given
    and available, the report text is also read by the model — catching
    measurements the regex layer misses in messy lab reports — and merged with
    the rule-based extraction. CSV stays purely structured/deterministic.
    """
    name = filename.lower()
    if name.endswith(".csv"):
        return _parse_csv(data, person, timestamp)
    if name.endswith(".pdf"):
        text = _parse_pdf(data)
        return _extract(text, person, filename, timestamp, provider), text
    if name.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
        text = _parse_image(data)
        return _extract(text, person, filename, timestamp, provider), text
    if name.endswith((".txt", ".md")):
        text = data.decode("utf-8", errors="ignore")
        return _extract(text, person, filename, timestamp, provider), text
    raise ValueError(f"Unsupported file type: {filename}")


def _extract(text: str, person: str, filename: str,
             timestamp: Optional[str], provider) -> List[Dict]:
    """Merge LLM document extraction (if available) with regex extraction."""
    records = _from_text(text, person, filename, timestamp)
    if provider is not None and getattr(provider, "available",
                                        lambda: False)():
        from llm.agent import extract_document
        from ingestion.metrics import REGISTRY
        llm_records = extract_document(text, provider, list(REGISTRY.keys()),
                                       person=person, source=f"file:{filename}")
        if llm_records:
            seen = {(r["metric"], r["numericValue"]) for r in records}
            for r in llm_records:
                if (r["metric"], r["numericValue"]) not in seen:
                    records.append(r)
    return records


def _from_text(text: str, person: str, filename: str,
               timestamp: Optional[str]) -> List[Dict]:
    records: List[Dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for rec in extract_observations(line, person=person,
                                        source=f"file:{filename}",
                                        timestamp=timestamp):
            records.append(rec)
    return records


def _parse_pdf(data: bytes) -> str:
    try:
        import pdfplumber
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PDF support needs pdfplumber (`pip install pdfplumber`)."
        ) from exc
    out = []
    with pdfplumber.open(io.BytesIO(data)) as pdf:
        for page in pdf.pages:
            out.append(page.extract_text() or "")
    return "\n".join(out)


def _parse_image(data: bytes) -> str:
    try:
        import pytesseract
        from PIL import Image
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Image OCR needs pytesseract + Pillow and a tesseract binary."
        ) from exc
    img = Image.open(io.BytesIO(data))
    return pytesseract.image_to_string(img)


def _parse_csv(data: bytes, person: str,
               timestamp: Optional[str]) -> Tuple[List[Dict], str]:
    """Parse a CSV of readings.

    Supports two shapes:
      * long form with columns like metric,value,unit,timestamp
      * wide form with one column per metric (glucose, systolic, ...)
    Falls back to line-by-line text extraction if neither matches.
    """
    df = pd.read_csv(io.BytesIO(data))
    cols = {c.lower().strip(): c for c in df.columns}
    records: List[Dict] = []

    def resolve_metric(name: str) -> Optional[str]:
        name = name.lower().strip()
        if name in REGISTRY:
            return name
        for key, mdef in REGISTRY.items():
            if name in [s.lower() for s in mdef.synonyms]:
                return key
        return None

    # Long form
    if "metric" in cols and ("value" in cols or "reading" in cols):
        val_col = cols.get("value") or cols.get("reading")
        for _, row in df.iterrows():
            key = resolve_metric(str(row[cols["metric"]]))
            if not key or pd.isna(row[val_col]):
                continue
            unit = str(row[cols["unit"]]) if "unit" in cols else None
            ts = (str(row[cols["timestamp"]]) if "timestamp" in cols
                  else timestamp or datetime.utcnow().isoformat())
            records.append(_csv_record(key, float(row[val_col]), unit,
                                       person, ts))
        return records, df.to_csv(index=False)

    # Wide form: match columns to known metrics
    ts_col = cols.get("timestamp") or cols.get("date") or cols.get("time")
    metric_cols = {c: resolve_metric(c) for c in cols}
    metric_cols = {c: k for c, k in metric_cols.items() if k}
    if metric_cols:
        for _, row in df.iterrows():
            ts = (str(row[cols[ts_col]]) if ts_col
                  else timestamp or datetime.utcnow().isoformat())
            for col_lower, key in metric_cols.items():
                raw = row[cols[col_lower]]
                if pd.isna(raw):
                    continue
                records.append(_csv_record(key, float(raw), None, person, ts))
        return records, df.to_csv(index=False)

    # Fallback: treat as text
    text = df.to_csv(index=False)
    return _from_text(text, person, "upload.csv", timestamp), text


def _csv_record(metric_key: str, value: float, unit: Optional[str],
                person: str, ts: str) -> Dict:
    mdef = REGISTRY[metric_key]
    norm_value, norm_unit = normalize_unit(metric_key, value, unit)
    return {
        "metric": metric_key,
        "type": mdef.ontology_class,
        "label": mdef.label,
        "category": mdef.category,
        "numericValue": norm_value,
        "unit": norm_unit,
        "observedFor": person,
        "source": "file:csv",
        "timestamp": ts,
        "raw_text": f"{metric_key}={value}{unit or ''}",
    }
