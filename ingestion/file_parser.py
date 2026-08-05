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
    """Extract observations from report text.

    Preferred path (when an LLM is available): parse the report's result
    *tables* into structured lab rows (any analyte, with reference ranges and
    H/L flags), ignoring threshold/footnote text. This is far more reliable on
    dense pathology PDFs than line-by-line regex. Falls back to regex metric
    extraction only when no lab rows are found or no provider is set.
    """
    if provider is not None and getattr(provider, "available",
                                        lambda: False)():
        from llm.agent import extract_lab_report
        labs = extract_lab_report(text, provider)
        if labs:
            return _labs_to_records(labs, person, f"file:{filename}", timestamp)
    # Fallback: chat-style regex extraction of the known metrics.
    return _from_text(text, person, filename, timestamp)


def _labs_to_records(labs: List[Dict], person: str, source: str,
                     timestamp: Optional[str]) -> List[Dict]:
    """Convert extracted lab rows into ontology-aligned observation records.

    A row that maps to a tracked metric (glucose, HbA1c, ...) becomes that
    metric's specific observation class so it flows into trends/reports; every
    other analyte becomes a generic phm:LaboratoryObservation carrying the
    report's own reference range and abnormal flag.
    """
    from ingestion.metrics import (REGISTRY, detect_context, normalize_unit,
                                    resolve_lab_metric)
    from datetime import datetime

    def num(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    records: List[Dict] = []
    for it in labs:
        test = str(it.get("test") or "").strip()
        value = num(it.get("value"))
        if not test or value is None:
            continue
        unit = it.get("unit")
        ts = it.get("collected") or timestamp or datetime.utcnow().isoformat()
        low, high = num(it.get("ref_low")), num(it.get("ref_high"))
        flag = (it.get("flag") or "").strip().upper() or None
        if flag not in ("H", "L", None):
            flag = None
        # infer flag from range if the report didn't print one
        if flag is None and (low is not None or high is not None):
            if low is not None and value < low:
                flag = "L"
            elif high is not None and value > high:
                flag = "H"

        metric = resolve_lab_metric(test)
        if metric:
            norm_v, norm_u = normalize_unit(metric, value, unit)
            mdef = REGISTRY[metric]
            rec = {
                "metric": metric, "type": mdef.ontology_class,
                "label": mdef.label, "category": mdef.category,
                "numericValue": norm_v, "unit": norm_u,
                "context": detect_context(test),
            }
        else:
            rec = {
                "metric": None, "type": "LaboratoryObservation",
                "label": test, "category": "lab",
                "numericValue": value, "unit": unit, "context": None,
            }
        rec.update({
            "observedFor": person, "source": source, "timestamp": ts,
            "raw_text": (f"{test} {value} {unit or ''}"
                         f" (ref {it.get('ref_text') or ''})").strip()[:280],
            "ref_low": low, "ref_high": high,
            "ref_text": it.get("ref_text"), "abnormal_flag": flag,
        })
        records.append(rec)
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
