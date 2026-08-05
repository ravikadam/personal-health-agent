"""Entity/observation extraction from free text.

Rule-based extractor (regex + synonym registry) that turns natural-language
health notes into ontology-aligned observation dicts. Works fully offline. If
an Anthropic API key is present, `llm_extract` can be used as an assist for
messier text, but the rule-based path is always the default so the app runs
with no external dependencies.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Dict, List, Optional

from .metrics import REGISTRY, normalize_unit, synonym_index

# Pre-compiled patterns -------------------------------------------------------

# Blood pressure like "120/80", "BP 120/80 mmHg"
_BP_RE = re.compile(
    r"(?:\bbp\b|\bblood\s*pressure\b)?\s*(\d{2,3})\s*/\s*(\d{2,3})",
    re.IGNORECASE,
)

_NUM = r"(\d+(?:\.\d+)?)"
_UNIT = r"([a-zA-Z%/]+(?:/[a-zA-Z]+)?)?"

_SYN_INDEX = synonym_index()


def _detect_datetime(text: str, default: datetime) -> datetime:
    """Very light relative-date handling (today/yesterday)."""
    low = text.lower()
    if "yesterday" in low:
        return default.replace(hour=12, minute=0, second=0, microsecond=0) \
            .fromtimestamp(default.timestamp() - 86400)
    return default


def extract_observations(
    text: str,
    person: str = "self",
    source: str = "chat",
    timestamp: Optional[str] = None,
) -> List[Dict]:
    """Extract observation records from a line/paragraph of text.

    Each record is an ontology-aligned dict (not yet an ontology-validated
    entity — that happens in the mapper/store).
    """
    now = datetime.utcnow()
    ts = timestamp or now.isoformat()
    records: List[Dict] = []
    low = text.lower()
    consumed_spans: List[tuple] = []

    def overlaps(start: int, end: int) -> bool:
        return any(not (end <= s or start >= e) for s, e in consumed_spans)

    # 1) Blood pressure (two numbers) -- handle before single-number metrics
    for m in _BP_RE.finditer(text):
        # Require some BP context to avoid matching e.g. dates or ratios
        window = low[max(0, m.start() - 12):m.start()]
        if ("bp" in window or "pressure" in window
                or "/" in m.group(0)) and _looks_like_bp(m):
            sys_v, dia_v = float(m.group(1)), float(m.group(2))
            for key, val in (("systolic", sys_v), ("diastolic", dia_v)):
                records.append(_make_record(key, val, "mmHg", person, source,
                                             ts, text))
            consumed_spans.append((m.start(), m.end()))
            break  # one BP reading per line is the common case

    # 2) Synonym-anchored single-value metrics
    for syn, key in _SYN_INDEX:
        if key in ("systolic", "diastolic"):
            continue  # handled by BP branch
        for sm in re.finditer(re.escape(syn), low):
            # look for a number near the synonym (after preferred, else before)
            after = low[sm.end():sm.end() + 25]
            before = low[max(0, sm.start() - 15):sm.start()]
            num_match = re.search(_NUM + r"\s*" + _UNIT, after)
            span_offset = sm.end()
            if not num_match:
                num_match = re.search(_NUM + r"\s*" + _UNIT + r"\s*$", before)
                span_offset = max(0, sm.start() - 15)
            if not num_match:
                continue
            gstart = span_offset + num_match.start()
            gend = span_offset + num_match.end()
            if overlaps(gstart, gend) or overlaps(sm.start(), sm.end()):
                continue
            value = float(num_match.group(1))
            unit = num_match.group(2)
            records.append(_make_record(key, value, unit, person, source, ts,
                                        text))
            consumed_spans.append((sm.start(), gend))
            break  # first occurrence of this metric per line

    return records


def _looks_like_bp(m) -> bool:
    a, b = int(m.group(1)), int(m.group(2))
    return 70 <= a <= 260 and 40 <= b <= 160 and a > b


def _make_record(metric_key: str, value: float, unit: Optional[str],
                 person: str, source: str, ts: str, raw: str) -> Dict:
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
        "source": source,
        "timestamp": ts,
        "raw_text": raw.strip()[:280],
    }


# --- Optional LLM assist (provider-agnostic) --------------------------------

def llm_extract(text: str, provider=None, person: str = "self",
                source: str = "chat") -> List[Dict]:
    """LLM-assisted extraction using any configured provider.

    `provider` is an `llm.base.LLMProvider`. Falls back to the rule-based
    extractor when no provider is available or on any error, so extraction is
    never worse than the deterministic path. The extraction prompt is grounded
    in the ontology's observation classes so the model maps to ontology terms.
    """
    if provider is None or not getattr(provider, "available", lambda: False)():
        return extract_observations(text, person, source)
    try:
        from ontology.grounding import build_llm_context

        metrics = ", ".join(REGISTRY.keys())
        system = (
            build_llm_context(list(REGISTRY.keys())) +
            "\n\nTASK: Extract health metrics from the user's note. Return ONLY "
            "a JSON array of objects with keys: metric (one of: "
            f"{metrics}), value (number), unit (string, optional)."
        )
        payload = provider.extract_json(system, text)
        if not isinstance(payload, list):
            return extract_observations(text, person, source)
        ts = datetime.utcnow().isoformat()
        out: List[Dict] = []
        for item in payload:
            key = item.get("metric")
            if key in REGISTRY and item.get("value") is not None:
                out.append(_make_record(key, float(item["value"]),
                                        item.get("unit"), person, source, ts,
                                        text))
        # Merge with rule-based to catch anything the model missed.
        rule = extract_observations(text, person, source)
        seen = {(r["metric"], r["numericValue"]) for r in out}
        for r in rule:
            if (r["metric"], r["numericValue"]) not in seen:
                out.append(r)
        return out or rule
    except Exception:
        return extract_observations(text, person, source)
