"""Ontology-aware retrieval.

Translates a natural-language query into a metric/ontology-class filter plus a
time window, then pulls the matching observations from memory. Retrieval is
keyword + synonym based (using the metric registry) with light ontology
awareness: querying a parent class (e.g. "vital signs") returns observations of
all descendant classes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from ingestion.metrics import REGISTRY, synonym_index
from ontology.ontology_loader import load_ontology

_SYN_INDEX = synonym_index()


@dataclass
class QuerySpec:
    metrics: List[str] = field(default_factory=list)
    ontology_types: List[str] = field(default_factory=list)
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    category: Optional[str] = None
    raw: str = ""


def parse_query(text: str) -> QuerySpec:
    """Map a NL question to metrics + time window + ontology class filter."""
    low = text.lower()
    spec = QuerySpec(raw=text)
    ont = load_ontology()

    # metrics via synonyms
    found = []
    for syn, key in _SYN_INDEX:
        if re.search(r"\b" + re.escape(syn) + r"\b", low) and key not in found:
            found.append(key)
    spec.metrics = found

    # ontology class mentions (e.g. "observation", "vital sign")
    for cls, label in ont.classes.items():
        if label.lower() in low or _camel_to_words(cls) in low:
            spec.ontology_types.append(cls)

    # category shortcuts
    for cat in {m.category for m in REGISTRY.values()}:
        if cat.replace("_", " ") in low:
            spec.category = cat

    spec.since, spec.until = _parse_time_window(low)
    return spec


def _camel_to_words(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).lower()


def _parse_time_window(low: str):
    now = datetime.utcnow()
    m = re.search(r"last\s+(\d+)\s*(day|days|week|weeks|month|months)", low)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        days = n * (7 if "week" in unit else 30 if "month" in unit else 1)
        return now - timedelta(days=days), now
    if "today" in low:
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start, now
    if "yesterday" in low:
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0,
                                                  microsecond=0)
        return start, start + timedelta(days=1)
    if "this week" in low or "past week" in low:
        return now - timedelta(days=7), now
    if "this month" in low or "past month" in low:
        return now - timedelta(days=30), now
    return None, None


def _parse_ts(ts: str) -> Optional[datetime]:
    for fmt in (None,):  # try ISO first
        try:
            return datetime.fromisoformat(ts.replace("Z", ""))
        except Exception:
            pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(ts, fmt)
        except Exception:
            continue
    return None


def retrieve(observations: List[Dict], spec: QuerySpec,
             person: str = "self") -> List[Dict]:
    """Filter observations by the query spec."""
    ont = load_ontology()
    # Expand ontology types to their descendant classes for parent queries
    type_filter: set = set()
    for t in spec.ontology_types:
        type_filter.add(t)
        type_filter |= ont.descendants(t)

    results = []
    for o in observations:
        if person and o.get("observedFor") not in (person, None):
            continue
        if spec.metrics and o.get("metric") not in spec.metrics:
            # allow ontology-type match even if metric doesn't match
            if not (type_filter and o.get("type") in type_filter):
                continue
        if not spec.metrics and type_filter and o.get("type") not in type_filter:
            continue
        if spec.category and o.get("category") != spec.category:
            continue
        ts = _parse_ts(o.get("timestamp", "")) or _parse_ts(
            o.get("recordedAt", ""))
        if spec.since and ts and ts < spec.since:
            continue
        if spec.until and ts and ts > spec.until:
            continue
        results.append(o)

    results.sort(key=lambda o: o.get("timestamp", ""), reverse=True)
    return results
