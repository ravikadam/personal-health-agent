"""LLM agent layer — where the model actively drives the pipeline.

This centralises every place the LLM does real work, always grounded in the
ontology and always with a deterministic fallback so the app still runs with no
provider:

  * `understand(message)`      — interpret a chat turn: log vs. query, which
                                 metrics/values, time window, memory facts
                                 (conditions/medications), and a chart spec.
  * `extract_document(text)`   — read a PDF/CSV/lab-report's text and pull
                                 structured observations the regex layer misses.
  * `compose_answer(...)`      — write the final response from the *retrieved,
                                 already-computed* numbers (never invents data),
                                 in ontology language.

Numbers are computed in code; the LLM handles intent, structure and prose.
Every prompt is seeded with the ontology via `build_llm_context`.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from ontology.grounding import build_llm_context

# Controlled vocabulary the model must map time expressions onto.
TIME_WINDOWS = {"today", "yesterday", "last_7_days", "last_30_days",
                "last_90_days", "all"}
CHART_TYPES = {"line", "bar", "scatter", "none"}


# --------------------------------------------------------------------------- #
# 1. Understand a chat turn
# --------------------------------------------------------------------------- #
def understand(message: str, provider, metrics: List[str]) -> Optional[Dict]:
    """Return a structured plan for a chat message, or None to fall back.

    Plan shape (all keys optional except intent):
      {
        "intent": "log" | "query" | "mixed" | "chat",
        "observations": [{"metric","value","unit","timestamp"?}],
        "memory_facts": [{"kind":"condition|medication",
                          "name","ontology_class","predicate"}],
        "query": {"metrics":[...], "ontology_types":[...],
                  "time_window": "<vocab>", "analysis":
                  "latest|average|trend|anomaly|correlation",
                  "correlate_with": "<metric>|null"},
        "chart": {"type":"line|bar|scatter|none","metrics":[...],"title":""},
        "response_style": "brief|detailed"
      }
    """
    if provider is None or not getattr(provider, "available",
                                       lambda: False)():
        return None
    system = build_llm_context(metrics) + (
        "\n\nYou are the router/parser of a personal-health agent. Read the "
        "user's message and return ONLY a JSON object describing what to do.\n"
        "Schema:\n"
        '{\n'
        '  "intent": "log" | "query" | "mixed" | "chat",\n'
        '  "observations": [{"metric": one of the known metrics, '
        '"value": number, "unit": string, "timestamp": ISO8601 optional}],\n'
        '  "memory_facts": [{"kind": "condition"|"medication", "name": string, '
        '"ontology_class": a phm class, "predicate": e.g. hasCondition}],\n'
        f'  "query": {{"metrics": [known metrics], "ontology_types": [phm '
        f'classes], "time_window": one of {sorted(TIME_WINDOWS)}, "analysis": '
        '"latest"|"average"|"trend"|"anomaly"|"correlation", "correlate_with": '
        'metric or null}},\n'
        f'  "chart": {{"type": one of {sorted(CHART_TYPES)}, "metrics": [known '
        'metrics], "title": string}},\n'
        '  "response_style": "brief" | "detailed"\n'
        "}\n"
        f"Known metrics: {', '.join(metrics)}.\n"
        "Rules: only use listed metrics; if the user reports numbers it's a "
        "'log' (or 'mixed' if they also ask something); pick a chart only when "
        "a trend/comparison over time helps, else type 'none'. Return JSON "
        "only, no prose."
    )
    plan = provider.extract_json(system, message)
    if not isinstance(plan, dict) or "intent" not in plan:
        return None
    return _sanitize_plan(plan, metrics)


def _sanitize_plan(plan: Dict, metrics: List[str]) -> Dict:
    mset = set(metrics)
    plan.setdefault("observations", [])
    plan["observations"] = [o for o in plan.get("observations", [])
                            if o.get("metric") in mset
                            and o.get("value") is not None]
    q = plan.get("query") or {}
    q["metrics"] = [m for m in (q.get("metrics") or []) if m in mset]
    if q.get("time_window") not in TIME_WINDOWS:
        q["time_window"] = "all"
    plan["query"] = q
    ch = plan.get("chart") or {}
    if ch.get("type") not in CHART_TYPES:
        ch["type"] = "none"
    ch["metrics"] = [m for m in (ch.get("metrics") or []) if m in mset]
    plan["chart"] = ch
    plan.setdefault("memory_facts", [])
    return plan


def window_to_dates(window: str) -> Tuple[Optional[str], Optional[str]]:
    now = datetime.utcnow()
    if window == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start.isoformat(), now.isoformat()
    if window == "yesterday":
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0,
                                                  microsecond=0)
        return start.isoformat(), (start + timedelta(days=1)).isoformat()
    days = {"last_7_days": 7, "last_30_days": 30, "last_90_days": 90}.get(window)
    if days:
        return (now - timedelta(days=days)).isoformat(), now.isoformat()
    return None, None


# --------------------------------------------------------------------------- #
# 2. Understand a document (file text)
# --------------------------------------------------------------------------- #
def extract_document(text: str, provider, metrics: List[str],
                     person: str = "self",
                     source: str = "file") -> Optional[List[Dict]]:
    """LLM extraction of observations from a report's text. None => fall back."""
    if provider is None or not getattr(provider, "available",
                                       lambda: False)():
        return None
    if not text or not text.strip():
        return None
    system = build_llm_context(metrics) + (
        "\n\nTASK: Extract every health measurement from this report text. "
        "Return ONLY a JSON array of objects with keys: metric (one of: "
        f"{', '.join(metrics)}), value (number), unit (string), timestamp "
        "(ISO8601 if a date is present, else omit). Ignore anything that isn't "
        "one of the listed metrics."
    )
    payload = provider.extract_json(system, text[:6000])
    if not isinstance(payload, list):
        return None

    from ingestion.extractor import _make_record
    from ingestion.metrics import REGISTRY
    ts_default = datetime.utcnow().isoformat()
    out: List[Dict] = []
    for item in payload:
        key = item.get("metric")
        if key in REGISTRY and item.get("value") is not None:
            out.append(_make_record(key, float(item["value"]),
                                    item.get("unit"), person, source,
                                    item.get("timestamp") or ts_default, text))
    return out or None


# --------------------------------------------------------------------------- #
# 3. Compose the final answer from computed facts
# --------------------------------------------------------------------------- #
def compose_answer(question: str, facts: Dict, provider,
                   metrics: List[str]) -> Optional[str]:
    """Write the response using ONLY the provided (already-computed) facts."""
    if provider is None or not getattr(provider, "available",
                                       lambda: False)():
        return None
    try:
        system = build_llm_context(metrics) + (
            "\n\nTASK: Answer the user's health question using ONLY the JSON "
            "facts (they are already computed — never change or invent a "
            "number). Reference the relevant ontology class and any linked "
            "condition. Use association, not causal, language. Keep it to the "
            "requested style; default 3-6 sentences.")
        user = (f"Question: {question}\n\nFacts (JSON):\n"
                f"{json.dumps(facts, default=str)}")
        res = provider.complete(system, user)
        return (res.text or "").strip() or None
    except Exception:
        return None
