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
def understand(message: str, provider, metrics: List[str],
               context: str = "") -> Optional[Dict]:
    """Return a structured plan for a chat message, or None to fall back.

    `context` is a short briefing of the person's recent readings and stored
    profile, so the model can (a) detect self-reported symptoms and (b) propose
    associations between a symptom and a recent reading.

    Plan shape (all keys optional except intent):
      {
        "intent": "log" | "query" | "mixed" | "chat",
        "observations": [{"metric","value","unit","timestamp"?,"context"?}],
        "symptoms": [{"name","note"?}],
        "associations": [{"exposure_metric","exposure_desc","outcome",
                          "relation":"association|causal_hypothesis",
                          "confidence":0..1,"rationale"}],
        "memory_facts": [...],
        "query": {...}, "chart": {...}, "response_style": "brief|detailed"
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
        '"value": number, "unit": string, "timestamp": ISO8601 optional, '
        '"context": "fasting"|"postprandial"|"random"|"bedtime" optional}],\n'
        '  "symptoms": [{"name": e.g. "dizziness"|"fatigue"|"headache", '
        '"note": optional}],\n'
        '  "associations": [{"exposure_metric": known metric or null, '
        '"exposure_desc": short phrase e.g. "low fasting glucose", "outcome": '
        'the symptom name, "relation": "association"|"causal_hypothesis", '
        '"confidence": 0..1, "rationale": short string}],\n'
        '  "memory_facts": [{"kind": '
        '"condition"|"medication"|"allergy"|"diet"|"lifestyle"|"risk_factor"'
        '|"goal"|"family_history"|"profile"|"other", "name": string, '
        '"ontology_class": best-fit phm class, "predicate": best-fit phm object '
        'property, "value": optional string or number, "note": optional short '
        'string}],\n'
        f'  "query": {{"metrics": [known metrics], "ontology_types": [phm '
        f'classes], "time_window": one of {sorted(TIME_WINDOWS)}, "analysis": '
        '"latest"|"average"|"trend"|"anomaly"|"correlation", "correlate_with": '
        'metric or null}},\n'
        f'  "chart": {{"type": one of {sorted(CHART_TYPES)}, "metrics": [known '
        'metrics], "title": string}},\n'
        '  "response_style": "brief" | "detailed"\n'
        "}\n"
        f"Known metrics: {', '.join(metrics)}.\n"
        "SYMPTOMS: if the user reports how they feel (dizzy, tired, headache, "
        "nausea, blurred vision, etc.), record it under 'symptoms' — this is a "
        "SelfReportedObservation, and often 'mixed' intent (they also ask "
        "why). ASSOCIATIONS: when a symptom plausibly relates to a recent "
        "reading in CONTEXT below, add an association (or causal_hypothesis) "
        "linking exposure_metric to the symptom, with a calibrated confidence "
        "— never assert certainty.\n"
        "Rules: only use listed metrics for observations; if the user reports "
        "numbers it's a 'log' (or 'mixed' if they also ask something); pick a "
        "chart only when a trend/comparison over time helps, else 'none'. "
        "IMPORTANT — memory_facts: capture ANY durable personal or health fact "
        "worth remembering long-term, not just numbers — who they are (name, "
        "age, sex), chronic conditions, medications, allergies, dietary "
        "pattern, lifestyle/habits, family history, and goals. Map each to the "
        "best-fit phm ontology class and object property. Ignore transient "
        "small talk. Return JSON only, no prose."
    )
    if context:
        system += f"\n\nCONTEXT (recent data & profile):\n{context}\n"
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
    plan["symptoms"] = [s for s in plan.get("symptoms", [])
                        if (s.get("name") or "").strip()]
    plan["associations"] = [a for a in plan.get("associations", [])
                            if (a.get("outcome") or "").strip()]
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
            "\n\nTASK: Answer the user's health question using the JSON facts. "
            "`person_profile` holds durable memory (conditions, medications, "
            "allergies, dietary pattern, goals, risk factors) mapped to phm "
            "classes — USE it to personalise the answer (e.g. a ChronicCondition "
            "like diabetes and a HealthGoal shape dietary guidance). "
            "`recent_symptoms` are SymptomObservations the person reported — "
            "acknowledge them. Each summary has an `escalation` tier "
            "(none/caution/urgent) with a clinical_name mapped to a phm "
            "Precaution/EscalationRule: if any reading is 'urgent' (e.g. "
            "hypoglycemia), LEAD with that and advise prompt action. `summaries` "
            "are computed stats — never change or invent a number. Reference the "
            "relevant ontology classes/conditions, use association (not causal) "
            "language, and add a brief 'not medical advice' note when giving "
            "guidance. Default 3-6 sentences.")
        user = (f"Question: {question}\n\nFacts (JSON):\n"
                f"{json.dumps(facts, default=str)}")
        res = provider.complete(system, user)
        return (res.text or "").strip() or None
    except Exception:
        return None
