"""Turn orchestration: LLM-planned, deterministically executed.

`handle_turn` is the single entry point the UI calls for every chat message. It:

  1. Asks the LLM agent to *understand* the message (intent, values, memory
     facts, query params, chart spec) — grounded in the ontology.
  2. Executes the plan deterministically: logs observations (LLM + regex
     merged), writes ontology memory facts (conditions/medications), retrieves
     and aggregates real numbers.
  3. Has the LLM *compose* the reply from those computed numbers, and follows
     the LLM's chart choice.

With no provider it falls back to the rule-based router so nothing breaks.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ingestion.extractor import _make_record, extract_observations
from ingestion.metrics import REGISTRY
from llm import agent
from ontology.grounding import ground
from reports.reasoning import aggregate, correlate, insights
from .retriever import QuerySpec, parse_query, retrieve

METRIC_KEYS = list(REGISTRY.keys())


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def handle_turn(store, message: str, provider) -> Dict:
    plan = agent.understand(message, provider, METRIC_KEYS)
    if plan is None:
        return _rule_based_turn(store, message, provider)
    return _planned_turn(store, message, plan, provider)


# --------------------------------------------------------------------------- #
# LLM-planned path
# --------------------------------------------------------------------------- #
def _planned_turn(store, message: str, plan: Dict, provider) -> Dict:
    result: Dict = {"mode": plan.get("intent", "chat"), "plan": plan,
                    "used_llm": True, "logged": None, "memory": None,
                    "answer": None}
    intent = plan.get("intent", "chat")

    # Persist whatever the LLM extracted, regardless of intent label: numeric
    # readings AND memory facts (conditions/medications). A pure statement like
    # "I'm diabetic and have BP issues" carries no numbers but must still update
    # memory.
    if plan.get("observations"):
        result["logged"] = _log_from_plan(store, message, plan)
    if plan.get("memory_facts"):
        result["memory"] = _apply_memory_facts(store, plan)

    # Compose an answer for questions, or when the turn was purely
    # conversational with nothing to store.
    nothing_stored = not (result["logged"] or result["memory"])
    if intent in ("query", "mixed") or (intent == "chat" and nothing_stored):
        result["answer"] = _answer_from_plan(store, message, plan, provider)

    return result


def _log_from_plan(store, message: str, plan: Dict) -> Dict:
    records: List[Dict] = []
    for o in plan.get("observations", []):
        key = o.get("metric")
        if key not in REGISTRY:
            continue
        records.append(_make_record(key, float(o["value"]), o.get("unit"),
                                    "self", "chat",
                                    o.get("timestamp")
                                    or _now(), message))
    # Merge with regex to catch anything the model missed.
    for r in extract_observations(message, person="self", source="chat"):
        if not any(x["metric"] == r["metric"]
                   and x["numericValue"] == r["numericValue"] for x in records):
            records.append(r)
    res = store.add_observations(records)
    res["records"] = records
    return res


# Sensible ontology fallbacks per fact kind: (class, object property).
_KIND_DEFAULTS = {
    "condition": ("ChronicCondition", "hasCondition"),
    "medication": ("Medication", "usesMedication"),
    "allergy": ("Contraindication", "hasContraindication"),
    "diet": ("DietaryPattern", "hasFacilitator"),
    "lifestyle": ("HealthConcept", "hasFacilitator"),
    "risk_factor": ("RiskFactor", "hasRiskFactor"),
    "family_history": ("RiskFactor", "hasRiskFactor"),
    "goal": ("HealthGoal", "pursuesGoal"),
    "profile": ("Person", "assertionSubject"),
    "other": ("HealthConcept", "assertionPredicate"),
}


def _apply_memory_facts(store, plan: Dict) -> Dict:
    """Persist ANY durable LLM-extracted fact (conditions, meds, allergies,
    diet, lifestyle, goals, profile, ...) as ontology entities + provenance-
    bearing MemoryAssertions. Falls back to a sensible class/property per kind
    when the LLM's suggestion isn't a known ontology term."""
    from ontology.ontology_loader import load_ontology
    ont = load_ontology()
    existing = {(a["predicate"], (a["object"] or "").lower())
                for a in store.assertions()}
    entities, assertions = [], []

    for fact in plan.get("memory_facts", []):
        name = (fact.get("name") or "").strip()
        if not name:
            continue
        kind = fact.get("kind", "other")
        def_cls, def_pred = _KIND_DEFAULTS.get(kind, _KIND_DEFAULTS["other"])

        cls = fact.get("ontology_class")
        if not cls or not ont.is_class(cls):
            cls = def_cls
        predicate = fact.get("predicate")
        if not predicate or predicate not in ont.object_properties:
            predicate = def_pred

        attrs = {}
        if fact.get("value") not in (None, ""):
            attrs["value"] = fact["value"]
        if fact.get("note"):
            attrs["note"] = fact["note"]
        ent = store.add_entity(cls, name, kind=kind, **attrs)
        entities.append(ent)

        if (predicate, name.lower()) not in existing:
            note_bits = [b for b in (fact.get("note"),
                         f"value={fact['value']}" if fact.get("value")
                         not in (None, "") else None) if b]
            assertions.append(store.add_assertion(
                "self", predicate, name, status="Candidate",
                evidence=note_bits))
            existing.add((predicate, name.lower()))

    return {"entities": entities, "assertions": assertions}


def _answer_from_plan(store, message: str, plan: Dict, provider) -> Dict:
    q = plan.get("query") or {}
    metrics = q.get("metrics") or []
    ont_types = q.get("ontology_types") or []
    since, until = agent.window_to_dates(q.get("time_window", "all"))

    spec = QuerySpec(metrics=metrics, ontology_types=ont_types, raw=message)
    if since:
        from datetime import datetime
        spec.since = datetime.fromisoformat(since)
        spec.until = datetime.fromisoformat(until) if until else None

    observations = store.all_observations()
    hits = retrieve(observations, spec, person="self")

    eff_metrics = metrics or sorted({h["metric"] for h in hits
                                     if h.get("metric")})
    summaries = [s for s in (aggregate(hits, m) for m in eff_metrics) if s]

    # Optional correlation analysis
    correlation = None
    if q.get("analysis") == "correlation" and q.get("correlate_with") \
            and eff_metrics:
        correlation = correlate(observations, eff_metrics[0],
                                q["correlate_with"])

    # Pull the person's stored profile (conditions, meds, allergies, diet,
    # goals, risk factors) so the answer is grounded in semantic memory, not
    # just numeric observations.
    profile = _profile_payload(store)
    profile_classes = list(profile.keys())
    grounding = ground(eff_metrics, ont_types + profile_classes)
    facts = _facts_payload(hits, summaries, correlation, profile)

    explanation = agent.compose_answer(message, facts, provider, eff_metrics)
    used_llm = explanation is not None
    if not explanation:
        explanation = _rule_explanation(spec, hits, summaries, profile)

    chart = _resolve_chart(plan.get("chart") or {}, summaries)
    return {"explanation": explanation, "summaries": summaries, "hits": hits,
            "grounding": grounding, "chart": chart, "correlation": correlation,
            "used_llm": used_llm}


# --------------------------------------------------------------------------- #
# Rule-based fallback path (no provider)
# --------------------------------------------------------------------------- #
def _rule_based_turn(store, message: str, provider) -> Dict:
    is_question = message.strip().endswith("?") or any(
        message.strip().lower().startswith(w) for w in
        ("what", "how", "when", "show", "why", "is ", "are ", "do ", "does",
         "trend", "average", "avg", "list"))
    extracted = extract_observations(message, person="self", source="chat")

    if extracted and not is_question:
        res = store.add_observations(extracted)
        res["records"] = extracted
        return {"mode": "log", "used_llm": False, "logged": res,
                "memory": {"entities": [], "assertions": []}, "answer": None,
                "plan": None}

    spec = parse_query(message)
    hits = retrieve(store.all_observations(), spec, person="self")
    metrics = spec.metrics or sorted({h["metric"] for h in hits
                                      if h.get("metric")})
    summaries = [s for s in (aggregate(hits, m) for m in metrics) if s]
    profile = _profile_payload(store)
    grounding = ground(metrics, spec.ontology_types + list(profile.keys()))
    answer = {
        "explanation": _rule_explanation(spec, hits, summaries, profile),
        "summaries": summaries, "hits": hits, "grounding": grounding,
        "chart": _resolve_chart({"type": "line"}, summaries),
        "correlation": None, "used_llm": False,
    }
    return {"mode": "query", "used_llm": False, "logged": None,
            "memory": None, "answer": answer, "plan": None}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _profile_payload(store) -> Dict:
    """The person's durable memory (conditions, meds, allergies, diet, goals,
    risk factors) grouped by ontology class, for grounding answers."""
    pred_by_obj = {(a.get("object") or "").lower(): a.get("predicate")
                   for a in store.assertions()}
    profile: Dict[str, List[Dict]] = {}
    for e in store.entities():
        item = {"name": e["name"],
                "predicate": pred_by_obj.get(e["name"].lower())}
        if e.get("note"):
            item["note"] = e["note"]
        if e.get("value") not in (None, ""):
            item["value"] = e.get("value")
        profile.setdefault(e["type"], []).append(item)
    return profile


def _facts_payload(hits: List[Dict], summaries: List[Dict],
                   correlation: Optional[Dict],
                   profile: Optional[Dict] = None) -> Dict:
    return {
        "person_profile": profile or {},
        "record_count": len(hits),
        "summaries": [
            {
                "label": s["label"],
                "ontology_class": REGISTRY[s["metric"]].ontology_class
                if s["metric"] in REGISTRY else None,
                "latest": s["latest"], "avg": s["avg"], "min": s["min"],
                "max": s["max"], "count": s["count"], "unit": s.get("unit"),
                "trend": s["trend"]["direction"],
                "normal_range": s.get("normal_range"),
                "anomaly_count": len(s.get("anomalies", [])),
            } for s in summaries
        ],
        "correlation": correlation,
    }


def _rule_explanation(spec: QuerySpec, hits: List[Dict],
                      summaries: List[Dict],
                      profile: Optional[Dict] = None) -> str:
    prefix = ""
    if profile:
        conds = [i["name"] for cls, items in profile.items()
                 for i in items if "Condition" in cls]
        if conds:
            prefix = f"On file: {', '.join(conds)}.\n"
    if not hits:
        base = ("No matching observations for that window. Log a reading (e.g. "
                "\"glucose 120 mg/dL\") for a data-grounded answer.")
        return prefix + base if prefix else base
    parts = [f"Found {len(hits)} matching record(s)"
             + (f" since {spec.since.date()}" if spec.since else "") + "."]
    for s in summaries:
        unit = s.get("unit") or ""
        parts.append(f"{s['label']}: latest {s['latest']}{unit} (avg "
                     f"{s['avg']}{unit}, min {s['min']}, max {s['max']}, "
                     f"{s['count']} readings, trend {s['trend']['direction']}).")
    parts.extend(insights(summaries))
    return prefix + "\n".join(parts)


def _resolve_chart(chart_spec: Dict, summaries: List[Dict]) -> Optional[Dict]:
    """Validate the LLM's chart choice against metrics that actually have
    enough data; fall back to a line chart of multi-point metrics."""
    chartable = [s for s in summaries if s["count"] > 1]
    if not chartable:
        return None
    ctype = chart_spec.get("type", "line")
    if ctype == "none":
        return None
    wanted = set(chart_spec.get("metrics") or [])
    chosen = [s for s in chartable if s["metric"] in wanted] or chartable
    return {
        "type": ctype if ctype in ("line", "bar", "scatter") else "line",
        "title": chart_spec.get("title") or "Trend",
        "summaries": chosen,
    }


def _now() -> str:
    from datetime import datetime
    return datetime.utcnow().isoformat()
