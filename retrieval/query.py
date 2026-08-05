"""Natural-language query answering — ontology-grounded, LLM-optional.

`answer_query` ties retrieval + reasoning + ontology grounding together:

  1. Parse the question into metrics / ontology classes / time window.
  2. Retrieve matching observations (ontology-aware: parent classes expand to
     their descendant classes).
  3. Summarise with the reasoning layer.
  4. Build an ontology **grounding trace** (classes, hierarchy paths, linked
     conditions, properties) that every answer is anchored to.
  5. Produce an explanation. If an LLM provider is available it writes the prose
     — but strictly from the retrieved numbers and the ontology context. With no
     provider, a deterministic rule-based explanation is used. Either way the
     grounding trace is attached.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from ontology.grounding import build_llm_context, ground
from ontology.ontology_loader import load_ontology
from reports.reasoning import aggregate, insights
from .retriever import QuerySpec, parse_query, retrieve


def answer_query(observations: List[Dict], text: str, provider=None,
                 person: str = "self") -> Dict:
    spec = parse_query(text)
    hits = retrieve(observations, spec, person=person)

    metrics = spec.metrics or sorted({h["metric"] for h in hits
                                      if h.get("metric")})
    summaries = [aggregate(hits, m) for m in metrics]
    summaries = [s for s in summaries if s]

    grounding = ground(metrics, spec.ontology_types)

    rule_expl = _rule_explanation(text, spec, hits, summaries)
    used_llm = False
    explanation = rule_expl
    if provider is not None and getattr(provider, "available",
                                        lambda: False)():
        llm_expl = _llm_explanation(provider, text, metrics, summaries, hits)
        if llm_expl:
            explanation, used_llm = llm_expl, True

    return {
        "query": text,
        "spec": spec,
        "hits": hits,
        "summaries": summaries,
        "grounding": grounding,
        "explanation": explanation,
        "used_llm": used_llm,
    }


def _rule_explanation(text: str, spec: QuerySpec, hits: List[Dict],
                      summaries: List[Dict]) -> str:
    if not hits:
        return ("No matching records yet. Try logging a reading first, e.g. "
                "\"glucose 120 mg/dL\".")
    parts: List[str] = []
    window = f" since {spec.since.date()}" if spec.since else ""
    parts.append(f"Found {len(hits)} matching record(s){window}.")
    for s in summaries:
        unit = s.get("unit") or ""
        parts.append(
            f"{s['label']}: latest {s['latest']}{unit} "
            f"(avg {s['avg']}{unit}, min {s['min']}, max {s['max']}, "
            f"{s['count']} readings, trend {s['trend']['direction']}).")
    parts.extend(insights(summaries))
    return "\n".join(parts)


def _llm_explanation(provider, text: str, metrics: List[str],
                     summaries: List[Dict], hits: List[Dict]) -> Optional[str]:
    """Have the LLM write the answer, constrained to the retrieved data and
    ontology. Returns None on failure (caller keeps the rule-based answer)."""
    try:
        # Compact, factual payload the model must not exceed.
        facts = {
            "summaries": [
                {
                    "label": s["label"],
                    "ontology_class": _class_for_metric(s["metric"]),
                    "latest": s["latest"], "avg": s["avg"], "min": s["min"],
                    "max": s["max"], "count": s["count"],
                    "unit": s.get("unit"), "trend": s["trend"]["direction"],
                    "normal_range": s.get("normal_range"),
                    "anomaly_count": len(s.get("anomalies", [])),
                }
                for s in summaries
            ],
            "record_count": len(hits),
        }
        system = build_llm_context(metrics) + (
            "\n\nTASK: Answer the user's health question using ONLY the JSON "
            "facts provided. Reference the relevant ontology class(es) and any "
            "linked condition. Do not invent numbers. 3-6 sentences.")
        user = (f"Question: {text}\n\nFacts (JSON):\n"
                f"{json.dumps(facts, default=str)}")
        res = provider.complete(system, user)
        return res.text.strip() or None
    except Exception:
        return None


def _class_for_metric(metric: Optional[str]) -> Optional[str]:
    from ingestion.metrics import REGISTRY
    mdef = REGISTRY.get(metric) if metric else None
    return mdef.ontology_class if mdef else None
