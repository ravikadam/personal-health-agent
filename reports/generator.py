"""Report generation.

Builds a structured health report from memory: per-metric summaries grouped by
condition/category, trends, anomalies, correlations and actionable insights.
Returns a plain dict so it can be rendered as Markdown (CLI/tests) or as
Streamlit widgets + charts in the UI.
"""

from __future__ import annotations

from datetime import datetime
from itertools import combinations
from typing import Dict, List

import pandas as pd

from ingestion.metrics import REGISTRY
from .reasoning import aggregate, correlate, insights


# Category -> friendly condition grouping used in the report
CATEGORY_TITLES = {
    "diabetes": "Diabetes / Glucose",
    "blood_pressure": "Blood Pressure",
    "cardio": "Cardiovascular",
    "sleep": "Sleep",
    "activity": "Activity",
    "wellbeing": "Wellbeing",
    "general": "General",
}


def build_report(observations: List[Dict], person: str = "self") -> Dict:
    """Assemble the full report structure."""
    obs = [o for o in observations if o.get("observedFor") in (person, None)]
    present_metrics = sorted({o.get("metric") for o in obs if o.get("metric")})

    summaries = {}
    for metric in present_metrics:
        s = aggregate(obs, metric)
        if s:
            summaries[metric] = s

    # Group by category/condition
    groups: Dict[str, List[Dict]] = {}
    for metric, s in summaries.items():
        cat = REGISTRY[metric].category if metric in REGISTRY else "general"
        groups.setdefault(cat, []).append(s)

    grouped = []
    for cat, items in groups.items():
        grouped.append({
            "category": cat,
            "title": CATEGORY_TITLES.get(cat, cat.title()),
            "summaries": sorted(items, key=lambda s: s["label"]),
        })
    grouped.sort(key=lambda g: g["title"])

    # Correlations across numeric metrics
    correlations = []
    for a, b in combinations(present_metrics, 2):
        c = correlate(obs, a, b)
        if c and c["strength"] in ("moderate", "strong"):
            correlations.append(c)

    all_insights = insights(list(summaries.values()))
    all_anomalies = [
        {**anom, "metric": s["label"]}
        for s in summaries.values() for anom in s["anomalies"]
    ]

    return {
        "person": person,
        "generatedAt": datetime.utcnow().isoformat(),
        "observation_count": len(obs),
        "metrics": present_metrics,
        "groups": grouped,
        "insights": all_insights,
        "anomalies": all_anomalies,
        "correlations": correlations,
        "summaries": summaries,
    }


def report_to_markdown(report: Dict) -> str:
    """Render the report dict to Markdown (used for export / CLI)."""
    lines = [f"# Health Report — {report['person']}",
             f"_Generated {report['generatedAt'][:19]} • "
             f"{report['observation_count']} observations_\n"]

    if report["insights"]:
        lines.append("## Key Insights")
        lines += [f"- {i}" for i in report["insights"]]
        lines.append("")

    for group in report["groups"]:
        lines.append(f"## {group['title']}")
        for s in group["summaries"]:
            unit = s.get("unit") or ""
            tr = s["trend"]["direction"]
            lines.append(
                f"- **{s['label']}**: latest {s['latest']}{unit}, "
                f"avg {s['avg']}{unit}, range {s['min']}–{s['max']}{unit}, "
                f"{s['count']} readings, trend {tr}."
            )
        lines.append("")

    if report["correlations"]:
        lines.append("## Correlations")
        for c in report["correlations"]:
            la = REGISTRY[c['metric_a']].label if c['metric_a'] in REGISTRY \
                else c['metric_a']
            lb = REGISTRY[c['metric_b']].label if c['metric_b'] in REGISTRY \
                else c['metric_b']
            lines.append(f"- {la} vs {lb}: {c['strength']} {c['direction']} "
                         f"correlation (r={c['r']}, n={c['n']}).")
        lines.append("")

    if report["anomalies"]:
        lines.append("## Anomalies")
        for a in report["anomalies"]:
            lines.append(f"- {a['metric']} = {a['value']} on "
                         f"{(a.get('timestamp') or '')[:16]} — {a['reason']}.")
        lines.append("")

    return "\n".join(lines)


def series_dataframe(summary: Dict) -> pd.DataFrame:
    """Time series for one metric as a DataFrame (for charting)."""
    df = pd.DataFrame(summary.get("series", []))
    if df.empty:
        return df
    df["t"] = pd.to_datetime(df["t"], errors="coerce")
    df = df.dropna(subset=["t"]).sort_values("t").set_index("t")
    df = df.rename(columns={"v": summary["label"]})
    return df
