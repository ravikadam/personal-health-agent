"""Reasoning layer: aggregation, trend and anomaly detection, insights.

Pure-Python / pandas statistics over retrieved observations. Rule-based and
explainable: every insight states the numbers behind it. Reference ranges come
from the metric registry so anomaly flags stay ontology/domain aligned.
"""

from __future__ import annotations

import statistics
from datetime import datetime
from typing import Dict, List, Optional

from ingestion.metrics import REGISTRY


def _num_series(observations: List[Dict], metric: str) -> List[Dict]:
    rows = [o for o in observations
            if o.get("metric") == metric and o.get("numericValue") is not None]
    rows.sort(key=lambda o: o.get("timestamp", ""))
    return rows


def aggregate(observations: List[Dict], metric: str) -> Optional[Dict]:
    """Return summary stats for one metric, or None if no data."""
    rows = _num_series(observations, metric)
    if not rows:
        return None
    values = [float(o["numericValue"]) for o in rows]
    mdef = REGISTRY.get(metric)
    summary = {
        "metric": metric,
        "label": mdef.label if mdef else metric,
        "unit": rows[-1].get("unit"),
        "count": len(values),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
        "avg": round(statistics.fmean(values), 2),
        "latest": values[-1],
        "latest_at": rows[-1].get("timestamp"),
        "first_at": rows[0].get("timestamp"),
        "trend": _trend(values),
        "anomalies": _anomalies(rows, metric),
        "normal_range": mdef.normal_range if mdef else None,
    }
    summary["series"] = [
        {"t": o.get("timestamp"), "v": float(o["numericValue"])} for o in rows
    ]
    return summary


def _trend(values: List[float]) -> Dict:
    """Simple least-squares slope + direction over the sequence."""
    n = len(values)
    if n < 2:
        return {"direction": "flat", "slope": 0.0, "change": 0.0}
    xs = list(range(n))
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(values)
    denom = sum((x - mean_x) ** 2 for x in xs) or 1e-9
    slope = sum((x - mean_x) * (y - mean_y)
                for x, y in zip(xs, values)) / denom
    change = values[-1] - values[0]
    direction = ("rising" if slope > 1e-6 else
                 "falling" if slope < -1e-6 else "flat")
    return {"direction": direction, "slope": round(slope, 4),
            "change": round(change, 2)}


def _anomalies(rows: List[Dict], metric: str) -> List[Dict]:
    """Flag out-of-range and statistical-outlier readings."""
    mdef = REGISTRY.get(metric)
    values = [float(o["numericValue"]) for o in rows]
    flags: List[Dict] = []

    # Reference-range breaches
    if mdef and mdef.normal_range:
        low, high = mdef.normal_range
        for o in rows:
            v = float(o["numericValue"])
            if v < low or v > high:
                flags.append({
                    "timestamp": o.get("timestamp"),
                    "value": v,
                    "reason": (f"outside normal range {low}–{high} "
                               f"{mdef.canonical_unit}"),
                    "severity": "high" if (v < low * 0.8 or v > high * 1.2)
                    else "moderate",
                })

    # Statistical outliers (z-score) when enough data
    if len(values) >= 5:
        mean = statistics.fmean(values)
        sd = statistics.pstdev(values) or 1e-9
        for o in rows:
            v = float(o["numericValue"])
            z = (v - mean) / sd
            if abs(z) >= 2.5 and not any(
                f["timestamp"] == o.get("timestamp") for f in flags
            ):
                flags.append({
                    "timestamp": o.get("timestamp"),
                    "value": v,
                    "reason": f"statistical outlier (z={z:.1f})",
                    "severity": "moderate",
                })
    return flags


def insights(summaries: List[Dict]) -> List[str]:
    """Turn summaries into short, explainable, actionable statements."""
    out: List[str] = []
    for s in summaries:
        if not s:
            continue
        label, unit = s["label"], s.get("unit") or ""
        rng = s.get("normal_range")
        latest = s["latest"]

        # Range status of the latest reading
        if rng:
            low, high = rng
            if latest < low:
                out.append(f"⚠️ Latest {label} is {latest}{unit}, below the "
                           f"typical {low}–{high}{unit} range.")
            elif latest > high:
                out.append(f"⚠️ Latest {label} is {latest}{unit}, above the "
                           f"typical {low}–{high}{unit} range.")
            else:
                out.append(f"✅ Latest {label} ({latest}{unit}) is within the "
                           f"typical {low}–{high}{unit} range.")

        # Trend note when there is enough history
        if s["count"] >= 3:
            tr = s["trend"]
            if tr["direction"] != "flat":
                out.append(f"📈 {label} is {tr['direction']} "
                           f"(net {tr['change']:+g}{unit} over {s['count']} "
                           f"readings, avg {s['avg']}{unit}).")

        # Anomaly summary
        if s["anomalies"]:
            n = len(s["anomalies"])
            out.append(f"🔎 {n} anomalous {label} reading(s) flagged — "
                       f"review the highlighted points.")
    return out


def correlate(observations: List[Dict], metric_a: str,
              metric_b: str, day_window: int = 0) -> Optional[Dict]:
    """Pearson correlation between two metrics matched by day.

    Useful for spec's "food vs glucose" style correlations when both are
    numeric. Returns None if fewer than 3 matched pairs.
    """
    from collections import defaultdict

    def by_day(metric):
        d = defaultdict(list)
        for o in observations:
            if o.get("metric") == metric and o.get("numericValue") is not None:
                day = (o.get("timestamp") or "")[:10]
                d[day].append(float(o["numericValue"]))
        return {k: statistics.fmean(v) for k, v in d.items()}

    a, b = by_day(metric_a), by_day(metric_b)
    common = sorted(set(a) & set(b))
    if len(common) < 3:
        return None
    xs = [a[d] for d in common]
    ys = [b[d] for d in common]
    try:
        r = statistics.correlation(xs, ys)
    except Exception:
        return None
    return {
        "metric_a": metric_a,
        "metric_b": metric_b,
        "n": len(common),
        "r": round(r, 3),
        "strength": ("strong" if abs(r) >= 0.7 else
                     "moderate" if abs(r) >= 0.4 else "weak"),
        "direction": "positive" if r > 0 else "negative",
    }
