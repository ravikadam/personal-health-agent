"""Metric registry.

Central definition of the health metrics the agent understands, each mapped to
an ontology class, canonical unit, synonyms (for the "sugar -> glucose" style
mapping the spec calls for) and a rough reference range used for anomaly /
interpretation hints. Adding a metric here makes it available to extraction,
retrieval, reasoning and reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class MetricDef:
    key: str                       # internal key, e.g. "glucose"
    ontology_class: str            # phm class local name
    label: str                     # human label
    canonical_unit: str
    synonyms: List[str] = field(default_factory=list)
    # inclusive (low, high) normal range in canonical unit; None => no range
    normal_range: Optional[Tuple[float, float]] = None
    # unit aliases mapping -> multiplier to canonical unit
    unit_aliases: Dict[str, float] = field(default_factory=dict)
    category: str = "general"      # grouping used in reports (e.g. "diabetes")


REGISTRY: Dict[str, MetricDef] = {
    "glucose": MetricDef(
        key="glucose",
        ontology_class="GlucoseObservation",
        label="Blood glucose",
        canonical_unit="mg/dL",
        synonyms=["glucose", "sugar", "blood sugar", "bg", "fasting sugar",
                  "fasting glucose", "random sugar"],
        normal_range=(70, 140),
        unit_aliases={"mg/dl": 1.0, "mmol/l": 18.0},  # mmol/L -> mg/dL
        category="diabetes",
    ),
    "hba1c": MetricDef(
        key="hba1c",
        ontology_class="LaboratoryObservation",
        label="HbA1c",
        canonical_unit="%",
        synonyms=["hba1c", "a1c", "glycated hemoglobin", "hemoglobin a1c"],
        normal_range=(4.0, 5.7),
        category="diabetes",
    ),
    "systolic": MetricDef(
        key="systolic",
        ontology_class="BloodPressureObservation",
        label="Systolic blood pressure",
        canonical_unit="mmHg",
        synonyms=["systolic", "sbp"],
        normal_range=(90, 120),
        category="blood_pressure",
    ),
    "diastolic": MetricDef(
        key="diastolic",
        ontology_class="BloodPressureObservation",
        label="Diastolic blood pressure",
        canonical_unit="mmHg",
        synonyms=["diastolic", "dbp"],
        normal_range=(60, 80),
        category="blood_pressure",
    ),
    "heart_rate": MetricDef(
        key="heart_rate",
        ontology_class="HeartRateObservation",
        label="Heart rate",
        canonical_unit="bpm",
        synonyms=["heart rate", "hr", "pulse", "resting heart rate"],
        normal_range=(60, 100),
        category="cardio",
    ),
    "weight": MetricDef(
        key="weight",
        ontology_class="BodyWeightObservation",
        label="Body weight",
        canonical_unit="kg",
        synonyms=["weight", "body weight", "wt"],
        normal_range=None,
        unit_aliases={"kg": 1.0, "kgs": 1.0, "lb": 0.453592, "lbs": 0.453592,
                      "pounds": 0.453592},
        category="general",
    ),
    "sleep": MetricDef(
        key="sleep",
        ontology_class="SleepObservation",
        label="Sleep duration",
        canonical_unit="hours",
        synonyms=["sleep", "slept", "sleep duration", "hours of sleep"],
        normal_range=(7, 9),
        unit_aliases={"h": 1.0, "hr": 1.0, "hrs": 1.0, "hour": 1.0,
                      "hours": 1.0, "min": 1 / 60.0, "mins": 1 / 60.0,
                      "minutes": 1 / 60.0},
        category="sleep",
    ),
    "steps": MetricDef(
        key="steps",
        ontology_class="ActivityObservation",
        label="Steps",
        canonical_unit="steps",
        synonyms=["steps", "step count", "walked"],
        normal_range=(7000, 20000),
        category="activity",
    ),
    "spo2": MetricDef(
        key="spo2",
        ontology_class="VitalSignObservation",
        label="Oxygen saturation",
        canonical_unit="%",
        synonyms=["spo2", "oxygen", "oxygen saturation", "o2 sat", "sao2"],
        normal_range=(95, 100),
        category="cardio",
    ),
    "temperature": MetricDef(
        key="temperature",
        ontology_class="VitalSignObservation",
        label="Body temperature",
        canonical_unit="C",
        synonyms=["temperature", "temp", "fever", "body temp"],
        normal_range=(36.1, 37.5),
        unit_aliases={"c": 1.0, "celsius": 1.0},
        category="general",
    ),
    "mood": MetricDef(
        key="mood",
        ontology_class="MeditationObservation",
        label="Mood / wellbeing",
        canonical_unit="score",
        synonyms=["mood", "stress", "energy", "wellbeing"],
        normal_range=None,
        category="wellbeing",
    ),
}


# Reverse index: synonym -> metric key (longest synonyms first for greedy match)
def synonym_index() -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for key, mdef in REGISTRY.items():
        for syn in mdef.synonyms:
            pairs.append((syn.lower(), key))
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return pairs


def normalize_unit(metric_key: str, value: float,
                   unit: Optional[str]) -> Tuple[float, str]:
    """Convert a value to the metric's canonical unit where possible."""
    mdef = REGISTRY[metric_key]
    if not unit:
        return value, mdef.canonical_unit
    u = unit.strip().lower()
    if u in mdef.unit_aliases:
        return round(value * mdef.unit_aliases[u], 4), mdef.canonical_unit
    # Already canonical (case-insensitive match) or unknown -> keep given unit
    if u == mdef.canonical_unit.lower():
        return value, mdef.canonical_unit
    return value, unit
