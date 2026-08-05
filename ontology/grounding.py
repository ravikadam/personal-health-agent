"""Ontology grounding — makes the ontology central to every turn.

Two jobs:

1. `build_llm_context(...)` renders a compact, ontology-derived briefing that is
   injected into *every* LLM system prompt: the relevant class hierarchy, object
   properties (relationships) and the condition/intervention links. This forces
   the model to reason and answer in ontology terms rather than free-form.

2. `ground(...)` produces a structured "grounding trace" for a set of
   observations / a query — the exact classes, hierarchy paths and properties
   the answer rests on. The UI renders this under every response as visible
   proof that the ontology drove the result.

Domain knowledge that isn't in the OWL file (which conditions a metric relates
to, and via which ontology property) lives in small, explicit maps here so it
stays auditable and easy to extend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .ontology_loader import OntologyModel, load_ontology

# Which chronic condition each metric category informs, and the ontology
# object property that expresses the link. Purely declarative and extensible.
CATEGORY_CONDITION = {
    "diabetes": ("Diabetes mellitus", "ChronicCondition"),
    "blood_pressure": ("Hypertension", "ChronicCondition"),
    "cardio": ("Cardiovascular risk", "RiskFactor"),
    "sleep": ("Sleep quality", "WellbeingDimension"),
    "activity": ("Physical inactivity", "RiskFactor"),
    "wellbeing": ("Stress / mood", "WellbeingDimension"),
    "general": (None, None),
}

# The ontology property that connects an observation's subject to a condition.
OBSERVATION_CONDITION_PROPERTY = "manifestsAs"   # HealthCondition -> HealthConcept
PERSON_CONDITION_PROPERTY = "hasCondition"       # Person -> HealthCondition
INTERVENTION_CONDITION_PROPERTY = "addressesCondition"


@dataclass
class Grounding:
    classes: List[Dict] = field(default_factory=list)     # {name,label,path}
    properties: List[Dict] = field(default_factory=list)  # {name,label,domain,range}
    conditions: List[Dict] = field(default_factory=list)  # {name,type,via,metric}
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "classes": self.classes,
            "properties": self.properties,
            "conditions": self.conditions,
            "notes": self.notes,
        }

    def as_markdown(self) -> str:
        lines: List[str] = []
        if self.classes:
            lines.append("**Ontology classes**")
            for c in self.classes:
                lines.append(f"- `{c['name']}` — {c['label']}  ·  "
                             f"⊂ {' ⊂ '.join(c['path']) or '—'}")
        if self.conditions:
            lines.append("\n**Linked conditions**")
            for c in self.conditions:
                lines.append(f"- {c['metric']} → *{c['name']}* "
                             f"(`{c['type']}`) via `{c['via']}`")
        if self.properties:
            lines.append("\n**Relationships in play**")
            for p in self.properties:
                lines.append(f"- `{p['name']}`: {p['domain']} → {p['range']}")
        return "\n".join(lines)


def _class_path(ont: OntologyModel, name: str) -> List[str]:
    """Ordered superclass path from immediate parent up to the root."""
    path: List[str] = []
    current = name
    seen: Set[str] = set()
    while True:
        parents = sorted(ont.subclass_of.get(current, set()))
        if not parents or parents[0] in seen:
            break
        parent = parents[0]
        path.append(parent)
        seen.add(parent)
        current = parent
    return path


def ground(metrics: List[str], ontology_types: Optional[List[str]] = None,
           registry: Optional[Dict] = None) -> Grounding:
    """Build a grounding trace for the given metrics / ontology types."""
    ont = load_ontology()
    if registry is None:
        from ingestion.metrics import REGISTRY as registry  # lazy to avoid cycle

    g = Grounding()
    seen_classes: Set[str] = set()

    def add_class(cls: str):
        cls = ont.local(cls)
        if cls in seen_classes or not ont.is_class(cls):
            return
        seen_classes.add(cls)
        g.classes.append({
            "name": cls,
            "label": ont.label(cls),
            "path": _class_path(ont, cls),
        })

    # Classes + condition links from metrics
    for m in metrics:
        mdef = registry.get(m)
        if not mdef:
            continue
        add_class(mdef.ontology_class)
        cond_name, cond_type = CATEGORY_CONDITION.get(mdef.category, (None, None))
        if cond_name:
            g.conditions.append({
                "metric": mdef.label,
                "name": cond_name,
                "type": cond_type,
                "via": OBSERVATION_CONDITION_PROPERTY,
            })

    for t in ontology_types or []:
        add_class(t)

    # Relevant object properties (observation + condition semantics)
    for pname in ("observes", "observedFor", "hasUnit", "manifestsAs",
                  "hasCondition", "addressesCondition", "interpretsObservation"):
        info = ont.object_properties.get(pname)
        if info:
            g.properties.append({
                "name": info.name,
                "label": info.label,
                "domain": info.domain or "—",
                "range": info.range or "—",
            })

    return g


def build_llm_context(metrics: Optional[List[str]] = None,
                      max_classes: int = 40) -> str:
    """Render an ontology briefing for injection into an LLM system prompt."""
    ont = load_ontology()

    # Observation branch (the part the agent reasons over most)
    obs_classes = sorted(ont.descendants("Observation"))
    interv_classes = sorted(ont.descendants("Intervention"))

    lines = [
        "You reason over the Personal Health Management (phm) ontology. "
        "Always describe data using its classes and relationships.",
        "",
        "OBSERVATION CLASS HIERARCHY (subset):",
    ]
    for c in obs_classes[:max_classes]:
        parents = ", ".join(sorted(ont.subclass_of.get(c, []))) or "—"
        lines.append(f"  - {c} ({ont.label(c)}) ⊂ {parents}")

    lines += ["", "KEY OBJECT PROPERTIES (relationships):"]
    for p in ("observedFor", "observes", "hasUnit", "manifestsAs",
              "hasCondition", "affectsPerson", "addressesCondition",
              "hasComorbidity", "interpretsObservation", "hasEvidence"):
        info = ont.object_properties.get(p)
        if info:
            lines.append(f"  - {info.name}: {info.domain or '—'} → "
                         f"{info.range or '—'}  ({info.label})")

    # Vocabulary for remembering general profile / situation (not just vitals)
    lines += ["", "PROFILE / MEMORY VOCABULARY (for durable personal facts):"]
    for cls in ("Person", "ChronicCondition", "Comorbidity", "Medication",
                "Contraindication", "DietaryPattern", "RiskFactor",
                "FunctionalLimitation", "HealthGoal", "Barrier", "Facilitator"):
        if ont.is_class(cls):
            lines.append(f"  - {cls} ({ont.label(cls)})")
    for p in ("hasCondition", "hasComorbidity", "usesMedication",
              "hasRiskFactor", "hasContraindication", "hasBarrier",
              "hasFacilitator", "pursuesGoal", "goalForPerson"):
        info = ont.object_properties.get(p)
        if info:
            lines.append(f"  - property {info.name}: {info.domain or '—'} → "
                         f"{info.range or '—'}")

    if metrics:
        lines += ["", "CONDITION LINKS FOR THIS CONTEXT:"]
        from ingestion.metrics import REGISTRY
        for m in metrics:
            mdef = REGISTRY.get(m)
            if not mdef:
                continue
            cond, ctype = CATEGORY_CONDITION.get(mdef.category, (None, None))
            if cond:
                lines.append(f"  - {mdef.ontology_class} relates to '{cond}' "
                             f"({ctype}) via {OBSERVATION_CONDITION_PROPERTY}")

    lines += [
        "",
        "RULES:",
        "  1. Ground every claim in the ontology: name the class and, when "
        "relevant, its parent class and any linked condition.",
        "  2. Separate observations from interpretations; do not assert "
        "causation — use association language (AssociationAssessment).",
        "  3. Never invent numeric values; use only the provided data.",
        "  4. Be concise and explainable.",
    ]
    return "\n".join(lines)
