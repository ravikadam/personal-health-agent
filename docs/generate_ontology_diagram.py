"""Generate docs/ontology.svg from the actual phm ontology.

Draws a faithful *subset* of the Personal Health Management ontology: the class
hierarchy (grouped into lanes) and the key object properties. Every subclass
relation and every property drawn is verified against the loaded OWL file, so
the diagram cannot silently drift from the source ontology.

The SVG is a self-contained light card (dark text on a pale panel) so it renders
legibly in both light and dark GitHub themes.

Run:  python docs/generate_ontology_diagram.py
"""

from __future__ import annotations

import os
import sys
from html import escape

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ontology.ontology_loader import load_ontology  # noqa: E402

ONT = load_ontology()

# --- palette (works on any background; boxes carry their own colour) -------
C = {
    "blue": "#2563eb", "red": "#dc2626", "redl": "#ef4444", "green": "#059669",
    "violet": "#7c3aed", "cyan": "#0891b2", "amber": "#d97706",
    "slate": "#475569", "line": "#64748b", "panel": "#f8fafc",
    "panelStroke": "#e2e8f0", "title": "#334155", "muted": "#64748b",
}

# Lanes: (title, colour, parent-branch label, [(class, shown_parent_or_None)])
# The first row of each lane is the branch head.
LANES = [
    ("Agents", "blue", "Entity", [
        ("Agent", "Entity"), ("Person", "Agent"), ("Clinician", "Person"),
        ("Caregiver", "Person"), ("SoftwareAgent", "Agent")]),
    ("Health state", "red", "HealthConcept", [
        ("HealthConcept", "Entity"), ("HealthCondition", "HealthConcept"),
        ("ChronicCondition", "HealthCondition"), ("Symptom", "HealthConcept"),
        ("RiskFactor", "HealthConcept")]),
    ("Observations", "green", "TemporalEntity", [
        ("Observation", "TemporalEntity"),
        ("QuantitativeObservation", "Observation"),
        ("GlucoseObservation", "QuantitativeObservation"),
        ("VitalSignObservation", "QuantitativeObservation"),
        ("BloodPressureObservation", "VitalSignObservation")]),
    ("Interventions", "violet", "HealthConcept", [
        ("Intervention", "HealthConcept"),
        ("MedicationIntervention", "ClinicalIntervention"),
        ("InterventionExecution", "TemporalEntity"),
        ("MedicationAdministration", "InterventionExecution"),
        ("ExerciseSession", "InterventionExecution")]),
    ("Plans & Goals", "cyan", "InformationArtifact", [
        ("Goal", "InformationArtifact"), ("HealthGoal", "Goal"),
        ("Plan", "InformationArtifact"), ("PersonalHealthPlan", "Plan"),
        ("Recommendation", "InformationArtifact")]),
    ("Memory & Evidence", "amber", "InformationArtifact", [
        ("Interpretation", "InformationArtifact"),
        ("MemoryAssertion", "InformationArtifact"),
        ("EvidenceItem", "InformationArtifact"),
        ("ProvenanceRecord", "InformationArtifact"),
        ("AssociationAssessment", "Interpretation")]),
]

# Relationship graph: node -> (cx, cy, colour)
RNODES = {
    "Person": (180, 520, "blue"),
    "HealthCondition": (490, 520, "red"),
    "Symptom": (800, 520, "redl"),
    "MemoryAssertion": (1090, 520, "amber"),
    "Observation": (180, 690, "green"),
    "Intervention": (490, 690, "violet"),
    "InterventionExecution": (800, 690, "violet"),
    "EvidenceItem": (1090, 690, "amber"),
}
# (subject, object, property) — drawn subject -> object
REDGES = [
    ("Person", "HealthCondition", "hasCondition"),
    ("HealthCondition", "Symptom", "manifestsAs"),
    ("Observation", "Person", "observedFor"),
    ("Intervention", "HealthCondition", "addressesCondition"),
    ("InterventionExecution", "Intervention", "executesIntervention"),
    ("MemoryAssertion", "EvidenceItem", "hasEvidence"),
]

# ---- geometry --------------------------------------------------------------
W = 1268
LANE_W, GAP, X0 = 196, 12, 16
NODE_W, NODE_H = 176, 46
lane_left = lambda i: X0 + i * (LANE_W + GAP)
lane_cx = lambda i: lane_left(i) + LANE_W / 2

HEAD_Y = 126
ROW_DY = 64
BOX_HW, BOX_HH = NODE_W / 2, NODE_H / 2


def verify():
    """Fail loudly if the curated subset claims a relation the OWL lacks."""
    problems = []
    for _, _, _, rows in LANES:
        for cls, parent in rows:
            if not ONT.is_class(cls):
                problems.append(f"missing class {cls}")
            elif parent and not ONT.is_subclass_of(cls, parent):
                problems.append(f"{cls} is not ⊑ {parent} in ontology")
    for _, _, prop in REDGES:
        if prop not in ONT.object_properties:
            problems.append(f"missing property {prop}")
    if problems:
        raise SystemExit("Diagram/ontology mismatch:\n  " +
                         "\n  ".join(problems))


def label_of(cls: str) -> str:
    return ONT.label(cls)


def box(cx, cy, text, colour, *, head=False, subtitle=""):
    x, y = cx - BOX_HW, cy - BOX_HH
    fill = C[colour]
    stroke = "#ffffff" if head else "rgba(255,255,255,0.35)"
    sw = 2.2 if head else 1
    parts = [f'<rect x="{x:.0f}" y="{y:.0f}" width="{NODE_W}" height="{NODE_H}"'
             f' rx="9" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>']
    if subtitle:
        parts.append(f'<text x="{cx:.0f}" y="{cy-3:.0f}" text-anchor="middle" '
                     f'font-size="12.5" font-weight="700" fill="#fff">'
                     f'{escape(text)}</text>')
        parts.append(f'<text x="{cx:.0f}" y="{cy+13:.0f}" text-anchor="middle"'
                     f' font-size="9.5" fill="rgba(255,255,255,0.85)">'
                     f'{escape(subtitle)}</text>')
    else:
        parts.append(f'<text x="{cx:.0f}" y="{cy+4:.0f}" text-anchor="middle" '
                     f'font-size="{13 if head else 12.5}" '
                     f'font-weight="{800 if head else 600}" fill="#fff">'
                     f'{escape(text)}</text>')
    return "".join(parts)


def clip_to_box(cx, cy, fromx, fromy):
    """Point on the border of box (cx,cy) along direction to (fromx,fromy)."""
    dx, dy = fromx - cx, fromy - cy
    tx = BOX_HW / abs(dx) if dx else 1e9
    ty = BOX_HH / abs(dy) if dy else 1e9
    t = min(tx, ty)
    return cx + dx * t, cy + dy * t


def arrow(sx, sy, ox, oy, prop):
    """Edge from subject border to object border with a labelled pill."""
    sxb, syb = clip_to_box(sx, sy, ox, oy)
    oxb, oyb = clip_to_box(ox, oy, sx, sy)
    mx, my = (sxb + oxb) / 2, (syb + oyb) / 2
    w = 7 * len(prop) + 14
    return (
        f'<line x1="{sxb:.1f}" y1="{syb:.1f}" x2="{oxb:.1f}" y2="{oyb:.1f}" '
        f'stroke="{C["line"]}" stroke-width="1.6" marker-end="url(#arw)"/>'
        f'<rect x="{mx-w/2:.1f}" y="{my-11:.1f}" width="{w}" height="18" '
        f'rx="9" fill="{C["slate"]}"/>'
        f'<text x="{mx:.1f}" y="{my+2:.1f}" text-anchor="middle" '
        f'font-size="10.5" fill="#fff" font-family="monospace">{prop}</text>')


def build() -> str:
    verify()
    H = 800
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
         f'font-family="Inter, Segoe UI, Helvetica, Arial, sans-serif">']
    s.append('<defs><marker id="arw" markerWidth="10" markerHeight="10" '
             'refX="8" refY="3" orient="auto" markerUnits="userSpaceOnUse">'
             f'<path d="M0,0 L9,3 L0,6 Z" fill="{C["line"]}"/></marker></defs>')
    # background card
    s.append(f'<rect x="0" y="0" width="{W}" height="{H}" rx="16" '
             f'fill="{C["panel"]}" stroke="{C["panelStroke"]}"/>')

    # titles
    s.append(f'<text x="28" y="34" font-size="17" font-weight="800" '
             f'fill="{C["title"]}">Personal Health Management ontology '
             f'<tspan font-weight="500" fill="{C["muted"]}">— faithful subset '
             f'(phm: prefix)</tspan></text>')
    s.append(f'<text x="28" y="56" font-size="12" fill="{C["muted"]}">'
             f'All classes are subclasses of phm:Entity. Boxes are owl:Class; '
             f'labelled arrows are owl:ObjectProperty (domain → range).</text>')

    # lane headers + nodes
    for i, (title, colour, _parent, rows) in enumerate(LANES):
        cx = lane_cx(i)
        s.append(f'<rect x="{lane_left(i)}" y="70" width="{LANE_W}" '
                 f'height="26" rx="7" fill="{C[colour]}" opacity="0.16"/>')
        s.append(f'<text x="{cx:.0f}" y="87" text-anchor="middle" '
                 f'font-size="12.5" font-weight="800" fill="{C[colour]}">'
                 f'{escape(title)}</text>')
        # vertical spine linking a head to its descendants
        top_y = HEAD_Y + BOX_HH
        bot_y = HEAD_Y + (len(rows) - 1) * ROW_DY - BOX_HH
        spine_x = lane_left(i) + 8
        s.append(f'<line x1="{spine_x}" y1="{top_y}" x2="{spine_x}" '
                 f'y2="{bot_y}" stroke="{C[colour]}" stroke-width="1.4" '
                 f'opacity="0.5"/>')
        for r, (cls, parent) in enumerate(rows):
            cy = HEAD_Y + r * ROW_DY
            if r > 0:
                s.append(f'<line x1="{spine_x}" y1="{cy}" '
                         f'x2="{cx-BOX_HW:.0f}" y2="{cy}" stroke="{C[colour]}" '
                         f'stroke-width="1.4" opacity="0.5"/>')
            sub = "" if (r == 0 or parent == rows[r-1][0]) else f"⊂ {parent}"
            s.append(box(cx, cy, label_of(cls), colour, head=(r == 0),
                         subtitle=sub))

    # relationships section
    s.append(f'<text x="28" y="452" font-size="15" font-weight="800" '
             f'fill="{C["title"]}">Key relationships '
             f'<tspan font-weight="500" fill="{C["muted"]}">'
             f'(object properties)</tspan></text>')

    # edges first (drawn under nodes)
    for subj, obj, prop in REDGES:
        sx, sy, _ = RNODES[subj]
        ox, oy, _ = RNODES[obj]
        s.append(arrow(sx, sy, ox, oy, prop))
    # performedBy routed through the empty margin to avoid crossings
    iex, iey, _ = RNODES["InterventionExecution"]
    px, py, _ = RNODES["Person"]
    s.append(
        f'<path d="M{iex},{iey+BOX_HH} L{iex},745 L44,745 L44,{py} '
        f'L{px-BOX_HW},{py}" fill="none" stroke="{C["line"]}" '
        f'stroke-width="1.6" marker-end="url(#arw)"/>'
        f'<rect x="360" y="736" width="104" height="18" rx="9" '
        f'fill="{C["slate"]}"/>'
        f'<text x="412" y="749" text-anchor="middle" font-size="10.5" '
        f'fill="#fff" font-family="monospace">performedBy</text>')

    for name, (cx, cy, colour) in RNODES.items():
        s.append(box(cx, cy, label_of(name), colour, head=True))

    s.append(f'<text x="28" y="782" font-size="11" fill="{C["muted"]}">'
             f'Generated from ontology/personal_health_management.owl · '
             f'{len(ONT.classes)} classes, {len(ONT.object_properties)} object '
             f'properties total. Diagram shows a curated subset.</text>')

    s.append('</svg>')
    return "\n".join(s)


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "ontology.svg")
    svg = build()
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(svg)
    print(f"Wrote {out} ({len(svg)} bytes)")
