"""Ontology loader and mapper.

Loads the Personal Health Management OWL/TTL ontology and exposes helpers to:
  * enumerate valid classes and their labels / hierarchy
  * validate that an entity type is a known ontology class
  * validate object-property (relationship) usage against domain/range
  * map free-text synonyms (e.g. "sugar" -> Glucose observation) to classes

The ontology is used for *structure*, not heavy reasoning, per the design
principles: we read the class hierarchy and property domains/ranges and use
them to keep memory records ontology-aligned.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

try:
    from rdflib import Graph, RDF, RDFS, OWL, URIRef
    from rdflib.namespace import Namespace
    _HAS_RDFLIB = True
except Exception:  # pragma: no cover - rdflib should be installed
    _HAS_RDFLIB = False


PHM = "https://w3id.org/phm/ontology#"

DEFAULT_ONTOLOGY_PATH = os.path.join(
    os.path.dirname(__file__), "personal_health_management.owl"
)


@dataclass
class PropertyInfo:
    name: str
    label: str
    domain: Optional[str]
    range: Optional[str]


@dataclass
class OntologyModel:
    """Parsed, queryable view of the ontology."""

    classes: Dict[str, str] = field(default_factory=dict)          # local_name -> label
    subclass_of: Dict[str, Set[str]] = field(default_factory=dict)  # local_name -> parents
    object_properties: Dict[str, PropertyInfo] = field(default_factory=dict)
    data_properties: Dict[str, PropertyInfo] = field(default_factory=dict)
    namespace: str = PHM

    # ---- class helpers -------------------------------------------------
    def is_class(self, name: str) -> bool:
        return self.local(name) in self.classes

    def local(self, name: str) -> str:
        """Normalise a possibly-prefixed / full-IRI name to its local part."""
        if not name:
            return name
        if name.startswith("phm:"):
            return name[len("phm:"):]
        if name.startswith(self.namespace):
            return name[len(self.namespace):]
        if "#" in name:
            return name.rsplit("#", 1)[-1]
        return name

    def ancestors(self, name: str) -> Set[str]:
        """All transitive superclasses of a class local name."""
        name = self.local(name)
        seen: Set[str] = set()
        stack = list(self.subclass_of.get(name, set()))
        while stack:
            parent = stack.pop()
            if parent in seen:
                continue
            seen.add(parent)
            stack.extend(self.subclass_of.get(parent, set()))
        return seen

    def is_subclass_of(self, name: str, ancestor: str) -> bool:
        name, ancestor = self.local(name), self.local(ancestor)
        return name == ancestor or ancestor in self.ancestors(name)

    def label(self, name: str) -> str:
        return self.classes.get(self.local(name), self.local(name))

    def descendants(self, name: str) -> Set[str]:
        name = self.local(name)
        result: Set[str] = set()
        for cls, parents in self.subclass_of.items():
            if name in self.ancestors(cls) or name in parents:
                result.add(cls)
        return result

    # ---- property helpers ----------------------------------------------
    def validate_relationship(self, prop: str, subject_type: str,
                              object_type: str) -> tuple[bool, str]:
        """Check a relationship against the property's domain and range.

        Returns (ok, message). Unknown properties are rejected; missing
        domain/range on a known property is treated as permissive.
        """
        p = self.local(prop)
        info = self.object_properties.get(p)
        if info is None:
            return False, f"Unknown object property '{prop}'."
        if info.domain and subject_type and not self.is_subclass_of(
            subject_type, info.domain
        ):
            return (
                False,
                f"Domain mismatch: {info.name} expects subject "
                f"{info.domain}, got {self.local(subject_type)}.",
            )
        if info.range and object_type and not self.is_subclass_of(
            object_type, info.range
        ):
            return (
                False,
                f"Range mismatch: {info.name} expects object "
                f"{info.range}, got {self.local(object_type)}.",
            )
        return True, "ok"


def _load_graph(path: str) -> "Graph":
    g = Graph()
    # The file uses Turtle syntax despite the .owl extension.
    for fmt in ("turtle", "xml", "n3"):
        try:
            g.parse(path, format=fmt)
            return g
        except Exception:
            continue
    raise ValueError(f"Could not parse ontology at {path}")


@functools.lru_cache(maxsize=4)
def load_ontology(path: str = DEFAULT_ONTOLOGY_PATH) -> OntologyModel:
    """Parse the ontology file into an OntologyModel (cached)."""
    if not _HAS_RDFLIB:
        raise RuntimeError(
            "rdflib is required to load the ontology. "
            "Install it with `pip install rdflib`."
        )
    g = _load_graph(path)
    ns = Namespace(PHM)
    model = OntologyModel(namespace=PHM)

    def local(uri: URIRef) -> str:
        s = str(uri)
        if s.startswith(PHM):
            return s[len(PHM):]
        if "#" in s:
            return s.rsplit("#", 1)[-1]
        return s

    # Classes + labels
    for cls in g.subjects(RDF.type, OWL.Class):
        if not isinstance(cls, URIRef):
            continue  # skip anonymous restriction nodes
        name = local(cls)
        label = g.value(cls, RDFS.label)
        model.classes[name] = str(label) if label else name
        model.subclass_of.setdefault(name, set())

    # Subclass edges (named superclasses only)
    for sub, sup in g.subject_objects(RDFS.subClassOf):
        if isinstance(sub, URIRef) and isinstance(sup, URIRef):
            model.subclass_of.setdefault(local(sub), set()).add(local(sup))

    # Object + data properties
    def read_prop(uri) -> PropertyInfo:
        dom = g.value(uri, RDFS.domain)
        rng = g.value(uri, RDFS.range)
        lbl = g.value(uri, RDFS.label)
        return PropertyInfo(
            name=local(uri),
            label=str(lbl) if lbl else local(uri),
            domain=local(dom) if isinstance(dom, URIRef) else None,
            range=local(rng) if isinstance(rng, URIRef) else None,
        )

    for p in g.subjects(RDF.type, OWL.ObjectProperty):
        if isinstance(p, URIRef):
            info = read_prop(p)
            model.object_properties[info.name] = info
    for p in g.subjects(RDF.type, OWL.DatatypeProperty):
        if isinstance(p, URIRef):
            info = read_prop(p)
            model.data_properties[info.name] = info

    return model


if __name__ == "__main__":
    m = load_ontology()
    print(f"Loaded {len(m.classes)} classes, "
          f"{len(m.object_properties)} object properties, "
          f"{len(m.data_properties)} data properties.")
    print("GlucoseObservation ancestors:", sorted(m.ancestors("GlucoseObservation")))
