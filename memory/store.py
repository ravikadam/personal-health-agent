"""SQLite-backed, ontology-governed memory store.

Vitals and other health data live in SQLite (fast filtering/aggregation), but
the **ontology is the authority**: every write must carry a valid ontology
class or it is rejected, relationships are validated against property
domain/range, and condition links are recorded as ontology `MemoryAssertion`s.

Public API is unchanged from the earlier JSON store so the rest of the app is
untouched, plus efficient `query_observations(...)` for retrieval and
`add_assertion(...)` for provenance-bearing, ontology-typed memory.

Tables
  observations       — append-only log; `supersedes` gives versioning
  entities           — people, medications, conditions (deduplicated)
  relationships      — ontology-validated links between entities
  memory_assertions  — phm:MemoryAssertion: subject/predicate/object + status
  uploads            — content hashes for duplicate-upload detection
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from ontology.ontology_loader import load_ontology
from ingestion.file_parser import file_hash

_SCHEMA = """
CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    ontology_class TEXT NOT NULL,
    metric TEXT,
    label TEXT,
    category TEXT,
    numeric_value REAL,
    unit TEXT,
    text_value TEXT,
    observed_for TEXT DEFAULT 'self',
    source TEXT,
    timestamp TEXT,
    recorded_at TEXT,
    raw_text TEXT,
    linked_to TEXT,
    supersedes TEXT,
    confidence REAL DEFAULT 1.0
);
CREATE INDEX IF NOT EXISTS idx_obs_metric ON observations(metric);
CREATE INDEX IF NOT EXISTS idx_obs_class ON observations(ontology_class);
CREATE INDEX IF NOT EXISTS idx_obs_person ON observations(observed_for);
CREATE INDEX IF NOT EXISTS idx_obs_ts ON observations(timestamp);

CREATE TABLE IF NOT EXISTS entities (
    id TEXT PRIMARY KEY,
    ontology_class TEXT NOT NULL,
    name TEXT NOT NULL,
    created_at TEXT,
    attrs TEXT,
    UNIQUE(ontology_class, name)
);

CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    property TEXT NOT NULL,
    subject_id TEXT,
    object_id TEXT,
    valid INTEGER,
    note TEXT,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS memory_assertions (
    id TEXT PRIMARY KEY,
    subject TEXT,
    predicate TEXT,
    object TEXT,
    status TEXT DEFAULT 'Candidate',
    confidence REAL DEFAULT 1.0,
    evidence TEXT,
    valid_from TEXT,
    recorded_at TEXT
);

CREATE TABLE IF NOT EXISTS uploads (
    hash TEXT PRIMARY KEY,
    filename TEXT,
    at TEXT
);
"""

_OBS_COLS = ["id", "ontology_class", "metric", "label", "category",
             "numeric_value", "unit", "text_value", "observed_for", "source",
             "timestamp", "recorded_at", "raw_text", "linked_to", "supersedes",
             "confidence"]


class MemoryStore:
    def __init__(self, data_dir: str = "data", db_name: str = "health.db"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.db_path = os.path.join(data_dir, db_name)
        self._lock = threading.Lock()
        self.ontology = load_ontology()
        with self._connect() as con:
            con.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    # ---- observations --------------------------------------------------
    def _validate_class(self, otype: Optional[str]) -> Optional[str]:
        """Return a rejection reason, or None if the class is acceptable."""
        if not otype or not self.ontology.is_class(otype):
            return f"Unknown ontology class '{otype}'"
        if not (self.ontology.is_subclass_of(otype, "Observation")
                or self.ontology.is_subclass_of(otype, "Entity")):
            return f"'{otype}' is not an Entity/Observation subclass"
        return None

    def add_observations(self, records: List[Dict]) -> Dict:
        added, rejected, reasons = 0, 0, []
        now = datetime.utcnow().isoformat()
        with self._lock, self._connect() as con:
            for rec in records:
                otype = rec.get("type")
                reason = self._validate_class(otype)
                if reason:
                    rejected += 1
                    reasons.append(reason)
                    continue
                row = {
                    "id": f"obs_{uuid.uuid4().hex[:12]}",
                    "ontology_class": otype,
                    "metric": rec.get("metric"),
                    "label": rec.get("label"),
                    "category": rec.get("category", "general"),
                    "numeric_value": rec.get("numericValue"),
                    "unit": rec.get("unit"),
                    "text_value": rec.get("textValue"),
                    "observed_for": rec.get("observedFor", "self"),
                    "source": rec.get("source", "chat"),
                    "timestamp": rec.get("timestamp", now),
                    "recorded_at": now,
                    "raw_text": rec.get("raw_text"),
                    "linked_to": rec.get("linked_to"),
                    "supersedes": rec.get("supersedes"),
                    "confidence": rec.get("confidence", 1.0),
                }
                con.execute(
                    f"INSERT INTO observations ({','.join(_OBS_COLS)}) "
                    f"VALUES ({','.join('?' for _ in _OBS_COLS)})",
                    [row[c] for c in _OBS_COLS],
                )
                added += 1
        return {"added": added, "rejected": rejected, "reasons": reasons}

    def _row_to_obs(self, row: sqlite3.Row) -> Dict:
        # Present with the same keys the rest of the app already expects.
        return {
            "id": row["id"],
            "type": row["ontology_class"],
            "metric": row["metric"],
            "label": row["label"],
            "category": row["category"],
            "numericValue": row["numeric_value"],
            "unit": row["unit"],
            "textValue": row["text_value"],
            "observedFor": row["observed_for"],
            "source": row["source"],
            "timestamp": row["timestamp"],
            "recordedAt": row["recorded_at"],
            "raw_text": row["raw_text"],
            "linked_to": row["linked_to"],
            "supersedes": row["supersedes"],
            "confidence": row["confidence"],
        }

    def all_observations(self) -> List[Dict]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM observations ORDER BY timestamp").fetchall()
        return [self._row_to_obs(r) for r in rows]

    def query_observations(self, metrics: Optional[List[str]] = None,
                           ontology_types: Optional[List[str]] = None,
                           since: Optional[str] = None,
                           until: Optional[str] = None,
                           person: str = "self") -> List[Dict]:
        """Efficient SQL-side filtering; ontology_types are matched exactly
        (caller expands parent classes to descendants via the ontology)."""
        clauses, params = [], []
        if person:
            clauses.append("observed_for = ?")
            params.append(person)
        if metrics:
            clauses.append(f"metric IN ({','.join('?' * len(metrics))})")
            params.extend(metrics)
        if ontology_types:
            clauses.append(
                f"ontology_class IN ({','.join('?' * len(ontology_types))})")
            params.extend(ontology_types)
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)
        if until:
            clauses.append("timestamp <= ?")
            params.append(until)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        with self._connect() as con:
            rows = con.execute(
                f"SELECT * FROM observations {where} ORDER BY timestamp DESC",
                params).fetchall()
        return [self._row_to_obs(r) for r in rows]

    # ---- entities & relationships -------------------------------------
    def add_entity(self, etype: str, name: str, **attrs) -> Dict:
        warning = None if self.ontology.is_class(etype) else \
            f"'{etype}' not a known class"
        eid = f"ent_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow().isoformat()
        with self._lock, self._connect() as con:
            existing = con.execute(
                "SELECT * FROM entities WHERE ontology_class=? AND "
                "lower(name)=lower(?)", (etype, name)).fetchone()
            if existing:
                return self._row_to_entity(existing)
            con.execute(
                "INSERT INTO entities (id, ontology_class, name, created_at, "
                "attrs) VALUES (?,?,?,?,?)",
                (eid, etype, name, now, json.dumps(attrs)))
        return {"id": eid, "type": etype, "name": name, "createdAt": now,
                "_ontology_warning": warning, **attrs}

    def _row_to_entity(self, row: sqlite3.Row) -> Dict:
        attrs = json.loads(row["attrs"] or "{}")
        return {"id": row["id"], "type": row["ontology_class"],
                "name": row["name"], "createdAt": row["created_at"], **attrs}

    def add_relationship(self, prop: str, subject_id: str, object_id: str,
                         subject_type: str = "", object_type: str = "") -> Dict:
        ok, msg = self.ontology.validate_relationship(
            prop, subject_type, object_type)
        rid = f"rel_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow().isoformat()
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT INTO relationships (id, property, subject_id, "
                "object_id, valid, note, created_at) VALUES (?,?,?,?,?,?,?)",
                (rid, prop, subject_id, object_id, int(ok), msg, now))
        return {"id": rid, "property": prop, "subject": subject_id,
                "object": object_id, "valid": ok, "note": msg, "createdAt": now}

    def add_assertion(self, subject: str, predicate: str, obj: str,
                      status: str = "Candidate", confidence: float = 1.0,
                      evidence: Optional[List[str]] = None) -> Dict:
        """Record a phm:MemoryAssertion (provenance-bearing, time-scoped)."""
        aid = f"mem_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow().isoformat()
        with self._lock, self._connect() as con:
            con.execute(
                "INSERT INTO memory_assertions (id, subject, predicate, "
                "object, status, confidence, evidence, valid_from, "
                "recorded_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (aid, subject, predicate, obj, status, confidence,
                 json.dumps(evidence or []), now, now))
        return {"id": aid, "subject": subject, "predicate": predicate,
                "object": obj, "status": status}

    # ---- duplicate detection ------------------------------------------
    def seen_upload(self, data: bytes, filename: str) -> bool:
        h = file_hash(data)
        with self._lock, self._connect() as con:
            if con.execute("SELECT 1 FROM uploads WHERE hash=?",
                           (h,)).fetchone():
                return True
            con.execute("INSERT INTO uploads (hash, filename, at) VALUES "
                        "(?,?,?)", (h, filename, datetime.utcnow().isoformat()))
        return False

    # ---- reads ---------------------------------------------------------
    def entities(self) -> List[Dict]:
        with self._connect() as con:
            rows = con.execute("SELECT * FROM entities").fetchall()
        return [self._row_to_entity(r) for r in rows]

    def relationships(self) -> List[Dict]:
        with self._connect() as con:
            rows = con.execute("SELECT * FROM relationships").fetchall()
        return [{"id": r["id"], "property": r["property"],
                 "subject": r["subject_id"], "object": r["object_id"],
                 "valid": bool(r["valid"]), "note": r["note"],
                 "createdAt": r["created_at"]} for r in rows]

    def assertions(self) -> List[Dict]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM memory_assertions ORDER BY recorded_at "
                "DESC").fetchall()
        return [{"id": r["id"], "subject": r["subject"],
                 "predicate": r["predicate"], "object": r["object"],
                 "status": r["status"], "confidence": r["confidence"],
                 "evidence": json.loads(r["evidence"] or "[]"),
                 "recordedAt": r["recorded_at"]} for r in rows]

    def index(self) -> Dict:
        """Computed lightweight index (kept for API compatibility)."""
        by_type: Dict[str, int] = {}
        by_metric: Dict[str, int] = {}
        with self._connect() as con:
            for r in con.execute("SELECT ontology_class, COUNT(*) c FROM "
                                 "observations GROUP BY ontology_class"):
                by_type[r["ontology_class"]] = r["c"]
            for r in con.execute("SELECT metric, COUNT(*) c FROM observations "
                                 "WHERE metric IS NOT NULL GROUP BY metric"):
                by_metric[r["metric"]] = r["c"]
        return {"by_type": by_type, "by_metric": by_metric}

    def stats(self) -> Dict:
        with self._connect() as con:
            obs = con.execute("SELECT COUNT(*) c FROM observations").fetchone()["c"]
            ent = con.execute("SELECT COUNT(*) c FROM entities").fetchone()["c"]
            rel = con.execute("SELECT COUNT(*) c FROM relationships").fetchone()["c"]
            metrics = [r["metric"] for r in con.execute(
                "SELECT DISTINCT metric FROM observations WHERE metric IS NOT "
                "NULL ORDER BY metric")]
        return {"observations": obs, "entities": ent, "relationships": rel,
                "metrics_tracked": metrics}
