"""End-to-end smoke test — runs the full pipeline without Streamlit.

Usage:  python -m tests.smoke_test
Uses a throwaway data dir so it never touches real memory.
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.extractor import extract_observations
from ingestion.file_parser import parse_file
from memory.store import MemoryStore
from reports.generator import build_report, report_to_markdown
from retrieval.query import answer_query


def main():
    tmp = tempfile.mkdtemp()
    store = MemoryStore(data_dir=tmp)

    # 1) Chat ingestion across several days
    notes = [
        ("2026-08-01T08:00:00", "fasting glucose 156 mg/dL, BP 128/85, slept 6h"),
        ("2026-08-02T08:00:00", "sugar 142, blood pressure 122/80, slept 7.5 hours"),
        ("2026-08-03T08:00:00", "glucose 168 mg/dL, pulse 74, weight 82 kg"),
        ("2026-08-04T08:00:00", "sugar was 7.9 mmol/L, slept 8 hours, 9000 steps"),
        ("2026-08-05T08:00:00", "glucose 210 mg/dL, BP 140/92, spo2 96%"),
    ]
    total = 0
    for ts, note in notes:
        recs = extract_observations(note, timestamp=ts)
        res = store.add_observations(recs)
        total += res["added"]
    print(f"[1] Ingested {total} observations from chat notes.")
    assert total > 10, "expected many observations"

    # 2) CSV ingestion (long form)
    csv = ("metric,value,unit,timestamp\n"
           "glucose,132,mg/dL,2026-08-06T08:00:00\n"
           "heart_rate,68,bpm,2026-08-06T08:00:00\n").encode()
    recs, _ = parse_file("labs.csv", csv)
    res = store.add_observations(recs)
    print(f"[2] CSV added {res['added']} observations.")
    assert res["added"] == 2

    # 3) Ontology validation rejects a bogus class
    bad = store.add_observations([{"type": "NotARealClass", "numericValue": 1}])
    print(f"[3] Bogus class rejected: {bad['rejected']} "
          f"({bad['reasons']}).")
    assert bad["rejected"] == 1

    # 4) Query with trend + time window
    ans = answer_query(store.all_observations(), "glucose trend last 7 days")
    print("[4] Query explanation:\n   " +
          ans["explanation"].replace("\n", "\n   "))
    assert ans["summaries"], "expected glucose summary"
    assert ans["summaries"][0]["metric"] == "glucose"

    # 5) Report with grouping, anomalies, correlations
    report = build_report(store.all_observations())
    print(f"[5] Report: {report['observation_count']} obs, "
          f"{len(report['groups'])} groups, "
          f"{len(report['anomalies'])} anomalies, "
          f"{len(report['insights'])} insights, "
          f"{len(report['correlations'])} correlations.")
    assert report["observation_count"] > 10
    assert any(a for a in report["anomalies"]), "expected some anomalies"

    # 6) Duplicate detection
    dup = store.seen_upload(csv, "labs.csv")
    store.seen_upload(csv, "labs.csv")
    assert store.seen_upload(csv, "labs.csv") is True

    # 7) SQLite specifics: DB file exists + efficient SQL-side query
    assert os.path.exists(os.path.join(tmp, "health.db")), "no sqlite db"
    glu = store.query_observations(metrics=["glucose"])
    print(f"[7] SQLite query_observations(glucose) -> {len(glu)} rows.")
    assert glu and all(o["metric"] == "glucose" for o in glu)

    # 8) Ontology grounding trace (proves ontology is central)
    from ontology.grounding import build_llm_context, ground
    g = ground(["glucose", "systolic"])
    print(f"[8] Grounding: {len(g.classes)} classes, "
          f"{len(g.conditions)} condition links, "
          f"{len(g.properties)} properties.")
    assert any(c["name"] == "GlucoseObservation" for c in g.classes)
    assert any("Diabetes" in c["name"] for c in g.conditions)
    assert "GlucoseObservation" in build_llm_context(["glucose"])

    # 9) Provider-agnostic LLM layer works with zero SDKs/keys
    from llm import default_config, get_provider, list_providers
    prov = get_provider(default_config())
    print(f"[9] Providers available: {list_providers()}; "
          f"active='{prov.name}', llm_used={prov.available()}.")
    assert set(list_providers()) >= {"none", "openai", "anthropic", "gemini"}
    # answer_query must work with no provider (rule-based) and with Null provider
    a2 = answer_query(store.all_observations(), "glucose last 7 days",
                      provider=prov)
    assert a2["grounding"].classes, "grounding missing on query"
    assert a2["used_llm"] is False  # no key in test env

    # 10) MemoryAssertion (ontology-typed provenance memory)
    store.add_assertion("self", "hasCondition", "Diabetes mellitus",
                        status="Candidate", evidence=[glu[0]["id"]])
    print(f"[10] Memory assertions stored: {len(store.assertions())}.")
    assert store.assertions()

    print("\n--- Markdown report preview ---\n")
    print(report_to_markdown(report)[:900])
    print("\n✅ All smoke checks passed.")


if __name__ == "__main__":
    main()
