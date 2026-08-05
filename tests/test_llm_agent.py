"""Exercise the LLM-driven pipeline with a MockProvider (no API calls).

Proves the agent path actually runs: intent understanding, memory updates from
natural language, LLM-composed answers, LLM chart selection, and LLM document
extraction — all deterministically, so it works in CI with no keys.

Usage:  python -m tests.test_llm_agent
"""

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm.base import LLMConfig, LLMProvider, LLMResult
from memory.store import MemoryStore
from ingestion.file_parser import parse_file
from retrieval.orchestrator import handle_turn


class MockProvider(LLMProvider):
    """Scripted provider: returns canned JSON/text based on the input."""
    name = "mock"

    def __init__(self):
        super().__init__(LLMConfig(provider="mock", model="mock-1"))

    def available(self) -> bool:
        return True

    def complete(self, system, user, history=None) -> LLMResult:
        # 1) Router/understand call -> JSON plan
        if '"intent"' in system:
            low = user.lower()
            if "glucose" in low and "?" in user:          # a question
                plan = {"intent": "query",
                        "query": {"metrics": ["glucose"],
                                  "time_window": "last_7_days",
                                  "analysis": "trend"},
                        "chart": {"type": "line", "metrics": ["glucose"],
                                  "title": "Glucose trend"}}
            elif "diabetes" in low or "metformin" in low:  # log + memory facts
                plan = {"intent": "mixed",
                        "observations": [{"metric": "glucose", "value": 180,
                                          "unit": "mg/dL"}],
                        "memory_facts": [
                            {"kind": "condition", "name": "Diabetes mellitus",
                             "ontology_class": "ChronicCondition",
                             "predicate": "hasCondition"},
                            {"kind": "medication", "name": "Metformin",
                             "ontology_class": "Medication",
                             "predicate": "usesMedication"}]}
            else:
                plan = {"intent": "log",
                        "observations": [{"metric": "glucose", "value": 156,
                                          "unit": "mg/dL"}]}
            return LLMResult(json.dumps(plan), self.name, self.config.model)

        # 2) Document extraction call -> JSON array
        if "Extract every health measurement" in system:
            return LLMResult(json.dumps([
                {"metric": "glucose", "value": 165, "unit": "mg/dL"},
                {"metric": "hba1c", "value": 7.1, "unit": "%"}]),
                self.name, self.config.model)

        # 3) Compose-answer call -> prose
        return LLMResult("Your GlucoseObservation readings are rising and the "
                         "latest is above the typical range (associated with "
                         "Diabetes mellitus).", self.name, self.config.model)

    def extract_json(self, system, user):
        from llm.base import _first_json
        return _first_json(self.complete(system, user).text)


def main():
    store = MemoryStore(data_dir=tempfile.mkdtemp())
    prov = MockProvider()

    # 1) Log + memory-fact extraction from natural language
    t1 = handle_turn(store, "I was diagnosed with diabetes and take metformin, "
                            "sugar today was high", prov)
    print(f"[1] mode={t1['mode']} logged={t1['logged']['added']} "
          f"entities={[e['name'] for e in t1['memory']['entities']]}")
    assert t1["used_llm"] and t1["logged"]["added"] >= 1
    assert any(e["name"] == "Metformin" for e in t1["memory"]["entities"])
    assert store.assertions(), "expected memory assertions"

    # add a couple more glucose points for a trend
    handle_turn(store, "glucose 156 mg/dL", prov)
    handle_turn(store, "sugar 168", prov)

    # 2) LLM-understood query -> LLM-composed answer + chosen chart
    t2 = handle_turn(store, "what is my glucose trend?", prov)
    ans = t2["answer"]
    chart_type = ans["chart"]["type"] if ans["chart"] else None
    print(f"[2] used_llm={ans['used_llm']} chart={chart_type} "
          f"summaries={len(ans['summaries'])}")
    print("    answer:", ans["explanation"][:80], "...")
    assert ans["used_llm"] and "Glucose" in ans["explanation"]
    assert ans["chart"] and ans["chart"]["type"] == "line"
    assert ans["grounding"].classes

    # 3) LLM document extraction on file text
    text = b"Lab report 2026-08-01. Blood sugar 165. A1c 7.1%. Notes: fasting."
    recs, _ = parse_file("report.txt", text, provider=prov)
    metrics = sorted({r["metric"] for r in recs})
    print(f"[3] file extraction metrics={metrics}")
    assert "hba1c" in metrics, "LLM should have pulled HbA1c the regex missed"

    print("\n✅ LLM-agent path verified with MockProvider (no API calls).")


if __name__ == "__main__":
    main()
