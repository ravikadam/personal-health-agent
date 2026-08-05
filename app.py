"""Personal Health Agent — Streamlit UI.

A single-file Streamlit front end over the ingestion / memory / retrieval /
reasoning / report modules. Local-first: all state lives in ./data as JSON.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os
from datetime import datetime

import pandas as pd
import streamlit as st

from ingestion.file_parser import parse_file
from ingestion.metrics import REGISTRY
from llm import (DEFAULT_MODELS, MODEL_CHOICES, LLMConfig, default_config,
                 env_key_for, get_provider, list_providers, sdk_installed)
from memory.store import MemoryStore
from ontology.grounding import ground
from ontology.ontology_loader import load_ontology
from reports.generator import (build_report, report_to_markdown,
                               series_dataframe)
from retrieval.orchestrator import handle_turn

st.set_page_config(page_title="Personal Health Agent", page_icon="🩺",
                   layout="wide")

PERSON = "self"


@st.cache_resource
def get_store() -> MemoryStore:
    return MemoryStore(data_dir="data")


@st.cache_resource
def get_ontology():
    return load_ontology()


store = get_store()
ontology = get_ontology()

if "chat" not in st.session_state:
    st.session_state.chat = []  # list of (role, text)
if "llm_cfg" not in st.session_state:
    st.session_state.llm_cfg = default_config()


def current_provider():
    """Build an LLM provider from the sidebar selection each run."""
    return get_provider(st.session_state.llm_cfg)


def render_grounding(grounding, key: str):
    """Show the ontology grounding trace under a response."""
    with st.expander("🔗 Ontology grounding for this answer", expanded=False):
        st.caption("Every answer is anchored to these ontology classes, "
                   "hierarchy paths and relationships.")
        st.markdown(grounding.as_markdown() or "_No classes matched._")


def render_chart(chart):
    """Render the chart the agent chose (line / bar / scatter)."""
    if not chart or not chart.get("summaries"):
        return
    frames = [series_dataframe(s) for s in chart["summaries"]]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return
    df = pd.concat(frames, axis=1)
    if chart.get("title"):
        st.caption(f"📊 {chart['title']}")
    ctype = chart.get("type", "line")
    if ctype == "bar":
        st.bar_chart(df, height=240)
    elif ctype == "scatter" and df.shape[1] >= 2:
        st.scatter_chart(df, height=240)
    else:
        st.line_chart(df, height=240)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🩺 Health Agent")
    st.caption("Local-first, ontology-aligned memory")

    stats = store.stats()
    st.metric("Observations", stats["observations"])
    c1, c2 = st.columns(2)
    c1.metric("Entities", stats["entities"])
    c2.metric("Links", stats["relationships"])

    st.divider()
    st.subheader("Ontology")
    st.write(f"**{len(ontology.classes)}** classes loaded")
    st.caption("Personal Health Management Ontology (phm)")
    if stats["metrics_tracked"]:
        st.write("**Tracked metrics:**")
        st.write(", ".join(stats["metrics_tracked"]))

    st.divider()
    st.subheader("🤖 LLM provider")
    st.caption("Bring your own model — nothing is hardcoded.")

    cfg: LLMConfig = st.session_state.llm_cfg
    providers = list_providers()
    labels = {"none": "None (rule-based)", "openai": "OpenAI",
              "anthropic": "Claude (Anthropic)", "gemini": "Gemini (Google)"}
    provider = st.selectbox(
        "Provider", providers, index=providers.index(cfg.provider),
        format_func=lambda p: labels.get(p, p) +
        ("" if p == "none" or sdk_installed(p) else "  ⚠️ SDK not installed"))

    model, api_key = cfg.model, cfg.api_key
    if provider != "none":
        choices = MODEL_CHOICES.get(provider, [""])
        default_model = cfg.model if cfg.model in choices else \
            DEFAULT_MODELS.get(provider, choices[0])
        model = st.selectbox("Model", choices,
                             index=choices.index(default_model)
                             if default_model in choices else 0)
        env_present = bool(env_key_for(provider))
        api_key = st.text_input(
            "API key", value="" if env_present else cfg.api_key,
            type="password",
            placeholder="Using environment variable" if env_present
            else "Paste API key",
            help="Kept in session only; never written to disk.")
        if env_present and not api_key:
            api_key = env_key_for(provider)

    # Persist selection
    st.session_state.llm_cfg = LLMConfig(
        provider=provider, model=model, api_key=api_key,
        temperature=cfg.temperature)

    _PKG = {"openai": "openai", "anthropic": "anthropic",
            "gemini": "google-generativeai"}
    prov = current_provider()
    if provider == "none":
        st.caption("Mode: ⚪ deterministic rule-based (no LLM)")
    elif not sdk_installed(provider):
        st.warning(f"Install the SDK: `pip install {_PKG.get(provider, provider)}`")
    elif prov.available():
        st.success(f"🟢 {labels.get(provider)} ready ({model})")
        if st.button("🔍 Test LLM connection"):
            try:
                r = prov.complete("You are a connectivity test. Reply with "
                                  "exactly: OK", "ping")
                st.success(f"Response: {r.text[:120] or '(empty)'}")
            except Exception as exc:  # surface the real, otherwise-swallowed error
                st.error(f"LLM call failed:\n\n{type(exc).__name__}: {exc}")
    else:
        st.info("Add an API key to enable this provider.")

    with st.expander("Supported metrics"):
        for key, m in REGISTRY.items():
            st.write(f"- **{m.label}** ({m.canonical_unit}) → `{m.ontology_class}`")


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_chat, tab_upload, tab_report, tab_memory = st.tabs(
    ["💬 Chat & Log", "📄 Upload", "📊 Report", "🧠 Memory"])


# ---- Chat & Log ----------------------------------------------------------- #
with tab_chat:
    st.subheader("Log readings or ask questions")
    st.caption("Log: “fasting glucose 156 mg/dL, BP 128/85, slept 6.5h”. "
               "Ask: “what's my glucose trend last 7 days?”")

    for role, text in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(text)

    prompt = st.chat_input("Log a reading or ask a question…")
    if prompt:
        st.session_state.chat.append(("user", prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        provider = current_provider()
        turn = handle_turn(store, prompt, provider)
        reply_parts = []

        with st.chat_message("assistant"):
            if turn.get("used_llm"):
                st.caption(f"↳ understood & answered by {provider.name}, "
                           f"grounded in the ontology (intent: {turn['mode']})")

            # Logged observations (with ontology-tiered severity)
            logged = turn.get("logged")
            urgent = False
            if logged and logged.get("records"):
                lines = [f"Logged **{logged['added']}** observation(s):"]
                for e in logged["records"]:
                    ctx = f" _({e['context']})_" if e.get("context") else ""
                    sev = e.get("severity") or {}
                    badge = ""
                    if sev.get("level") == "urgent":
                        urgent = True
                        badge = (f"  🚨 **{sev.get('clinical_name') or 'critical'}"
                                 f"** (`{sev.get('ontology_class')}`)")
                    elif sev.get("level") == "caution":
                        badge = (f"  ⚠️ {sev.get('clinical_name') or 'out of range'}"
                                 f" (`{sev.get('ontology_class')}`)")
                    lines.append(f"- {e['label']}: **{e['numericValue']} "
                                 f"{e['unit']}**{ctx} → `{e['type']}`{badge}")
                if logged.get("rejected"):
                    lines.append(f"\n_Rejected {logged['rejected']}: "
                                 f"{'; '.join(logged['reasons'])}_")
                block = "\n".join(lines)
                st.markdown(block)
                if urgent:
                    st.error("🚨 A reading is in the urgent range — please act "
                             "promptly and consider contacting a clinician.")
                reply_parts.append(block)

            # Self-reported symptoms → SymptomObservation
            sym = turn.get("symptoms")
            if sym and sym.get("records"):
                names = ", ".join(f"**{r['label']}**" for r in sym["records"])
                block = f"🩹 Noted symptom(s): {names} → `SymptomObservation`"
                st.markdown(block)
                reply_parts.append(block)

            # Symptom↔reading associations → AssociationAssessment / hypothesis
            assoc = turn.get("associations")
            if assoc and assoc.get("entities"):
                lines = ["🔗 Recorded association(s):"]
                for e in assoc["entities"]:
                    conf = e.get("confidence")
                    ctag = f" · conf {conf}" if conf is not None else ""
                    lines.append(f"- **{e['name']}** (`{e['type']}`){ctag}")
                block = "\n".join(lines)
                st.markdown(block)
                reply_parts.append(block)

            # Memory facts (anything durable the LLM decided to remember)
            mem = turn.get("memory")
            if mem and mem.get("entities"):
                preds = {(a["object"] or "").lower(): a["predicate"]
                         for a in mem.get("assertions", [])}
                lines = ["🧠 Remembered:"]
                for e in mem["entities"]:
                    pred = preds.get(e["name"].lower())
                    via = f" · `self {pred}`" if pred else ""
                    extra = f" — {e.get('note')}" if e.get("note") else ""
                    lines.append(f"- **{e['name']}** (`{e['type']}`){via}{extra}")
                block = "\n".join(lines)
                st.markdown(block)
                reply_parts.append(block)

            # Answer to a question
            ans = turn.get("answer")
            if ans:
                st.markdown(ans["explanation"])
                reply_parts.append(ans["explanation"])
                render_chart(ans.get("chart"))
                if ans.get("correlation"):
                    c = ans["correlation"]
                    st.caption(f"Correlation r={c['r']} ({c['strength']} "
                               f"{c['direction']}, n={c['n']}).")
                render_grounding(ans["grounding"], key="qry")
            elif logged:
                render_grounding(
                    ground([r["metric"] for r in logged["records"]]),
                    key="log")

            if not reply_parts:
                fallback = ("I didn't catch a reading or a question there. "
                            "Try “glucose 130 mg/dL” or “glucose trend last "
                            "7 days?”.")
                st.markdown(fallback)
                reply_parts.append(fallback)

        st.session_state.chat.append(("assistant", "\n\n".join(reply_parts)))
        st.rerun()


# ---- Upload --------------------------------------------------------------- #
with tab_upload:
    st.subheader("Upload health reports")
    st.caption("PDF, CSV, image (OCR), or text. Data is mapped to ontology "
               "observations and appended to memory. Duplicates are detected.")

    upload_provider = current_provider()
    if upload_provider.available():
        st.caption(f"🤖 {upload_provider.name} will help read report text.")

    files = st.file_uploader(
        "Choose file(s)", type=["pdf", "csv", "png", "jpg", "jpeg", "txt",
                                "md", "tiff", "bmp"],
        accept_multiple_files=True)

    if files and st.button("Ingest files", type="primary"):
        for f in files:
            data = f.getvalue()
            if store.seen_upload(data, f.name):
                st.warning(f"⏭️ `{f.name}` looks like a duplicate — skipped.")
                continue
            try:
                records, text = parse_file(f.name, data, person=PERSON,
                                           provider=upload_provider)
            except Exception as exc:
                st.error(f"❌ `{f.name}`: {exc}")
                continue
            result = store.add_observations(records)
            st.success(f"✅ `{f.name}`: added {result['added']} observation(s)"
                       + (f", rejected {result['rejected']}"
                          if result['rejected'] else ""))
            if records:
                st.dataframe(pd.DataFrame(records)[
                    ["label", "numericValue", "unit", "type", "timestamp"]],
                    use_container_width=True, hide_index=True)
            with st.expander(f"Extracted text — {f.name}"):
                st.text((text or "")[:4000])
        st.rerun()


# ---- Report --------------------------------------------------------------- #
with tab_report:
    st.subheader("Structured health report")
    observations = store.all_observations()
    if not observations:
        st.info("No data yet. Log readings or upload a report first.")
    else:
        report = build_report(observations, person=PERSON)

        top = st.columns(3)
        top[0].metric("Observations", report["observation_count"])
        top[1].metric("Metrics tracked", len(report["metrics"]))
        top[2].metric("Anomalies", len(report["anomalies"]))

        # Ontology-grounded LLM narrative (optional)
        provider = current_provider()
        if provider.available():
            if st.button("🤖 Generate ontology-grounded narrative",
                         type="secondary"):
                from ontology.grounding import build_llm_context
                facts = {g["title"]: [
                    {"metric": s["label"],
                     "ontology_class": REGISTRY[s["metric"]].ontology_class
                     if s["metric"] in REGISTRY else None,
                     "latest": s["latest"], "avg": s["avg"],
                     "trend": s["trend"]["direction"],
                     "anomalies": len(s["anomalies"])}
                    for s in g["summaries"]] for g in report["groups"]}
                system = build_llm_context(report["metrics"]) + (
                    "\n\nTASK: Write a brief health-report narrative from the "
                    "JSON facts. Group by condition, reference ontology "
                    "classes and linked conditions, use association (not "
                    "causal) language. Do not invent numbers.")
                with st.spinner(f"Writing with {provider.name}…"):
                    import json as _json
                    res = provider.complete(system, _json.dumps(facts,
                                                                default=str))
                st.session_state["report_narrative"] = res.text
            if st.session_state.get("report_narrative"):
                st.markdown("### 📝 Narrative")
                st.markdown(st.session_state["report_narrative"])

        if report["insights"]:
            st.markdown("### 💡 Key insights")
            for i in report["insights"]:
                st.markdown(f"- {i}")

        with st.expander("🔗 Ontology grounding for this report"):
            st.markdown(ground(report["metrics"]).as_markdown()
                        or "_No classes matched._")

        st.markdown("### 📈 Trends by condition")
        for group in report["groups"]:
            st.markdown(f"#### {group['title']}")
            for s in group["summaries"]:
                cols = st.columns([3, 1])
                with cols[0]:
                    df = series_dataframe(s)
                    if not df.empty and len(df) > 1:
                        st.line_chart(df, height=200)
                    else:
                        st.caption(f"{s['label']}: single reading "
                                   f"{s['latest']} {s.get('unit') or ''}")
                with cols[1]:
                    unit = s.get("unit") or ""
                    st.metric(s["label"], f"{s['latest']}{unit}",
                              delta=f"{s['trend']['change']:+g}{unit}"
                              if s["count"] > 1 else None)
                    st.caption(f"avg {s['avg']} • n={s['count']}")

        if report["correlations"]:
            st.markdown("### 🔗 Correlations")
            for c in report["correlations"]:
                la = REGISTRY[c['metric_a']].label
                lb = REGISTRY[c['metric_b']].label
                st.markdown(f"- **{la} vs {lb}**: {c['strength']} "
                            f"{c['direction']} (r={c['r']}, n={c['n']})")

        if report["anomalies"]:
            st.markdown("### ⚠️ Anomalies")
            st.dataframe(pd.DataFrame(report["anomalies"]),
                         use_container_width=True, hide_index=True)

        md = report_to_markdown(report)
        st.download_button("⬇️ Download report (Markdown)", md,
                           file_name=f"health_report_"
                           f"{datetime.now():%Y%m%d}.md")


# ---- Memory --------------------------------------------------------------- #
with tab_memory:
    st.subheader("Ontology-aligned memory")
    observations = store.all_observations()
    st.caption("Append-only observation log. Every record is validated "
               "against the ontology on write.")
    if observations:
        df = pd.DataFrame(observations)
        show_cols = [c for c in ["timestamp", "label", "numericValue", "unit",
                                 "type", "source", "id"] if c in df.columns]
        st.dataframe(df[show_cols].sort_values("timestamp", ascending=False),
                     use_container_width=True, hide_index=True)
    else:
        st.info("Memory is empty.")

    with st.expander("Entities, relationships & memory assertions"):
        st.write("**Entities**")
        st.json(store.entities() or [])
        st.write("**Relationships** (validated against property domain/range)")
        st.json(store.relationships() or [])
        st.write("**Memory assertions** (phm:MemoryAssertion)")
        st.json(store.assertions() or [])

    with st.expander("Ontology class hierarchy (Observation branch)"):
        obs_classes = sorted(ontology.descendants("Observation"))
        for c in obs_classes:
            parents = ", ".join(sorted(ontology.subclass_of.get(c, []))) or "—"
            st.write(f"- **{ontology.label(c)}** (`{c}`) ⊂ {parents}")

    st.divider()
    if st.button("🗑️ Reset memory (delete all data)"):
        import shutil
        shutil.rmtree("data", ignore_errors=True)
        st.cache_resource.clear()
        st.session_state.chat = []
        st.rerun()
