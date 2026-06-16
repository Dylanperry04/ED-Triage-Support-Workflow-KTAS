"""
AI Triage Agentic System — Streamlit UI (Kaggle KTAS research mode).

Six tabs:
  1. Triage Review     — run the deterministic workflow on a selected case,
                          view the rules engine output, safety flags, and ML
                          research estimate, and submit a clinician review.
  2. Clinician Chat     — ask the real AutoGen chat agent (app/agents/autogen_team.py)
                          about a selected case. The agent only explains
                          already-computed evidence; it never assigns a
                          category or invents a value.
  3. Governance         — five-stage responsible-AI review gate, adapted for
                          KTAS, summarising whether this system is ready for
                          anything beyond a research demo (it is not).
  4. Review Queue       — cases with missing triage-time data that need a
                          clinician's attention before being relied on.
  5. Audit Log          — full history of saved clinician reviews.
  6. Model Performance  — the two trained KTAS research models (5-class and
                          emergency binary) and their cross-validated metrics.

NOT FOR CLINICAL USE. Research prototype only. Every output requires
clinician confirmation. KTAS is not Manchester Triage Scale; no mapping
between them exists anywhere in this codebase.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.storage.jsonl_repository import read_jsonl
from app.schemas.internal import EDTriageCase
from app.schemas.review import HumanReviewRecord
from app.agents.orchestrator import run_workflow
from app.storage.human_review_repository import (
    append_human_review,
    get_reviews_for_stay,
    read_human_reviews,
)
from app.agents.autogen_team import load_azure_config, run_single_question


st.set_page_config(
    page_title="AI Triage Agentic System — KTAS",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Helper functions ──────────────────────────────────────────────────────────

def load_cases() -> list[dict]:
    path = settings.processed_dir / "triage_cases_sample.jsonl"
    if not path.exists():
        st.error(
            "No processed cases found. Run:\n\n```\npython scripts/run_ktas_pipeline.py\n```"
        )
        st.stop()
    return read_jsonl(path)


def load_json_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model_registry() -> dict | None:
    return load_json_file(settings.model_registry_path)


def fmt_pct(value) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def _status_badge(status: str) -> str:
    if status in ("PASS", "TRIAGE_INPUT_DATA_COMPLETE"):
        return f"✅ {status}"
    if status in ("NOT_CONFIGURED", "NOT_REQUESTED", "NOT_RUN"):
        return f"⚪ {status}"
    if "FAIL" in status or "MISSING" in status or "NEEDS" in status or "ERROR" in status:
        return f"⚠️ {status}"
    return f"ℹ️ {status}"


REASON_CODE_SEVERITY_PREFIXES = ("CRITICAL", "FORBIDDEN")


def _flag_icon(flag: str) -> str:
    upper = flag.upper()
    if any(p in upper for p in REASON_CODE_SEVERITY_PREFIXES):
        return "🔴"
    if "MISSING" in upper or "CONCERN" in upper:
        return "🟡"
    return "ℹ️"


# ── Safety banner ─────────────────────────────────────────────────────────────

st.markdown(
    """
<div style='background:#fef3cd;padding:14px 18px;border-radius:8px;
            border-left:5px solid #f0a500;margin-bottom:18px;'>
<b>🚨 NOT FOR CLINICAL USE — Research Prototype Only</b><br>
This version uses the public Kaggle KTAS dataset. KTAS is not Manchester
Triage Scale, and no KTAS-to-Manchester mapping exists anywhere in this
codebase. This system does not make autonomous triage decisions. Every
output requires clinician review, confirmation, and sign-off.
</div>
""",
    unsafe_allow_html=True,
)

st.title("🏥 AI Triage Agentic System — Kaggle KTAS Research Mode")
st.caption(
    "Multi-agent triage decision support, orchestrated with AutoGen for the "
    "explanation layer. Dataset phase: public KTAS research data. Future "
    "phase: MIMIC-IV-ED / UHL validation, subject to approvals."
)

records = load_cases()

tab_triage, tab_chat, tab_governance, tab_queue, tab_audit, tab_models = st.tabs(
    [
        "🩺 Triage Review",
        "💬 Clinician Chat",
        "🔒 Governance",
        "📋 Review Queue",
        "📜 Audit Log",
        "📊 Model Performance",
    ]
)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — TRIAGE REVIEW
# ═══════════════════════════════════════════════════════════════════════════
with tab_triage:
    st.subheader("Select ED Stay for Review")

    case_options = {
        f"Stay {r['stay_id']} — "
        f"{(r.get('triage') or {}).get('chiefcomplaint') or 'No complaint'}": r
        for r in records
    }
    selected_label = st.selectbox(
        "ED Stay", list(case_options.keys()), label_visibility="collapsed"
    )
    selected_record = case_options[selected_label]
    case = EDTriageCase(**selected_record)

    with st.spinner("Running deterministic triage workflow..."):
        result = run_workflow(case, include_llm_explanation=False)

    ti = result.triage_input
    labels = result.retrospective_labels

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Stay ID", str(result.stay_id))
    col2.metric(
        "Data Status", result.data_validation.validation_status.replace("_", " ")
    )
    col3.metric(
        "Rules Engine Status",
        result.decision.classification_status.replace("_", " "),
        help="No Manchester category is ever assigned without a clinician-approved ruleset.",
    )
    col4.metric(
        "ML Emergency Estimate",
        fmt_pct(result.ml_prediction.emergency_research_estimate),
        help=(
            f"Research-grade estimate only, model: {result.ml_prediction.model_name}. "
            "P(KTAS_expert in {1,2,3}) — not a clinical decision."
        ),
    )

    st.markdown("---")
    with st.expander(
        "📋 Triage-Time Inputs (available at triage — no retrospective data)",
        expanded=True,
    ):
        v_col1, v_col2, v_col3, v_col4 = st.columns(4)
        v_col1.metric("Chief Complaint", ti.chiefcomplaint or "⚠️ MISSING")
        v_col2.metric("Arrival", ti.arrival_transport or "Unknown")
        v_col3.metric("Gender", ti.gender or "Not recorded")
        v_col4.metric("Age", ti.age if ti.age is not None else "Not recorded")

        st.markdown("**Vital Signs**")
        v1, v2, v3, v4, v5, v6, v7 = st.columns(7)
        v1.metric(f"Temp (°{ti.temperature_unit})", ti.temperature if ti.temperature is not None else "⚠️ MISSING")
        v2.metric("HR (bpm)", ti.heartrate if ti.heartrate is not None else "⚠️ MISSING")
        v3.metric("RR (/min)", ti.resprate if ti.resprate is not None else "⚠️ MISSING")
        v4.metric("SpO2 (%)", ti.o2sat if ti.o2sat is not None else "⚠️ MISSING")
        v5.metric("SBP (mmHg)", ti.sbp if ti.sbp is not None else "⚠️ MISSING")
        v6.metric("DBP (mmHg)", ti.dbp if ti.dbp is not None else "⚠️ MISSING")
        v7.metric(
            "Pain (0-10)",
            ti.nrs_pain if ti.nrs_pain is not None else (ti.pain or "⚠️ MISSING"),
        )

        if result.data_validation.missing_required_fields:
            st.warning(
                f"Missing fields: {', '.join(result.data_validation.missing_required_fields)}"
            )
        if result.data_validation.non_informative_fields:
            st.info(
                f"Non-informative fields: {', '.join(result.data_validation.non_informative_fields)}"
            )

    st.markdown("---")
    st.subheader("🔍 Clinical Safety Assessment")
    st.caption(
        "Deterministic physiology and pathway analysis — no LLM. No MTS category "
        "assigned without an approved clinical ruleset. Clinician confirmation ALWAYS required."
    )

    dec = result.decision
    if dec.priority is not None:
        st.markdown(
            f"**{dec.category}** — max wait {dec.max_wait_minutes} minutes — "
            f"status: {dec.classification_status}"
        )
    else:
        st.error(
            f"⚠️ **{dec.classification_status.replace('_', ' ')}**\n\n"
            "No Manchester category assigned. Clinician review required."
        )

    if dec.reason_codes:
        with st.expander("Rules engine reason codes"):
            for code in dec.reason_codes:
                st.code(f"{_flag_icon(code)} {code}")

    st.info("🔒 **Clinician confirmation is ALWAYS required** before any action on this output.")

    safety = result.safety_review
    if safety.data_quality_flags:
        with st.expander(
            f"⚠️ Safety Flags ({len(safety.data_quality_flags)} issues detected)",
            expanded=True,
        ):
            for flag in safety.data_quality_flags:
                st.write(f"{_flag_icon(flag)} {flag}")
    else:
        st.success("✅ Safety review: no flags")

    ml = result.ml_prediction
    st.markdown("---")
    st.subheader("🤖 ML Research Estimate")
    st.caption(
        "Research-grade estimate trained on the public Kaggle KTAS dataset. "
        "NOT a Manchester triage category. NOT a validated clinical risk score."
    )

    if ml.prediction_available:
        ml1, ml2, ml3 = st.columns(3)
        ml1.metric(
            "Predicted KTAS Class",
            ml.predicted_ktas_class if ml.predicted_ktas_class is not None else "N/A",
            help=f"1=most critical .. 5=least critical (research estimate). Model: {ml.model_name}",
        )
        ml2.metric(
            "Emergency Estimate",
            fmt_pct(ml.emergency_research_estimate),
            help="P(KTAS_expert in {1,2,3}) — research only",
        )
        ml3.metric(
            "Model Confidence",
            fmt_pct(ml.top_class_confidence),
            help="Top-class probability from the trained model — not a clinical probability",
        )
        if ml.ktas_class_probabilities:
            with st.expander("Full class probability distribution"):
                st.bar_chart(
                    pd.DataFrame(
                        {"probability": ml.ktas_class_probabilities}
                    ).sort_index()
                )
        st.warning(
            f"⚠️ Model: **{ml.model_name}** v{ml.model_version} — "
            "research estimate only. Clinical validation required before any use."
        )
    else:
        st.info(
            f"ML prediction not available: {ml.model_note}. "
            "Train models with: `python ml_training/train_all_models.py`"
        )

    st.markdown("---")
    st.subheader("✅ Clinician Review")

    review_log_path = settings.processed_dir / "human_reviews.jsonl"
    existing_reviews = get_reviews_for_stay(review_log_path, case.stay_id)

    if existing_reviews:
        st.markdown(f"**{len(existing_reviews)} existing review(s) for this stay:**")
        for rev in existing_reviews:
            with st.expander(
                f"{rev.reviewer_role.upper()} — {rev.review_status} — {rev.created_at_utc[:19]}"
            ):
                st.json(rev.model_dump(mode="json"))
    else:
        st.info("No reviews saved for this stay yet.")

    with st.form(f"review_form_{case.stay_id}"):
        st.markdown("**Submit clinician review**")

        r1, r2 = st.columns(2)
        reviewer_role = r1.selectbox(
            "Reviewer role",
            ["triage_nurse", "emergency_physician", "researcher", "supervisor"],
            key=f"role_{case.stay_id}",
        )
        review_status = r2.selectbox(
            "Review decision",
            [
                "ACCEPTED_AS_PRESENTED",
                "OVERRIDE_REQUIRED",
                "ESCALATION_REQUIRED",
                "REQUEST_MORE_INFORMATION",
                "REJECTED_DATA_QUALITY",
                "NOT_REVIEWED",
            ],
            key=f"status_{case.stay_id}",
        )

        default_comment = (
            "Missing or limited triage fields — human data review required."
            if result.data_validation.requires_human_data_review
            else "Triage data complete. Review completed."
        )
        review_comment = st.text_area(
            "Review notes", value=default_comment, key=f"comment_{case.stay_id}"
        )

        submitted = st.form_submit_button("💾 Save Review to Audit Log")

        if submitted:
            record = HumanReviewRecord(
                review_id=str(uuid4()),
                stay_id=case.stay_id,
                reviewer_role=reviewer_role,
                review_status=review_status,
                review_comment=review_comment,
                created_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            append_human_review(review_log_path, record)
            st.success("✅ Review saved to audit log.")
            st.rerun()

    with st.expander("🔍 Full workflow output (JSON)"):
        st.json(result.model_dump(mode="json"))


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — CLINICIAN CHAT (real AutoGen agent)
# ═══════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.subheader("💬 Clinician Chat Agent")
    st.caption(
        "Powered by a real AutoGen AssistantAgent (autogen-agentchat). The agent's "
        "only tool looks up already-computed, verified evidence for the case you "
        "select — it cannot invent a vital sign, assign a triage category, "
        "diagnose, or recommend treatment or disposition. Every reply is checked "
        "by a deterministic safety filter before being shown."
    )

    chat_case_options = {
        f"Stay {r['stay_id']} — {(r.get('triage') or {}).get('chiefcomplaint') or 'No complaint'}": r
        for r in records
    }
    selected_chat_label = st.selectbox(
        "Select case for chat context", list(chat_case_options.keys()), key="chat_case_select"
    )
    selected_chat_record = chat_case_options[selected_chat_label]
    chat_stay_id = int(selected_chat_record["stay_id"])

    with st.spinner("Loading case context..."):
        chat_case = EDTriageCase(**selected_chat_record)
        chat_context_result = run_workflow(chat_case, include_llm_explanation=False)

    cti = chat_context_result.triage_input
    st.info(
        f"**Case context:** Stay {cti.stay_id} — "
        f"Chief complaint: {cti.chiefcomplaint or 'MISSING'} | "
        f"Rules engine status: {chat_context_result.decision.classification_status}"
    )

    azure_configured = load_azure_config() is not None

    if not azure_configured:
        st.warning(
            "Azure OpenAI is not configured, so the AutoGen chat agent cannot run "
            "right now. To enable it, add credentials to your `.env` file:\n\n"
            "```\nAZURE_OPENAI_ENDPOINT=...\n"
            "AZURE_OPENAI_API_KEY=...\n"
            "AZURE_OPENAI_DEPLOYMENT=...\n"
            "AZURE_OPENAI_API_VERSION=2024-10-21\n```\n\n"
            "The deterministic evidence above is still complete and usable without "
            "the chat agent — this only disables the conversational explanation layer."
        )
    else:
        chat_key = f"chat_history_{chat_stay_id}"
        if chat_key not in st.session_state:
            st.session_state[chat_key] = []

        for message in st.session_state[chat_key]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if prompt := st.chat_input(f"Ask about stay {chat_stay_id}..."):
            full_prompt = f"Regarding stay {chat_stay_id}: {prompt}"
            st.session_state[chat_key].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    cases_path = settings.processed_dir / "triage_cases_sample.jsonl"
                    chat_result = asyncio.run(
                        run_single_question(full_prompt, cases_path=cases_path)
                    )

                    if chat_result["status"] == "SAFETY_FAIL":
                        st.error("🚨 Chat agent reply failed safety checks and was blocked:")
                        for failure in chat_result["safety_failures"]:
                            st.error(f"• {failure}")
                        with st.expander("Raw reply (failed safety — do not act on this)"):
                            st.write(chat_result["reply_text"])
                        reply_for_history = (
                            "[This reply was blocked by the safety filter and is not shown. "
                            "Please ask a clinician directly.]"
                        )
                    else:
                        st.write(chat_result["reply_text"])
                        reply_for_history = chat_result["reply_text"]

                    st.session_state[chat_key].append(
                        {"role": "assistant", "content": reply_for_history}
                    )

        if st.button("🗑️ Clear chat history", key=f"clear_chat_{chat_stay_id}"):
            st.session_state[chat_key] = []
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — GOVERNANCE DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
with tab_governance:
    st.subheader("🔒 Responsible AI Governance Dashboard")
    st.caption(
        "Five-stage review gate. This is evidence for review — not a clinical certification."
    )

    dataset_audit = load_json_file(settings.processed_dir / "dataset_audit_report.json")
    missing_inputs = load_json_file(settings.processed_dir / "missing_triage_inputs_report.json")
    schema_report = load_json_file(settings.processed_dir / "schema_report.json")
    model_eval = load_json_file(settings.processed_dir / "model_evaluation_report.json")
    review_log_path = settings.processed_dir / "human_reviews.jsonl"
    human_reviews = read_human_reviews(review_log_path)

    reviewed_stay_ids = {int(r.stay_id) for r in human_reviews}
    missing_cases = (missing_inputs or {}).get("missing_cases", [])
    missing_stay_ids = {int(c["stay_id"]) for c in missing_cases if c.get("stay_id")}
    unreviewed_missing = missing_stay_ids - reviewed_stay_ids

    blocking_issues = ["No clinician-approved Manchester triage ruleset configured."]
    if unreviewed_missing:
        blocking_issues.append(
            f"{len(unreviewed_missing)} cases with missing triage data have no human review."
        )
    if not schema_report:
        blocking_issues.append("Schema verification report not found.")

    st.error("🔴 Governance Verdict: **NOT_READY_FOR_CLINICAL_USE**")
    st.markdown("**Blocking Issues:**")
    for issue in blocking_issues:
        st.error(f"• {issue}")

    st.markdown("---")
    st.markdown("### Five-Stage Review Gate")

    stages = {
        "1. Intake": {
            "status": "PASS",
            "description": "System purpose, dataset, deployment context, and risk classification documented.",
            "evidence": {
                "system_name": "AI Triage Agentic System",
                "purpose": "Research prototype for emergency department triage decision support",
                "intended_use": "Support triage nurses with structured evidence — not autonomous decisions",
                "not_intended_use": "Autonomous triage, diagnosis, treatment, or clinical routing",
                "dataset": "Kaggle Emergency Service - KTAS Triage Application (public, 1267 rows)",
                "orchestration_framework": "autogen-agentchat 0.7.5 for the explanation/chat layer only",
                "deployment_status": "Research prototype — not deployed clinically",
                "risk_classification": "HIGH RISK — emergency care system",
            },
        },
        "2. Scope": {
            "status": "PASS",
            "description": "Risk tier set. High-risk because it relates to emergency care.",
            "evidence": {
                "risk_tier": "HIGH",
                "reason": "Emergency department triage; future UHL/MIMIC validation will involve real patient data",
                "required_evaluations": [
                    "Data completeness and quality", "Leakage guard verification",
                    "Under-triage rate", "Subgroup performance",
                    "Forbidden-phrase detection in LLM/AutoGen outputs",
                    "Missing-vital sensitivity analysis",
                ],
            },
        },
        "3. Assess": {
            "status": "PASS" if dataset_audit else "WARNING",
            "description": "Evaluation pipeline: data quality, leakage, missing data, ML metrics.",
            "evidence": {
                "dataset_audit": "PASS" if dataset_audit else "MISSING — run scripts/run_ktas_pipeline.py",
                "schema_verification": "PASS" if schema_report else "MISSING",
                "leakage_guard": "PASS — retrospective KTAS fields excluded from triage input and ML features",
                "missing_data_report": f"{len(missing_cases)} cases with missing triage fields",
                "human_reviews": f"{len(human_reviews)} clinician reviews logged",
                "ktas_model_report": "PASS" if model_eval else "MISSING",
                "unit_tests": "Run: pytest (135+ tests as of the last review pass)",
            },
        },
        "4. Probe": {
            "status": "PARTIAL",
            "description": "Clinician or domain reviewer manually tests realistic, edge-case, and unsafe inputs.",
            "evidence": {
                "human_review_records": len(human_reviews),
                "missing_cases_reviewed": len(missing_stay_ids & reviewed_stay_ids),
                "missing_cases_unreviewed": len(unreviewed_missing),
                "red_team_testing": "Pending — adversarial triage and adversarial chat-agent prompts not yet systematically tested",
                "edge_case_testing": "Use Review Queue to probe missing-data and high-risk cases",
            },
        },
        "5. Decide": {
            "status": "NOT_READY_FOR_CLINICAL_USE",
            "description": "Release decision with rationale and evidence links.",
            "evidence": {
                "decision": "NOT_READY_FOR_CLINICAL_USE",
                "rationale": [
                    "System is a research prototype — no clinical validation completed",
                    "Manchester rules engine not configured (no clinician-approved ruleset)",
                    "ML models trained on a small public KTAS dataset (1267 rows), not validated against UHL ground truth",
                    "AutoGen chat agent has not been exercised against a live model in this environment — only against scripted test responses",
                    "Formal EU AI Act Annex IV documentation, post-market monitoring, and qualified legal review required for regulated deployment",
                ],
                "approved_for": "Research, demonstration, and further development only",
            },
        },
    }

    for stage_name, stage_data in stages.items():
        status = stage_data["status"]
        icon = "✅" if status == "PASS" else "⚠️" if status in ("WARNING", "PARTIAL") else "🔴"
        with st.expander(f"{icon} {stage_name}: {stage_data['description']}"):
            st.markdown(f"**Status:** {status}")
            st.json(stage_data["evidence"])

    st.markdown("---")
    st.markdown("### Governance Controls")

    controls = {
        "Leakage guard": ("PASS", "Retrospective KTAS fields (KTAS_expert, KTAS_RN, mistriage, Error_group, Diagnosis in ED, Disposition, Length of stay, KTAS duration) excluded from all triage inputs and ML features."),
        "Data validation agent": ("PASS", "Missing and non-informative fields flagged on every case."),
        "Clinical safety rules": ("ACTIVE", "Vital-sign safety detection always active. MTS pathway engine gated — requires an approved clinical ruleset to assign categories. No categories assigned by default."),
        "MTS pathway status": ("AWAITING_APPROVED_RULESET", "Complaint pathways implemented as heuristics, not licensed MTS discriminators."),
        "KTAS-to-Manchester mapping": ("NOT_IMPLEMENTED", "KTAS and Manchester are different scales. No conversion exists anywhere in this codebase."),
        "AutoGen chat agent safety filter": ("PASS", "Every chat reply is checked against the shared forbidden-phrase filter and a human-review-reference requirement before being shown."),
        "AutoGen agent tool scope": ("PASS", "The agent's only tool reads already-computed deterministic evidence; it has no tool that can set a triage category or modify a vital sign."),
        "Clinician review requirement": ("PASS", "requires_clinician_review=True on ALL rules-engine outputs."),
        "Audit logging": ("PASS", "All clinician reviews logged with timestamp, role, and reason."),
        "Schema verification": ("PASS" if schema_report else "MISSING", "Column headers verified against the KTAS CSV adapter's expected schema."),
        "Clinical use guardrail": ("PASS", "System explicitly declares NOT_FOR_CLINICAL_USE everywhere."),
    }

    for control_name, (status, detail) in controls.items():
        if status == "PASS":
            st.success(f"✅ **{control_name}**: {detail}")
        elif status == "MISSING":
            st.warning(f"⚠️ **{control_name}**: {detail}")
        else:
            st.info(f"ℹ️ **{control_name}** ({status}): {detail}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — REVIEW QUEUE
# ═══════════════════════════════════════════════════════════════════════════
with tab_queue:
    st.subheader("📋 Human Review Queue")
    st.caption("Cases with missing triage data that require clinician attention")

    missing_inputs_data = load_json_file(
        settings.processed_dir / "missing_triage_inputs_report.json"
    )
    review_log_path = settings.processed_dir / "human_reviews.jsonl"
    queue_reviews = read_human_reviews(review_log_path)
    queue_reviewed_ids = {int(r.stay_id) for r in queue_reviews}

    if missing_inputs_data is None:
        st.error(
            "Missing triage inputs report not found. Run: "
            "`python scripts/inspect_missing_triage_inputs.py`"
        )
    else:
        queue_cases = missing_inputs_data.get("missing_cases", [])

        total = len(queue_cases)
        reviewed = sum(
            1 for c in queue_cases if int(c.get("stay_id", 0)) in queue_reviewed_ids
        )
        unreviewed = total - reviewed

        q1, q2, q3 = st.columns(3)
        q1.metric("Total missing-data cases", total)
        q2.metric("Reviewed", reviewed)
        q3.metric("⚠️ Needs review", unreviewed)

        queue_table = []
        for c in queue_cases:
            stay_id_int = int(c.get("stay_id", 0))
            review_status = "REVIEWED" if stay_id_int in queue_reviewed_ids else "PENDING"
            queue_table.append(
                {
                    "Stay ID": stay_id_int,
                    "Chief Complaint": c.get("chiefcomplaint", "?"),
                    "Missing Fields": ", ".join(c.get("missing_fields", [])),
                    "Review Status": review_status,
                }
            )

        st.dataframe(queue_table, use_container_width=True)

        pending = [c for c in queue_cases if int(c.get("stay_id", 0)) not in queue_reviewed_ids]
        if pending:
            st.markdown("---")
            st.subheader("Review a pending case")

            queue_options = {
                f"Stay {c['stay_id']} — {c.get('chiefcomplaint', '?')}": c for c in pending
            }
            selected_queue = st.selectbox("Select pending case", list(queue_options.keys()))
            selected_case_data = queue_options[selected_queue]
            selected_stay_id = int(selected_case_data["stay_id"])

            st.write(
                f"**Missing fields:** {', '.join(selected_case_data.get('missing_fields', []))}"
            )

            with st.form(f"queue_review_{selected_stay_id}"):
                qr1, qr2 = st.columns(2)
                q_role = qr1.selectbox(
                    "Role", ["triage_nurse", "emergency_physician", "researcher"]
                )
                q_status = qr2.selectbox(
                    "Decision",
                    [
                        "REQUEST_MORE_INFORMATION", "OVERRIDE_REQUIRED",
                        "REJECTED_DATA_QUALITY", "NOT_REVIEWED",
                    ],
                )
                q_comment = st.text_area(
                    "Notes",
                    value=(
                        f"Missing fields: {', '.join(selected_case_data.get('missing_fields', []))}. "
                        "Review required."
                    ),
                )
                if st.form_submit_button("Save"):
                    record = HumanReviewRecord(
                        review_id=str(uuid4()),
                        stay_id=selected_stay_id,
                        reviewer_role=q_role,
                        review_status=q_status,
                        review_comment=q_comment,
                        created_at_utc=datetime.now(timezone.utc).isoformat(),
                    )
                    append_human_review(review_log_path, record)
                    st.success(f"Review saved for stay {selected_stay_id}")
                    st.rerun()
        else:
            st.success("✅ All missing-data cases have been reviewed.")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — AUDIT LOG
# ═══════════════════════════════════════════════════════════════════════════
with tab_audit:
    st.subheader("📜 Clinician Review Audit Log")
    st.caption("Complete history of all clinician reviews. This log is append-only.")

    audit_log_path = settings.processed_dir / "human_reviews.jsonl"
    all_reviews = read_human_reviews(audit_log_path)

    dataset_audit = load_json_file(settings.processed_dir / "dataset_audit_report.json")
    missing_report = load_json_file(settings.processed_dir / "missing_triage_inputs_report.json")

    if dataset_audit:
        with st.expander("Dataset audit report"):
            st.json(dataset_audit)
    if missing_report:
        with st.expander("Missing triage input report"):
            st.metric("Cases with missing inputs", missing_report.get("cases_with_missing_triage_inputs"))
            st.metric("Missing case percent", f"{missing_report.get('missing_case_percent')}%")

    st.markdown("---")
    if not all_reviews:
        st.info("No reviews logged yet. Submit a review from the Triage Review tab.")
    else:
        st.markdown(f"**Total reviews logged: {len(all_reviews)}**")

        audit_table = [
            {
                "Stay ID": r.stay_id,
                "Reviewer Role": r.reviewer_role,
                "Decision": r.review_status,
                "Override": r.clinician_override or "",
                "Timestamp": r.created_at_utc[:19],
            }
            for r in reversed(all_reviews)
        ]
        st.dataframe(audit_table, use_container_width=True)

        st.markdown("---")
        for review in reversed(all_reviews):
            with st.expander(
                f"Stay {review.stay_id} — {review.review_status} — "
                f"{review.reviewer_role} — {review.created_at_utc[:19]}"
            ):
                st.json(review.model_dump(mode="json"))


# ═══════════════════════════════════════════════════════════════════════════
# TAB 6 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════
with tab_models:
    st.subheader("📊 ML Model Performance")
    st.caption(
        "Training results from the public Kaggle KTAS dataset (1267 rows). "
        "NOT validated against UHL clinical ground truth. For research only."
    )

    registry = load_model_registry()
    eval_report = load_json_file(settings.processed_dir / "model_evaluation_report.json")

    if not registry:
        st.warning(
            "No trained models found.\n\n"
            "Run the full pipeline:\n```\npython scripts/run_ktas_pipeline.py\n```"
        )
    else:
        st.success(f"✅ Models trained — version: {registry.get('version')}")

        best_ktas = registry.get("best_ktas_model", {})
        best_em = registry.get("best_emergency_model", {})
        ktas_metrics = best_ktas.get("metrics", {})
        em_metrics = best_em.get("metrics", {})

        st.markdown("### Best Models Summary")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"**5-class KTAS model: {best_ktas.get('name', '?')}**")
            st.metric("Macro F1", f"{ktas_metrics.get('macro_f1', 0):.3f}")
            st.metric(
                "Under-triage rate ⚠️",
                f"{ktas_metrics.get('under_triage_rate', 0):.3f}",
                help="Fraction of cases where the model predicted LESS urgent than the expert label — the clinically dangerous direction.",
            )
            st.metric("Over-triage rate", f"{ktas_metrics.get('over_triage_rate', 0):.3f}")
        with m2:
            st.markdown(f"**Emergency binary model: {best_em.get('name', '?')}**")
            st.metric("AUROC", f"{em_metrics.get('macro_auroc', 0):.3f}")
            st.metric(
                "False-negative emergency rate ⚠️",
                f"{em_metrics.get('false_negative_emergency_rate', 0):.3f}",
                help="Fraction of true emergencies (KTAS 1-3) the model predicted as non-emergency.",
            )
            st.metric("Weighted F1", f"{em_metrics.get('weighted_f1', 0):.3f}")

        under_rate = ktas_metrics.get("under_triage_rate", 1.0)
        fn_rate = em_metrics.get("false_negative_emergency_rate", 1.0)
        if under_rate > 0.15 or fn_rate > 0.15:
            st.error(
                f"⚠️ Under-triage rate {under_rate:.1%} / false-negative-emergency rate "
                f"{fn_rate:.1%} — both well above what would be acceptable for any "
                "clinical use. This is a research metric on a 1267-row public dataset "
                "using only triage-time vitals and demographics. More data and "
                "clinical validation are required before these numbers mean anything "
                "beyond a research baseline."
            )
        else:
            st.warning(
                f"Under-triage rate {under_rate:.1%}, false-negative-emergency rate "
                f"{fn_rate:.1%}. Clinical validation still required before any use."
            )

        st.markdown("---")
        st.markdown("### All 5-Class KTAS Models Compared")
        all_ktas = registry.get("all_ktas_models", [])
        if all_ktas:
            comparison = [
                {
                    "Model": m["name"],
                    "Macro F1": round(m["metrics"].get("macro_f1", 0), 3),
                    "Under-triage rate": round(m["metrics"].get("under_triage_rate", 0), 3),
                    "Over-triage rate": round(m["metrics"].get("over_triage_rate", 0), 3),
                    "Selection score": round(m["metrics"].get("selection_score", 0), 3),
                }
                for m in all_ktas
            ]
            st.dataframe(comparison, use_container_width=True)

        st.markdown("---")
        st.markdown("### All Emergency Binary Models Compared")
        all_em = registry.get("all_emergency_models", [])
        if all_em:
            comparison_em = [
                {
                    "Model": m["name"],
                    "AUROC": round(m["metrics"].get("macro_auroc", 0), 3),
                    "False-negative rate": round(m["metrics"].get("false_negative_emergency_rate", 0), 3),
                    "Selection score": round(m["metrics"].get("selection_score", 0), 3),
                }
                for m in all_em
            ]
            st.dataframe(comparison_em, use_container_width=True)

        st.markdown("---")
        st.markdown("### Training Data Summary")
        st.markdown(
            f"""
| Property | Value |
|---|---|
| Dataset | {registry.get('dataset', '?')} |
| Training samples | {registry.get('n_samples', '?')} |
| Features | {len(registry.get('feature_names', []))} |
| Version | {registry.get('version', '?')} |
| Trained at | {str(registry.get('created_at_utc', '?'))[:19]} |
            """
        )

        with st.expander("Blocked leakage features (never used as model inputs)"):
            st.json(registry.get("blocked_leakage_features", []))

        st.warning("⚠️ **Research Note:** " + registry.get("research_note", ""))

        with st.expander("Full registry JSON"):
            st.json(registry)

    if eval_report:
        with st.expander("Full model evaluation report JSON"):
            st.json(eval_report)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### System Status")

    has_models = settings.model_registry_path.exists()
    st.markdown(f"**ML Models:** {'✅ Trained' if has_models else '⬜ Not trained'}")

    azure_ok = load_azure_config() is not None
    st.markdown(f"**Azure OpenAI / AutoGen chat:** {'✅ Configured' if azure_ok else '⬜ Not configured'}")

    mimic_demo_path = settings.raw_demo_dir / "edstays.csv.gz"
    mimic_full_path = settings.raw_ed_dir / "edstays.csv.gz"
    st.markdown(f"**MIMIC-IV-ED Demo:** {'✅ Loaded' if mimic_demo_path.exists() else '⬜ Not loaded'}")
    st.markdown(f"**MIMIC-IV-ED Full:** {'✅ Loaded' if mimic_full_path.exists() else '⬜ Awaiting approval'}")

    st.markdown("---")
    st.markdown("### Quick Start")
    st.code(
        """# 1. Run the full KTAS pipeline
python scripts/run_ktas_pipeline.py

# 2. Run tests
pytest

# 3. Run the API (separate terminal)
uvicorn app.main:app --reload

# 4. Run this UI (separate terminal)
streamlit run frontend/app.py
""",
        language="bash",
    )

    st.markdown("---")
    st.caption(
        "NOT FOR CLINICAL USE\n\n"
        "Research prototype — KTAS is not Manchester Triage Scale. "
        "No KTAS-to-Manchester mapping exists. AutoGen orchestrates the "
        "explanation/chat layer only; it has no authority over any clinical decision."
    )
