"""
Workflow Orchestrator.

Sequences all agents in the correct clinical safety order:

  1. Data Validation Agent   — completeness check (deterministic, no LLM)
  2. Leakage guard           — via Safety Review Agent
  3. Case Summary Agent      — structured evidence summary (deterministic, no LLM)
  4. Manchester Rules Engine — deterministic triage classification (no LLM)
  5. Safety Review Agent     — data quality and high-risk flags (deterministic, no LLM)
  6. ML Prediction Agent     — research-grade risk estimates (model inference)
  7. LLM Explanation Agent   — clinician-facing explanation (Azure OpenAI)

The LLM is called LAST and ONLY after all deterministic checks have passed.
The LLM never modifies the Manchester decision or the ML prediction.
The LLM explains verified evidence — it does not create it.

Every stage is logged in the audit dict.
"""
from datetime import datetime, timezone

from app.schemas.internal import EDTriageCase
from app.schemas.workflow import WorkflowResult, ExplanationResult
from app.agents.data_validation_agent import run_data_validation_agent
from app.agents.case_summary_agent import run_case_summary_agent
from app.agents.safety_review_agent import run_safety_review
from app.agents.ml_prediction_agent import run_ml_prediction
from app.rules.manchester_engine import run_manchester_engine


def run_workflow(
    case: EDTriageCase,
    include_llm_explanation: bool = False,
) -> WorkflowResult:
    """
    Run the full triage workflow for one ED case.

    Parameters
    ----------
    case : EDTriageCase
        Full ED stay container with all source tables.
    include_llm_explanation : bool
        If True, call the LLM Explanation Agent after all deterministic
        checks. Defaults to False so the workflow is fast and works without
        Azure OpenAI configured.

    Returns
    -------
    WorkflowResult
        Complete, auditable workflow output.
    """
    run_start = datetime.now(timezone.utc).isoformat()

    # ── Extract triage-time inputs (enforces leakage boundary) ───────────────
    triage_input = case.to_triage_time_input()
    retrospective_labels = case.to_retrospective_labels()

    # ── Step 1: Data validation ───────────────────────────────────────────────
    data_validation = run_data_validation_agent(triage_input)

    # ── Step 2: Case summary (no LLM) ────────────────────────────────────────
    case_summary = run_case_summary_agent(triage_input, data_validation)

    # ── Step 3: Manchester rules engine (deterministic) ───────────────────────
    decision = run_manchester_engine(triage_input)

    # ── Step 4: Safety review (deterministic) ─────────────────────────────────
    safety_review = run_safety_review(triage_input)

    # ── Step 5: ML risk prediction ────────────────────────────────────────────
    ml_prediction = run_ml_prediction(triage_input)

    # ── Step 6: LLM Explanation (optional, only if requested) ─────────────────
    if include_llm_explanation:
        from app.agents.llm_explanation_agent import run_llm_explanation
        evidence_package = {
            "stay_id": triage_input.stay_id,
            "chief_complaint": triage_input.chiefcomplaint,
            "triage_vitals": {
                "temperature_F": triage_input.temperature,
                "heartrate_bpm": triage_input.heartrate,
                "resprate_per_min": triage_input.resprate,
                "o2sat_pct": triage_input.o2sat,
                "systolic_bp_mmhg": triage_input.sbp,
                "diastolic_bp_mmhg": triage_input.dbp,
                "pain_score_0_to_10": triage_input.pain,
            },
            "arrival_transport": triage_input.arrival_transport,
            "data_validation": data_validation.model_dump(mode="json"),
            "manchester_decision": decision.model_dump(mode="json"),
            "safety_flags": safety_review.data_quality_flags,
            "missing_vitals": safety_review.critical_missing_vitals,
            "ml_prediction": ml_prediction.model_dump(mode="json"),
        }
        explanation = run_llm_explanation(evidence_package)
    else:
        explanation = ExplanationResult(explanation_status="NOT_REQUESTED")

    run_end = datetime.now(timezone.utc).isoformat()

    # ── Audit record ──────────────────────────────────────────────────────────
    audit = {
        "workflow_version": "1.0.0",
        "run_start_utc": run_start,
        "run_end_utc": run_end,
        "source_dataset": case.source_dataset,
        "clinical_decision_policy": (
            "No automated Manchester classification is performed. Kaggle KTAS outputs are research estimates only. "
            "A clinician must accept, override, escalate, or reject every output."
        ),
        "leakage_policy": (
            "Outcome and retrospective fields are excluded from triage_input. "
            "Leakage guard is checked on every case."
        ),
        "llm_policy": (
            "LLM is called ONLY to explain verified evidence. "
            "LLM does not assign triage categories or make clinical decisions."
        ),
        "ml_policy": (
            "ML predictions are research-grade KTAS estimates trained on public Kaggle data. "
            "They do not replace deterministic safety review or clinician judgement. "
            "KTAS is not Manchester Triage Scale; no KTAS-to-Manchester mapping is implemented."
        ),
        "safety_guardrail": (
            "requires_clinician_review=True on all rules-engine outputs. "
            "All outputs require clinician confirmation before any action."
        ),
        "governance_status": "RESEARCH_PROTOTYPE_NOT_FOR_CLINICAL_USE",
    }

    return WorkflowResult(
        stay_id=case.stay_id,
        triage_input=triage_input,
        data_validation=data_validation,
        case_summary=case_summary,
        retrospective_labels=retrospective_labels,
        decision=decision,
        safety_review=safety_review,
        ml_prediction=ml_prediction,
        explanation=explanation,
        audit=audit,
    )
