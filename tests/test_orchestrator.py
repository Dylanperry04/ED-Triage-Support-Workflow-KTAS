"""
Tests for the workflow orchestrator.

Verifies end-to-end workflow correctness and clinical safety invariants.

Run with: pytest tests/test_orchestrator.py -v
"""
import pytest
from app.schemas.internal import EDTriageCase, EDStaySource, TriageSource
from app.agents.orchestrator import run_workflow


def _make_case(chiefcomplaint="chest pain", **triage_kwargs) -> EDTriageCase:
    triage_defaults = {
        "subject_id": 1, "stay_id": 1,
        "temperature": 98.6, "heartrate": 80.0,
        "resprate": 16.0, "o2sat": 98.0,
        "sbp": 120.0, "dbp": 80.0, "pain": "5",
        "chiefcomplaint": chiefcomplaint,
        "acuity": 2.0,  # retrospective — must NOT appear in triage input
    }
    triage_defaults.update(triage_kwargs)
    return EDTriageCase(
        stay_id=1,
        subject_id=1,
        edstay=EDStaySource(
            subject_id=1, stay_id=1,
            disposition="ADMITTED",  # retrospective — must NOT appear in triage input
            outtime="2024-01-01 12:00:00",
        ),
        triage=TriageSource(**triage_defaults),
    )


class TestOrchestrator:
    def test_workflow_completes_without_error(self):
        case = _make_case()
        result = run_workflow(case)
        assert result is not None
        assert result.stay_id == 1

    def test_workflow_output_contains_all_required_fields(self):
        case = _make_case()
        result = run_workflow(case)
        assert result.triage_input is not None
        assert result.data_validation is not None
        assert result.case_summary is not None
        assert result.decision is not None
        assert result.safety_review is not None
        assert result.retrospective_labels is not None
        assert result.audit is not None

    def test_retrospective_data_not_in_triage_input(self):
        case = _make_case()
        result = run_workflow(case)
        triage_dict = result.triage_input.model_dump()
        # acuity and disposition must NOT appear in triage_input
        assert "acuity" not in triage_dict
        assert "disposition" not in triage_dict
        assert "outtime" not in triage_dict

    def test_retrospective_labels_preserved_separately(self):
        case = _make_case()
        result = run_workflow(case)
        # Retrospective labels should be preserved for audit
        assert result.retrospective_labels.original_acuity == 2.0
        assert result.retrospective_labels.disposition == "ADMITTED"

    def test_clinician_review_always_required(self):
        case = _make_case()
        result = run_workflow(case)
        assert result.decision.requires_clinician_review is True

    def test_no_mts_category_without_approved_ruleset(self):
        """Without approved ruleset, no MTS category should ever be assigned."""
        case = _make_case()
        result = run_workflow(case)
        assert result.decision.category is None
        assert result.decision.priority is None
        assert result.decision.classification_status in (
            "AWAITING_APPROVED_CLINICAL_RULESET",
            "REQUIRES_CLINICIAN_REVIEW",
            "CRITICAL_PHYSIOLOGY_FLAGGED",
            "PHYSIOLOGY_CONCERN_FLAGGED",
        )

    def test_audit_record_populated(self):
        case = _make_case()
        result = run_workflow(case)
        assert "workflow_version" in result.audit
        assert "run_start_utc" in result.audit
        assert "clinical_decision_policy" in result.audit

    def test_workflow_without_llm_does_not_raise(self):
        """Workflow must complete safely without Azure OpenAI configured."""
        case = _make_case()
        result = run_workflow(case, include_llm_explanation=False)
        assert result.explanation.explanation_status == "NOT_REQUESTED"

    def test_missing_vitals_case_handled_safely(self):
        """Workflow must not crash when vital signs are absent."""
        case = _make_case(
            o2sat=None, heartrate=None, temperature=None
        )
        result = run_workflow(case)
        assert result is not None
        assert result.safety_review.is_safe_to_present is False
        assert len(result.safety_review.critical_missing_vitals) > 0
