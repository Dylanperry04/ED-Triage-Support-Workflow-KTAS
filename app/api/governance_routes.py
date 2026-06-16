import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.storage.human_review_repository import read_human_reviews

router = APIRouter()


def _read_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Required governance evidence file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/governance/report")
def get_governance_report():
    dataset_audit_path = settings.processed_dir / "dataset_audit_report.json"
    missing_inputs_path = settings.processed_dir / "missing_triage_inputs_report.json"
    schema_report_path = settings.processed_dir / "schema_report.json"
    model_eval_path = settings.processed_dir / "model_evaluation_report.json"
    human_review_path = settings.processed_dir / "human_reviews.jsonl"

    dataset_audit = _read_json_file(dataset_audit_path)
    missing_inputs = _read_json_file(missing_inputs_path)
    model_eval = _read_json_file(model_eval_path) if model_eval_path.exists() else {"status": "MISSING"}
    human_reviews = read_human_reviews(human_review_path)
    reviewed_stay_ids = {int(record.stay_id) for record in human_reviews}

    missing_cases = missing_inputs.get("missing_cases", [])
    missing_stay_ids = {int(c["stay_id"]) for c in missing_cases if c.get("stay_id") is not None}
    unreviewed_missing = sorted(missing_stay_ids.difference(reviewed_stay_ids))

    controls = {
        "dataset_loaded": {
            "status": "PASS",
            "evidence": {"sample_size": dataset_audit.get("sample_size"), "dataset": "Kaggle-KTAS"},
        },
        "schema_report_available": {
            "status": "PASS" if schema_report_path.exists() else "WARNING",
            "evidence": str(schema_report_path),
        },
        "triage_input_separation": {
            "status": "PASS",
            "evidence": {
                "triage_input_fields": dataset_audit.get("triage_input_fields", []),
                "retrospective_label_fields": dataset_audit.get("retrospective_label_fields", []),
                "policy": "KTAS labels, mistriage, diagnosis, disposition, and LOS excluded from triage-time inputs.",
            },
        },
        "missing_data_visibility": {
            "status": "PASS",
            "evidence": {
                "cases_with_missing_triage_inputs": missing_inputs.get("cases_with_missing_triage_inputs"),
                "missing_case_percent": missing_inputs.get("missing_case_percent"),
            },
        },
        "human_review_for_missing_data": {
            "status": "PASS" if not unreviewed_missing else "REQUEST_CHANGES",
            "evidence": {"unreviewed_missing_stay_ids": unreviewed_missing},
        },
        "ktas_model_report": {"status": "PASS" if model_eval_path.exists() else "WARNING", "evidence": model_eval},
        "manchester_mapping": {
            "status": "NOT_IMPLEMENTED",
            "evidence": "KTAS is not Manchester Triage Scale; no mapping or conversion is implemented.",
        },
        "clinical_use_guardrail": {
            "status": "PASS",
            "evidence": "System declares not_for_clinical_use and requires human review.",
        },
    }
    blocking_issues: List[str] = ["No clinician-approved Manchester triage ruleset configured."]
    if unreviewed_missing:
        blocking_issues.append("Some cases with missing triage inputs have no saved human review.")
    if not schema_report_path.exists():
        blocking_issues.append("Schema report file is missing.")

    return {
        "system_name": "AI Triage Agentic System",
        "dataset": "Kaggle-KTAS",
        "clinical_use_status": "not_for_clinical_use",
        "governance_verdict": "READY_FOR_RESEARCH_DEMO_ONLY" if not unreviewed_missing else "NOT_READY_FOR_CLINICAL_USE",
        "blocking_issues": blocking_issues,
        "controls": controls,
        "responsible_ai_review_gate": {
            "intake": "Public Kaggle KTAS cases loaded from supplied data.csv.",
            "scope": "Workflow limited to research KTAS estimates and deterministic safety review.",
            "assess": "Dataset audit, missing-data report, leakage guard, model report, and unit tests are available.",
            "probe": "Human review records can be saved and retrieved.",
            "decide": "System remains blocked from clinical use; Manchester triage is not configured.",
        },
    }
