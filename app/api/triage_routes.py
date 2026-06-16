"""
Triage API routes.

GET /triage/cases         — list all available cases
GET /triage/run/{stay_id} — run full workflow for one case (no LLM)
POST /triage/run/{stay_id}/explain — run full workflow + LLM explanation
"""
from fastapi import APIRouter, HTTPException

from app.config import settings
from app.storage.jsonl_repository import read_jsonl
from app.schemas.internal import EDTriageCase
from app.agents.orchestrator import run_workflow

router = APIRouter()


@router.get("/triage/cases")
def list_cases():
    """List all available processed triage cases."""
    path = settings.processed_dir / "triage_cases_sample.jsonl"
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "No processed cases found. "
                "Run: python scripts/run_ktas_pipeline.py"
            ),
        )
    records = read_jsonl(path)
    return [
        {
            "stay_id": r["stay_id"],
            "subject_id": r["subject_id"],
            "chiefcomplaint": (
                r.get("triage", {}).get("chiefcomplaint")
                if r.get("triage") else None
            ),
            "source_dataset": r.get("source_dataset"),
            "age": r.get("triage", {}).get("age") if r.get("triage") else None,
        }
        for r in records
    ]


def _find_case(stay_id: int) -> EDTriageCase:
    """Load a case by stay_id from the processed JSONL file."""
    path = settings.processed_dir / "triage_cases_sample.jsonl"
    if not path.exists():
        raise HTTPException(status_code=404, detail="No processed data found.")
    records = read_jsonl(path)
    for r in records:
        if int(r["stay_id"]) == stay_id:
            return EDTriageCase(**r)
    raise HTTPException(status_code=404, detail=f"stay_id not found: {stay_id}")


@router.get("/triage/run/{stay_id}")
def run_case(stay_id: int):
    """Run the triage workflow for one case (deterministic agents only, no LLM)."""
    case = _find_case(stay_id)
    result = run_workflow(case, include_llm_explanation=False)
    return result.model_dump(mode="json")


@router.post("/triage/run/{stay_id}/explain")
def run_case_with_explanation(stay_id: int):
    """
    Run the full triage workflow including LLM Explanation Agent.
    Requires Azure OpenAI to be configured in .env.
    """
    case = _find_case(stay_id)
    result = run_workflow(case, include_llm_explanation=True)
    return result.model_dump(mode="json")
