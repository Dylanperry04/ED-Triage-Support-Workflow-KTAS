"""
Workflow output schemas.

One schema per agent output. Pydantic validates on construction.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.schemas.internal import TriageTimeInput, RetrospectiveLabels
from app.schemas.validation import DataValidationResult
from app.schemas.summary import CaseSummaryResult


class ManchesterDecision(BaseModel):
    """
    Output of the clinical safety rules engine.

    classification_status values:
      CRITICAL_PHYSIOLOGY_FLAGGED              — dangerous vital signs detected; no MTS category assigned
      PHYSIOLOGY_CONCERN_FLAGGED               — concerning vital signs; no MTS category assigned
      MTS_CATEGORY_ASSIGNED_PENDING_CLINICIAN_REVIEW — approved ruleset active; clinician confirms
      AWAITING_APPROVED_CLINICAL_RULESET       — pathway matched but no approved ruleset registered
      REQUIRES_CLINICIAN_REVIEW                — missing data; engine cannot classify
      NO_AUTOMATED_MANCHESTER_CLASSIFICATION_CONFIGURED — engine disabled

    requires_clinician_review is ALWAYS True. The clinician confirms every result.
    ruleset_id is None when no approved ruleset is active.
    """
    classification_status: str = "NO_AUTOMATED_MANCHESTER_CLASSIFICATION_CONFIGURED"
    category: Optional[str] = None
    priority: Optional[int] = None
    max_wait_minutes: Optional[int] = None
    reason_codes: List[str] = Field(default_factory=list)
    requires_clinician_review: bool = True
    ruleset_id: Optional[str] = None


class SafetyReviewResult(BaseModel):
    """
    Output of the Safety Review Agent.
    Flags data quality issues and confirms leakage guard passed.
    Does NOT assign or modify triage categories.
    """
    data_quality_flags: List[str] = Field(default_factory=list)
    leakage_guard_passed: bool = True
    is_safe_to_present: bool = True           # False if critical flags present
    critical_missing_vitals: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class MLPredictionResult(BaseModel):
    """
    Output of the ML research prediction layer.

    These are KTAS research estimates from the public Kaggle dataset. They are
    not Manchester triage categories, not clinical risk scores, and not approved
    for patient care.
    """
    model_config = {"protected_namespaces": ()}

    model_version: str = "not_loaded"
    model_name: str = "not_loaded"
    prediction_available: bool = False

    predicted_ktas_class: Optional[int] = None
    ktas_class_probabilities: Dict[str, float] = Field(default_factory=dict)
    emergency_research_estimate: Optional[float] = None
    non_emergency_research_estimate: Optional[float] = None

    # Backward-compatible display fields
    high_acuity_research_estimate: Optional[float] = None
    admission_risk_estimate: Optional[float] = None
    top_class_confidence: Optional[float] = None

    research_label_only: bool = True
    requires_clinical_validation: bool = True
    model_note: str = (
        "Research-grade estimates trained on public Kaggle KTAS data using KTAS_expert. "
        "KTAS is not Manchester Triage Scale. No KTAS-to-Manchester mapping is implemented. "
        "Not for clinical use; human clinical review is required."
    )


class ExplanationResult(BaseModel):
    """
    Output of the LLM Explanation Agent.
    The LLM explains verified evidence only — it never assigns triage categories.
    """
    explanation_status: str = "NOT_RUN"
    explanation_text: str = ""
    safety_failures: List[str] = Field(default_factory=list)
    clinical_use_allowed: bool = False
    automated_manchester_triage_allowed: bool = False
    manchester_category_assigned: bool = False
    human_review_required: bool = True
    model: str = "not_configured"
    deployment: str = "not_configured"


class WorkflowResult(BaseModel):
    """
    Full output of one triage workflow run.
    Everything flows through this schema for auditability.
    """
    stay_id: int
    triage_input: TriageTimeInput
    data_validation: DataValidationResult
    case_summary: CaseSummaryResult
    retrospective_labels: RetrospectiveLabels
    decision: ManchesterDecision
    safety_review: SafetyReviewResult
    ml_prediction: MLPredictionResult = Field(default_factory=MLPredictionResult)
    explanation: ExplanationResult = Field(default_factory=ExplanationResult)
    audit: Dict[str, Any] = Field(default_factory=dict)
