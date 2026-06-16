from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health():
    return {
        "status": "ok",
        "clinical_use": "not_for_clinical_use",
        "active_dataset": "Kaggle-KTAS",
        "ktas_model_status": "research_only",
        "manchester_mapping": "not_implemented",
        "rules_status": "NO_AUTOMATED_MANCHESTER_CLASSIFICATION_CONFIGURED",
        "human_review_required": True,
    }
