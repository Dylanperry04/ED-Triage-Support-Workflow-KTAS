from typing import Optional
from pydantic import BaseModel


class HumanReviewRequest(BaseModel):
    """
    Request body for POST /review/submit.
    The API receives this from the Streamlit UI or any client submitting a review.
    """
    stay_id: int
    reviewer_role: str
    review_status: str
    review_comment: str
    clinician_override: Optional[str] = None
    override_reason: Optional[str] = None


class HumanReviewRecord(BaseModel):
    """
    Stored audit record. Adds server-generated fields (review_id, timestamp)
    to the incoming HumanReviewRequest.
    """
    review_id: str
    stay_id: int
    reviewer_role: str
    review_status: str
    review_comment: str
    clinician_override: Optional[str] = None
    override_reason: Optional[str] = None
    created_at_utc: str
