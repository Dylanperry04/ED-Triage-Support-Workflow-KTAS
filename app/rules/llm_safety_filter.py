"""
Shared LLM-output phrase-blocking safety checks.

This module holds the part of the safety filter that is genuinely about
output safety regardless of response format: never let an LLM-generated
response assign a Manchester triage category, or give clinical advice
(diagnosis, treatment, disposition) in those exact terms.

It deliberately does NOT include format-completeness checks like "must
mention missing data" or "must state no category assigned" -- those are
specific to the single-shot LLM Explanation Agent's mandated five-section
response structure (see llm_explanation_agent.py), and applying them to a
free-form conversational reply (e.g. the AutoGen-based clinician chat agent)
would produce constant false-positive failures on completely benign short
answers such as "the heart rate is 84 bpm", which has no missing-data
statement and no category-assignment statement to make, because it was
never asked to discuss either. A safety flag that fires constantly on benign
output trains people to ignore it, which is worse than not having one.

Each consumer (llm_explanation_agent.py, autogen_team.py) calls
`check_forbidden_phrases()` and may add its own additional, format-specific
checks on top of the result.
"""
from __future__ import annotations


FORBIDDEN_ASSIGNMENT_PHRASES = [
    "assigned red", "assigned orange", "assigned yellow", "assigned green", "assigned blue",
    "category red", "category orange", "category yellow", "category green", "category blue",
    "triage level red", "triage level orange", "triage level yellow",
    "immediate (red)", "very urgent (orange)", "urgent (yellow)",
    "standard (green)", "non-urgent (blue)",
    "triage category is", "triage category:",
]

FORBIDDEN_CLINICAL_ADVICE = [
    "diagnose", "diagnosis is", "the diagnosis",
    "treat with", "administer", "give the patient",
    "discharge the patient", "send home", "safe to go home",
    "prescribe", "order a", "should receive",
]


def check_forbidden_phrases(text: str) -> list[str]:
    """
    Returns a list of safety failure descriptions for forbidden phrases found
    in `text`. Empty list = no forbidden phrases detected. This check alone
    does not mean a response is fully safe -- it only means it did not
    contain a known-dangerous phrase pattern. Callers may add further,
    format-specific checks on top.
    """
    failures: list[str] = []
    lower = text.lower()

    for phrase in FORBIDDEN_ASSIGNMENT_PHRASES:
        if phrase in lower:
            failures.append(f"FORBIDDEN_TRIAGE_ASSIGNMENT_PHRASE: '{phrase}'")

    for phrase in FORBIDDEN_CLINICAL_ADVICE:
        if phrase in lower:
            failures.append(f"FORBIDDEN_CLINICAL_ADVICE_PHRASE: '{phrase}'")

    return failures
