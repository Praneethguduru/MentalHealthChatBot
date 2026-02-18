# phq8.py — OPTIMIZED
# Changes:
#   - PHQ8_FIELDS tuple for DRY access (used by calculate_phq8 and the API schema)
#   - calculate_phq8 now uses the tuple to avoid repetition and silent key typos
#   - No logic changes — scores/thresholds unchanged

from typing import Dict

# Canonical field order — single source of truth
PHQ8_FIELDS = (
    "no_interest",
    "depressed",
    "sleep",
    "tired",
    "appetite",
    "failure",
    "concentrating",
    "moving",
)


def calculate_phq8(responses: Dict[str, int]) -> Dict:
    """
    Calculate PHQ-8 score and severity.

    Args:
        responses: Dict with keys matching PHQ8_FIELDS (values 0-3)
    Returns:
        Dict with score, severity, binary classification, and depressed flag
    """
    total_score = sum(responses.get(field, 0) for field in PHQ8_FIELDS)

    if total_score <= 4:
        severity = "None/Minimal"
    elif total_score <= 9:
        severity = "Mild"
    elif total_score <= 14:
        severity = "Moderate"
    elif total_score <= 19:
        severity = "Moderately Severe"
    else:
        severity = "Severe"

    binary = 1 if total_score >= 10 else 0

    return {
        "total_score": total_score,
        "severity": severity,
        "binary": binary,
        "depressed": binary == 1,
    }


# ── Questions & options (used by the /phq8/questions endpoint) ────────────────

PHQ8_QUESTIONS = [
    {"id": "no_interest",    "question": "Little interest or pleasure in doing things?",             "field": "no_interest"},
    {"id": "depressed",      "question": "Feeling down, depressed, or hopeless?",                    "field": "depressed"},
    {"id": "sleep",          "question": "Trouble falling or staying asleep, or sleeping too much?", "field": "sleep"},
    {"id": "tired",          "question": "Feeling tired or having little energy?",                   "field": "tired"},
    {"id": "appetite",       "question": "Poor appetite or overeating?",                             "field": "appetite"},
    {"id": "failure",        "question": "Feeling bad about yourself or that you are a failure?",    "field": "failure"},
    {"id": "concentrating",  "question": "Trouble concentrating on things?",                         "field": "concentrating"},
    {"id": "moving",         "question": "Moving or speaking slowly, or being fidgety/restless?",   "field": "moving"},
]

PHQ8_OPTIONS = [
    {"value": 0, "label": "Not at all"},
    {"value": 1, "label": "Several days"},
    {"value": 2, "label": "More than half the days"},
    {"value": 3, "label": "Nearly every day"},
]