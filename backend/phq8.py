# phq8.py
from typing import Dict

def calculate_phq8(responses: Dict[str, int]) -> Dict:
    """
    Calculate PHQ-8 score and severity.
    
    Args:
        responses: Dict with keys matching PHQ-8 questions (values 0-3)
    
    Returns:
        Dict with score, severity, and binary classification
    """
    # Sum all responses
    total_score = sum([
        responses.get('no_interest', 0),
        responses.get('depressed', 0),
        responses.get('sleep', 0),
        responses.get('tired', 0),
        responses.get('appetite', 0),
        responses.get('failure', 0),
        responses.get('concentrating', 0),
        responses.get('moving', 0)
    ])
    
    # Determine severity
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
    
    # Binary classification (≥10 = depressed)
    binary = 1 if total_score >= 10 else 0
    
    return {
        'total_score': total_score,
        'severity': severity,
        'binary': binary,
        'depressed': binary == 1
    }

# PHQ-8 Questions
PHQ8_QUESTIONS = [
    {
        'id': 'no_interest',
        'question': 'Little interest or pleasure in doing things?',
        'field': 'no_interest'
    },
    {
        'id': 'depressed',
        'question': 'Feeling down, depressed, or hopeless?',
        'field': 'depressed'
    },
    {
        'id': 'sleep',
        'question': 'Trouble falling or staying asleep, or sleeping too much?',
        'field': 'sleep'
    },
    {
        'id': 'tired',
        'question': 'Feeling tired or having little energy?',
        'field': 'tired'
    },
    {
        'id': 'appetite',
        'question': 'Poor appetite or overeating?',
        'field': 'appetite'
    },
    {
        'id': 'failure',
        'question': 'Feeling bad about yourself or that you are a failure?',
        'field': 'failure'
    },
    {
        'id': 'concentrating',
        'question': 'Trouble concentrating on things?',
        'field': 'concentrating'
    },
    {
        'id': 'moving',
        'question': 'Moving or speaking slowly, or being fidgety/restless?',
        'field': 'moving'
    }
]

PHQ8_OPTIONS = [
    {'value': 0, 'label': 'Not at all'},
    {'value': 1, 'label': 'Several days'},
    {'value': 2, 'label': 'More than half the days'},
    {'value': 3, 'label': 'Nearly every day'}
]