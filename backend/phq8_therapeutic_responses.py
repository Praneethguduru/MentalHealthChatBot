# phq8_therapeutic_responses.py

def get_phq8_therapeutic_context(score: int, severity: str) -> str:
    """
    Generate therapeutic context based on PHQ-8 score.
    This will be added to the prompt to guide the therapist's response.
    """
    
    if score <= 4:  # None/Minimal
        return """
ASSESSMENT CONTEXT: User has minimal/no depression (PHQ-8: {score}/24)

THERAPEUTIC APPROACH:
- Acknowledge their self-awareness in taking the assessment
- Validate that low scores don't mean absence of struggles
- Encourage preventive self-care and emotional awareness
- Normalize checking in with mental health
- Be supportive but don't minimize if they express concerns
- Focus on maintaining wellbeing and building resilience

TONE: Encouraging, validating, preventive
AVOID: Dismissing concerns, implying "nothing is wrong"
""".format(score=score)
    
    elif score <= 9:  # Mild
        return """
ASSESSMENT CONTEXT: User has mild depression (PHQ-8: {score}/24)

THERAPEUTIC APPROACH:
- Validate that "mild" doesn't mean their experience isn't real or important
- Explore what specific symptoms are present
- Discuss self-care strategies and coping mechanisms
- Gently assess if symptoms are affecting daily functioning
- Monitor for worsening without being alarmist
- Encourage healthy habits (sleep, exercise, social connection)
- Be supportive of seeking help if desired, but don't pressure

TONE: Supportive, practical, normalizing
AVOID: Minimizing their experience, being overly clinical
""".format(score=score)
    
    elif score <= 14:  # Moderate
        return """
ASSESSMENT CONTEXT: User has moderate depression (PHQ-8: {score}/24)

THERAPEUTIC APPROACH:
- Validate the significance of their struggles
- Express genuine concern while avoiding alarm
- Gently recommend considering professional support
- Explore what's been most difficult lately
- Ask about support systems and resources available
- Discuss both immediate coping and longer-term help
- Be present with their pain without rushing to "fix"
- Normalize therapy/counseling as a helpful step

TONE: Warm, concerned, supportive, gently directive
AVOID: Being overly clinical, creating panic, judging
""".format(score=score)
    
    elif score <= 19:  # Moderately Severe
        return """
ASSESSMENT CONTEXT: User has moderately severe depression (PHQ-8: {score}/24)

THERAPEUTIC APPROACH:
- Express clear concern while maintaining hope
- Strongly encourage professional help (therapist, counselor, doctor)
- Ask about safety and support systems
- Validate the courage it took to complete assessment
- Be direct about the importance of seeking help
- Offer to discuss barriers to getting help
- Provide resources if they're open to it
- Balance urgency with compassion

TONE: Caring, concerned, gently firm, hopeful
AVOID: Creating panic, being judgmental, minimizing severity
SAFETY CHECK: If user mentions self-harm thoughts, provide crisis resources
""".format(score=score)
    
    else:  # Severe (20-24)
        return """
ASSESSMENT CONTEXT: User has severe depression (PHQ-8: {score}/24) - HIGH CONCERN

THERAPEUTIC APPROACH:
- Express clear, compassionate concern
- STRONGLY recommend immediate professional help
- Assess safety - ask directly about self-harm thoughts if appropriate
- Emphasize that help is available and effective
- Validate how hard things must be right now
- Be directive while remaining compassionate
- Provide crisis resources
- Don't leave them alone - encourage reaching out to someone trusted

TONE: Deeply caring, urgent but not panicked, hopeful, firm
AVOID: Minimizing, being casual, overwhelming with information

CRITICAL: If ANY mention of self-harm/suicide:
- Take it seriously
- Encourage immediate professional contact
- Provide crisis hotline: 988 Suicide & Crisis Lifeline
- Suggest emergency room if in immediate danger

SAFETY IS PRIORITY
""".format(score=score)


def get_phq8_follow_up_prompt(score: int, severity: str) -> str:
    """
    Generate the initial follow-up message after PHQ-8 completion.
    """
    
    if score <= 4:  # None/Minimal
        return f"""I just took a PHQ-8 assessment and my score is {score} out of 24, which indicates minimal or no depression. I'm not sure what to make of this result. Can you help me understand what this means?"""
    
    elif score <= 9:  # Mild
        return f"""I just took a PHQ-8 assessment and my score is {score} out of 24, which indicates mild depression. I'm feeling a bit concerned about this. What does this mean for me?"""
    
    elif score <= 14:  # Moderate
        return f"""I just took a PHQ-8 assessment and my score is {score} out of 24, which indicates moderate depression. I'm worried about what this means. Can you help me understand what I should do?"""
    
    elif score <= 19:  # Moderately Severe
        return f"""I just took a PHQ-8 assessment and my score is {score} out of 24, which indicates moderately severe depression. I'm really concerned about this result. What should I do?"""
    
    else:  # Severe
        return f"""I just took a PHQ-8 assessment and my score is {score} out of 24, which indicates severe depression. I'm very worried and don't know what to do. Can you help me?"""