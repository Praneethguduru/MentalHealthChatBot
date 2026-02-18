# phq8_therapeutic_responses.py — OPTIMIZED
# Changes:
#   - Replaced string .format(score=score) with f-strings (cleaner, safer)
#   - Extracted repeated boilerplate into _ctx() helper to DRY up the code
#   - No changes to therapeutic content — prompts are identical

def get_phq8_therapeutic_context(score: int, severity: str) -> str:
    """Generate therapeutic context based on PHQ-8 score for the LLM prompt."""

    def _ctx(label: str, approach: str, tone: str, avoid: str, extra: str = "") -> str:
        return (
            f"ASSESSMENT CONTEXT: {label} (PHQ-8: {score}/24)\n"
            f"THERAPEUTIC APPROACH:\n{approach}\n"
            f"TONE: {tone}\n"
            f"AVOID: {avoid}\n"
            + (f"{extra}\n" if extra else "")
        )

    if score <= 4:
        return _ctx(
            label="User has minimal/no depression",
            approach=(
                "- Acknowledge their self-awareness in taking the assessment\n"
                "- Validate that low scores don't mean absence of struggles\n"
                "- Encourage preventive self-care and emotional awareness\n"
                "- Normalize checking in with mental health\n"
                "- Be supportive but don't minimize if they express concerns\n"
                "- Focus on maintaining wellbeing and building resilience"
            ),
            tone="Encouraging, validating, preventive",
            avoid='Dismissing concerns, implying "nothing is wrong"',
        )

    if score <= 9:
        return _ctx(
            label="User has mild depression",
            approach=(
                '- Validate that "mild" doesn\'t mean their experience isn\'t real or important\n'
                "- Explore what specific symptoms are present\n"
                "- Discuss self-care strategies and coping mechanisms\n"
                "- Gently assess if symptoms are affecting daily functioning\n"
                "- Monitor for worsening without being alarmist\n"
                "- Encourage healthy habits (sleep, exercise, social connection)\n"
                "- Be supportive of seeking help if desired, but don't pressure"
            ),
            tone="Supportive, practical, normalizing",
            avoid="Minimizing their experience, being overly clinical",
        )

    if score <= 14:
        return _ctx(
            label="User has moderate depression",
            approach=(
                "- Validate the significance of their struggles\n"
                "- Express genuine concern while avoiding alarm\n"
                "- Gently recommend considering professional support\n"
                "- Explore what's been most difficult lately\n"
                "- Ask about support systems and resources available\n"
                "- Discuss both immediate coping and longer-term help\n"
                "- Be present with their pain without rushing to 'fix'\n"
                "- Normalize therapy/counseling as a helpful step"
            ),
            tone="Warm, concerned, supportive, gently directive",
            avoid="Being overly clinical, creating panic, judging",
        )

    if score <= 19:
        return _ctx(
            label="User has moderately severe depression",
            approach=(
                "- Express clear concern while maintaining hope\n"
                "- Strongly encourage professional help (therapist, counselor, doctor)\n"
                "- Ask about safety and support systems\n"
                "- Validate the courage it took to complete assessment\n"
                "- Be direct about the importance of seeking help\n"
                "- Offer to discuss barriers to getting help\n"
                "- Provide resources if they're open to it\n"
                "- Balance urgency with compassion"
            ),
            tone="Caring, concerned, gently firm, hopeful",
            avoid="Creating panic, being judgmental, minimizing severity",
            extra="SAFETY CHECK: If user mentions self-harm thoughts, provide crisis resources",
        )

    # Severe (20–24)
    return _ctx(
        label="User has SEVERE depression — HIGH CONCERN",
        approach=(
            "- Express clear, compassionate concern\n"
            "- STRONGLY recommend immediate professional help\n"
            "- Assess safety — ask directly about self-harm thoughts if appropriate\n"
            "- Emphasize that help is available and effective\n"
            "- Validate how hard things must be right now\n"
            "- Be directive while remaining compassionate\n"
            "- Provide crisis resources\n"
            "- Don't leave them alone — encourage reaching out to someone trusted"
        ),
        tone="Deeply caring, urgent but not panicked, hopeful, firm",
        avoid="Minimizing, being casual, overwhelming with information",
        extra=(
            "CRITICAL: If ANY mention of self-harm/suicide:\n"
            "- Take it seriously\n"
            "- Encourage immediate professional contact\n"
            "- Provide crisis hotline: 988 Suicide & Crisis Lifeline\n"
            "- Suggest emergency room if in immediate danger\n"
            "SAFETY IS PRIORITY"
        ),
    )


def get_phq8_follow_up_prompt(score: int, severity: str) -> str:
    """Generate the initial follow-up message inserted into chat after PHQ-8 completion."""
    score_str = f"{score} out of 24"

    if score <= 4:
        return (
            f"I just took a PHQ-8 assessment and my score is {score_str}, which indicates "
            "minimal or no depression. I'm not sure what to make of this result. Can you help me understand what this means?"
        )
    if score <= 9:
        return (
            f"I just took a PHQ-8 assessment and my score is {score_str}, which indicates mild "
            "depression. I'm feeling a bit concerned about this. What does this mean for me?"
        )
    if score <= 14:
        return (
            f"I just took a PHQ-8 assessment and my score is {score_str}, which indicates moderate "
            "depression. I'm worried about what this means. Can you help me understand what I should do?"
        )
    if score <= 19:
        return (
            f"I just took a PHQ-8 assessment and my score is {score_str}, which indicates moderately "
            "severe depression. I'm really concerned about this result. What should I do?"
        )
    return (
        f"I just took a PHQ-8 assessment and my score is {score_str}, which indicates severe "
        "depression. I'm very worried and don't know what to do. Can you help me?"
    )