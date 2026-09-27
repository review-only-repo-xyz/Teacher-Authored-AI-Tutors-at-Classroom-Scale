def get_prompt_conversation(TEACHER_SETUP_PROMPT_TEXT, STUDENT_STARTER_TEXT, Dok, CONVERSATION):
    prompt = f"""
        SYSTEM
        You are an expert learning-sciences coder and human–AI interaction analyst. You will code a single STUDENT–AI conversation (messages with student requests and AI responses) that occurred under a teacher-designed Teaching Aide. Your goal is to extract reproducible, evidence-based interaction codes aligned to our study: adoption/engagement signatures, epistemic moves, deviation/misalignment, risk/shortcut behavior, and student demonstrated DOK.

        Rules:
        - Be conservative: code “1” only if the conversation provides explicit evidence.
        - Do NOT infer student demographics, identity, or learning outcomes.
        - Use the teacher prompt + student-facing starter ONLY as context to judge alignment; do not code features that are not evidenced in the conversation.
        - Focus on student turns for student-move codes and student DOK. Use AI turns for AI-behavior/guardrail adherence codes.
        - Output valid JSON only (no markdown, no commentary, no extra keys).

        USER
        You will receive:
        (1) teacher_setup_prompt (hidden discussionContent)
        (2) student_starter_message (what students saw first)
        (3) a full conversation transcript as an ordered list of messages with fields:
            - role: "student" or "ai"
            - text: message content
            - timestamp: optional

        Task:
        Code the conversation using the rubric below.

        RUBRIC (conversation-level)

        A) Basic structure (numeric)
        - n_student_turns
        - n_ai_turns
        - n_total_turns
        - student_total_tokens_proxy (whitespace count)
        - ai_total_tokens_proxy
        - duration_seconds (if timestamps present; else null)

        B) Student interaction moves (0/1 each; mark present if occurs at least once)
        - student_asks_questions
        - student_explains_reasoning
        - student_uses_evidence_or_sources
        - student_compares_alternatives
        - student_requests_revision_or_iteration
        - student_requests_hints_or_scaffolding
        - student_expresses_confusion_or_stuck
        - student_reflects_metacognitively (e.g., strategy, monitoring understanding)

        C) Risk / shortcut / misalignment signals (0/1 each; student side)
        - student_requests_direct_answer
        - student_requests_completion_of_work (e.g., “write the whole essay/problem set for me”)
        - student_copy_paste_like_input (very long pasted text or “here is my essay:”)
        - student_off_topic_or_non_instructional (chat unrelated to task)
        - student_attempts_to_bypass_constraints (asks AI to ignore rules, or repeats prohibited request)

        D) AI behavior / scaffold adherence (0/1 each; AI side)
        - ai_asks_socratic_questions
        - ai_provides_stepwise_guidance
        - ai_provides_hints_first
        - ai_requests_student_attempt_first
        - ai_refuses_or_limits_direct_answers
        - ai_requests_evidence_or_reasoning
        - ai_redirects_off_topic_back_to_task
        - ai_provides_final_answer_or_solution (mark 1 if AI gives direct final answer/solution)
        - ai_off_safeguard (mark 1 if AI gives content that is prohibited by teacher prompt)

        E) Alignment to teacher objective (conversation-level)
        Code alignment using teacher_setup_prompt as context.
        - alignment_overall: one of ["on_track","partially_on_track","off_track","unclear"]
        Definitions:
        on_track = student requests and AI responses stay focused on the teacher’s stated task/objective and constraints
        partially_on_track = mostly on task but with noticeable drift, constraint violations, or generic interaction that weakly serves the objective
        off_track = primarily unrelated to the objective OR dominated by shortcut requests/violations
        unclear = objective not recoverable from provided context

        - deviation_types: list containing any of:
        ["off_topic","shortcut_answer_seeking","format_deliverable_mismatch","role_mismatch","other","none"]
        Use "none" if alignment_overall is on_track and no deviations occur.

        F) Demonstrated cognitive demand (student demonstrated DOK)
        Assign DOK based on what the STUDENT demonstrates in their turns (NOT what AI writes).
        - student_dok_max: 1|2|3|4
        - student_dok_final: 1|2|3|4 (based on final student turn; if no student turn, null)
        DOK definitions:
        1 = recall/reproduce (facts, copying, simple retrieval)
        2 = skills/concepts (explain concept, summarize, apply procedure, describe relationships)
        3 = strategic thinking (justify reasoning, analyze, compare, use evidence, critique, revise with rationale)
        4 = extended thinking (synthesize across sources/time, design investigation, multi-stage sustained reasoning)
        If ambiguous, choose the lower plausible level and set dok_uncertainty = true.

        G) Rigor alignment (target vs demonstrated)
        Given prompt_target_dok (if provided), compute:
        - target_dok: 1|2|3|4|null
        - dok_alignment: one of ["aligned","under_target","over_target","unknown"]
        Rules:
        aligned if student_dok_max == target_dok
        under_target if student_dok_max < target_dok
        over_target if student_dok_max > target_dok
        unknown if target_dok is null

        H) Evidence spans (verbatim snippets)
        Provide up to 3 short verbatim snippets (<=15 words each) that justify key codes.
        Include:
        - student_move_evidence: snippets from student turns
        - ai_behavior_evidence: snippets from AI turns
        - alignment_evidence: snippets showing on-track or deviation
        - dok_evidence: snippets showing DOK level

        I) Starters objective & actionability
        Code objective and actionability of student_starter_message only.
        - starter_objective_clarity:
            0 = unclear (no clear task/objective)
            1 = partial (general topic/task hinted but ambiguous)
            2 = clear (explicit task or learning objective stated)
        - starter_first_action:
            0 = not actionable (no clear next step for student)
            1 = somewhat actionable (suggests what to do but not specific)
            2 = actionable (explicit instruction for first student message or choices)
        - starter_deliverable_present:
            0 = no explicit deliverable/output
            1 = explicit deliverable/output (e.g., “write X”, “submit Y”, “produce Z”)

        OUTPUT JSON (exact keys)
        {{
        "basic_structure": {{
            "n_student_turns": <int>,
            "n_ai_turns": <int>,
            "n_total_turns": <int>,
            "student_total_tokens_proxy": <int>,
            "ai_total_tokens_proxy": <int>,
            "duration_seconds": <number|null>
        }},
        "student_moves": {{
            "student_asks_questions": 0|1,
            "student_explains_reasoning": 0|1,
            "student_uses_evidence_or_sources": 0|1,
            "student_compares_alternatives": 0|1,
            "student_requests_revision_or_iteration": 0|1,
            "student_requests_hints_or_scaffolding": 0|1,
            "student_expresses_confusion_or_stuck": 0|1,
            "student_reflects_metacognitively": 0|1
        }},
        "risk_misalignment_signals": {{
            "student_requests_direct_answer": 0|1,
            "student_requests_completion_of_work": 0|1,
            "student_copy_paste_like_input": 0|1,
            "student_off_topic_or_non_instructional": 0|1,
            "student_attempts_to_bypass_constraints": 0|1
        }},
        "ai_scaffold_adherence": {{
            "ai_asks_socratic_questions": 0|1,
            "ai_provides_stepwise_guidance": 0|1,
            "ai_provides_hints_first": 0|1,
            "ai_requests_student_attempt_first": 0|1,
            "ai_refuses_or_limits_direct_answers": 0|1,
            "ai_requests_evidence_or_reasoning": 0|1,
            "ai_redirects_off_topic_back_to_task": 0|1,
            "ai_provides_final_answer_or_solution": 0|1,
            "ai_off_safeguard": 0|1,

        }},
        "alignment": {{
            "alignment_overall": "on_track"|"partially_on_track"|"off_track"|"unclear",
            "deviation_types": ["off_topic"|"shortcut_answer_seeking"|"format_deliverable_mismatch"|"role_mismatch"|"other"|"none"]
        }},
        "dok": {{
            "student_dok_max": 1|2|3|4|null,
            "student_dok_final": 1|2|3|4|null,
            "dok_uncertainty": true|false,
            "target_dok": 1|2|3|4|null,
            "dok_alignment": "aligned"|"under_target"|"over_target"|"unknown"
        }},
        "evidence_spans": {{
            "student_move_evidence": [<snippets>],
            "ai_behavior_evidence": [<snippets>],
            "alignment_evidence": [<snippets>],
            "dok_evidence": [<snippets>]
        }}
        "starter_quality": {{
            "starter_objective_clarity": 0|1|2,
            "starter_first_action": 0|1|2,
            "starter_deliverable_present": 0|1,
        }}
        }}

        Now code the following inputs.

        teacher_setup_prompt:
        {TEACHER_SETUP_PROMPT_TEXT}

        student_starter_message:
        {STUDENT_STARTER_TEXT}

        prompt_target_dok:
        {Dok}

        conversation_transcript (ordered):
        {CONVERSATION}
    """
    return prompt