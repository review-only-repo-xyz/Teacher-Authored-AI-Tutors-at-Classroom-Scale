"""
Turn-level coding rubric (AI moves, student DOK 0-4, student moves).
Verbatim codebook used for the turn-level coding pass.
"""
import json

AI_MOVES = [
    "socratic_question",   # open question asking the student to reason, explain, predict or evaluate
    "stepwise_guidance",   # breaks the task into an explicit next step or ordered steps
    "hint",                # partial information or a nudge that stops short of the answer
    "request_attempt",     # asks the student to try, draft or answer before help is given
    "request_evidence",    # asks for a source, example, quotation, data or a reason ("why?")
    "affirm_extend",       # accepts or praises a student idea and pushes it further
    "final_answer",        # provides the answer, solution, or the deliverable itself
    "redirect",            # returns an off-task or shortcut request back to the task or rules
    "recap",               # summarizes what has been established so far
    "closure",             # signals that the task or finish line is complete
    "other",               # none of the above (greeting, logistics, generic encouragement)
]

STUDENT_MOVES = [
    "asks_question",       # asks the AI a question
    "explains",            # explains, describes, or summarizes in own words
    "argues_with_evidence",  # makes a claim supported by evidence, data or a quotation
    "compares_evaluates",  # compares alternatives or evaluates with criteria
    "submits_work",        # pastes or types a draft, answer, paragraph or solution
    "requests_answer",     # asks for the answer or for the work to be done
    "requests_help",       # asks for a hint, clarification or next step
    "agrees_acknowledges",  # short agreement, thanks, "ok", "yes"
    "off_task",            # unrelated to the task
    "other",
]

SYSTEM_PROMPT = f"""You are an expert learning-sciences coder. You code each turn of a student-AI conversation that took place under a teacher-authored activity prompt.

You will code TWO kinds of turns.

1. AI turns. Assign every AI-move code that the turn clearly exhibits, from this list:
{json.dumps(AI_MOVES)}
Definitions:
- socratic_question: an open question that asks the student to reason, explain, predict, or evaluate (not a yes/no or factual check).
- stepwise_guidance: breaks the task into an explicit next step or an ordered sequence of steps.
- hint: partial information or a nudge toward the answer that stops short of giving it.
- request_attempt: asks the student to try, draft, or answer before further help is given.
- request_evidence: asks for a source, example, quotation, data point, or a reason for a claim.
- affirm_extend: accepts or praises a student idea AND pushes it further (a follow-up that builds on what the student said).
- final_answer: gives the answer, the solution, or the deliverable itself (a complete paragraph the student was asked to write, the solved problem, the direct factual answer to the task question).
- redirect: returns an off-task or shortcut request back to the task or the rules.
- recap: summarizes what has been established so far in the conversation.
- closure: signals that the task, deliverable, or finish line is complete.
- other: greeting, logistics, or generic encouragement with none of the above.
Use "other" only when no other code applies. Most substantive AI turns carry 1 to 3 codes.

2. Student turns. Assign (a) the DOK level demonstrated IN THAT TURN and (b) every student-move code that applies.
DOK levels, based ONLY on what the student writes in that turn:
- 0: non-substantive (empty, "Start Classroom Discussion", "ok", "hi", a single word with no content, a bare request).
- 1: recall or reproduce (facts, one-word or short answers, yes/no, copying, simple identification, a factual question).
- 2: skills and concepts (explains a concept in own words, summarizes, applies a procedure, describes a relationship, surface comparison).
- 3: strategic thinking (constructs an argument with evidence, justifies with substantive rationale, analyzes, synthesizes sources or perspectives, compares with explicit criteria, critiques, revises with stated rationale).
- 4: extended thinking (integrates multiple frameworks over sustained reasoning, designs an investigation or solution approach, multi-step reasoning across many ideas within the turn).
Decision rules: a factual question is DOK 1 regardless of topic; "I think X because Y" with a substantive Y is DOK 3, with a trivial Y is DOK 2; if ambiguous choose the LOWER level; do not raise the DOK because the AI's question was demanding.
Student-move codes:
{json.dumps(STUDENT_MOVES)}

Output valid JSON only, no markdown, no commentary. Code ONLY the turns listed under TURNS TO CODE. Turns under CONTEXT are for understanding and must not appear in the output."""

USER_TEMPLATE = """Teacher's activity prompt (context only):
{teacher_prompt}

Target DOK the teacher intended (context only, do not use it to raise student DOK): {target_dok}

CONTEXT (preceding turns, do not code):
{context_turns}

TURNS TO CODE:
{turns}

Return JSON of the form:
{{"turns": [
  {{"turn": <int>, "role": "ai", "codes": [<ai move codes>]}},
  {{"turn": <int>, "role": "student", "dok": <0|1|2|3|4>, "moves": [<student move codes>], "evidence": "<up to 12 words quoted from the turn that justify the DOK>"}}
]}}
Include exactly one object per turn listed under TURNS TO CODE, in order."""
