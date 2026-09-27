"""
Configuration for the shared replication code.

data/ holds the de-identified coded datasets (real codes, no text, no
names, hashed identifiers). synthetic/ holds invented example content with
worked coding results for illustration only.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Prompt feature columns (activity-level binary codes plus task specificity
# and target DOK, coded from each teacher-authored prompt).
PROMPT_FEATURES = [
    "task_specificity",
    "finish_line",
    "scaffold_socratic",
    "scaffold_stepwise",
    "scaffold_hints_first",
    "scaffold_attempt_first",
    "scaffold_worked_example_prohibition",
    "epistemic_explain_reasoning",
    "epistemic_use_evidence",
    "epistemic_compare_alternatives",
    "epistemic_justify_claims",
    "constraint_short_turn",
    "constraint_one_question_per_turn",
    "constraint_structured_format",
    "constraint_language_level",
    "guardrail_no_direct_answers",
    "guardrail_integrity_language",
    "guardrail_privacy",
    "guardrail_respectful",
    "prompt_target_dok",
]
