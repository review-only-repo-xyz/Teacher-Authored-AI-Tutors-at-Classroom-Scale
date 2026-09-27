# Replication package: design–enactment gaps in student–AI dialogue

This folder contains the coded data, the coding instruments, and the
analysis code behind the study of teacher-authored AI tutoring activities,
shared for reviewers and interested researchers. Running `python run_all.py`
reproduces the reported RQ1 and RQ2 results from the de-identified coded
data in `data/`.

## What is shared and what is not

**Shared (real, de-identified).** All codes produced by the three coding
layers, for every activity, conversation, and turn in the study:

- `data/activities.csv` — 74 activities: prompt feature codes, target DOK,
  role/tone codes, subject, grade band (one row per activity)
- `data/conversations.csv` — 1,479 conversations (1,362 with
  conversation-level codes): conversation-level student-move, AI-adherence,
  alignment, and DOK codes, plus the activity's prompt features
- `data/turn_codes.csv` — 23,114 turns of the 1,213 primary-sample
  conversations: AI-move codes per AI turn, DOK 0–4 and student-move codes
  per student turn
- `data/validation_codes.csv` — 3,218 turns of the 136-conversation human
  validation sample: student DOK and AI-move codes from two independent
  human coders (`hc1`, `hc2`) and the LLM turn coder (`llm`), with subject
  and target DOK. Codes only, no text. Conversation identifiers in this file
  were hashed with a separate discarded salt, so the validation sample cannot
  be joined to the other data files.

**Not shared, under any circumstances.** The student–AI conversation text.
No transcript, starter message, message excerpt, rationale, or evidence
quote appears anywhere in this package. Teacher prompt text is likewise
withheld from the datasets; three screened examples appear in
`example_prompts/`.

**De-identification applied to the data files.** Every column containing
text was dropped (transcripts, starters, prompt wording, LLM rationales and
evidence spans, raw model output). Teacher names were replaced with random
pseudonyms `T01`–`T16`. Conversation, activity, classroom, and student
identifiers were replaced with salted one-way hashes; the salt was discarded,
so the hashes cannot be joined back to platform records. District, timestamps,
and class sizes were removed. Subject and grade band are retained because the
analyses stratify on them.

**Synthetic (invented) content, labeled as such.** `synthetic/` contains
three invented teacher prompts and four invented conversations, together
with hand-assigned codes for each, so that readers can see what each coding
layer produces from content. See `synthetic/README.md`.

Researchers interested in further examples of real teacher prompts may
contact the authors.

## Layout

```
data/                      de-identified coded datasets (real codes, no text)
example_prompts/           three real teacher prompts, screened, one per pattern
coding/
  rubrics/                 verbatim coding instruments (Layers 1–3)
  code_prompts.py          Layer 1: prompt -> features + target DOK (LLM, needs API key)
  code_conversation_dok.py Layer 2: conversation -> student DOK label (LLM, needs API key)
  code_turns.py            Layer 3: turn -> AI moves / student DOK (LLM, needs API key)
synthetic/                 invented content + hand-assigned codes, all three layers
01_descriptives.py         sample descriptives
02_prompt_patterns.py      RQ1: k-modes prompt patterns; mixed-effects and
                           cluster-robust models; minimum detectable effects
03_turn_level_analysis.py  RQ2: time to target, hazard models, persistence,
                           DOK transition networks (bootstrap CIs), PrefixSpan,
                           within-activity AI-move comparison
04_revision_analyses.py    inter-rater reliability (two human coders and the
                           LLM); timing and persistence under human codes;
                           hazard and AI-move models with and without log
                           length; student clustering and one-conversation-
                           per-student checks; lagged-state hazard; drops
                           after attainment; pattern collinearity
04b_glmm_student_re.R      GLMMs with student random intercepts (R, lme4)
run_all.py                 runs 01 -> 02 -> 03 -> 04 (-> 04b if R is available)
config.py                  paths and the prompt-feature column list
```

## Reproducing the analyses

```bash
pip install -r requirements.txt
python run_all.py
```

Outputs land in `outputs/` (CSV tables) and `outputs/figures/` (PNG). The
primary specification excludes the World Language stratum and uses the
turn-level DOK maximum as the outcome; the scripts' docstrings list the
flags for the sensitivity specifications (conversation-level label,
discipline split, alternative target strata).

## The coding instruments

`coding/rubrics/` holds the four instruments verbatim:

- `prompt_coding_rubric.py` — Layer 1. Codes a teacher prompt for task
  specificity, finish line, five scaffolding strategies, four epistemic
  framings, four constraints, role, tone, four guardrails, and the target
  DOK level implied by the prompt's objective (with an uncertainty flag).
  In the study, LLM output was a first pass and every assignment was
  reviewed to full human consensus.
- `conversation_coding_rubric.py` — Layer 2, full conversation rubric
  (student moves, risk signals, AI adherence, alignment, DOK, starter quality).
- `dok_recoding_rubric.py` — Layer 2, the student-only DOK rubric that
  produced the validated conversation-level DOK label (`recoded_*` columns).
- `turn_coding_rubric.py` — Layer 3. AI-move codes per AI turn; DOK 0–4 and
  student-move codes per student turn.

The three `code_*.py` scripts apply these instruments with an LLM and write
outputs in the same schemas as the files in `data/`. They default to the
synthetic examples; pass `--input` to point them elsewhere. The default
coder is `claude-sonnet-4-5-20250929`, the model that produced the codes in
`data/` (recorded in the `coder_model` column); `--model` accepts any
Anthropic or OpenAI model name (see `coding/llm_client.py`).

## Turn indexing

The platform records the student's start button ("Start Classroom
Discussion") as student turn 1 of every conversation. That turn has no
preceding AI turn and cannot be at target. `04_revision_analyses.py`
reports the hazard models on the corrected risk set (student turns 2
onward) alongside the specification that includes turn 1, and section H
indexes attainment timing from the first typed student turn.

## Requirements

Python 3.10+ and the packages in `requirements.txt`. `04b_glmm_student_re.R`
additionally needs R 4.1+ with `lme4`; `run_all.py` skips it when `Rscript`
is not found. The analysis pipeline
takes a few minutes on a laptop. The coding scripts additionally require
`ANTHROPIC_API_KEY` (or `OPENAI_API_KEY` for an OpenAI model) in the
environment or a `.env` file.
