"""
Layer 1 coding: teacher-authored prompt -> structural features + target DOK.

Applies the prompt-coding rubric (rubrics/prompt_coding_rubric.py) to each
teacher prompt with an LLM and writes one row per prompt in the schema of
data/activities.csv (feature columns, prompt_target_dok, dok_uncertainty,
prompt_role, prompt_tone). In the study, LLM output was a first pass; every
code was then reviewed to full human consensus.

Input  (default): ../synthetic/prompts.csv  columns: discussionId, discussionContent
Output (default): ../synthetic/prompts_llm_coded.csv

Usage
    python code_prompts.py [--input CSV] [--output CSV] [--model MODEL]
Requires ANTHROPIC_API_KEY (Claude models) or OPENAI_API_KEY (OpenAI models)
in the environment or a .env file; see llm_client.py.
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "rubrics"))
from llm_client import JsonLLM  # noqa: E402
from prompt_coding_rubric import prompt_content_coding  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["task_specificity", "finish_line", "scaffold_socratic", "scaffold_stepwise",
            "scaffold_hints_first", "scaffold_attempt_first", "scaffold_worked_example_prohibition",
            "epistemic_explain_reasoning", "epistemic_use_evidence", "epistemic_compare_alternatives",
            "epistemic_justify_claims", "constraint_short_turn", "constraint_one_question_per_turn",
            "constraint_structured_format", "constraint_language_level", "guardrail_no_direct_answers",
            "guardrail_integrity_language", "guardrail_privacy", "guardrail_respectful"]


def code_prompt(client, model, text):
    system, user = prompt_content_coding.split("USER", 1)
    system = system.replace("SYSTEM", "", 1).strip()
    user = user.replace("<DISCUSSION_CONTENT_HERE>", text)
    return client.complete_json(system, user)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(HERE, "..", "synthetic", "prompts.csv"))
    ap.add_argument("--output", default=os.path.join(HERE, "..", "synthetic", "prompts_llm_coded.csv"))
    ap.add_argument("--model", default="claude-sonnet-4-5-20250929")
    args = ap.parse_args()

    client = JsonLLM(args.model)

    df = pd.read_csv(args.input)
    rows = []
    for _, r in df.iterrows():
        out = code_prompt(client, args.model, str(r["discussionContent"]))
        row = {"discussionId": r["discussionId"], "coder_model": args.model}
        for f in FEATURES + ["prompt_target_dok", "dok_uncertainty"]:
            row[f] = out.get(f)
        row["prompt_role"] = out.get("role")
        row["prompt_tone"] = out.get("tone")
        rows.append(row)
        print(f"coded {r['discussionId']}: target DOK {row['prompt_target_dok']}")
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
