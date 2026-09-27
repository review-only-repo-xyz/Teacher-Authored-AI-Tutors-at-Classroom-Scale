"""
Layer 2 coding: conversation-level student DOK (the validated conversation label).

Applies the conversation-level DOK rubric (rubrics/dok_recoding_rubric.py) to
each transcript. DOK is judged from student turns only; AI turns are shown as
context. Writes student_dok_max, student_dok_final, confidence, and the
derived dok_alignment against the prompt's target DOK, in the schema of the
recoded_* columns of data/conversations.csv. The rationale and evidence
quotes the rubric returns are NOT written to the output, because in the real
data they contain student text.

Input  (default): ../synthetic/transcripts.jsonl
Output (default): ../synthetic/conversation_dok_llm_coded.csv

Usage
    python code_conversation_dok.py [--input JSONL] [--output CSV] [--model MODEL]
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
from dok_recoding_rubric import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def split_roles(transcript):
    st = [f"[Turn {i+1}] {m.get('text', '')}" for i, m in enumerate(transcript) if m.get("role") == "student"]
    ai = [f"[Turn {i+1}] {m.get('text', '')}" for i, m in enumerate(transcript) if m.get("role") == "ai"]
    return st, ai


def fmt(msgs, max_chars):
    s = "\n---\n".join(msgs)
    return s if len(s) <= max_chars else s[:max_chars] + "\n[...truncated...]"


def code_one(client, model, rec):
    st, ai = split_roles(rec["conversation_transcript"])
    if not st:
        return {"student_dok_max": None, "student_dok_final": None, "confidence": "low"}
    target = rec.get("prompt_target_dok", "unknown")
    user = USER_PROMPT_TEMPLATE.format(
        teacher_prompt=str(rec.get("discussionContent", ""))[:1500], target_dok=target,
        student_messages=fmt(st, 6000), ai_messages=fmt(ai, 4000))
    return client.complete_json(SYSTEM_PROMPT, user)


def alignment(dok_max, target):
    if dok_max is None or target in (None, "unknown"):
        return "unknown"
    return "aligned" if dok_max == target else "under_target" if dok_max < target else "over_target"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(HERE, "..", "synthetic", "transcripts.jsonl"))
    ap.add_argument("--output", default=os.path.join(HERE, "..", "synthetic", "conversation_dok_llm_coded.csv"))
    ap.add_argument("--model", default="claude-sonnet-4-5-20250929")
    args = ap.parse_args()

    client = JsonLLM(args.model)

    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            out = code_one(client, args.model, rec)
            target = rec.get("prompt_target_dok")
            rows.append({
                "conversationId": rec["conversationId"], "discussionId": rec["discussionId"],
                "prompt_target_dok": target, "coder_model": args.model,
                "recoded_student_dok_max": out.get("student_dok_max"),
                "recoded_student_dok_final": out.get("student_dok_final"),
                "recoded_confidence": out.get("confidence"),
                "recoded_dok_alignment": alignment(out.get("student_dok_max"), target),
            })
            print(f"coded {rec['conversationId']}: max DOK {out.get('student_dok_max')}")
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
