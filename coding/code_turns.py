"""
Layer 3 coding: turn-level AI moves and student DOK.

Applies the turn-level codebook (rubrics/turn_coding_rubric.py) to every
turn of every transcript. AI turns receive one or more AI-move codes;
student turns receive a DOK level 0-4 and student-move codes. Long
conversations are coded in windows of WINDOW turns with CONTEXT preceding
turns shown as context. Writes the long-format CSV consumed by
03_turn_level_analysis.py (schema of data/turn_codes.csv). The short
evidence quote the codebook returns per student turn is NOT written, because
in the real data it contains student text.

The codebook and windowing here are those of the study's coding procedure.
The model is a parameter; the paper reports validation for each coder used.

Input  (default): ../synthetic/transcripts.jsonl
Output (default): ../synthetic/turn_codes_llm_coded.csv

Usage
    python code_turns.py [--input JSONL] [--output CSV] [--model MODEL]
Requires ANTHROPIC_API_KEY (Claude models) or OPENAI_API_KEY (OpenAI models)
in the environment or a .env file; see llm_client.py.
"""
import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "rubrics"))
from llm_client import JsonLLM  # noqa: E402
from turn_coding_rubric import AI_MOVES, STUDENT_MOVES, SYSTEM_PROMPT, USER_TEMPLATE  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WINDOW = 40          # turns coded per API call
CONTEXT = 4          # preceding turns shown as context (not coded)
MAX_TURN_CHARS = 1500


def fmt_turn(i, m):
    text = str(m.get("text", "")).replace("\n", " ").strip()
    if len(text) > MAX_TURN_CHARS:
        text = text[:MAX_TURN_CHARS] + " [...truncated]"
    return f"[Turn {i}] ({m.get('role', '?')}) {text}"


def windows(n):
    start = 0
    while start < n:
        end = min(start + WINDOW, n)
        yield list(range(max(0, start - CONTEXT), start)), list(range(start, end))
        start = end


def code_window(client, model, teacher_prompt, target, transcript, ctx, idx, retries=4):
    user = USER_TEMPLATE.format(
        teacher_prompt=str(teacher_prompt)[:1500], target_dok=target,
        context_turns="\n".join(fmt_turn(i + 1, transcript[i]) for i in ctx) or "(none)",
        turns="\n".join(fmt_turn(i + 1, transcript[i]) for i in idx))
    delay = 2
    for _ in range(retries):
        try:
            return client.complete_json(SYSTEM_PROMPT, user, max_tokens=8192).get("turns", [])
        except Exception:
            time.sleep(delay)
            delay *= 2
    return []


def code_conversation(client, model, rec):
    tr = rec["conversation_transcript"]
    target = rec.get("prompt_target_dok", "unknown")
    rows = []
    for ctx, idx in windows(len(tr)):
        for t in code_window(client, model, rec.get("discussionContent", ""), target, tr, ctx, idx):
            if t.get("turn") not in {i + 1 for i in idx}:
                continue
            i = t["turn"] - 1
            role = tr[i].get("role")
            r = {"conversationId": rec["conversationId"], "discussionId": rec["discussionId"],
                 "user_id": rec.get("user_id", ""), "prompt_target_dok": target,
                 "turn": t["turn"], "role": role, "n_chars": len(str(tr[i].get("text", ""))),
                 "student_dok": t.get("dok") if role == "student" else None}
            codes, moves = set(t.get("codes") or []), set(t.get("moves") or [])
            for c in AI_MOVES:
                r[f"ai_{c}"] = int(c in codes) if role == "ai" else None
            for c in STUDENT_MOVES:
                r[f"st_{c}"] = int(c in moves) if role == "student" else None
            rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(HERE, "..", "synthetic", "transcripts.jsonl"))
    ap.add_argument("--output", default=os.path.join(HERE, "..", "synthetic", "turn_codes_llm_coded.csv"))
    ap.add_argument("--model", default="claude-sonnet-4-5-20250929")
    args = ap.parse_args()

    client = JsonLLM(args.model)

    rows = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rows += code_conversation(client, args.model, rec)
            print(f"coded {rec['conversationId']}")
    pd.DataFrame(rows).sort_values(["conversationId", "turn"]).to_csv(args.output, index=False)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
