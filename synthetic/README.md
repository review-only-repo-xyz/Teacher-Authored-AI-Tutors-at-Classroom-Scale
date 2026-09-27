# SYNTHETIC worked examples

**Everything in this folder is invented.** The three teacher prompts, the
four student–AI conversations, and every code attached to them were written
for this package to show what the three coding layers look like when applied
to content. Nothing here is an excerpt, paraphrase, or composite of any real
prompt or conversation; the topics do not correspond to any study activity.
Codes were assigned by hand against the rubrics in `../coding/rubrics/` and
are illustrative, not validated.

| File | Layer | Role |
|---|---|---|
| `prompts.csv` | 1 (activity) | Input: prompt text |
| `prompts_coded.csv` | 1 | Output: feature codes, target DOK (schema of `../data/activities.csv`) |
| `transcripts.jsonl` | 2, 3 | Input: conversations as ordered `{role, text}` lists |
| `conversation_codes.csv` | 2 (conversation) | Output: conversation-level codes (schema of `../data/conversations.csv`) |
| `turn_codes.csv` | 3 (turn) | Output: AI moves per AI turn, DOK 0–4 and moves per student turn (schema of `../data/turn_codes.csv`) |
| `transcripts_coded.md` | — | The four conversations with their codes shown inline, for reading |
| `build_synthetic_examples.py` | — | Regenerates all of the above from the literals in the script |

To see the LLM coders run on this content (requires an OpenAI API key):

```bash
cd ../coding
python code_prompts.py             # -> ../synthetic/prompts_llm_coded.csv
python code_conversation_dok.py    # -> ../synthetic/conversation_dok_llm_coded.csv
python code_turns.py               # -> ../synthetic/turn_codes_llm_coded.csv
```

The `*_llm_coded.csv` outputs can be compared against the hand-assigned
files above.
