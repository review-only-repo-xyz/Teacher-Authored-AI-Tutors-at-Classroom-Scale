"""
Build the SYNTHETIC worked examples.

EVERYTHING in this folder is invented. The three teacher prompts, the four
student-AI conversations, and the codes attached to them were written for
this package to show what the three coding layers look like when applied
to content. They are not excerpts, paraphrases, or composites of any real
prompt or conversation, and the topics do not correspond to any study
activity. Codes were assigned by hand against the rubrics in ../coding/rubrics
and are illustrative, not validated.

Writes
    prompts.csv               teacher prompt text (Layer 1 input)
    prompts_coded.csv         hand-assigned prompt codes (Layer 1 output schema)
    transcripts.jsonl         conversations (Layer 2 and 3 input)
    conversation_codes.csv    hand-assigned conversation-level codes (Layer 2 output schema)
    turn_codes.csv            hand-assigned turn-level codes (Layer 3 output schema)
    transcripts_coded.md      the same conversations with codes shown inline, for reading

Usage
    python build_synthetic_examples.py
"""
import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

FEATURES = ["task_specificity", "finish_line", "scaffold_socratic", "scaffold_stepwise",
            "scaffold_hints_first", "scaffold_attempt_first", "scaffold_worked_example_prohibition",
            "epistemic_explain_reasoning", "epistemic_use_evidence", "epistemic_compare_alternatives",
            "epistemic_justify_claims", "constraint_short_turn", "constraint_one_question_per_turn",
            "constraint_structured_format", "constraint_language_level", "guardrail_no_direct_answers",
            "guardrail_integrity_language", "guardrail_privacy", "guardrail_respectful"]
AI_MOVES = ["socratic_question", "stepwise_guidance", "hint", "request_attempt", "request_evidence",
            "affirm_extend", "final_answer", "redirect", "recap", "closure", "other"]
ST_MOVES = ["asks_question", "explains", "argues_with_evidence", "compares_evaluates", "submits_work",
            "requests_answer", "requests_help", "agrees_acknowledges", "off_task", "other"]

# --------------------------------------------------------------------------
# Layer 1: three synthetic teacher prompts with hand-assigned codes
# --------------------------------------------------------------------------
PROMPTS = [
    {
        "discussionId": "SYN-ACT-01", "style": "minimal",
        "discussionContent": "Students will talk about renewable energy sources and share which one they think is most useful.",
        "codes": {"task_specificity": 1, "prompt_target_dok": 2, "dok_uncertainty": True,
                  "prompt_role": "unknown", "prompt_tone": "unknown"},
    },
    {
        "discussionId": "SYN-ACT-02", "style": "structured task",
        "discussionContent": ("Guide the student, one step at a time, to write a paragraph explaining how the water "
                              "cycle moves water from the ocean to a mountain stream. First check that they can name "
                              "the four stages, then have them describe each stage in order, then have them write the "
                              "paragraph. Keep your replies short."),
        "codes": {"task_specificity": 2, "finish_line": 1, "scaffold_stepwise": 1, "constraint_short_turn": 1,
                  "epistemic_explain_reasoning": 1, "prompt_target_dok": 2, "dok_uncertainty": False,
                  "prompt_role": "unknown", "prompt_tone": "unknown"},
    },
    {
        "discussionId": "SYN-ACT-03", "style": "epistemic-guarded",
        "discussionContent": ("Act as a Socratic tutor. Students are evaluating whether their town should replace a "
                              "parking lot with a community garden. Never give them a position or write their argument "
                              "for them. Ask questions that make them state a claim, support it with evidence from the "
                              "two readings, and respond to the strongest counterargument. Ask one question at a time."),
        "codes": {"task_specificity": 2, "finish_line": 0, "scaffold_socratic": 1,
                  "scaffold_worked_example_prohibition": 1, "epistemic_use_evidence": 1,
                  "epistemic_justify_claims": 1, "constraint_one_question_per_turn": 1,
                  "guardrail_no_direct_answers": 1, "prompt_target_dok": 3, "dok_uncertainty": False,
                  "prompt_role": "tutor", "prompt_tone": "unknown"},
    },
]

# --------------------------------------------------------------------------
# Layers 2 and 3: four synthetic conversations. Each turn: (role, text,
# ai_codes | (dok, student_moves)).
# --------------------------------------------------------------------------
CONVERSATIONS = [
    {
        "conversationId": "SYN-CONV-01", "discussionId": "SYN-ACT-03", "user_id": "SYN-STU-01",
        "note": "reaches the DOK 3 target on the fourth student turn, then regresses",
        "turns": [
            ("ai", "Should the town replace the parking lot with a community garden? Start by telling me the position you lean toward.", ["socratic_question", "request_attempt"]),
            ("student", "I think the garden is better.", (1, ["explains"])),
            ("ai", "Why do you lean that way? What in the readings supports the garden?", ["socratic_question", "request_evidence"]),
            ("student", "The second reading said the town has almost no green space downtown and the lot is only half full most days.", (2, ["explains", "argues_with_evidence"])),
            ("ai", "Good, that is evidence. What would someone who wants to keep the parking lot say back to you?", ["affirm_extend", "socratic_question"]),
            ("student", "They would say businesses need the parking.", (2, ["explains"])),
            ("ai", "How would you answer that with what you know from the readings?", ["socratic_question", "request_evidence"]),
            ("student", "The first reading said the lot is half empty most days, so losing it would not really hurt the shops, and the garden would bring more people downtown on weekends, which actually helps the shops more than empty parking spaces do. So the counterargument is weaker than it sounds.", (3, ["argues_with_evidence", "compares_evaluates"])),
            ("ai", "You have used evidence to turn the counterargument around. Which of your two points is stronger, and why?", ["affirm_extend", "socratic_question"]),
            ("student", "probably the half empty one", (1, ["explains"])),
            ("ai", "Say more about that: what makes it the stronger point?", ["socratic_question", "request_evidence"]),
            ("student", "because it is a fact from the reading not just an opinion", (2, ["explains"])),
        ],
        "conv_codes": {"student_asks_questions": 0, "student_explains_reasoning": 1, "student_uses_evidence_or_sources": 1,
                       "student_compares_alternatives": 1, "student_requests_revision_or_iteration": 0,
                       "student_requests_hints_or_scaffolding": 0, "student_expresses_confusion_or_stuck": 0,
                       "student_reflects_metacognitively": 0, "student_requests_direct_answer": 0,
                       "student_requests_completion_of_work": 0, "student_copy_paste_like_input": 0,
                       "student_off_topic_or_non_instructional": 0, "student_attempts_to_bypass_constraints": 0,
                       "ai_asks_socratic_questions": 1, "ai_provides_stepwise_guidance": 0, "ai_provides_hints_first": 0,
                       "ai_requests_student_attempt_first": 1, "ai_refuses_or_limits_direct_answers": 0,
                       "ai_requests_evidence_or_reasoning": 1, "ai_redirects_off_topic_back_to_task": 0,
                       "ai_provides_final_answer_or_solution": 0, "ai_off_safeguard": 0,
                       "alignment_overall": "on_track", "deviation_types": "none",
                       "recoded_student_dok_max": 3, "recoded_student_dok_final": 2, "recoded_confidence": "high"},
    },
    {
        "conversationId": "SYN-CONV-02", "discussionId": "SYN-ACT-03", "user_id": "SYN-STU-02",
        "note": "stays below the DOK 3 target; recall and answer-seeking",
        "turns": [
            ("ai", "Should the town replace the parking lot with a community garden? Start by telling me the position you lean toward.", ["socratic_question", "request_attempt"]),
            ("student", "idk", (0, ["other"])),
            ("ai", "That is fine. What is one thing you remember from either reading about the parking lot or the garden?", ["socratic_question", "hint"]),
            ("student", "the garden would have vegetables", (1, ["explains"])),
            ("ai", "Yes. Does that make you lean toward the garden or the parking lot?", ["affirm_extend", "socratic_question"]),
            ("student", "garden", (1, ["explains"])),
            ("ai", "What evidence from the readings supports choosing the garden?", ["socratic_question", "request_evidence"]),
            ("student", "can you just tell me what to write", (0, ["requests_answer"])),
            ("ai", "I cannot write it for you, but I can help you find it. Look at the second reading. What does it say about green space downtown?", ["redirect", "hint", "socratic_question"]),
            ("student", "there isnt much green space", (1, ["explains"])),
            ("ai", "Good. Now put those together in one sentence: your claim and that piece of evidence.", ["affirm_extend", "stepwise_guidance", "request_attempt"]),
            ("student", "the town should build the garden because there is not much green space downtown", (2, ["explains", "argues_with_evidence"])),
        ],
        "conv_codes": {"student_asks_questions": 1, "student_explains_reasoning": 1, "student_uses_evidence_or_sources": 1,
                       "student_compares_alternatives": 0, "student_requests_revision_or_iteration": 0,
                       "student_requests_hints_or_scaffolding": 0, "student_expresses_confusion_or_stuck": 1,
                       "student_reflects_metacognitively": 0, "student_requests_direct_answer": 1,
                       "student_requests_completion_of_work": 1, "student_copy_paste_like_input": 0,
                       "student_off_topic_or_non_instructional": 0, "student_attempts_to_bypass_constraints": 0,
                       "ai_asks_socratic_questions": 1, "ai_provides_stepwise_guidance": 1, "ai_provides_hints_first": 1,
                       "ai_requests_student_attempt_first": 1, "ai_refuses_or_limits_direct_answers": 1,
                       "ai_requests_evidence_or_reasoning": 1, "ai_redirects_off_topic_back_to_task": 1,
                       "ai_provides_final_answer_or_solution": 0, "ai_off_safeguard": 0,
                       "alignment_overall": "partially_on_track", "deviation_types": "shortcut_answer_seeking",
                       "recoded_student_dok_max": 2, "recoded_student_dok_final": 2, "recoded_confidence": "high"},
    },
    {
        "conversationId": "SYN-CONV-03", "discussionId": "SYN-ACT-02", "user_id": "SYN-STU-03",
        "note": "structured task; reaches the DOK 2 target and closes on the deliverable",
        "turns": [
            ("ai", "Let's start with step one. Can you name the four stages of the water cycle?", ["stepwise_guidance", "request_attempt"]),
            ("student", "evaporation, condensation, precipitation, collection", (1, ["explains"])),
            ("ai", "All four, nicely done. Step two: describe what happens in evaporation, in one or two sentences.", ["affirm_extend", "stepwise_guidance"]),
            ("student", "The sun heats the ocean and water turns into vapor that rises up.", (2, ["explains"])),
            ("ai", "Right. Now condensation: what happens to that vapor as it rises?", ["affirm_extend", "stepwise_guidance", "socratic_question"]),
            ("student", "it gets cold high up so it turns back into tiny drops and makes clouds", (2, ["explains"])),
            ("ai", "Good. Precipitation and collection next: how does the water get from the cloud to a mountain stream?", ["affirm_extend", "stepwise_guidance", "socratic_question"]),
            ("student", "the drops get heavy and fall as rain or snow on the mountain, then it runs downhill and collects in streams", (2, ["explains"])),
            ("ai", "You have all four stages in order. Step three: write them as one paragraph.", ["recap", "stepwise_guidance", "request_attempt"]),
            ("student", "The water cycle starts when the sun heats the ocean and water evaporates into vapor. The vapor rises, cools, and condenses into clouds. When the drops get heavy they fall as rain or snow on the mountains. The water runs downhill and collects in streams.", (2, ["explains", "submits_work"])),
            ("ai", "That paragraph explains all four stages in order. You have completed the task.", ["affirm_extend", "closure"]),
        ],
        "conv_codes": {"student_asks_questions": 0, "student_explains_reasoning": 1, "student_uses_evidence_or_sources": 0,
                       "student_compares_alternatives": 0, "student_requests_revision_or_iteration": 0,
                       "student_requests_hints_or_scaffolding": 0, "student_expresses_confusion_or_stuck": 0,
                       "student_reflects_metacognitively": 0, "student_requests_direct_answer": 0,
                       "student_requests_completion_of_work": 0, "student_copy_paste_like_input": 0,
                       "student_off_topic_or_non_instructional": 0, "student_attempts_to_bypass_constraints": 0,
                       "ai_asks_socratic_questions": 1, "ai_provides_stepwise_guidance": 1, "ai_provides_hints_first": 0,
                       "ai_requests_student_attempt_first": 1, "ai_refuses_or_limits_direct_answers": 0,
                       "ai_requests_evidence_or_reasoning": 0, "ai_redirects_off_topic_back_to_task": 0,
                       "ai_provides_final_answer_or_solution": 0, "ai_off_safeguard": 0,
                       "alignment_overall": "on_track", "deviation_types": "none",
                       "recoded_student_dok_max": 2, "recoded_student_dok_final": 2, "recoded_confidence": "high"},
    },
    {
        "conversationId": "SYN-CONV-04", "discussionId": "SYN-ACT-01", "user_id": "SYN-STU-04",
        "note": "minimal prompt; drifts off task, then a brief DOK 2 explanation",
        "turns": [
            ("ai", "Which renewable energy source do you think is most useful?", ["socratic_question"]),
            ("student", "solar", (1, ["explains"])),
            ("ai", "What makes solar stand out to you compared with the others?", ["socratic_question"]),
            ("student", "do you know what time lunch is", (0, ["off_task"])),
            ("ai", "I can't help with the schedule, but let's get back to energy. Why solar rather than wind or hydro?", ["redirect", "socratic_question"]),
            ("student", "solar works almost anywhere because most places get sun, but wind only works where it is windy and hydro needs a river", (2, ["explains", "compares_evaluates"])),
            ("ai", "That is a real comparison. Is there a downside to solar you would want to mention?", ["affirm_extend", "socratic_question"]),
            ("student", "it doesnt work at night", (1, ["explains"])),
            ("ai", "True. Thanks for sharing your thinking on this.", ["closure", "other"]),
        ],
        "conv_codes": {"student_asks_questions": 1, "student_explains_reasoning": 1, "student_uses_evidence_or_sources": 0,
                       "student_compares_alternatives": 1, "student_requests_revision_or_iteration": 0,
                       "student_requests_hints_or_scaffolding": 0, "student_expresses_confusion_or_stuck": 0,
                       "student_reflects_metacognitively": 0, "student_requests_direct_answer": 0,
                       "student_requests_completion_of_work": 0, "student_copy_paste_like_input": 0,
                       "student_off_topic_or_non_instructional": 1, "student_attempts_to_bypass_constraints": 0,
                       "ai_asks_socratic_questions": 1, "ai_provides_stepwise_guidance": 0, "ai_provides_hints_first": 0,
                       "ai_requests_student_attempt_first": 0, "ai_refuses_or_limits_direct_answers": 0,
                       "ai_requests_evidence_or_reasoning": 0, "ai_redirects_off_topic_back_to_task": 1,
                       "ai_provides_final_answer_or_solution": 0, "ai_off_safeguard": 0,
                       "alignment_overall": "partially_on_track", "deviation_types": "off_topic",
                       "recoded_student_dok_max": 2, "recoded_student_dok_final": 1, "recoded_confidence": "high"},
    },
]


def main():
    # Layer 1 files
    pd.DataFrame([{"discussionId": p["discussionId"], "discussionContent": p["discussionContent"]}
                  for p in PROMPTS]).to_csv(os.path.join(HERE, "prompts.csv"), index=False)
    rows = []
    for p in PROMPTS:
        r = {"discussionId": p["discussionId"], "synthetic": True, "style": p["style"]}
        for f in FEATURES:
            r[f] = p["codes"].get(f, 0)
        for k in ["prompt_target_dok", "dok_uncertainty", "prompt_role", "prompt_tone"]:
            r[k] = p["codes"][k]
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "prompts_coded.csv"), index=False)

    # Layer 2 and 3 files
    target = {p["discussionId"]: p["codes"]["prompt_target_dok"] for p in PROMPTS}
    text = {p["discussionId"]: p["discussionContent"] for p in PROMPTS}
    conv_rows, turn_rows, md = [], [], ["# SYNTHETIC transcripts with codes\n",
                                        "**Everything here is invented.** See build_synthetic_examples.py.\n"]
    with open(os.path.join(HERE, "transcripts.jsonl"), "w", encoding="utf-8") as f:
        for c in CONVERSATIONS:
            did, cid = c["discussionId"], c["conversationId"]
            tr = [{"role": role, "text": t} for role, t, _ in c["turns"]]
            f.write(json.dumps({"conversationId": cid, "discussionId": did, "user_id": c["user_id"],
                                "synthetic": True, "discussionContent": text[did],
                                "prompt_target_dok": target[did], "conversation_transcript": tr},
                               ensure_ascii=False) + "\n")
            n_st = sum(r == "student" for r, _, _ in c["turns"])
            tgt = target[did]
            mx = c["conv_codes"]["recoded_student_dok_max"]
            conv_rows.append({"conversationId": cid, "discussionId": did, "user_id": c["user_id"], "synthetic": True,
                              "n_messages": len(tr), "n_student_msgs": n_st, "n_ai_msgs": len(tr) - n_st,
                              "prompt_target_dok": tgt, **c["conv_codes"],
                              "recoded_dok_alignment": "aligned" if mx == tgt else "under_target" if mx < tgt else "over_target"})
            md.append(f"\n## {cid} (activity {did}, target DOK {tgt}) — {c['note']}\n")
            md.append("| Turn | Speaker | Text (synthetic) | Code |\n|---|---|---|---|")
            for i, (role, t, code) in enumerate(c["turns"], start=1):
                r = {"conversationId": cid, "discussionId": did, "user_id": c["user_id"], "prompt_target_dok": tgt,
                     "turn": i, "role": role, "n_chars": len(t), "student_dok": None}
                if role == "ai":
                    for m in AI_MOVES:
                        r[f"ai_{m}"] = int(m in code)
                    for m in ST_MOVES:
                        r[f"st_{m}"] = None
                    label = ", ".join(code)
                else:
                    dok, moves = code
                    r["student_dok"] = dok
                    for m in AI_MOVES:
                        r[f"ai_{m}"] = None
                    for m in ST_MOVES:
                        r[f"st_{m}"] = int(m in moves)
                    label = f"DOK {dok}; " + ", ".join(moves)
                turn_rows.append(r)
                md.append(f"| {i} | {'AI' if role == 'ai' else 'Student'} | {t} | {label} |")
    pd.DataFrame(conv_rows).to_csv(os.path.join(HERE, "conversation_codes.csv"), index=False)
    pd.DataFrame(turn_rows).to_csv(os.path.join(HERE, "turn_codes.csv"), index=False)
    open(os.path.join(HERE, "transcripts_coded.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(f"wrote {len(PROMPTS)} prompts, {len(CONVERSATIONS)} conversations, {len(turn_rows)} turns")


if __name__ == "__main__":
    main()
