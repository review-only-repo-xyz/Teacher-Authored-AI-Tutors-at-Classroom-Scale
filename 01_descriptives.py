"""
REPLICATION VERSION.

Descriptive summary of the analytic dataset (sample sizes, target DOK
distribution, DOK reached, and achievement by target stratum) computed from the de-identified coded data in data/.

Usage
    python 01_descriptives.py [--exclude-subject "World Language"]

Outputs (outputs/)
    descriptives_sample.csv
    descriptives_by_target.csv
"""
import argparse
import os

import pandas as pd

from config import DATA_DIR, OUTPUT_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--exclude-subject", action="append", default=[])
args = ap.parse_args()

os.makedirs(OUTPUT_DIR, exist_ok=True)
conv = pd.read_csv(os.path.join(DATA_DIR, "conversations.csv"))
turns = pd.read_csv(os.path.join(DATA_DIR, "turn_codes.csv"))
if args.exclude_subject:
    conv = conv[~conv["subject"].isin(args.exclude_subject)].copy()
    turns = turns[turns["conversationId"].isin(conv["conversationId"])].copy()

tmax = (turns[turns["role"] == "student"].groupby("conversationId")["student_dok"]
        .max().rename("dok_reached"))
conv = conv.merge(tmax, on="conversationId", how="inner")
conv["achieved"] = (conv["dok_reached"] >= conv["prompt_target_dok"]).astype(int)

sample = pd.Series({
    "teachers": conv["teacher_name"].nunique(),
    "activities": conv["discussionId"].nunique(),
    "conversations": len(conv),
    "students": conv["user_id"].nunique(),
    "turns_coded": len(turns),
    "median_messages_per_conversation": conv["n_messages"].median(),
}, name="value")
by_target = conv.groupby("prompt_target_dok").agg(
    n_activities=("discussionId", "nunique"), n_conversations=("conversationId", "size"),
    mean_dok_reached=("dok_reached", "mean"), achievement_rate=("achieved", "mean"),
).round(3)

sample.to_csv(os.path.join(OUTPUT_DIR, "descriptives_sample.csv"))
by_target.to_csv(os.path.join(OUTPUT_DIR, "descriptives_by_target.csv"))
print(sample.to_string(), "\n")
print(by_target.to_string())
