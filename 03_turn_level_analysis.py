"""
REPLICATION VERSION.

This script is shared for replication and review. It runs on the
de-identified coded datasets in data/ (activities.csv, conversations.csv,
turn_codes.csv), which contain the study's real codes but no conversation
text, no prompt text, no names, and no platform identifiers. See README.md.

RQ2: Turn-level analyses of AI moves and student DOK trajectories.

Consumes the long-format turn codes (data/turn_codes.csv) and the
conversation-level analytic dataset. All group comparisons contrast
goal-achieved with goal-not-achieved conversations inside one target-DOK
stratum (default DOK 3) and, for the AI-move comparisons, inside activities
that contain both outcomes, so that differences in AI behavior are not
differences in teacher prompts.

Analyses
  A. Agreement between turn-level and conversation-level achievement labels.
  B. Time to target: first student turn at or above target (index and share
     of student turns), cumulative attainment curve, discrete-time hazard
     model with the preceding AI move as time-varying predictor.
  C. Persistence at target: post-attainment turns at target, sustained /
     regressed / intermittent classification, final-turn attainment share,
     AI move following first attainment.
  D. DOK transition networks (achieved vs not achieved): transition
     probabilities, pruning, bootstrap CIs, edge differences, strength
     centralities, two-panel figure.
  E. Sequential pattern mining (PrefixSpan) on AI-move sequences: cohort
     patterns, per-group patterns at a minsup sweep, unique patterns,
     per-conversation pattern counts.

Usage
    python 03_turn_level_analysis.py data/turn_codes.csv --exclude-subject "World Language"
    python 03_turn_level_analysis.py <csv> --attainment conv --tag _convlabel     # sensitivity
    python 03_turn_level_analysis.py <csv> --target 3 --minsup 0.5 0.6 0.7 --maxlen 4

Outputs (outputs/rq2_*.csv, outputs/figures/fig_rq2_*.png)
"""
import argparse
import itertools
import os
import sys
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, OUTPUT_DIR  # noqa: E402

warnings.filterwarnings("ignore")
rng = np.random.default_rng(2026)

DATA_FILE = os.path.join(DATA_DIR, "conversations.csv")
PATTERN_FILE = os.path.join(OUTPUT_DIR, "rq1_activity_features.csv")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)
SUFFIX = ""          # set to "_turnrule" for the sensitivity run (after args are parsed)

PAL = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
       "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e6e5e1"}
GROUP_COL = {"achieved": PAL["blue"], "not_achieved": PAL["orange"]}

AI_MOVES = ["socratic_question", "stepwise_guidance", "hint", "request_attempt", "request_evidence",
            "affirm_extend", "final_answer", "redirect", "recap", "closure", "other"]
# priority used to assign one primary move per AI turn for sequence mining
AI_PRIORITY = ["final_answer", "redirect", "request_evidence", "request_attempt", "stepwise_guidance",
               "hint", "socratic_question", "affirm_extend", "recap", "closure", "other"]
AI_ABBR = {"socratic_question": "SQ", "stepwise_guidance": "SG", "hint": "HT", "request_attempt": "RA",
           "request_evidence": "RE", "affirm_extend": "AE", "final_answer": "FA", "redirect": "RD",
           "recap": "RC", "closure": "CL", "other": "OT"}
STATES = ["D0", "D1", "D2", "D3+"]        # DOK 4 merged into 3+ (rare)
PRUNE_QUANTILE = 0.15                     # drop the weakest 15% of edges (FTNA convention)
N_BOOT = 999


def style_axes(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(PAL["grid"])
    ax.tick_params(colors=PAL["ink2"], labelsize=9)
    ax.yaxis.grid(True, color=PAL["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


# ==========================================================================
# 0. Load and join
# ==========================================================================
ap = argparse.ArgumentParser()
ap.add_argument("turn_csv")
ap.add_argument("--target", type=int, default=3)
ap.add_argument("--minsup", type=float, nargs="+", default=[0.5, 0.6, 0.7])
ap.add_argument("--maxlen", type=int, default=4)
ap.add_argument("--attainment", choices=["conv", "turn"], default="turn",
                help="turn (primary): goal achieved and group membership are defined from the maximum "
                     "turn-level student DOK; conv: the validated conversation-level label defines "
                     "achievement and grouping, and turn codes only locate the first at-target turn")
ap.add_argument("--exclude-subject", action="append", default=[], help="subject to drop (repeatable)")
ap.add_argument("--include-subject", action="append", default=[],
                help="keep only these subjects (repeatable); applied after exclusions")
ap.add_argument("--pattern-file", default=None, help="rq1_activity_features*.csv with pattern labels")
ap.add_argument("--tag", default=None, help="suffix for output files (default: '' for turn, '_convlabel' for conv)")
args = ap.parse_args()
TARGET = args.target
SUFFIX = args.tag if args.tag is not None else ("" if args.attainment == "turn" else "_convlabel")
if args.pattern_file:
    PATTERN_FILE = args.pattern_file

turns = pd.read_csv(args.turn_csv)
conv = pd.read_csv(DATA_FILE)
conv = conv[conv["recoded_dok_alignment"].notna()].copy()
if args.exclude_subject:
    conv = conv[~conv["subject"].isin(args.exclude_subject)].copy()
    print(f"Excluded subjects {args.exclude_subject}; analytic sample now {len(conv)} conversations")
if args.include_subject:
    conv = conv[conv["subject"].isin(args.include_subject)].copy()
    print(f"Restricted to subjects {args.include_subject}; analytic sample now {len(conv)} conversations")
conv["target"] = conv["prompt_target_dok"].astype(int)
conv["achieved_conv"] = conv["recoded_dok_alignment"].isin(["aligned", "over_target"]).astype(int)
if os.path.exists(PATTERN_FILE):
    pat = pd.read_csv(PATTERN_FILE, index_col=0)[["pattern"]]
    conv = conv.merge(pat, left_on="discussionId", right_index=True, how="left")
else:
    conv["pattern"] = "NA"
meta = conv.set_index("conversationId")[["discussionId", "teacher_name", "subject", "target",
                                         "achieved_conv", "pattern", "n_messages"]]

turns = turns[turns["conversationId"].isin(meta.index)].copy()
turns = turns.drop(columns=[c for c in ["discussionId", "user_id", "prompt_target_dok"] if c in turns.columns])
turns = turns.merge(meta, left_on="conversationId", right_index=True, how="left")
turns = turns.sort_values(["conversationId", "turn"]).reset_index(drop=True)

# student-turn ordinal (1..n) within conversation, ignoring DOK-0 turns for the
# "substantive" ordinal but keeping the raw ordinal too
st = turns[turns["role"] == "student"].copy()
st["k"] = st.groupby("conversationId").cumcount() + 1
st["dok"] = st["student_dok"].fillna(0).astype(int)
st["at_target"] = (st["dok"] >= st["target"]).astype(int)
n_student_turns = st.groupby("conversationId")["k"].max().rename("n_student_turns")

# AI move immediately preceding each student turn (the move the student responds to)
ai = turns[turns["role"] == "ai"].copy()
for m in AI_MOVES:
    ai[f"ai_{m}"] = ai[f"ai_{m}"].fillna(0).astype(int)


def primary_move(row):
    for m in AI_PRIORITY:
        if row[f"ai_{m}"] == 1:
            return m
    return "other"


ai["primary_move"] = ai.apply(primary_move, axis=1)
prev_ai = {}
for cid, g in turns.groupby("conversationId"):
    last = None
    for _, r in g.iterrows():
        if r["role"] == "ai":
            last = r["turn"]
        elif r["role"] == "student":
            prev_ai[(cid, r["turn"])] = last
st["prev_ai_turn"] = [prev_ai.get((c, t)) for c, t in zip(st["conversationId"], st["turn"])]
ai_idx = ai.set_index(["conversationId", "turn"])
for m in AI_MOVES:
    st[f"prev_{m}"] = [ai_idx.loc[(c, t), f"ai_{m}"] if pd.notna(t) and (c, t) in ai_idx.index else 0
                       for c, t in zip(st["conversationId"], st["prev_ai_turn"])]

# conversation-level turn-derived summary
cs = st.groupby("conversationId").agg(
    n_student_turns=("k", "max"), max_dok_turn=("dok", "max"),
    target=("target", "first"), achieved_conv=("achieved_conv", "first"),
    discussionId=("discussionId", "first"), teacher=("teacher_name", "first"),
    pattern=("pattern", "first"))
cs["achieved_turn"] = (cs["max_dok_turn"] >= cs["target"]).astype(int)
first_hit = st[st["at_target"] == 1].groupby("conversationId")["k"].min().rename("k_first_turn")
cs = cs.join(first_hit)
if args.attainment == "conv":
    cs["k_first"] = np.where(cs["achieved_conv"] == 1, cs["k_first_turn"], np.nan)
    cs["achieved"] = cs["achieved_conv"]
else:
    cs["k_first"] = cs["k_first_turn"]
    cs["achieved"] = cs["achieved_turn"]
cs["share_first"] = cs["k_first"] / cs["n_student_turns"]
cs["group"] = np.where(cs["achieved"] == 1, "achieved", "not_achieved")
print(f"Attainment and grouping rule: {args.attainment}")

# ==========================================================================
# A. Agreement between turn-level and conversation-level labels
# ==========================================================================
agree = pd.crosstab(cs["achieved_conv"], cs["achieved_turn"], rownames=["conv_level"], colnames=["turn_level"])
agree.to_csv(os.path.join(OUTPUT_DIR, f"rq2_label_agreement{SUFFIX}.csv"))
acc = (cs["achieved_conv"] == cs["achieved_turn"]).mean()
print(f"Conversations with turn codes: {len(cs)} | turn-level vs conversation-level achievement agreement: {acc:.1%}")
print(agree)
cs = cs.join(meta[["subject"]])
dis = cs[(cs["achieved_conv"] == 0) & (cs["achieved_turn"] == 1)]
n_hits = st[st["at_target"] == 1].groupby("conversationId").size().rename("n_at_target_turns")
dis = dis.join(n_hits)
dis_tab = pd.DataFrame({
    "n_conversations": cs.groupby("subject").size(),
    "n_disagree_turn_higher": dis.groupby("subject").size(),
    "share_disagree": (dis.groupby("subject").size() / cs.groupby("subject").size()),
    "median_at_target_turns_in_disagreeing": dis.groupby("subject")["n_at_target_turns"].median(),
}).fillna(0).sort_values("share_disagree", ascending=False)
dis_tab.to_csv(os.path.join(OUTPUT_DIR, f"rq2_label_disagreement_by_subject{SUFFIX}.csv"))
print(f"\nAttainment rule: {args.attainment}. Conversations where turn-level exceeds conversation-level: {len(dis)}; "
      f"share with exactly one at-target turn: {(dis['n_at_target_turns'] == 1).mean():.2f}")
print(dis_tab.round(2).to_string())
n_ach_nohit = int(((cs["achieved_conv"] == 1) & cs["k_first_turn"].isna()).sum())
print(f"Conversation-level achieved with no turn-level at-target turn (excluded from timing): {n_ach_nohit}")

strat = cs[cs["target"] == TARGET].copy()
# mixed-outcome activities: those containing both outcomes under the grouping rule in use
mixed_acts = strat.groupby("discussionId")["achieved"].agg(["min", "max"])
mixed_acts = mixed_acts[(mixed_acts["min"] == 0) & (mixed_acts["max"] == 1)].index
strat["mixed_activity"] = strat["discussionId"].isin(mixed_acts)
print(f"\nStratum target DOK {TARGET}: {len(strat)} conversations, "
      f"{strat['achieved'].sum()} achieved; {strat['mixed_activity'].sum()} in mixed-outcome activities "
      f"({len(mixed_acts)} activities)")

# ==========================================================================
# B. Time to target
# ==========================================================================
ach = strat[strat["k_first"].notna()].copy()
tt = ach.groupby("pattern").agg(n=("k_first", "size"), median_k=("k_first", "median"),
                                q1_k=("k_first", lambda s: s.quantile(.25)), q3_k=("k_first", lambda s: s.quantile(.75)),
                                median_share=("share_first", "median"),
                                first_turn_share=("k_first", lambda s: (s == 1).mean()))
tt.loc["all"] = [len(ach), ach["k_first"].median(), ach["k_first"].quantile(.25), ach["k_first"].quantile(.75),
                 ach["share_first"].median(), (ach["k_first"] == 1).mean()]
tt.to_csv(os.path.join(OUTPUT_DIR, f"rq2_time_to_target{SUFFIX}.csv"))
print("\nTime to target (achieved conversations, student-turn ordinal):\n", tt.round(2).to_string())

# cumulative attainment curve (discrete Kaplan-Meier style)
kmax = int(strat["n_student_turns"].quantile(.95))
curve = []
for k in range(1, kmax + 1):
    at_risk = ((strat["n_student_turns"] >= k) & ((strat["k_first"].isna()) | (strat["k_first"] >= k))).sum()
    hit = (strat["k_first"] == k).sum()
    curve.append({"k": k, "at_risk": at_risk, "hit": hit, "hazard": hit / at_risk if at_risk else np.nan})
curve = pd.DataFrame(curve)
curve["survival"] = (1 - curve["hazard"].fillna(0)).cumprod()
curve["cum_attainment_km"] = 1 - curve["survival"]          # Kaplan-Meier, treats ended conversations as censored
curve["cum_share_raw"] = curve["hit"].cumsum() / len(strat)   # share of all stratum conversations attained by turn k
curve.to_csv(os.path.join(OUTPUT_DIR, f"rq2_attainment_curve{SUFFIX}.csv"), index=False)

# person-period data for the discrete-time hazard model (until first attainment)
pp = st[st["conversationId"].isin(strat.index)].merge(cs[["k_first"]], left_on="conversationId", right_index=True)
pp = pp[(pp["k_first"].isna()) | (pp["k"] <= pp["k_first"])].copy()
pp["event"] = ((pp["k"] == pp["k_first"])).astype(int)
pp["log_k"] = np.log(pp["k"])
pp = pp.merge(n_student_turns, left_on="conversationId", right_index=True, suffixes=("", "_y"))
pp["log_len"] = np.log(pp["n_student_turns"])
pp["activity"] = pp["discussionId"].astype(str)
big_pats = strat["pattern"].value_counts()
pp = pp[pp["pattern"].isin(big_pats[big_pats >= 20].index)]
haz_terms = " + ".join(f"prev_{m}" for m in AI_MOVES
                       if m != "other" and pp.loc[pp["event"] == 1, f"prev_{m}"].sum() >= 10)
formula = (f"event ~ log_k + log_len + C(pattern) + {haz_terms}" if pp["pattern"].nunique() > 1
           else f"event ~ log_k + log_len + {haz_terms}")
try:
    haz = smf.logit(formula, pp).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pp["activity"]})
    hz = pd.DataFrame({"term": haz.params.index, "log_odds": haz.params.values, "se": haz.bse.values,
                       "odds_ratio": np.exp(haz.params.values), "p": haz.pvalues.values,
                       "ci_low_or": np.exp(haz.conf_int()[0].values), "ci_high_or": np.exp(haz.conf_int()[1].values)})
    hz.to_csv(os.path.join(OUTPUT_DIR, f"rq2_hazard_model{SUFFIX}.csv"), index=False)
    print(f"\nDiscrete-time hazard of first attainment (person-periods = {len(pp)}, SE clustered by activity):\n",
          hz.round(3).to_string(index=False))
except Exception as e:
    print("Hazard model failed:", e)

# robustness: activity fixed effects instead of pattern, mixed-outcome activities only
try:
    ppm = pp[pp["discussionId"].isin(mixed_acts)].copy()
    f_fe = f"event ~ log_k + log_len + C(activity) + {haz_terms}"
    haz_fe = smf.logit(f_fe, ppm).fit(disp=0, maxiter=200, cov_type="cluster", cov_kwds={"groups": ppm["activity"]})
    keep = [t for t in haz_fe.params.index if not t.startswith("C(activity)")]
    hz_fe = pd.DataFrame({"term": keep, "log_odds": haz_fe.params[keep].values, "se": haz_fe.bse[keep].values,
                          "odds_ratio": np.exp(haz_fe.params[keep].values), "p": haz_fe.pvalues[keep].values,
                          "ci_low_or": np.exp(haz_fe.conf_int().loc[keep, 0].values),
                          "ci_high_or": np.exp(haz_fe.conf_int().loc[keep, 1].values)})
    hz_fe.to_csv(os.path.join(OUTPUT_DIR, f"rq2_hazard_model_activity_fe{SUFFIX}.csv"), index=False)
    print(f"\nHazard model with activity fixed effects (mixed-outcome activities, person-periods = {len(ppm)}):\n",
          hz_fe.round(3).to_string(index=False))
except Exception as e:
    print("Hazard FE model failed:", e)

# ==========================================================================
# C. Persistence at target
# ==========================================================================
def persistence(cid):
    s = st[st["conversationId"] == cid].sort_values("k")
    k0 = cs.loc[cid, "k_first"]
    if pd.isna(k0):
        return None
    post = s[s["k"] > k0]["at_target"].values
    hit_last = int(s["at_target"].values[-1] == 1)
    if len(post) == 0:
        cls = "final_turn"                       # reached on the last student turn
    elif post.min() == 1:
        cls = "sustained"
    elif post[-1] == 1:
        cls = "intermittent"
    else:
        # fell below target; did it come back at all?
        cls = "intermittent" if post.sum() > 0 else "regressed"
    # AI move that follows the first at-target turn
    after = ai[(ai["conversationId"] == cid) & (ai["turn"] > s[s["k"] == k0]["turn"].iloc[0])].sort_values("turn")
    nxt = after["primary_move"].iloc[0] if len(after) else "none"
    return {"conversationId": cid, "k_first": k0, "n_post": len(post),
            "n_post_at_target": int(post.sum()), "share_post_at_target": post.mean() if len(post) else np.nan,
            "ends_at_target": hit_last, "persistence_class": cls, "ai_move_after_first_hit": nxt}


pers = pd.DataFrame([r for r in (persistence(c) for c in ach.index) if r])
pers = pers.merge(cs[["pattern", "n_student_turns"]], left_on="conversationId", right_index=True)
pers.to_csv(os.path.join(OUTPUT_DIR, f"rq2_persistence_conversations{SUFFIX}.csv"), index=False)
psum = pers.groupby("pattern").agg(n=("conversationId", "size"), mean_post=("n_post", "mean"),
                                   mean_share_post=("share_post_at_target", "mean"),
                                   ends_at_target=("ends_at_target", "mean"))
psum.loc["all"] = [len(pers), pers["n_post"].mean(), pers["share_post_at_target"].mean(), pers["ends_at_target"].mean()]
pcls = pd.crosstab(pers["pattern"], pers["persistence_class"], normalize="index")
pcls.loc["all"] = pers["persistence_class"].value_counts(normalize=True)
psum = psum.join(pcls)
psum.to_csv(os.path.join(OUTPUT_DIR, f"rq2_persistence_summary{SUFFIX}.csv"))
print("\nPersistence at target:\n", psum.round(2).to_string())
xt = pd.crosstab(pers["ai_move_after_first_hit"], pers["persistence_class"])
xt.to_csv(os.path.join(OUTPUT_DIR, f"rq2_persistence_by_next_ai_move{SUFFIX}.csv"))
print("\nAI move after first at-target turn x persistence class:\n", xt)

# ==========================================================================
# D. DOK transition networks
# ==========================================================================
def state(d):
    return "D3+" if d >= 3 else f"D{d}"


seqs = {}
for cid, g in st.groupby("conversationId"):
    seqs[cid] = [state(d) for d in g.sort_values("k")["dok"].values]


S_IDX = {s_: i for i, s_ in enumerate(STATES)}
conv_counts = {}                                   # per-conversation transition count matrices
for cid, s_ in seqs.items():
    M = np.zeros((len(STATES), len(STATES)))
    for a, b in zip(s_[:-1], s_[1:]):
        M[S_IDX[a], S_IDX[b]] += 1
    conv_counts[cid] = M


def trans_matrix(cids):
    M = np.zeros((len(STATES), len(STATES)))
    for c in cids:
        if c in conv_counts:
            M += conv_counts[c]
    rs = M.sum(axis=1, keepdims=True)
    P = np.divide(M, rs, out=np.zeros_like(M), where=rs > 0)
    return pd.DataFrame(M, index=STATES, columns=STATES), pd.DataFrame(P, index=STATES, columns=STATES)


groups = {"achieved": strat[strat["group"] == "achieved"].index.tolist(),
          "not_achieved": strat[strat["group"] == "not_achieved"].index.tolist()}
net_rows, boot_store = [], {}
for gname, cids in groups.items():
    M, P = trans_matrix(cids)
    boots = np.zeros((N_BOOT, len(STATES), len(STATES)))
    cids_arr = np.array(cids)
    for b in range(N_BOOT):
        samp = rng.choice(cids_arr, size=len(cids_arr), replace=True)
        boots[b] = trans_matrix(samp)[1].values
    boot_store[gname] = boots
    thr = np.quantile(P.values[P.values > 0], PRUNE_QUANTILE) if (P.values > 0).any() else 0
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    for i, a in enumerate(STATES):
        for j, b in enumerate(STATES):
            net_rows.append({"group": gname, "from": a, "to": b, "count": M.loc[a, b], "prob": P.loc[a, b],
                             "ci_low": lo[i, j], "ci_high": hi[i, j],
                             "retained_after_pruning": P.loc[a, b] >= thr and M.loc[a, b] > 0,
                             "n_conversations": len(cids)})
net = pd.DataFrame(net_rows)
# edge differences (achieved - not achieved) with bootstrap CI
diff = boot_store["achieved"] - boot_store["not_achieved"]
dlo, dhi = np.percentile(diff, [2.5, 97.5], axis=0)
Pa = net[net["group"] == "achieved"].set_index(["from", "to"])["prob"]
Pn = net[net["group"] == "not_achieved"].set_index(["from", "to"])["prob"]
drows = []
for i, a in enumerate(STATES):
    for j, b in enumerate(STATES):
        drows.append({"from": a, "to": b, "prob_achieved": Pa[(a, b)], "prob_not_achieved": Pn[(a, b)],
                      "difference": Pa[(a, b)] - Pn[(a, b)], "ci_low": dlo[i, j], "ci_high": dhi[i, j],
                      "ci_excludes_zero": (dlo[i, j] > 0) or (dhi[i, j] < 0)})
diffs = pd.DataFrame(drows)
net.to_csv(os.path.join(OUTPUT_DIR, f"rq2_transition_networks{SUFFIX}.csv"), index=False)
diffs.to_csv(os.path.join(OUTPUT_DIR, f"rq2_transition_differences{SUFFIX}.csv"), index=False)
# strength centralities
cent = []
for gname in groups:
    P = net[net["group"] == gname].pivot(index="from", columns="to", values="prob").loc[STATES, STATES]
    for s_ in STATES:
        cent.append({"group": gname, "state": s_, "out_strength": P.loc[s_].sum() - P.loc[s_, s_],
                     "in_strength": P[s_].sum() - P.loc[s_, s_], "self_loop": P.loc[s_, s_]})
pd.DataFrame(cent).to_csv(os.path.join(OUTPUT_DIR, f"rq2_transition_centrality{SUFFIX}.csv"), index=False)
print("\nTransition differences with bootstrap CI excluding zero:\n",
      diffs[diffs["ci_excludes_zero"]].round(3).to_string(index=False))

# figure: two-panel network
pos = {"D0": (0, 0), "D1": (1, 0), "D2": (1, 1), "D3+": (0, 1)}
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.8))
for ax, gname in zip(axes, groups):
    sub = net[(net["group"] == gname) & net["retained_after_pruning"]]
    n_conv = int(sub["n_conversations"].iloc[0]) if len(sub) else 0
    col = GROUP_COL[gname]
    for s_, (x, y) in pos.items():
        ax.add_patch(plt.Circle((x, y), 0.13, color=col, zorder=3))
        ax.text(x, y, s_, ha="center", va="center", color="white", fontsize=9, fontweight="bold", zorder=4)
        loop = sub[(sub["from"] == s_) & (sub["to"] == s_)]
        if len(loop):
            p = loop["prob"].iloc[0]
            dx, dy = (x - 0.5) * 0.4, (y - 0.5) * 0.4
            ax.text(x + dx, y + dy, f"{p:.2f}", ha="center", va="center", fontsize=8, color=PAL["ink2"],
                    bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=col, lw=1 + 3 * p))
    for _, e in sub[sub["from"] != sub["to"]].iterrows():
        (x1, y1), (x2, y2) = pos[e["from"]], pos[e["to"]]
        rad = 0.25
        arr = FancyArrowPatch((x1, y1), (x2, y2), connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
                              mutation_scale=14, lw=0.6 + 5 * e["prob"], color=col, alpha=0.85,
                              shrinkA=22, shrinkB=22, zorder=2)
        ax.add_patch(arr)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        nxv, nyv = -(y2 - y1), (x2 - x1)
        ax.text(mx + nxv * rad * 0.5, my + nyv * rad * 0.5, f"{e['prob']:.2f}", fontsize=7,
                ha="center", va="center", color=PAL["ink2"],
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9), zorder=5)
    ax.set_xlim(-0.55, 1.55); ax.set_ylim(-0.55, 1.55); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"{gname.replace('_', ' ').capitalize()} (n = {n_conv})", fontsize=10, loc="left", color=PAL["ink"])
fig.suptitle(f"Student DOK transitions, target DOK {TARGET}. Edge labels are row-normalized transition "
             f"probabilities; weakest {int(PRUNE_QUANTILE*100)}% of edges pruned", fontsize=10, x=0.01, ha="left", color=PAL["ink"])
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f"fig_rq2_transition_networks{SUFFIX}.png"), dpi=300)
plt.close(fig)

# figure: cumulative attainment
fig, ax = plt.subplots(figsize=(5.4, 3.4))
cplot = curve[curve["at_risk"] >= 10]           # stop the curves where fewer than 10 conversations remain at risk
ax.step(cplot["k"], cplot["cum_share_raw"], where="post", color=PAL["blue"], lw=2, label="Share of all conversations")
ax.step(cplot["k"], cplot["cum_attainment_km"], where="post", color=PAL["orange"], lw=2, ls="--",
        label="Kaplan-Meier estimate (ended conversations censored)")
ax.set_xlabel("Student turn", fontsize=9, color=PAL["ink2"])
ax.set_ylabel("Cumulative share reaching target", fontsize=9, color=PAL["ink2"])
ax.set_ylim(0, 1)
ax.set_title(f"Cumulative target attainment by student turn (target DOK {TARGET}, n = {len(strat)})",
             fontsize=9.5, loc="left", color=PAL["ink"])
ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.02, 0.98))
style_axes(ax)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, f"fig_rq2_attainment_curve{SUFFIX}.png"), dpi=300)
plt.close(fig)

# ==========================================================================
# E. Sequential pattern mining on AI-move sequences (PrefixSpan)
# ==========================================================================
def prefixspan(sequences, minsup_count, maxlen):
    """Classic PrefixSpan over sequences of single items. Returns {pattern: support}."""
    results = {}

    def project(db, item):
        out = []
        for seq in db:
            try:
                i = seq.index(item)
                out.append(seq[i + 1:])
            except ValueError:
                pass
        return out

    def mine(prefix, db):
        if len(prefix) >= maxlen:
            return
        counts = defaultdict(int)
        for seq in db:
            for it in set(seq):
                counts[it] += 1
        for it, c in counts.items():
            if c >= minsup_count:
                pat = prefix + (it,)
                results[pat] = c
                mine(pat, project(db, it))

    mine((), sequences)
    return results


ai_seq = {cid: [AI_ABBR[m] for m in g.sort_values("turn")["primary_move"].values]
          for cid, g in ai.groupby("conversationId")}
mix = strat[strat["mixed_activity"]]
sets = {"cohort": strat.index.tolist(),
        "achieved": mix[mix["group"] == "achieved"].index.tolist(),
        "not_achieved": mix[mix["group"] == "not_achieved"].index.tolist()}
spm_rows = []
for ms in args.minsup:
    pats = {}
    for name, cids in sets.items():
        db = [ai_seq.get(c, []) for c in cids]
        pats[name] = prefixspan(db, int(np.ceil(ms * len(db))), args.maxlen)
        for p, sup in pats[name].items():
            spm_rows.append({"minsup": ms, "set": name, "pattern": " > ".join(p), "length": len(p),
                             "support": sup, "n_sequences": len(db), "support_share": sup / len(db)})
    # unique patterns between the two groups
    for a, b in [("achieved", "not_achieved"), ("not_achieved", "achieved")]:
        for p in set(pats[a]) - set(pats[b]):
            spm_rows.append({"minsup": ms, "set": f"unique_to_{a}", "pattern": " > ".join(p), "length": len(p),
                             "support": pats[a][p], "n_sequences": len(sets[a]), "support_share": pats[a][p] / len(sets[a])})
spm = pd.DataFrame(spm_rows)


def is_subseq(p, s):
    it = iter(s)
    return all(any(x == y for y in it) for x in p)


# support of every pattern in both groups, so "unique" can be read against the other group's share
def share_in(p, cids):
    db = [ai_seq.get(c, []) for c in cids]
    return np.mean([is_subseq(p.split(" > "), s) for s in db]) if db else np.nan


spm["share_achieved"] = spm["pattern"].apply(lambda p: share_in(p, sets["achieved"]))
spm["share_not_achieved"] = spm["pattern"].apply(lambda p: share_in(p, sets["not_achieved"]))
spm = spm.sort_values(["minsup", "set", "support"], ascending=[True, True, False])
spm.to_csv(os.path.join(OUTPUT_DIR, f"rq2_prefixspan_patterns{SUFFIX}.csv"), index=False)
print(f"\nPrefixSpan (primary AI move per turn; legend {AI_ABBR}); mixed-outcome activities: "
      f"{len(sets['achieved'])} achieved vs {len(sets['not_achieved'])} not achieved")
for ms in args.minsup:
    u = spm[(spm["minsup"] == ms) & spm["set"].str.startswith("unique")]
    print(f"  minsup {ms}: {len(u)} unique patterns")
    if len(u):
        print(u[["set", "pattern", "share_achieved", "share_not_achieved"]].round(2).to_string(index=False))

# per-conversation pattern counts (occurrences with gaps allowed) for cohort patterns at the lowest minsup
cohort_pats = spm[(spm["minsup"] == min(args.minsup)) & (spm["set"] == "cohort")]["pattern"].tolist()


def count_occ(p, s):
    """Count non-overlapping occurrences of pattern p (list) in sequence s, gaps allowed."""
    n, i = 0, 0
    while True:
        j = i
        ok = True
        for x in p:
            while j < len(s) and s[j] != x:
                j += 1
            if j >= len(s):
                ok = False
                break
            j += 1
        if not ok:
            break
        n += 1
        i = j
    return n


pc = pd.DataFrame({"conversationId": strat.index})
for p in cohort_pats:
    pc[p] = [count_occ(p.split(" > "), ai_seq.get(c, [])) for c in pc["conversationId"]]
pc = pc.merge(strat[["group", "pattern", "mixed_activity"]], left_on="conversationId", right_index=True)
pc.to_csv(os.path.join(OUTPUT_DIR, f"rq2_pattern_counts_per_conversation{SUFFIX}.csv"), index=False)
if cohort_pats:
    means = pc[pc["mixed_activity"]].groupby("group")[cohort_pats].mean().T
    means.to_csv(os.path.join(OUTPUT_DIR, f"rq2_pattern_means_by_group{SUFFIX}.csv"))
    print("\nMean pattern occurrences per conversation (mixed-outcome activities):\n", means.round(2).to_string())

# AI move prevalence per turn by group (descriptive)
ai_g = ai[ai["conversationId"].isin(mix.index)].merge(strat[["group"]], left_on="conversationId", right_index=True)
prev = ai_g.groupby("group")[[f"ai_{m}" for m in AI_MOVES]].mean().T
prev.to_csv(os.path.join(OUTPUT_DIR, f"rq2_ai_move_prevalence_by_group{SUFFIX}.csv"))
print("\nShare of AI turns carrying each move (mixed-outcome activities):\n", prev.round(2).to_string())

# within-activity comparison: linear probability model per move with activity fixed effects,
# SEs clustered by conversation (the AI turn is the unit; the prompt is held constant by the FE)
ai_g["achieved"] = (ai_g["group"] == "achieved").astype(int)
ai_g["activity"] = ai_g["discussionId"].astype(str)
ai_g["log_len"] = np.log(ai_g.groupby("conversationId")["turn"].transform("size"))
wa_rows = []
for m in AI_MOVES:
    y = f"ai_{m}"
    if ai_g[y].sum() < 20:
        continue
    fit = smf.ols(f"{y} ~ achieved + log_len + C(activity)", ai_g).fit(
        cov_type="cluster", cov_kwds={"groups": ai_g["conversationId"]})
    wa_rows.append({"move": m, "share_achieved": prev.loc[y, "achieved"], "share_not_achieved": prev.loc[y, "not_achieved"],
                    "within_activity_diff": fit.params["achieved"], "se": fit.bse["achieved"],
                    "ci_low": fit.conf_int().loc["achieved", 0], "ci_high": fit.conf_int().loc["achieved", 1],
                    "p": fit.pvalues["achieved"]})
wa = pd.DataFrame(wa_rows)
wa.to_csv(os.path.join(OUTPUT_DIR, f"rq2_ai_move_within_activity{SUFFIX}.csv"), index=False)
print("\nWithin-activity difference (achieved minus not achieved) in the share of AI turns carrying each move, "
      "activity fixed effects, log length covariate, SE clustered by conversation:\n", wa.round(3).to_string(index=False))
n_ai = ai_g.groupby(["group", "conversationId"]).size().groupby("group").median()
print("\nMedian AI turns per conversation by group (mixed-outcome activities):", n_ai.to_dict())
print("\nDone. Outputs written to", OUTPUT_DIR)
