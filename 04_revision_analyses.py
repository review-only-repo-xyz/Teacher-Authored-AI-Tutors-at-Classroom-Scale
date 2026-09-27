"""
REPLICATION VERSION.

This script is shared for replication and review. It runs on the
de-identified coded datasets in data/ (conversations.csv, turn_codes.csv,
validation_codes.csv) and on outputs/rq1_activity_features.csv written by
02_prompt_patterns.py. The data contain the study's real codes but no
conversation text, no prompt text, no names, and no platform identifiers.
See README.md.

Revision analyses. Each section writes outputs/rev_<section>_*.csv and a
log at outputs/rev_run_log.txt. The primary specification matches
03_turn_level_analysis.py (World Language excluded, turn-level attainment,
DOK 3 stratum).

  A  Inter-rater reliability on the 136-conversation validation sample.
     Second human coder (HC2) against the first (HC1), and each human
     against the LLM turn coder.
  B  RQ2 timing and persistence re-run descriptively on the human-coded
     DOK 3 conversations of the validation sample, under HC1, HC2 and the LLM.
  C  Hazard model and within-activity LPM with and without log conversation
     length; fixed-horizon check; discipline split and conversation-level label.
  D  Conversations per student; student-clustered versions of M1, M2 and the
     hazard model; one-conversation-per-student check over random draws.
     Frequentist GLMMs with student random intercepts are in 04b_glmm_student_re.R.
  E  Lagged-state hazard: move effects conditional on the prior student
     turn's DOK, move-by-state interaction, and a transition model.
  F  Persistence: first drop below target classified as task-driven,
     loss of rigor, or disengagement, by persistence class and position.
  G  Collinearity between prompt pattern, target DOK and teacher.
  H  Attainment timing indexed from the first typed student turn.

Risk set. Student turn 1 in every conversation is the platform's start
button ("Start Classroom Discussion"). It has no preceding AI turn and
cannot be at target, so the corrected hazard models (turns 2+) exclude it;
the specifications that include it are also reported for comparison.

Usage
    python 04_revision_analyses.py                    # all sections
    python 04_revision_analyses.py --sections A B     # selected sections
    Rscript 04b_glmm_student_re.R                     # after section D

Outputs (outputs/rev_*.csv, outputs/figures/fig_rev_*.png)
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.metrics import cohen_kappa_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, OUTPUT_DIR  # noqa: E402

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(2026)

ap = argparse.ArgumentParser()
ap.add_argument("--sections", nargs="+", default=list("ABCDEFGH"))
ap.add_argument("--turn-csv", default=os.path.join(DATA_DIR, "turn_codes.csv"))
ap.add_argument("--pattern-file", default=os.path.join(OUTPUT_DIR, "rq1_activity_features.csv"))
ap.add_argument("--target", type=int, default=3)
args = ap.parse_args()
SECTIONS = set(s.upper() for s in args.sections)
TARGET = args.target

DATA_FILE = os.path.join(DATA_DIR, "conversations.csv")
VALIDATION_FILE = os.path.join(DATA_DIR, "validation_codes.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)
AI_MOVES = ["socratic_question", "stepwise_guidance", "hint", "request_attempt", "request_evidence",
            "affirm_extend", "final_answer", "redirect", "recap", "closure", "other"]
AI_PRIORITY = ["final_answer", "redirect", "request_evidence", "request_attempt", "stepwise_guidance",
               "hint", "socratic_question", "affirm_extend", "recap", "closure", "other"]
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    LOG.append(s)


def out(name):
    return os.path.join(OUTPUT_DIR, name)


def gwet_ac1(a, b):
    """Gwet's AC1 for nominal codes (robust to skewed prevalence)."""
    a, b = np.asarray(a), np.asarray(b)
    cats = np.union1d(a, b)
    q = len(cats)
    if q < 2:
        return np.nan
    pa = (a == b).mean()
    pi = np.array([((a == c).mean() + (b == c).mean()) / 2 for c in cats])
    pe = (pi * (1 - pi)).sum() / (q - 1)
    return (pa - pe) / (1 - pe)


def agree_stats(a, b, ordinal=False):
    a, b = np.asarray(a), np.asarray(b)
    r = {"n": len(a), "pct_agree": (a == b).mean(), "kappa": cohen_kappa_score(a, b), "ac1": gwet_ac1(a, b)}
    if ordinal:
        r["qwk"] = cohen_kappa_score(a, b, weights="quadratic")
        r["pct_within_one"] = (np.abs(a - b) <= 1).mean()
        r["n_disagree"] = int((a != b).sum())
        r["first_higher"] = int((a > b).sum())
    return r


# ==========================================================================
# Shared preparation (mirrors 11_turn_level_analysis.py, primary specification)
# ==========================================================================
def prepare():
    turns = pd.read_csv(args.turn_csv)
    conv = pd.read_csv(DATA_FILE)
    conv = conv[conv["recoded_dok_alignment"].notna()].copy()
    conv = conv[conv["subject"] != "World Language"].copy()
    conv["target"] = conv["prompt_target_dok"].astype(int)
    conv["achieved_conv"] = conv["recoded_dok_alignment"].isin(["aligned", "over_target"]).astype(int)
    pat = pd.read_csv(args.pattern_file, index_col=0)[["pattern"]]
    conv = conv.merge(pat, left_on="discussionId", right_index=True, how="left")
    meta = conv.set_index("conversationId")[["discussionId", "teacher_name", "subject", "target", "achieved_conv",
                                             "pattern", "user_id"]]
    turns = turns[turns["conversationId"].isin(meta.index)].copy()
    turns = turns.drop(columns=[c for c in ["discussionId", "user_id", "prompt_target_dok"] if c in turns.columns])
    turns = turns.merge(meta, left_on="conversationId", right_index=True, how="left")
    turns = turns.sort_values(["conversationId", "turn"]).reset_index(drop=True)

    st = turns[turns["role"] == "student"].copy()
    st["k"] = st.groupby("conversationId").cumcount() + 1
    st["dok"] = st["student_dok"].fillna(0).astype(int)
    st["at_target"] = (st["dok"] >= st["target"]).astype(int)
    ai = turns[turns["role"] == "ai"].copy()
    for m in AI_MOVES:
        ai[f"ai_{m}"] = ai[f"ai_{m}"].fillna(0).astype(int)
    ai["primary_move"] = ai.apply(lambda r: next((m for m in AI_PRIORITY if r[f"ai_{m}"] == 1), "other"), axis=1)

    # preceding AI turn for each student turn, and the student turn preceding it
    turns_sorted = turns[["conversationId", "turn", "role"]]
    prev_ai = {}
    for cid, g in turns_sorted.groupby("conversationId"):
        last = None
        for t, r in zip(g["turn"], g["role"]):
            if r == "ai":
                last = t
            elif r == "student":
                prev_ai[(cid, t)] = last
    st["prev_ai_turn"] = [prev_ai.get((c, t)) for c, t in zip(st["conversationId"], st["turn"])]
    ai_idx = ai.set_index(["conversationId", "turn"])
    for m in AI_MOVES:
        col = ai_idx[f"ai_{m}"]
        st[f"prev_{m}"] = [col.get((c, t), 0) if pd.notna(t) else 0
                           for c, t in zip(st["conversationId"], st["prev_ai_turn"])]
    st["prev_dok"] = st.groupby("conversationId")["dok"].shift(1)       # DOK of the prior student turn
    st["prev_dok_state"] = st["prev_dok"].map(lambda d: "none" if pd.isna(d) else f"D{int(min(d, 2))}")

    cs = st.groupby("conversationId").agg(
        n_student_turns=("k", "max"), max_dok_turn=("dok", "max"), target=("target", "first"),
        achieved_conv=("achieved_conv", "first"), discussionId=("discussionId", "first"),
        teacher=("teacher_name", "first"), pattern=("pattern", "first"), subject=("subject", "first"),
        user_id=("user_id", "first"))
    cs["achieved"] = (cs["max_dok_turn"] >= cs["target"]).astype(int)
    cs = cs.join(st[st["at_target"] == 1].groupby("conversationId")["k"].min().rename("k_first"))
    cs["group"] = np.where(cs["achieved"] == 1, "achieved", "not_achieved")
    return conv, turns, st, ai, cs


def person_period(st, cs, strat):
    pp = st[st["conversationId"].isin(strat.index)].merge(cs[["k_first", "n_student_turns"]],
                                                          left_on="conversationId", right_index=True)
    pp = pp[(pp["k_first"].isna()) | (pp["k"] <= pp["k_first"])].copy()
    pp["event"] = (pp["k"] == pp["k_first"]).astype(int)
    pp["log_k"] = np.log(pp["k"])
    pp["log_len"] = np.log(pp["n_student_turns"])
    pp["activity"] = pp["discussionId"].astype(str)
    return pp


def hazard_terms(pp):
    return [f"prev_{m}" for m in AI_MOVES if m != "other" and pp.loc[pp["event"] == 1, f"prev_{m}"].sum() >= 10]


def tidy_logit(res, label, keep_prefix=("prev_", "log_", "C(pattern)", "C(prev_dok_state)", "prev_dok")):
    ci = res.conf_int()
    rows = []
    for t in res.params.index:
        if not t.startswith(keep_prefix) and t != "Intercept":
            continue
        rows.append({"model": label, "term": t, "log_odds": res.params[t], "se": res.bse[t],
                     "odds_ratio": np.exp(res.params[t]), "ci_low_or": np.exp(ci.loc[t, 0]),
                     "ci_high_or": np.exp(ci.loc[t, 1]), "p": res.pvalues[t], "n_obs": int(res.nobs)})
    return rows


def cluster2_cov(model, res, g1, g2):
    """Two-way cluster-robust covariance (Cameron, Gelbach and Miller 2011) for a fitted logit."""
    def meat(groups):
        return res.model.fit(disp=0, cov_type="cluster", cov_kwds={"groups": groups}).cov_params()
    inter = pd.factorize(pd.Series(g1).astype(str) + "|" + pd.Series(g2).astype(str))[0]
    v = meat(pd.factorize(g1)[0]) + meat(pd.factorize(g2)[0]) - meat(inter)
    return v


NEED_MAIN = SECTIONS & set("BCDEFGH")
if NEED_MAIN:
    conv, turns, st, ai, cs = prepare()
    strat = cs[cs["target"] == TARGET].copy()
    mixed = strat.groupby("discussionId")["achieved"].agg(["min", "max"])
    mixed_acts = mixed[(mixed["min"] == 0) & (mixed["max"] == 1)].index
    strat["mixed_activity"] = strat["discussionId"].isin(mixed_acts)
    log(f"Primary sample: {len(cs)} conversations with student turns; DOK {TARGET} stratum {len(strat)} "
        f"({strat['achieved'].sum()} achieved); mixed-outcome activities {len(mixed_acts)} "
        f"({strat['mixed_activity'].sum()} conversations)")


# ==========================================================================
# A. Inter-rater reliability
# ==========================================================================
def load_validation():
    """data/validation_codes.csv: one row per turn of the 136 validation conversations, with the
    codes of both human coders (hc1, hc2) and the LLM turn coder (llm). Conversation identifiers
    are hashed separately from the other data files, so the sample cannot be joined to them."""
    v = pd.read_csv(VALIDATION_FILE)
    v["primary_sample"] = v["subject"] != "World Language"

    def parse(x):
        if pd.isna(x) or str(x).strip() == "":
            return set()
        return {t.strip() for t in str(x).split(";") if t.strip()}
    for c in ["hc1", "hc2", "llm"]:
        v[f"mset_{c}"] = v[f"moves_{c}"].apply(parse)
    return v, None


if "A" in SECTIONS:
    log("\n" + "=" * 78 + "\nA. Inter-rater reliability\n" + "=" * 78)
    v, samp = load_validation()
    pairs = [("hc1", "hc2", "HC1 vs HC2"), ("hc1", "llm", "HC1 vs LLM"), ("hc2", "llm", "HC2 vs LLM")]
    rows = []
    stu = v[v["role"] == "student"].dropna(subset=["dok_hc1", "dok_hc2", "dok_llm"]).copy()
    for c in ["hc1", "hc2", "llm"]:
        stu[f"dok_{c}"] = stu[f"dok_{c}"].astype(int)
        stu[f"at_{c}"] = (stu[f"dok_{c}"] >= stu["target"]).astype(int)
    for scope, sub in [("all 136 conversations", stu), ("primary sample (World Language excluded)",
                                                           stu[stu["primary_sample"]])]:
        for a, b, lab in pairs:
            r = agree_stats(sub[f"dok_{a}"], sub[f"dok_{b}"], ordinal=True)
            rows.append({"scope": scope, "unit": "student turn", "measure": "turn DOK (0-4)", "pair": lab, **r})
            r = agree_stats(sub[f"at_{a}"], sub[f"at_{b}"])
            rows.append({"scope": scope, "unit": "student turn", "measure": "at or above target", "pair": lab, **r})
            # DOK 3 or higher, regardless of target (the strategic-thinking indicator)
            r = agree_stats((sub[f"dok_{a}"] >= 3).astype(int), (sub[f"dok_{b}"] >= 3).astype(int))
            rows.append({"scope": scope, "unit": "student turn", "measure": "DOK 3 or higher", "pair": lab, **r})
    # conversation-level DOK reached and goal achieved
    cv = stu.groupby("conversationId").agg(target=("target", "first"), subject=("subject", "first"),
                                           primary=("primary_sample", "first"),
                                           **{f"max_{c}": (f"dok_{c}", "max") for c in ["hc1", "hc2", "llm"]})
    for c in ["hc1", "hc2", "llm"]:
        cv[f"ach_{c}"] = (cv[f"max_{c}"] >= cv["target"]).astype(int)
    for scope, sub in [("all 136 conversations", cv), ("primary sample (World Language excluded)", cv[cv["primary"]]),
                       ("DOK 3 target conversations", cv[cv["target"] == 3])]:
        for a, b, lab in pairs:
            r = agree_stats(sub[f"max_{a}"], sub[f"max_{b}"], ordinal=True)
            rows.append({"scope": scope, "unit": "conversation", "measure": "DOK reached (max)", "pair": lab, **r})
            r = agree_stats(sub[f"ach_{a}"], sub[f"ach_{b}"])
            rows.append({"scope": scope, "unit": "conversation", "measure": "goal achieved", "pair": lab, **r})
    # AI moves (binary per move)
    aiv = v[v["role"] == "ai"].copy()
    aiv = aiv[aiv["moves_hc1"].notna() & aiv["moves_hc2"].notna() & aiv["moves_llm"].notna()]
    for m in AI_MOVES:
        for c in ["hc1", "hc2", "llm"]:
            aiv[f"{m}_{c}"] = aiv[f"mset_{c}"].apply(lambda s: int(m in s))
    for scope, sub in [("all 136 conversations", aiv), ("primary sample (World Language excluded)",
                                                        aiv[aiv["primary_sample"]])]:
        for m in AI_MOVES:
            for a, b, lab in pairs:
                r = agree_stats(sub[f"{m}_{a}"], sub[f"{m}_{b}"])
                r.update({f"prev_{a}": sub[f"{m}_{a}"].mean(), f"prev_{b}": sub[f"{m}_{b}"].mean()})
                rows.append({"scope": scope, "unit": "AI turn", "measure": m, "pair": lab, **r})
    irr = pd.DataFrame(rows)
    irr.to_csv(out("rev_A_interrater_reliability.csv"), index=False)

    main = irr[(irr["scope"] == "all 136 conversations")]
    log(f"Validation sample: {stu['conversationId'].nunique()} conversations, {len(stu)} student turns, "
        f"{len(aiv)} AI turns coded by HC1, HC2 and the LLM")
    log(main[main["unit"] != "AI turn"][["unit", "measure", "pair", "n", "pct_agree", "kappa", "ac1", "qwk",
                                          "pct_within_one", "n_disagree", "first_higher"]].round(3).to_string(index=False))
    mv = main[main["unit"] == "AI turn"].pivot(index="measure", columns="pair", values="kappa").loc[AI_MOVES]
    log("\nAI-move kappa by pair (all 136 conversations):\n" + mv.round(2).to_string())
    log("Range of HC1-HC2 kappa across moves: "
        f"{mv['HC1 vs HC2'].min():.2f} to {mv['HC1 vs HC2'].max():.2f}; median {mv['HC1 vs HC2'].median():.2f}")

    # LLM agreement on turns where the two humans agree (a consensus benchmark)
    cons = stu[stu["dok_hc1"] == stu["dok_hc2"]]
    r = agree_stats(cons["dok_llm"], cons["dok_hc1"], ordinal=True)
    log(f"\nStudent turns where HC1 = HC2: {len(cons)} of {len(stu)} ({len(cons)/len(stu):.1%}); "
        f"LLM agreement with the consensus code {r['pct_agree']:.1%}, kappa {r['kappa']:.2f}, QWK {r['qwk']:.2f}")
    pd.DataFrame([{"subset": "turns where HC1 = HC2", **r}]).to_csv(out("rev_A_llm_vs_human_consensus.csv"), index=False)

    # direction of disagreement between humans, and the DOK 2/3 boundary
    d = stu[stu["dok_hc1"] != stu["dok_hc2"]]
    log(f"HC1-HC2 student-DOK disagreements: {len(d)}; HC1 higher in {(d['dok_hc1'] > d['dok_hc2']).sum()}; "
        f"adjacent {(abs(d['dok_hc1'] - d['dok_hc2']) == 1).mean():.1%}")
    ct = pd.crosstab(stu["dok_hc1"], stu["dok_hc2"], rownames=["HC1"], colnames=["HC2"])
    ct.to_csv(out("rev_A_student_dok_crosstab_hc1_hc2.csv"))
    log("HC1 x HC2 student DOK:\n" + ct.to_string())
    b23 = stu[stu[["dok_hc1", "dok_hc2"]].isin([2, 3]).any(axis=1) & stu[["dok_hc1", "dok_hc2"]].isin([2, 3, 4]).all(axis=1)]
    log(f"DOK 2 vs DOK 3+ boundary (turns both humans coded 2 or higher, n = {len(b23)}): "
        f"kappa {cohen_kappa_score(b23['dok_hc1'] >= 3, b23['dok_hc2'] >= 3):.2f}")

    # achievement rates in the DOK 3 stratum under each coder
    d3 = cv[cv["target"] == 3]
    rates = {c: d3[f"ach_{c}"].mean() for c in ["hc1", "hc2", "llm"]}
    rates_p = {c: d3[d3["primary"]][f"ach_{c}"].mean() for c in ["hc1", "hc2", "llm"]}
    log(f"DOK 3 conversations in the validation sample: {len(d3)} (primary sample {int(d3['primary'].sum())}). "
        f"Achievement: HC1 {rates['hc1']:.1%}, HC2 {rates['hc2']:.1%}, LLM {rates['llm']:.1%} | primary only: "
        f"HC1 {rates_p['hc1']:.1%}, HC2 {rates_p['hc2']:.1%}, LLM {rates_p['llm']:.1%}")
    pd.DataFrame([{"scope": "DOK 3, all", "n": len(d3), **{f"ach_{k}": x for k, x in rates.items()}},
                  {"scope": "DOK 3, primary", "n": int(d3["primary"].sum()), **{f"ach_{k}": x for k, x in rates_p.items()}}]
                 ).to_csv(out("rev_A_dok3_achievement_by_coder.csv"), index=False)
    VALID = (v, stu, cv)


# ==========================================================================
# B. RQ2 timing and persistence on the human-coded DOK 3 conversations
# ==========================================================================
def classify_persistence(at):
    """at: array of at-target indicators over student turns. Returns (k_first, class, n_post, share_post, ends)."""
    at = np.asarray(at)
    hits = np.flatnonzero(at == 1)
    if len(hits) == 0:
        return np.nan, "not_achieved", np.nan, np.nan, 0
    k0 = hits[0]
    post = at[k0 + 1:]
    if len(post) == 0:
        cls = "final_turn"
    elif post.min() == 1:
        cls = "sustained"
    elif post.sum() > 0:
        cls = "intermittent"
    else:
        cls = "regressed"
    return k0 + 1, cls, len(post), (post.mean() if len(post) else np.nan), int(at[-1] == 1)


def timing_persistence_summary(seqs, label):
    """seqs: dict cid -> array of at-target indicators. Returns one summary row."""
    recs = []
    for cid, at in seqs.items():
        k, cls, npost, sh, ends = classify_persistence(at)
        recs.append({"conversationId": cid, "n_turns": len(at), "k_first": k, "class": cls,
                     "n_post": npost, "share_post": sh, "ends_at_target": ends})
    d = pd.DataFrame(recs)
    a = d[d["class"] != "not_achieved"]
    n = len(d)
    row = {"coder": label, "n_conversations": n, "n_achieved": len(a), "achievement_rate": len(a) / n,
           "median_first_turn": a["k_first"].median(), "q1_first_turn": a["k_first"].quantile(.25),
           "q3_first_turn": a["k_first"].quantile(.75),
           "median_share_first": (a["k_first"] / a["n_turns"]).median(),
           "first_turn_share": (a["k_first"] == 1).mean()}
    for kk in [2, 3, 5, 10]:
        row[f"cum_by_turn_{kk}"] = (d["k_first"] <= kk).sum() / n
    # student turn 1 is the platform's "Start Classroom Discussion" button; index from the first typed turn
    row["median_first_typed_turn"] = (a["k_first"] - 1).median()
    row["q1_first_typed_turn"] = (a["k_first"] - 1).quantile(.25)
    row["q3_first_typed_turn"] = (a["k_first"] - 1).quantile(.75)
    for kk in [1, 2, 4, 9]:
        row[f"cum_by_typed_turn_{kk}"] = (d["k_first"] - 1 <= kk).sum() / n
    row["share_of_final_reached_by_turn_5"] = (a["k_first"] <= 5).mean()
    for c in ["sustained", "intermittent", "regressed", "final_turn"]:
        row[f"pers_{c}"] = (a["class"] == c).mean()
    row["ends_at_target"] = a["ends_at_target"].mean()
    row["median_post_turns"] = a["n_post"].median()
    row["median_share_post_at_target"] = a["share_post"].median()
    return row, d


if "B" in SECTIONS:
    log("\n" + "=" * 78 + "\nB. RQ2 timing and persistence on the human-coded DOK 3 conversations\n" + "=" * 78)
    if "A" not in SECTIONS:
        v, samp = load_validation()
    vs = v[(v["role"] == "student") & (v["target"] == 3)].sort_values(["conversationId", "turn"]).copy()
    rows, per_conv = [], {}
    for scope, sub in [("all DOK 3 validation conversations", vs),
                       ("primary sample only (World Language excluded)", vs[vs["primary_sample"]])]:
        for c, lab in [("hc1", "HC1"), ("hc2", "HC2"), ("llm", "LLM (Claude)")]:
            seqs = {cid: (g[f"dok_{c}"].fillna(0).astype(int) >= 3).astype(int).values
                    for cid, g in sub.groupby("conversationId")}
            r, d = timing_persistence_summary(seqs, lab)
            rows.append({"scope": scope, **r})
            if scope.startswith("all"):
                per_conv[c] = d.set_index("conversationId")
    # full-sample reference (LLM codes, 741 conversations)
    if NEED_MAIN:
        seqs = {cid: g.sort_values("k")["at_target"].values for cid, g in st[st["conversationId"].isin(strat.index)].groupby("conversationId")}
        r, _ = timing_persistence_summary(seqs, "LLM (Claude), full DOK 3 stratum")
        rows.append({"scope": "full primary DOK 3 stratum", **r})
    tp = pd.DataFrame(rows)
    tp.to_csv(out("rev_B_timing_persistence_human_coded.csv"), index=False)
    show = ["scope", "coder", "n_conversations", "n_achieved", "achievement_rate", "median_first_turn",
            "q1_first_turn", "q3_first_turn", "first_turn_share", "cum_by_turn_2", "cum_by_turn_3", "cum_by_turn_5",
            "cum_by_turn_10", "share_of_final_reached_by_turn_5", "pers_sustained", "pers_intermittent",
            "pers_regressed", "pers_final_turn", "ends_at_target", "median_post_turns", "median_share_post_at_target"]
    log(tp[show].round(3).T.to_string())
    # agreement between coders on the first-attainment turn (conversations achieved under both codings)
    for a_, b_ in [("hc1", "hc2"), ("hc1", "llm"), ("hc2", "llm")]:
        j = per_conv[a_][["k_first", "class"]].join(per_conv[b_][["k_first", "class"]], lsuffix="_a", rsuffix="_b")
        both = j.dropna(subset=["k_first_a", "k_first_b"])
        log(f"{a_.upper()} vs {b_.upper()}: achieved under both {len(both)}; identical first-attainment turn "
            f"{(both['k_first_a'] == both['k_first_b']).mean():.1%}; median |difference| "
            f"{(both['k_first_a'] - both['k_first_b']).abs().median():.1f}; identical persistence class "
            f"{(both['class_a'] == both['class_b']).mean():.1%}")
    pd.concat({k: d for k, d in per_conv.items()}, axis=1).to_csv(out("rev_B_per_conversation_by_coder.csv"))


# ==========================================================================
# C. Hazard and within-activity LPM with and without log conversation length
# ==========================================================================
def fit_hazard(pp, rhs, label, groups=None, fe=False):
    data = pp.copy()
    f = f"event ~ {rhs}" + (" + C(activity)" if fe else "")
    g = data["activity"] if groups is None else groups
    res = smf.logit(f, data).fit(disp=0, maxiter=300, cov_type="cluster", cov_kwds={"groups": pd.factorize(g)[0]})
    return res, tidy_logit(res, label)


if NEED_MAIN:
    PP = person_period(st, cs, strat)
    PP = PP[PP["pattern"].isin(strat["pattern"].value_counts()[lambda s: s >= 20].index)]
    HT = hazard_terms(PP)
    PPM = PP[PP["discussionId"].isin(mixed_acts)].copy()
    MOVES_RHS = " + ".join(HT)
    # Student turn 1 is the platform's "Start Classroom Discussion" button in every conversation: it has no
    # preceding AI turn (all move indicators 0) and cannot be at target. The corrected risk set starts at turn 2.
    PP2 = PP[PP["k"] >= 2].copy()
    PPM2 = PPM[PPM["k"] >= 2].copy()

if "C" in SECTIONS:
    log("\n" + "=" * 78 + "\nC. Hazard and within-activity LPM with and without log length\n" + "=" * 78)
    log(f"Person-periods: {len(PP)} (events {int(PP['event'].sum())}); FE subset {len(PPM)}. Move terms: {HT}")
    rows = []
    specs = [
        ("H1 pattern, with log length (original specification)", PP, f"log_k + log_len + C(pattern) + {MOVES_RHS}", False),
        ("H2 pattern, without log length", PP, f"log_k + C(pattern) + {MOVES_RHS}", False),
        ("H3 activity FE, with log length (original specification)", PPM, f"log_k + log_len + {MOVES_RHS}", True),
        ("H4 activity FE, without log length", PPM, f"log_k + {MOVES_RHS}", True),
        ("H1c pattern, with log length, turns 2+", PP2, f"log_k + log_len + C(pattern) + {MOVES_RHS}", False),
        ("H2c pattern, without log length, turns 2+", PP2, f"log_k + C(pattern) + {MOVES_RHS}", False),
        ("H3c activity FE, with log length, turns 2+", PPM2, f"log_k + log_len + {MOVES_RHS}", True),
        ("H4c activity FE, without log length, turns 2+", PPM2, f"log_k + {MOVES_RHS}", True),
    ]
    for lab, d, rhs, fe in specs:
        _, r = fit_hazard(d, rhs, lab, fe=fe)
        rows += r
    # fixed-horizon check: conversations with at least H student turns, risk set limited to turns 1..H,
    # so every conversation contributes the same exposure and length cannot act through the risk set
    for H in [6, 10]:
        long_ids = strat[strat["n_student_turns"] >= H].index
        d = PPM2[PPM2["conversationId"].isin(long_ids) & (PPM2["k"] <= H)]
        try:
            _, r = fit_hazard(d, f"log_k + {MOVES_RHS}", f"H5c activity FE, turns 2..{H}, "
                              f"conversations with >= {H} student turns (n = {d['conversationId'].nunique()})", fe=True)
            rows += r
        except Exception as e:
            log(f"fixed-horizon {H} failed: {e}")
    # discipline split and conversation-level sensitivity label, original and corrected risk sets
    disc = np.where(strat["subject"] == "ELA", "ELA", "STEM/CTE")
    disc = pd.Series(disc, index=strat.index)
    for dname in ["ELA", "STEM/CTE"]:
        ids = disc[disc == dname].index
        for lab, d in [("all turns", PPM), ("turns 2+", PPM2)]:
            dd = d[d["conversationId"].isin(ids)]
            try:
                _, r = fit_hazard(dd, f"log_k + log_len + {MOVES_RHS}", f"H6 {dname}, activity FE, {lab}", fe=True)
                rows += r
            except Exception as e:
                log(f"H6 {dname} {lab} failed: {e}")
    # conversation-level label defines achievement; turn codes locate the first at-target turn
    sc = strat.copy()
    sc["k_first_conv"] = np.where(sc["achieved_conv"] == 1, sc["k_first"], np.nan)
    mxc = sc.groupby("discussionId")["achieved_conv"].agg(["min", "max"])
    macts_c = mxc[(mxc["min"] == 0) & (mxc["max"] == 1)].index
    ppc = st[st["conversationId"].isin(sc.index)].merge(sc[["k_first_conv", "n_student_turns"]],
                                                        left_on="conversationId", right_index=True)
    ppc = ppc[(ppc["k_first_conv"].isna()) | (ppc["k"] <= ppc["k_first_conv"])].copy()
    ppc["event"] = (ppc["k"] == ppc["k_first_conv"]).astype(int)
    ppc["log_k"], ppc["log_len"] = np.log(ppc["k"]), np.log(ppc["n_student_turns"])
    ppc["activity"] = ppc["discussionId"].astype(str)
    ppc = ppc[ppc["discussionId"].isin(macts_c)]
    for lab, d in [("all turns", ppc), ("turns 2+", ppc[ppc["k"] >= 2])]:
        _, r = fit_hazard(d, f"log_k + log_len + {MOVES_RHS}", f"H7 conversation-level label, activity FE, {lab}", fe=True)
        rows += r
    hz = pd.DataFrame(rows)
    hz.to_csv(out("rev_C_hazard_log_length.csv"), index=False)
    log(hz[hz["model"].str.startswith(("H6", "H7")) & (hz["term"] == "prev_socratic_question")]
        [["model", "odds_ratio", "ci_low_or", "ci_high_or", "p"]].round(2).to_string(index=False))
    piv = hz[hz["term"].str.startswith(("prev_", "log_"))].pivot(index="term", columns="model", values="odds_ratio")
    log("Odds ratios by specification:\n" + piv.round(2).to_string())
    ci = hz[hz["term"] == "prev_socratic_question"][["model", "odds_ratio", "ci_low_or", "ci_high_or", "n_obs"]]
    log("Socratic question:\n" + ci.round(2).to_string(index=False))

    # why length matters: length is the number of turns the conversation lasted, which is fixed only at its end;
    # report its association with the outcome and with move prevalence
    ln = strat.assign(log_len=np.log(strat["n_student_turns"]))
    log(f"Median student turns: achieved {ln[ln['achieved']==1]['n_student_turns'].median():.0f}, "
        f"not achieved {ln[ln['achieved']==0]['n_student_turns'].median():.0f}")

    # within-activity LPM, AI-move prevalence per AI turn, with and without log length
    mix = strat[strat["mixed_activity"]]
    ai_g = ai[ai["conversationId"].isin(mix.index)].merge(mix[["group"]], left_on="conversationId", right_index=True)
    ai_g["achieved"] = (ai_g["group"] == "achieved").astype(int)
    ai_g["activity"] = ai_g["discussionId"].astype(str)
    ai_g["log_len"] = np.log(ai_g.groupby("conversationId")["turn"].transform("size"))
    wrows = []
    for m in AI_MOVES:
        y = f"ai_{m}"
        if ai_g[y].sum() < 20:
            continue
        for lab, f in [("with log length (original specification)", f"{y} ~ achieved + log_len + C(activity)"),
                       ("without log length", f"{y} ~ achieved + C(activity)")]:
            fit = smf.ols(f, ai_g).fit(cov_type="cluster", cov_kwds={"groups": ai_g["conversationId"]})
            ci_ = fit.conf_int().loc["achieved"]
            wrows.append({"move": m, "spec": lab, "diff_pp": 100 * fit.params["achieved"],
                          "ci_low_pp": 100 * ci_[0], "ci_high_pp": 100 * ci_[1], "p": fit.pvalues["achieved"],
                          "log_len_coef_pp": 100 * fit.params.get("log_len", np.nan)})
    wa = pd.DataFrame(wrows)
    wa.to_csv(out("rev_C_within_activity_lpm_log_length.csv"), index=False)
    log("\nWithin-activity LPM (pp), with vs without log length:\n" +
        wa.pivot(index="move", columns="spec", values="diff_pp").round(1).join(
            wa[wa["spec"] == "without log length"].set_index("move")[["ci_low_pp", "ci_high_pp", "p"]].round(3)).to_string())
    log(f"Achieved vs not-achieved AI turns per conversation (median): "
        f"{ai_g.groupby(['group','conversationId']).size().groupby('group').median().to_dict()}")


# ==========================================================================
# D. Student dependence
# ==========================================================================
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM  # noqa: E402


def rq1_frame(cs_):
    d = cs_.copy().reset_index()
    d["dok_reached"] = d["max_dok_turn"]
    d["target_c"] = d["target"] - 3
    d["activity"] = d["discussionId"].astype(str)
    d["student"] = d["user_id"].astype(str)
    d["teacher"] = d["teacher"].astype(str)
    return d


def fit_m1(d, student_re=False):
    vc = {"activity": "0 + C(activity)"}
    if student_re:
        vc["student"] = "0 + C(student)"
    return smf.mixedlm("dok_reached ~ C(pattern) + target_c", d, groups=d["teacher"], re_formula="1",
                       vc_formula=vc).fit(reml=True)


def fit_m2(d, groups):
    d23 = d[d["target"] >= 2]
    g = groups(d23)
    return smf.logit("achieved ~ C(pattern) + target_c", d23).fit(disp=0, cov_type="cluster",
                                                                 cov_kwds={"groups": g})


def m2_twoway(d):
    d23 = d[d["target"] >= 2].reset_index(drop=True)
    res = smf.logit("achieved ~ C(pattern) + target_c", d23).fit(disp=0)
    V = cluster2_cov(None, res, d23["activity"].values, d23["student"].values)
    return res, V


def rows_from(res, label, terms, cov=None, exp=True):
    se = np.sqrt(np.diag(cov)) if cov is not None else res.bse.values
    se = pd.Series(se, index=res.params.index)
    out_ = []
    for t in terms:
        if t not in res.params.index:
            continue
        b, s = res.params[t], se[t]
        lo, hi = b - 1.96 * s, b + 1.96 * s
        out_.append({"model": label, "term": t, "estimate": b, "se": s,
                     "effect": np.exp(b) if exp else b, "ci_low": np.exp(lo) if exp else lo,
                     "ci_high": np.exp(hi) if exp else hi, "p": 2 * (1 - stats.norm.cdf(abs(b / s)))})
    return out_


def pattern_wald(res, cov=None):
    terms = [t for t in res.params.index if t.startswith("C(pattern)")]
    idx = [list(res.params.index).index(t) for t in terms]
    b = res.params.values[idx]
    V = (cov if cov is not None else res.cov_params().values)
    V = np.asarray(V)[np.ix_(idx, idx)]
    w = float(b @ np.linalg.solve(V, b))
    return w, 1 - stats.chi2.cdf(w, len(idx))


if "D" in SECTIONS:
    log("\n" + "=" * 78 + "\nD. Student dependence\n" + "=" * 78)
    D = rq1_frame(cs)
    cps = D.groupby("student").size()
    dist = cps.value_counts().sort_index()
    cps3 = D[D["target"] == 3].groupby("student").size()
    dist_tab = pd.DataFrame({"conversations_per_student": dist.index, "n_students_primary": dist.values})
    dist_tab = dist_tab.merge(pd.DataFrame({"conversations_per_student": cps3.value_counts().sort_index().index,
                                            "n_students_dok3": cps3.value_counts().sort_index().values}), how="outer")
    dist_tab.to_csv(out("rev_D_conversations_per_student.csv"), index=False)
    log(f"Primary sample: {len(D)} conversations from {D['student'].nunique()} students; conversations per student "
        f"mean {cps.mean():.2f}, median {cps.median():.0f}, max {cps.max()}; distribution {dist.to_dict()}; "
        f"students with one conversation {(cps == 1).mean():.1%}; conversations from students with 2+ "
        f"{(D['student'].map(cps) > 1).mean():.1%}")
    log(f"DOK 3 stratum: {int(cps3.sum())} conversations from {len(cps3)} students; distribution {cps3.value_counts().sort_index().to_dict()}")
    log(f"Students with conversations under more than one teacher: {(D.groupby('student')['teacher'].nunique() > 1).sum()}; "
        f"student-activity pairs with more than one conversation: {(D.groupby(['student', 'activity']).size() > 1).sum()}")

    rows = []
    PT = ["C(pattern)[T.P2]", "C(pattern)[T.P3]", "target_c"]
    # M1
    m1a = fit_m1(D, student_re=False)
    m1b = fit_m1(D, student_re=True)
    rows += rows_from(m1a, "M1 LMM, teacher + activity RI (original specification)", PT, exp=False)
    rows += rows_from(m1b, "M1 LMM, teacher + activity + student RI", PT, exp=False)
    vc_names = m1b.model.exog_vc.names if hasattr(m1b.model, "exog_vc") else []
    log(f"M1 variance components with student RI: teacher {m1b.cov_re.iloc[0,0]:.3f}; "
        + "; ".join(f"{n} {v:.3f}" for n, v in zip(vc_names, m1b.vcomp)))
    # M2
    m2a = fit_m2(D, lambda d: pd.factorize(d["activity"])[0])
    m2b = fit_m2(D, lambda d: pd.factorize(d["student"])[0])
    m2c, V2 = m2_twoway(D)
    rows += rows_from(m2a, "M2 logit, SE clustered by activity (original specification)", PT)
    rows += rows_from(m2b, "M2 logit, SE clustered by student", PT)
    rows += rows_from(m2c, "M2 logit, two-way SE clustered by activity and student", PT, cov=V2)
    for lab, res, cov in [("activity", m2a, None), ("student", m2b, None), ("two-way", m2c, V2)]:
        w, p = pattern_wald(res, cov)
        rows.append({"model": f"M2 Wald test pattern block, {lab} clustering", "term": "pattern", "estimate": w, "p": p})
    # M3 with student random intercept
    d23 = D[D["target"] >= 2]
    try:
        m3 = BinomialBayesMixedGLM.from_formula(
            "achieved ~ C(pattern) + target_c",
            {"teacher": "0 + C(teacher)", "activity": "0 + C(activity)", "student": "0 + C(student)"}, d23).fit_vb()
        for i, t in enumerate(m3.model.exog_names):
            if t in PT:
                b, s = m3.fe_mean[i], m3.fe_sd[i]
                rows.append({"model": "M3 Bayesian mixed GLM, teacher + activity + student RI", "term": t, "estimate": b,
                             "se": s, "effect": np.exp(b), "ci_low": np.exp(b - 1.96 * s), "ci_high": np.exp(b + 1.96 * s),
                             "p": 2 * (1 - stats.norm.cdf(abs(b / s)))})
        log("M3 variance components (posterior mean log SD): " +
            "; ".join(f"{n} {v:.2f}" for n, v in zip(m3.model.vcp_names, m3.vcp_mean)))
    except Exception as e:
        log("M3 with student RI failed:", e)

    # hazard: student and two-way clustering
    HZ = ["prev_socratic_question", "prev_request_evidence", "prev_request_attempt", "prev_stepwise_guidance",
          "prev_hint", "prev_affirm_extend", "log_k", "log_len"]
    for lab, d, rhs, fe in [("pattern", PP, f"log_k + log_len + C(pattern) + {MOVES_RHS}", False),
                            ("activity FE", PPM, f"log_k + log_len + {MOVES_RHS}", True),
                            ("pattern, turns 2+", PP2, f"log_k + log_len + C(pattern) + {MOVES_RHS}", False),
                            ("activity FE, turns 2+", PPM2, f"log_k + log_len + {MOVES_RHS}", True)]:
        d = d.reset_index(drop=True)
        f = f"event ~ {rhs}" + (" + C(activity)" if fe else "")
        base = smf.logit(f, d).fit(disp=0, maxiter=300)
        for cl, cov in [("activity (original specification)", base.model.fit(disp=0, maxiter=300, cov_type="cluster",
                                                               cov_kwds={"groups": pd.factorize(d["activity"])[0]}).cov_params()),
                        ("student", base.model.fit(disp=0, maxiter=300, cov_type="cluster",
                                                   cov_kwds={"groups": pd.factorize(d["user_id"])[0]}).cov_params()),
                        ("two-way activity and student", cluster2_cov(None, base, d["activity"].values, d["user_id"].values))]:
            rows += rows_from(base, f"Hazard, {lab}, SE clustered by {cl}", HZ, cov=np.asarray(cov))
    # hazard with student random intercept (pattern specification, Bayesian mixed GLM)
    try:
        hb = BinomialBayesMixedGLM.from_formula(
            f"event ~ log_k + log_len + C(pattern) + {MOVES_RHS}",
            {"activity": "0 + C(activity)", "student": "0 + C(user_id)"}, PP.reset_index(drop=True)).fit_vb()
        for i, t in enumerate(hb.model.exog_names):
            if t in HZ:
                b, s = hb.fe_mean[i], hb.fe_sd[i]
                rows.append({"model": "Hazard, pattern, Bayesian mixed GLM with activity + student RI", "term": t,
                             "estimate": b, "se": s, "effect": np.exp(b), "ci_low": np.exp(b - 1.96 * s),
                             "ci_high": np.exp(b + 1.96 * s), "p": 2 * (1 - stats.norm.cdf(abs(b / s)))})
    except Exception as e:
        log("Hazard mixed GLM failed:", e)
    # model frames for the frequentist GLMMs in 04b_glmm_student_re.R
    D[["conversationId", "achieved", "dok_reached", "pattern", "target", "target_c", "teacher", "activity", "student"]
      ].to_csv(out("rev_D_modeldata_conversations.csv"), index=False)
    PP.assign(student=PP["user_id"].astype(str), mixed=PP["discussionId"].isin(mixed_acts).astype(int))[
        ["conversationId", "event", "log_k", "log_len", "pattern", "activity", "student", "mixed", "prev_dok_state"] + HT
    ].to_csv(out("rev_D_modeldata_person_period.csv"), index=False)
    dep = pd.DataFrame(rows)
    dep.to_csv(out("rev_D_student_clustering.csv"), index=False)
    log(dep[dep["term"].isin(PT + ["prev_socratic_question", "prev_request_evidence", "pattern"])]
        [["model", "term", "effect", "ci_low", "ci_high", "p"]].round(3).to_string(index=False))

    # one conversation per student: first conversation (chronological) and random draws
    def one_per_student_models(Dsub):
        r = {}
        try:
            m1 = fit_m1(Dsub)
            r.update({"M1_P2": m1.params["C(pattern)[T.P2]"], "M1_P3": m1.params["C(pattern)[T.P3]"],
                      "M1_P2_p": m1.pvalues["C(pattern)[T.P2]"], "M1_P3_p": m1.pvalues["C(pattern)[T.P3]"]})
        except Exception:
            pass
        m2 = fit_m2(Dsub, lambda d: pd.factorize(d["activity"])[0])
        w, p = pattern_wald(m2)
        r.update({"M2_OR_P2": np.exp(m2.params["C(pattern)[T.P2]"]), "M2_OR_P3": np.exp(m2.params["C(pattern)[T.P3]"]),
                  "M2_OR_target": np.exp(m2.params["target_c"]), "M2_target_p": m2.pvalues["target_c"],
                  "M2_wald_p": p})
        ids = set(Dsub["conversationId"])
        s3 = strat[strat.index.isin(ids)]
        mx = s3.groupby("discussionId")["achieved"].agg(["min", "max"])
        macts = mx[(mx["min"] == 0) & (mx["max"] == 1)].index
        dpp = PP2[PP2["conversationId"].isin(ids) & PP2["discussionId"].isin(macts)]
        h = smf.logit(f"event ~ log_k + log_len + {MOVES_RHS} + C(activity)", dpp).fit(
            disp=0, maxiter=300, cov_type="cluster", cov_kwds={"groups": pd.factorize(dpp["activity"])[0]})
        ci = h.conf_int().loc["prev_socratic_question"]
        r.update({"HZ_FE_OR_socratic": np.exp(h.params["prev_socratic_question"]),
                  "HZ_FE_socratic_lo": np.exp(ci[0]), "HZ_FE_socratic_hi": np.exp(ci[1]),
                  "HZ_FE_OR_evidence": np.exp(h.params["prev_request_evidence"]),
                  "n_conv": len(Dsub), "n_dok3": len(s3)})
        return r

    # Timestamps are not shared, so the check uses random draws of one conversation per student.
    draws = []
    for b in range(200):
        samp_ = D.sample(frac=1, random_state=int(RNG.integers(1e9))).groupby("student").head(1)
        try:
            draws.append(one_per_student_models(samp_))
        except Exception:
            continue
    dr = pd.DataFrame(draws)
    summ = dr.describe(percentiles=[.025, .5, .975]).T[["count", "2.5%", "50%", "97.5%"]]
    summ["share_p_below_05"] = [((dr[c] < .05).mean() if c.endswith("_p") else np.nan) for c in summ.index]
    summ.loc["share_socratic_ci_excludes_1", "50%"] = (dr["HZ_FE_socratic_lo"] > 1).mean()
    summ.loc["share_M2_wald_p_above_05", "50%"] = (dr["M2_wald_p"] > .05).mean()
    summ.to_csv(out("rev_D_one_per_student_random_draws.csv"))
    dr.to_csv(out("rev_D_one_per_student_draws_raw.csv"), index=False)
    log(f"Random one-per-student draws ({len(dr)}):\n" + summ.round(3).to_string())


# ==========================================================================
# E. Lagged-state hazard
# ==========================================================================
if "E" in SECTIONS:
    log("\n" + "=" * 78 + "\nE. Lagged-state hazard\n" + "=" * 78)
    # Before first attainment the prior student turn is below DOK 3 by construction, so the state is
    # none (first student turn), D0, D1 or D2. cum_max is the highest DOK reached so far.
    E = PPM.sort_values(["conversationId", "k"]).copy()
    Ep = PP.sort_values(["conversationId", "k"]).copy()
    for d in (E, Ep):
        d["prev_dok_state"] = pd.Categorical(d["prev_dok_state"], categories=["D1", "none", "D0", "D2"])
        d["cum_max_prior"] = (d.groupby("conversationId")["dok"].cummax().groupby(d["conversationId"]).shift(1)
                              .fillna(-1).astype(int))
        d["cum_max_state"] = d["cum_max_prior"].map({-1: "none", 0: "D0", 1: "D1", 2: "D2"})
        d["cum_max_state"] = pd.Categorical(d["cum_max_state"], categories=["D1", "none", "D0", "D2"])
    log("Person-periods by prior student state (mixed-outcome activities):\n" +
        pd.crosstab(E["prev_dok_state"], E["event"], margins=True).to_string())
    log("Attainment rate and Socratic prevalence by prior state:\n" +
        E.groupby("prev_dok_state").agg(n=("event", "size"), attain=("event", "mean"),
                                        socratic=("prev_socratic_question", "mean")).round(3).to_string())

    # The first student turn has no preceding coded AI turn (the opening message is the teacher-written
    # starter), so it carries no move, no prior state and, empirically, no attainment events. The lagged-state
    # models therefore use student turns 2 onward; L0b shows the original specification on that same risk set.
    n_first = int((E["k"] == 1).sum())
    log(f"First student turns in the risk set: {n_first}, events {int(E.loc[E['k'] == 1, 'event'].sum())}, "
        f"Socratic-preceded {int(E.loc[E['k'] == 1, 'prev_socratic_question'].sum())}")
    E = E[E["k"] >= 2].copy()
    Ep = Ep[Ep["k"] >= 2].copy()
    for d in (E, Ep):
        d["prev_dok_state"] = d["prev_dok_state"].cat.remove_unused_categories()
        d["cum_max_state"] = d["cum_max_state"].cat.remove_unused_categories()
    rows = []
    _, r = fit_hazard(PPM, f"log_k + log_len + {MOVES_RHS}", "L0a activity FE, all turns (original specification)", fe=True)
    rows += r
    specs = [
        ("L0b activity FE, turns 2+, no state", E, f"log_k + log_len + {MOVES_RHS}", True),
        ("L1 activity FE, turns 2+, + prior student DOK", E, f"log_k + log_len + C(prev_dok_state) + {MOVES_RHS}", True),
        ("L2 activity FE, turns 2+, + prior DOK + highest DOK so far", E,
         f"log_k + log_len + C(prev_dok_state) + C(cum_max_state) + {MOVES_RHS}", True),
        ("L3 activity FE, turns 2+, + prior DOK, without log length", E, f"log_k + C(prev_dok_state) + {MOVES_RHS}", True),
        ("L4 pattern, turns 2+, + prior student DOK", Ep, f"log_k + log_len + C(pattern) + C(prev_dok_state) + {MOVES_RHS}", False),
    ]
    for lab, d, rhs, fe in specs:
        _, r = fit_hazard(d, rhs, lab, fe=fe)
        rows += r
    # Socratic-by-state interaction and stratified estimates
    Ei = E.copy()
    res_i, r = fit_hazard(Ei, f"log_k + log_len + C(prev_dok_state) * prev_socratic_question + "
                          + " + ".join(t for t in HT if t != "prev_socratic_question"),
                          "L5 activity FE, Socratic x prior state interaction", fe=True)
    rows += r
    inter = [t for t in res_i.params.index if ":prev_socratic_question" in t]
    w = res_i.wald_test(", ".join(f"{t} = 0" for t in inter), scalar=True)
    log(f"Socratic x prior-state interaction, Wald chi2 = {float(w.statistic):.2f}, p = {float(w.pvalue):.3f}")
    rows.append({"model": "L5 Wald test, Socratic x prior state", "term": "interaction block",
                 "log_odds": float(w.statistic), "p": float(w.pvalue)})
    for s_ in ["D0", "D1", "D2"]:
        d = E[E["prev_dok_state"] == s_]
        if d["event"].sum() < 15:
            log(f"State {s_}: {int(d['event'].sum())} events, too few for a stratified model")
            continue
        try:
            _, r = fit_hazard(d, f"log_k + log_len + {MOVES_RHS}", f"L6 stratified, prior state {s_} "
                              f"(n = {len(d)}, events = {int(d['event'].sum())})", fe=True)
            rows += r
        except Exception as e:
            try:
                _, r = fit_hazard(d, f"log_k + log_len + {MOVES_RHS}", f"L6 stratified, prior state {s_}, no FE "
                                  f"(n = {len(d)}, events = {int(d['event'].sum())})", fe=False)
                rows += r
            except Exception as e2:
                log(f"State {s_} failed: {e2}")
    lag = pd.DataFrame(rows)
    lag.to_csv(out("rev_E_lagged_state_hazard.csv"), index=False)
    show = lag[lag["term"].isin(["prev_socratic_question", "prev_request_evidence", "prev_request_attempt",
                                 "prev_stepwise_guidance", "prev_affirm_extend", "prev_hint"])]
    log(show.pivot(index="model", columns="term", values="odds_ratio").round(2).to_string())
    log(lag[lag["term"].str.contains("prev_dok_state|cum_max")][["model", "term", "odds_ratio", "ci_low_or",
                                                                  "ci_high_or"]].round(2).to_string(index=False))
    log(lag[lag["term"] == "prev_socratic_question"][["model", "odds_ratio", "ci_low_or", "ci_high_or", "p"]]
        .round(3).to_string(index=False))

    # selection check: does the student's prior DOK predict whether the AI asks a Socratic question?
    sel = []
    for m in ["prev_socratic_question", "prev_request_evidence", "prev_affirm_extend", "prev_hint", "prev_request_attempt"]:
        f = smf.logit(f"{m} ~ C(prev_dok_state) + log_k + C(activity)", E).fit(
            disp=0, maxiter=300, cov_type="cluster", cov_kwds={"groups": pd.factorize(E["activity"])[0]})
        for t in f.params.index:
            if t.startswith("C(prev_dok_state)"):
                ci = f.conf_int().loc[t]
                sel.append({"move": m, "term": t, "odds_ratio": np.exp(f.params[t]), "ci_low": np.exp(ci[0]),
                            "ci_high": np.exp(ci[1]), "p": f.pvalues[t]})
    sel = pd.DataFrame(sel)
    sel.to_csv(out("rev_E_move_selection_on_state.csv"), index=False)
    log("Selection: odds of the AI move given the prior student DOK (reference D1), activity FE:\n" +
        sel.round(3).to_string(index=False))

    # lagged-state transition model over all student turns (not only pre-attainment): next-turn DOK >= 3
    # given the prior student DOK and the intervening AI move, stratum conversations, activity FE
    Z = st[st["conversationId"].isin(strat.index) & st["prev_dok"].notna()].copy()
    Z["activity"] = Z["discussionId"].astype(str)
    Z["up"] = (Z["dok"] >= 3).astype(int)
    Z["prev_state"] = pd.Categorical(Z["prev_dok"].clip(upper=3).astype(int).map(lambda x: f"D{x}"),
                                     categories=["D1", "D0", "D2", "D3"])
    fz = smf.logit(f"up ~ C(prev_state) + {MOVES_RHS} + C(activity)", Z[Z["discussionId"].isin(mixed_acts)]).fit(
        disp=0, maxiter=300, cov_type="cluster", cov_kwds={"groups": pd.factorize(Z[Z["discussionId"].isin(mixed_acts)]["activity"])[0]})
    rz = tidy_logit(fz, "T1 transition model, all student turns: next turn at DOK 3+ ~ prior state + preceding move, activity FE",
                    keep_prefix=("prev_", "C(prev_state)"))
    fz2 = smf.logit(f"up ~ C(prev_state) * prev_socratic_question + "
                    + " + ".join(t for t in HT if t != "prev_socratic_question") + " + C(activity)",
                    Z[Z["discussionId"].isin(mixed_acts)]).fit(
        disp=0, maxiter=300, cov_type="cluster", cov_kwds={"groups": pd.factorize(Z[Z["discussionId"].isin(mixed_acts)]["activity"])[0]})
    rz += tidy_logit(fz2, "T2 transition model with Socratic x prior state", keep_prefix=("prev_", "C(prev_state)"))
    tz = pd.DataFrame(rz)
    tz.to_csv(out("rev_E_transition_model.csv"), index=False)
    log("Transition model (all student turns in mixed-outcome DOK 3 activities):\n" +
        tz[["model", "term", "odds_ratio", "ci_low_or", "ci_high_or", "p"]].round(3).to_string(index=False))
    # marginal: Socratic effect within each prior state from T2
    b = fz2.params
    V = fz2.cov_params()
    for s_ in ["D0", "D1", "D2", "D3"]:
        terms = ["prev_socratic_question"] + ([f"C(prev_state)[T.{s_}]:prev_socratic_question"] if s_ != "D1" else [])
        est = sum(b[t] for t in terms)
        var = sum(V.loc[t1, t2] for t1 in terms for t2 in terms)
        log(f"  Socratic OR for next turn DOK 3+ given prior state {s_}: {np.exp(est):.2f} "
            f"[{np.exp(est - 1.96*np.sqrt(var)):.2f}, {np.exp(est + 1.96*np.sqrt(var)):.2f}]")


# ==========================================================================
# F. Persistence: task-driven descent versus loss of rigor
# ==========================================================================
PRESS = {"socratic_question", "request_evidence", "affirm_extend"}
ADVANCE = {"stepwise_guidance", "final_answer"}
CLOSING = {"closure", "recap"}
ST_PROC = ["st_submits_work", "st_agrees_acknowledges", "st_off_task", "st_requests_help", "st_requests_answer"]


def ai_moves_set(row):
    return {m for m in AI_MOVES if row.get(f"ai_{m}", 0) == 1}


if "F" in SECTIONS:
    log("\n" + "=" * 78 + "\nF. Persistence: task-driven descent versus loss of rigor\n" + "=" * 78)
    ach3 = strat[strat["achieved"] == 1]
    S = st[st["conversationId"].isin(ach3.index)].sort_values(["conversationId", "k"])
    A = ai[ai["conversationId"].isin(ach3.index)].sort_values(["conversationId", "turn"])
    for c in ST_PROC:
        S[c] = S[c].fillna(0).astype(int)
    Aidx = A.set_index(["conversationId", "turn"])
    recs = []
    for cid, g in S.groupby("conversationId"):
        at = g["at_target"].values
        k0 = cs.loc[cid, "k_first"]
        i0 = int(k0) - 1
        post = at[i0 + 1:]
        if len(post) == 0:
            cls = "final_turn"
        elif post.min() == 1:
            cls = "sustained"
        elif post.sum() > 0:
            cls = "intermittent"
        else:
            cls = "regressed"
        first_hit_turn = g["turn"].values[i0]
        after = A[(A["conversationId"] == cid) & (A["turn"] > first_hit_turn)]
        mv_after = ai_moves_set(after.iloc[0]) if len(after) else set()
        rec = {"conversationId": cid, "class": cls, "k_first": k0, "n_turns": len(at), "n_post": len(post),
               "move_after_first_hit": after.iloc[0]["primary_move"] if len(after) else "none",
               "after_press": int(bool(mv_after & PRESS)), "after_advance": int(bool(mv_after & ADVANCE)),
               "after_closing": int(bool(mv_after & CLOSING)),
               "ai_closing_after_hit": int(after[[f"ai_{m}" for m in CLOSING]].to_numpy().sum() > 0) if len(after) else 0,
               "pattern": cs.loc[cid, "pattern"], "subject": cs.loc[cid, "subject"]}
        if cls in ("intermittent", "regressed"):
            j = i0 + 1 + int(np.flatnonzero(post == 0)[0])          # first below-target turn after attainment
            drop = g.iloc[j]
            pt = drop["prev_ai_turn"]
            pm = ai_moves_set(Aidx.loc[(cid, pt)]) if pd.notna(pt) and (cid, pt) in Aidx.index else set()
            proc = any(drop[c] == 1 for c in ["st_submits_work", "st_agrees_acknowledges", "st_off_task"])
            if drop["dok"] == 0 and (proc or pm & CLOSING or j == len(at) - 1):
                dtype = "procedural or closing turn"
            elif pm & CLOSING:
                dtype = "procedural or closing turn"
            elif pm & ADVANCE and not (pm & {"socratic_question", "request_evidence"}):
                dtype = "task advanced to next step"
            elif drop["dok"] == 0:
                dtype = "non-substantive turn under press" if pm & PRESS else "non-substantive turn, other"
            elif pm & PRESS:
                dtype = "lower-level answer under continued press"
            else:
                dtype = "lower-level answer, other move"
            rec.update({"drop_k": j + 1, "drop_dok": int(drop["dok"]), "drop_position": (j + 1) / len(at),
                        "drop_is_last_turn": int(j == len(at) - 1), "drop_prev_ai_moves": ";".join(sorted(pm)),
                        "drop_student_procedural": int(proc), "drop_type": dtype,
                        "post_share_at_target": post.mean()})
        recs.append(rec)
    P = pd.DataFrame(recs)
    P.to_csv(out("rev_F_persistence_descent_per_conversation.csv"), index=False)
    dropped = P[P["class"].isin(["intermittent", "regressed"])]
    log(f"Achieved DOK 3 conversations: {len(P)}; classes {P['class'].value_counts().to_dict()}")
    log(f"Conversations that fall below target after attainment: {len(dropped)} ({len(dropped)/len(P):.1%})")
    log("DOK of the first below-target turn after attainment:\n" +
        pd.crosstab(dropped["class"], dropped["drop_dok"], margins=True).to_string())
    t1 = pd.crosstab(dropped["drop_type"], dropped["class"], margins=True)
    t1.to_csv(out("rev_F_descent_type_by_class.csv"))
    log("Type of first descent x persistence class:\n" + t1.to_string())
    # grouped: task-driven versus rigor loss
    grp = {"procedural or closing turn": "task-driven", "task advanced to next step": "task-driven",
           "lower-level answer under continued press": "loss of rigor (substantive answer below target)",
           "lower-level answer, other move": "loss of rigor (substantive answer below target)",
           "non-substantive turn under press": "disengagement (non-substantive turn)",
           "non-substantive turn, other": "disengagement (non-substantive turn)"}
    dropped = dropped.assign(descent=dropped["drop_type"].map(grp))
    t2 = pd.crosstab(dropped["descent"], dropped["class"], margins=True)
    t2n = pd.crosstab(dropped["descent"], dropped["class"], normalize="columns")
    t2.to_csv(out("rev_F_descent_grouped_by_class.csv"))
    log("Grouped descent x class (counts):\n" + t2.to_string() + "\n(column shares):\n" + t2n.round(3).to_string())
    # task phase: position of the first descent in the conversation
    dropped = dropped.assign(phase=pd.cut(dropped["drop_position"], [0, .5, .8, 1.0001],
                                          labels=["first half", "50-80%", "final 20%"]))
    t3 = pd.crosstab([dropped["phase"]], [dropped["descent"]], margins=True)
    t3.to_csv(out("rev_F_descent_by_phase.csv"))
    log("Position of first descent (share of conversation) x descent type:\n" + t3.to_string())
    # AI move immediately after first attainment, grouped, x class
    P["move_group_after_hit"] = np.select([P["after_closing"] == 1, (P["after_advance"] == 1) & (P["after_press"] == 0),
                                           P["after_press"] == 1], ["closing", "advance procedure", "press"], "other")
    t4 = pd.crosstab(P["move_group_after_hit"], P["class"], margins=True)
    t4r = pd.crosstab(P["move_group_after_hit"], P["class"], normalize="index")
    t4.to_csv(out("rev_F_move_after_attainment_by_class.csv"))
    log("AI move after first attainment (grouped) x class:\n" + t4.to_string() + "\n(row shares):\n" + t4r.round(3).to_string())
    # headline shares of all achieved conversations
    n = len(P)
    summary = {
        "n_achieved": n,
        "share_sustained": (P["class"] == "sustained").mean(),
        "share_final_turn": (P["class"] == "final_turn").mean(),
        "share_fall_below": len(dropped) / n,
        "share_fall_below_task_driven": (dropped["descent"] == "task-driven").sum() / n,
        "share_fall_below_rigor_loss": dropped["descent"].str.startswith("loss").sum() / n,
        "share_fall_below_disengagement": dropped["descent"].str.startswith("disengagement").sum() / n,
        "share_regressed_no_return": (P["class"] == "regressed").mean(),
        "share_regressed_no_return_rigor_loss": ((dropped["class"] == "regressed") & dropped["descent"].str.startswith("loss")).sum() / n,
        "share_regressed_no_return_task_driven": ((dropped["class"] == "regressed") & (dropped["descent"] == "task-driven")).sum() / n,
        "share_first_drop_is_dok0": (dropped["drop_dok"] == 0).sum() / n,
    }
    pd.Series(summary).to_csv(out("rev_F_persistence_summary.csv"))
    log("Summary shares of achieved conversations:\n" + pd.Series(summary).round(3).to_string())


# ==========================================================================
# G. Collinearity between pattern, target DOK and teacher
# ==========================================================================
def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    return np.sqrt(chi2 / (n * (min(ct.shape) - 1)))


if "G" in SECTIONS:
    log("\n" + "=" * 78 + "\nG. Collinearity between pattern, target DOK and teacher\n" + "=" * 78)
    act = pd.read_csv(args.pattern_file, index_col=0)
    act = act[act["subject"] != "World Language"]
    t_pt = pd.crosstab(act["pattern"], act["target"], margins=True)
    t_ptt = pd.crosstab(act["teacher"], act["pattern"], margins=True)
    t_pt.to_csv(out("rev_G_pattern_by_target_activities.csv"))
    log("Activities, pattern x target DOK:\n" + t_pt.to_string())
    tp = pd.crosstab(act["teacher"], act["pattern"])
    # teacher labels in data/ are already pseudonyms (T01-T16)
    tp.to_csv(out("rev_G_teacher_by_pattern_activities.csv"))
    log("Activities, teacher (anonymized) x pattern:\n" + tp.to_string())
    p3_teachers = tp[tp["P3"] > 0]
    log(f"P3: {int(tp['P3'].sum())} activities from {len(p3_teachers)} teachers; largest teacher share "
        f"{p3_teachers['P3'].max() / tp['P3'].sum():.0%}; P3 teachers who also author P1 or P2: "
        f"{int(((p3_teachers[['P1', 'P2']].sum(axis=1)) > 0).sum())}; teachers authoring more than one pattern: "
        f"{int((tp.gt(0).sum(axis=1) > 1).sum())} of {len(tp)}")
    log(f"Share of P3 activities (and conversations) authored by the top two teachers: "
        f"{p3_teachers['P3'].nlargest(2).sum() / tp['P3'].sum():.0%}")
    cv_rows = [{"pair": "pattern-target DOK (activities)", "cramers_v": cramers_v(act["pattern"], act["target"])},
               {"pair": "pattern-teacher (activities)", "cramers_v": cramers_v(act["pattern"], act["teacher"])},
               {"pair": "target DOK-teacher (activities)", "cramers_v": cramers_v(act["target"], act["teacher"])}]
    # conversation level
    D = rq1_frame(cs)
    cv_rows += [{"pair": "pattern-target DOK (conversations)", "cramers_v": cramers_v(D["pattern"], D["target"])},
                {"pair": "pattern-teacher (conversations)", "cramers_v": cramers_v(D["pattern"], D["teacher"])}]
    # variance inflation for the M2 design (DOK 2-3 targets): VIF per term and GVIF for the pattern block
    d23 = D[D["target"] >= 2].copy()
    X = pd.get_dummies(d23["pattern"], drop_first=True).astype(float)
    X["target_c"] = d23["target_c"].astype(float)
    R = np.corrcoef(X.values, rowvar=False)
    vif = pd.Series(np.diag(np.linalg.inv(R)), index=X.columns)
    pat_idx = [0, 1]
    gvif = np.linalg.det(R[np.ix_(pat_idx, pat_idx)]) * np.linalg.det(R[np.ix_([2], [2])]) / np.linalg.det(R)
    for t, vv in vif.items():
        cv_rows.append({"pair": f"VIF {t} in M2 design (conversations, DOK 2-3)", "cramers_v": vv})
    cv_rows.append({"pair": "GVIF^(1/(2 df)) pattern block in M2 design", "cramers_v": gvif ** (1 / 4)})
    # activity level (the level at which pattern and target vary)
    a23 = act[act["target"] >= 2]
    Xa = pd.get_dummies(a23["pattern"], drop_first=True).astype(float)
    Xa["target_c"] = a23["target"].astype(float) - 3
    Ra = np.corrcoef(Xa.values, rowvar=False)
    via = pd.Series(np.diag(np.linalg.inv(Ra)), index=Xa.columns)
    for t, vv in via.items():
        cv_rows.append({"pair": f"VIF {t} in activity-level design (DOK 2-3)", "cramers_v": vv})
    log(f"Correlation P3 indicator with target DOK (conversations, DOK 2-3): "
        f"{np.corrcoef(X['P3'], X['target_c'])[0, 1]:.2f}; activities: {np.corrcoef(Xa['P3'], Xa['target_c'])[0, 1]:.2f}")
    coll = pd.DataFrame(cv_rows).rename(columns={"cramers_v": "value"})
    coll.to_csv(out("rev_G_collinearity.csv"), index=False)
    log(coll.round(3).to_string(index=False))

    # how much of the P3 standard error comes from target DOK and teacher?
    rows = []
    m_full = smf.logit("achieved ~ C(pattern) + target_c", d23).fit(disp=0, cov_type="cluster",
                                                                   cov_kwds={"groups": pd.factorize(d23["activity"])[0]})
    m_not = smf.logit("achieved ~ C(pattern)", d23).fit(disp=0, cov_type="cluster",
                                                       cov_kwds={"groups": pd.factorize(d23["activity"])[0]})
    d3 = D[D["target"] == 3]
    m_d3 = smf.logit("achieved ~ C(pattern)", d3).fit(disp=0, cov_type="cluster",
                                                     cov_kwds={"groups": pd.factorize(d3["activity"])[0]})
    for lab, m in [("M2 pattern + target (original specification)", m_full), ("M2 pattern only", m_not),
                   ("DOK 3 stratum, pattern only (P3 vs P1 at equal target)", m_d3)]:
        for t in ["C(pattern)[T.P2]", "C(pattern)[T.P3]"]:
            ci = m.conf_int().loc[t]
            rows.append({"model": lab, "term": t, "or": np.exp(m.params[t]), "se": m.bse[t],
                         "ci_low": np.exp(ci[0]), "ci_high": np.exp(ci[1]), "p": m.pvalues[t]})
    # teacher fixed effects: pattern contrasts identified only from teachers who author more than one pattern
    try:
        m_t = smf.logit("achieved ~ C(pattern) + target_c + C(teacher)", d23).fit(
            disp=0, maxiter=300, cov_type="cluster", cov_kwds={"groups": pd.factorize(d23["activity"])[0]})
        for t in ["C(pattern)[T.P2]", "C(pattern)[T.P3]"]:
            ci = m_t.conf_int().loc[t]
            rows.append({"model": "M2 + teacher fixed effects (within-teacher contrast)", "term": t,
                         "or": np.exp(m_t.params[t]), "se": m_t.bse[t], "ci_low": np.exp(ci[0]),
                         "ci_high": np.exp(ci[1]), "p": m_t.pvalues[t]})
    except Exception as e:
        log("teacher FE model failed:", e)
    se_tab = pd.DataFrame(rows)
    se_tab.to_csv(out("rev_G_p3_se_by_specification.csv"), index=False)
    log("Pattern contrasts under alternative specifications:\n" + se_tab.round(3).to_string(index=False))
    # MDE under each specification (80% power, two-sided .05, 67 residual df as in 12_prompt_patterns.py)
    mult = stats.t.ppf(0.975, 67) + stats.t.ppf(0.80, 67)
    se_tab["mde_or"] = np.exp(mult * se_tab["se"])
    log("MDE (odds ratio) by specification:\n" + se_tab[["model", "term", "mde_or"]].round(2).to_string(index=False))
    se_tab.to_csv(out("rev_G_p3_se_by_specification.csv"), index=False)


# ==========================================================================
# H. Timing re-indexed from the first typed student turn (applies to every section)
# ==========================================================================
if "H" in SECTIONS or "B" in SECTIONS:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    log("\n" + "=" * 78 + "\nH. Timing indexed from the first typed student turn\n" + "=" * 78)
    s3 = strat.copy()
    s3["n_typed"] = s3["n_student_turns"] - 1
    s3["j_first"] = s3["k_first"] - 1                          # typed-turn index of first attainment
    a3 = s3[s3["j_first"].notna()]
    log(f"Achieved {len(a3)}: median typed turn {a3['j_first'].median():.0f} (IQR {a3['j_first'].quantile(.25):.0f} "
        f"to {a3['j_first'].quantile(.75):.0f}); on the first typed turn {(a3['j_first'] == 1).mean():.1%} of achieved, "
        f"{(a3['j_first'] == 1).sum() / len(s3):.1%} of all; median position {(a3['j_first'] / a3['n_typed']).median():.0%} "
        f"of typed turns; reached by typed turn 4: {(a3['j_first'] <= 4).mean():.1%} of achieved")
    curve = []
    for j in range(1, int(s3["n_typed"].quantile(.95)) + 1):
        at_risk = ((s3["n_typed"] >= j) & (s3["j_first"].isna() | (s3["j_first"] >= j))).sum()
        hit = (s3["j_first"] == j).sum()
        curve.append({"typed_turn": j, "at_risk": at_risk, "hit": hit, "hazard": hit / at_risk if at_risk else np.nan})
    curve = pd.DataFrame(curve)
    curve["cum_attainment_km"] = 1 - (1 - curve["hazard"].fillna(0)).cumprod()
    curve["cum_share_raw"] = curve["hit"].cumsum() / len(s3)
    curve.to_csv(out("rev_H_attainment_curve_typed_turns.csv"), index=False)
    log(curve[curve["typed_turn"].isin([1, 2, 3, 4, 5, 9, 10, 14])].round(3).to_string(index=False))
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    cp = curve[curve["at_risk"] >= 10]
    ax.step(cp["typed_turn"], cp["cum_share_raw"], where="post", color="#2a78d6", lw=2, label="Share of all conversations")
    ax.step(cp["typed_turn"], cp["cum_attainment_km"], where="post", color="#eb6834", lw=2, ls="--",
            label="Kaplan-Meier estimate (ended conversations censored)")
    ax.set_xlabel("Typed student turn", fontsize=9, color="#52514e")
    ax.set_ylabel("Cumulative share reaching target", fontsize=9, color="#52514e")
    ax.set_ylim(0, 1)
    ax.set_title(f"Cumulative target attainment by typed student turn (target DOK 3, n = {len(s3)})",
                 fontsize=9.5, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.yaxis.grid(True, color="#e6e5e1")
    ax.set_axisbelow(True)
    fig.tight_layout()
    os.makedirs(os.path.join(OUTPUT_DIR, "figures"), exist_ok=True)
    fig.savefig(os.path.join(OUTPUT_DIR, "figures", "fig_rev_attainment_curve_typed.png"), dpi=300)
    plt.close(fig)
    # persistence classes are unchanged by the re-indexing (they depend on order, not on the index)


# ==========================================================================
# write log
# ==========================================================================
def finish():
    with open(out("rev_run_log.txt"), "w") as f:
        f.write("\n".join(LOG) + "\n")
finish()
