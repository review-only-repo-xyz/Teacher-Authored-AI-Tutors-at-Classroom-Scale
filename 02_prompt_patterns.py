"""
REPLICATION VERSION.

This script is shared for replication and review. It runs on the
de-identified coded datasets in data/ (activities.csv, conversations.csv,
turn_codes.csv), which contain the study's real codes but no conversation
text, no prompt text, no names, and no platform identifiers. See README.md.

RQ1: Prompt patterns, DOK reached, and goal achievement.

Identifies configurations ("patterns") of teacher-authored prompt features
across the 74 analyzed activities and tests whether the pattern an activity
belongs to is associated with (a) the DOK level students reach and (b) whether
students achieve the teacher's target DOK, controlling for target DOK and for
nesting in activities and teachers.

Steps
  1. Build the activity-level binary feature matrix (features with >= 7 and
     <= 67 of 74 activities present; near-constant features are dropped).
  2. Cluster activities with k-modes (Hamming) and, as a sensitivity check,
     agglomerative clustering on Jaccard distance. k chosen by silhouette
     over k = 2..6. Agreement between methods reported (adjusted Rand index).
  3. Describe each pattern (feature profile, n activities, n conversations,
     mean DOK reached, achievement rate by target DOK).
  4. Models, all with target DOK as covariate:
       M1  DOK reached ~ pattern + target DOK, linear mixed model with
           random intercepts for teacher and for activity within teacher.
       M2  Goal achieved ~ pattern + target DOK, logistic regression with
           cluster-robust SEs by activity (74 clusters).
       M2b Same as M2 on the DOK 3 stratum only (patterns with >= 20
           conversations in the stratum).
       M3  Goal achieved ~ pattern + target DOK, Bayesian binomial mixed GLM
           with random intercepts for teacher and activity (check on M2;
           variational fit, posterior SDs are approximate and tend to be small).
       M4  Feature-count model: DOK reached and goal achieved regressed on
           counts of scaffolding, epistemic, constraint and guardrail features
           (a pattern-free alternative that does not depend on clustering).
  5. Minimum detectable effects for pattern contrasts at 80% power.

Usage
    python 02_prompt_patterns.py                         # conversation-level outcome, all subjects
    python 02_prompt_patterns.py --turn-csv data/turn_codes.csv \
        --exclude-subject "World Language"                       # primary specification (turn-level DOK)
    python 02_prompt_patterns.py --tag _convlabel        # suffix outputs

Outputs (outputs/)
    rq1_activity_features.csv      activity x feature matrix with pattern labels
    rq1_pattern_profiles.csv       feature prevalence within each pattern
    rq1_pattern_outcomes.csv       outcomes by pattern and target DOK
    rq1_cluster_selection.csv      silhouette by k, both methods
    rq1_models.csv                 coefficient tables, all models
    rq1_mde.csv                    minimum detectable effects
    figures/fig_rq1_pattern_profiles.png
    figures/fig_rq1_achievement_by_pattern.png
    figures/fig_rq1_forest.png
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from kmodes.kmodes import KModes
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, OUTPUT_DIR, PROMPT_FEATURES  # noqa: E402

import argparse

warnings.filterwarnings("ignore")
np.random.seed(2026)

ap = argparse.ArgumentParser()
ap.add_argument("--turn-csv", default=None,
                help="long-format turn codes (data/turn_codes.csv); when given, DOK reached is the maximum "
                     "turn-level student DOK and goal achieved is that maximum at or above target")
ap.add_argument("--exclude-subject", action="append", default=[],
                help="subject to drop from the analytic sample (repeatable)")
ap.add_argument("--tag", default="", help="suffix appended to every output file name")
ap.add_argument("--discipline-split", action="store_true",
                help="add ELA versus STEM/CTE discipline analyses (interaction and stratified models)")
args = ap.parse_args()
TAG = args.tag

DATA_FILE = os.path.join(DATA_DIR, "conversations.csv")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def outpath(name):
    base, ext = os.path.splitext(name)
    return os.path.join(OUTPUT_DIR, f"{base}{TAG}{ext}")


def figpath(name):
    base, ext = os.path.splitext(name)
    return os.path.join(FIG_DIR, f"{base}{TAG}{ext}")

MIN_PREV, MAX_PREV = 7, 67          # keep features present in 7..67 of 74 activities
K_RANGE = range(2, 7)
MIN_CLUSTER_ACTIVITIES = 5
MIN_CLUSTER_TEACHERS = 3

# Palette (validated categorical slots; see dataviz reference)
PAL = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
     "yellow": "#eda100", "magenta": "#e87ba4", "violet": "#4a3aa7",
     "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#e6e5e1"}
PATTERN_COLORS = [PAL["blue"], PAL["orange"], PAL["aqua"], PAL["yellow"], PAL["magenta"], PAL["violet"]]

FEATURE_LABELS = {
    "high_specificity": "High task specificity",
    "finish_line": "Finish line",
    "scaffold_socratic": "Socratic questioning",
    "scaffold_stepwise": "Stepwise guidance",
    "scaffold_worked_example_prohibition": "No worked examples",
    "epistemic_explain_reasoning": "Explain reasoning",
    "epistemic_use_evidence": "Use evidence",
    "epistemic_compare_alternatives": "Compare alternatives",
    "epistemic_justify_claims": "Justify claims",
    "constraint_short_turn": "Short turns",
    "constraint_one_question_per_turn": "One question per turn",
    "constraint_structured_format": "Structured format",
    "constraint_language_level": "Language level",
    "guardrail_no_direct_answers": "No direct answers",
    "guardrail_respectful": "Respectful tone",
}
FAMILY = {
    "scaffold": ["scaffold_socratic", "scaffold_stepwise", "scaffold_hints_first",
                 "scaffold_attempt_first", "scaffold_worked_example_prohibition"],
    "epistemic": ["epistemic_explain_reasoning", "epistemic_use_evidence",
                  "epistemic_compare_alternatives", "epistemic_justify_claims"],
    "constraint": ["constraint_short_turn", "constraint_one_question_per_turn",
                   "constraint_structured_format", "constraint_language_level"],
    "guardrail": ["guardrail_no_direct_answers", "guardrail_integrity_language",
                  "guardrail_privacy", "guardrail_respectful"],
}


def style_axes(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(PAL["grid"])
    ax.tick_params(colors=PAL["ink2"], labelsize=9)
    ax.yaxis.grid(True, color=PAL["grid"], linewidth=0.8)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------
# 1. Data
# --------------------------------------------------------------------------
df = pd.read_csv(DATA_FILE)
df = df[df["recoded_dok_alignment"].notna()].copy()          # analytic sample, N = 1,362
if args.exclude_subject:
    n0 = len(df)
    df = df[~df["subject"].isin(args.exclude_subject)].copy()
    print(f"Excluded subjects {args.exclude_subject}: {n0 - len(df)} conversations dropped")
df["target"] = df["prompt_target_dok"].astype(int)
if args.turn_csv:
    tc = pd.read_csv(args.turn_csv)
    tmax = tc[tc["role"] == "student"].groupby("conversationId")["student_dok"].max()
    df = df.merge(tmax.rename("dok_reached_turn"), left_on="conversationId", right_index=True, how="left")
    n_missing = int(df["dok_reached_turn"].isna().sum())
    df = df[df["dok_reached_turn"].notna()].copy()
    df["dok_reached"] = df["dok_reached_turn"].astype(int)
    df["achieved"] = (df["dok_reached"] >= df["target"]).astype(int)
    print(f"Outcome: turn-level (max student DOK over turns); {n_missing} conversations without coded student turns dropped")
else:
    df["dok_reached"] = df["recoded_student_dok_max"].astype(int)
    df["achieved"] = df["recoded_dok_alignment"].isin(["aligned", "over_target"]).astype(int)
    print("Outcome: conversation-level (recoded_student_dok_max)")
df["high_specificity"] = (df["task_specificity"] == 2).astype(int)

raw_feats = [f for f in PROMPT_FEATURES if f not in ("prompt_target_dok", "task_specificity")]
act = df.groupby("discussionId").agg(
    **{f: (f, "first") for f in raw_feats + ["high_specificity"]},
    target=("target", "first"), teacher=("teacher_name", "first"),
    subject=("subject", "first"), n_conv=("conversationId", "size"),
    mean_dok=("dok_reached", "mean"), ach_rate=("achieved", "mean"),
)
for f in raw_feats:
    act[f] = act[f].fillna(0).astype(int)

prev = act[raw_feats + ["high_specificity"]].sum()
cluster_feats = [f for f in prev.index if MIN_PREV <= prev[f] <= MAX_PREV]
print(f"Activities: {len(act)}  Conversations: {len(df)}")
print(f"Features used for clustering ({len(cluster_feats)}): {cluster_feats}")

X = act[cluster_feats].values.astype(int)

# --------------------------------------------------------------------------
# 2. Clustering and k selection
# --------------------------------------------------------------------------
D_jac = squareform(pdist(X, metric="jaccard"))
D_ham = squareform(pdist(X, metric="hamming"))
sel_rows = []
km_labels, ag_labels = {}, {}
for k in K_RANGE:
    km = KModes(n_clusters=k, init="Cao", n_init=10, random_state=2026, verbose=0)
    km_labels[k] = km.fit_predict(X)
    ag = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
    ag_labels[k] = ag.fit_predict(D_jac)
    sel_rows.append({
        "k": k,
        "silhouette_kmodes_hamming": silhouette_score(D_ham, km_labels[k], metric="precomputed"),
        "silhouette_agglom_jaccard": silhouette_score(D_jac, ag_labels[k], metric="precomputed"),
        "ari_kmodes_vs_agglom": adjusted_rand_score(km_labels[k], ag_labels[k]),
        "min_cluster_size_kmodes": int(np.bincount(km_labels[k]).min()),
        "min_cluster_size_agglom": int(np.bincount(ag_labels[k]).min()),
        "min_teachers_per_cluster_kmodes": int(
            act.assign(c=km_labels[k]).groupby("c")["teacher"].nunique().min()),
    })
sel = pd.DataFrame(sel_rows)
sel.to_csv(outpath("rq1_cluster_selection.csv"), index=False)
print("\nCluster selection:\n", sel.round(3).to_string(index=False))

# choose k: the largest k for which every cluster has >= 5 activities authored by
# >= 3 teachers (so no pattern is one teacher's idiom); ties broken by silhouette.
# k = 2 (silhouette-optimal) is retained as a sensitivity solution.
ok = sel[(sel["min_cluster_size_kmodes"] >= MIN_CLUSTER_ACTIVITIES) &
         (sel["min_teachers_per_cluster_kmodes"] >= MIN_CLUSTER_TEACHERS)]
k_best = int(ok.sort_values(["k", "silhouette_kmodes_hamming"], ascending=False).iloc[0]["k"])
k_sens = int(sel.sort_values("silhouette_kmodes_hamming", ascending=False).iloc[0]["k"])
labels = km_labels[k_best]

# relabel patterns by descending mean number of features so "P1" is the leanest
order = pd.Series(X.sum(axis=1)).groupby(labels).mean().sort_values().index.tolist()
remap = {old: f"P{i+1}" for i, old in enumerate(order)}
act["pattern"] = [remap[l] for l in labels]
act["pattern_agglom_k"] = [f"A{l+1}" for l in ag_labels[k_best]]
act["pattern_sens_k"] = [f"S{l+1}" for l in km_labels[k_sens]]
act.to_csv(outpath("rq1_activity_features.csv"))
print(f"\nChosen k = {k_best} (sensitivity k = {k_sens}); pattern sizes: {act['pattern'].value_counts().sort_index().to_dict()}")

# --------------------------------------------------------------------------
# 3. Pattern profiles and outcomes
# --------------------------------------------------------------------------
prof = act.groupby("pattern")[cluster_feats].mean().T
prof.index = [FEATURE_LABELS.get(f, f) for f in prof.index]
prof["overall"] = act[cluster_feats].mean().values
prof.to_csv(outpath("rq1_pattern_profiles.csv"))
print("\nPattern profiles (share of activities with feature):\n", prof.round(2).to_string())

df = df.merge(act[["pattern"]], left_on="discussionId", right_index=True, how="left")
out = (df.groupby(["pattern", "target"])
         .agg(n_conv=("conversationId", "size"),
              n_act=("discussionId", "nunique"),
              n_teachers=("teacher_name", "nunique"),
              mean_dok=("dok_reached", "mean"),
              sd_dok=("dok_reached", "std"),
              ach_rate=("achieved", "mean"))
         .reset_index())
out_all = (df.groupby("pattern")
             .agg(n_conv=("conversationId", "size"), n_act=("discussionId", "nunique"),
                  n_teachers=("teacher_name", "nunique"), mean_dok=("dok_reached", "mean"),
                  sd_dok=("dok_reached", "std"), ach_rate=("achieved", "mean"))
             .reset_index().assign(target="all"))
out = pd.concat([out_all, out], ignore_index=True)
out["target_share_dok3"] = out["pattern"].map(
    act.groupby("pattern")["target"].apply(lambda s: (s == 3).mean()))
out.to_csv(outpath("rq1_pattern_outcomes.csv"), index=False)
print("\nOutcomes by pattern:\n", out.round(3).to_string(index=False))

# --------------------------------------------------------------------------
# 4. Models
# --------------------------------------------------------------------------
rows = []


def tidy(res, model, terms=None, exp=False):
    ci = res.conf_int()
    for t in res.params.index:
        if terms is not None and t not in terms:
            continue
        rows.append({"model": model, "term": t, "estimate": res.params[t],
                     "se": res.bse[t], "ci_low": ci.loc[t, 0], "ci_high": ci.loc[t, 1],
                     "p": res.pvalues[t],
                     "odds_ratio": np.exp(res.params[t]) if exp else np.nan})


df["target_c"] = df["target"] - 3          # centred at DOK 3 so intercept = P1 at DOK 3
df["teacher"] = df["teacher_name"].astype("category")
df["activity"] = df["discussionId"].astype(str)

# M1: linear mixed model, DOK reached
m1 = smf.mixedlm("dok_reached ~ C(pattern) + target_c", df, groups=df["teacher"],
                 re_formula="1", vc_formula={"activity": "0 + C(activity)"}).fit(reml=True)
tidy(m1, "M1 DOK reached, LMM (teacher + activity RI)")
print("\nM1\n", m1.summary().tables[1])

# M2: logit, goal achieved, cluster-robust by activity
d23 = df[df["target"] >= 2].copy()
m2 = smf.logit("achieved ~ C(pattern) + target_c", d23).fit(
    disp=0, cov_type="cluster", cov_kwds={"groups": d23["activity"]})
tidy(m2, "M2 goal achieved, logit, SE clustered by activity (DOK 2-3 targets)", exp=True)
print("\nM2\n", m2.summary().tables[1])

# M2b: DOK 3 stratum
# patterns with fewer than 20 conversations in the stratum are excluded (P3 has 3)
d3 = df[df["target"] == 3].copy()
keep = d3["pattern"].value_counts()
d3 = d3[d3["pattern"].isin(keep[keep >= 20].index)].copy()
print(f"\nM2b stratum: {len(d3)} conversations; patterns kept: {sorted(d3['pattern'].unique())}")
m2b = smf.logit("achieved ~ C(pattern)", d3).fit(
    disp=0, cov_type="cluster", cov_kwds={"groups": d3["activity"]})
tidy(m2b, "M2b goal achieved, logit, DOK 3 stratum", exp=True)
print("\nM2b\n", m2b.summary().tables[1])

# Wald test of the pattern block in M2 and M2b
for name, res in [("M2", m2), ("M2b", m2b)]:
    pat_terms = [t for t in res.params.index if t.startswith("C(pattern)")]
    w = res.wald_test(" = ".join(pat_terms) + " = 0" if len(pat_terms) == 1
                      else ", ".join(f"{t} = 0" for t in pat_terms), scalar=True)
    rows.append({"model": f"{name} Wald test, pattern block", "term": "all pattern contrasts",
                 "estimate": float(w.statistic), "se": np.nan, "ci_low": np.nan,
                 "ci_high": np.nan, "p": float(w.pvalue), "odds_ratio": np.nan})
    print(f"{name} Wald pattern block: stat = {float(w.statistic):.2f}, p = {float(w.pvalue):.3f}")

# M2s: sensitivity, silhouette-optimal k
df = df.merge(act[["pattern_sens_k"]], left_on="discussionId", right_index=True, how="left")
d23 = df[df["target"] >= 2].copy()
m2s = smf.logit("achieved ~ C(pattern_sens_k) + target_c", d23).fit(
    disp=0, cov_type="cluster", cov_kwds={"groups": d23["activity"]})
tidy(m2s, f"M2s goal achieved, logit, sensitivity solution k = {k_sens}", exp=True)
prof_s = act.groupby("pattern_sens_k")[cluster_feats].mean().T
prof_s.index = [FEATURE_LABELS.get(f, f) for f in prof_s.index]
prof_s.to_csv(outpath("rq1_pattern_profiles_sensitivity.csv"))

# M3: Bayesian binomial mixed GLM, random intercepts for teacher and activity
try:
    m3 = BinomialBayesMixedGLM.from_formula(
        "achieved ~ C(pattern) + target_c",
        {"teacher": "0 + C(teacher)", "activity": "0 + C(activity)"}, d23).fit_vb()
    names = list(m3.model.exog_names)
    for i, t in enumerate(names):
        est, sd = m3.fe_mean[i], m3.fe_sd[i]
        rows.append({"model": "M3 goal achieved, Bayesian mixed GLM (teacher + activity RI)",
                     "term": t, "estimate": est, "se": sd, "ci_low": est - 1.96 * sd,
                     "ci_high": est + 1.96 * sd,
                     "p": 2 * (1 - stats.norm.cdf(abs(est / sd))), "odds_ratio": np.exp(est)})
    for j, vn in enumerate(m3.model.vcp_names):
        rows.append({"model": "M3 variance components (log SD)", "term": vn,
                     "estimate": m3.vcp_mean[j], "se": m3.vcp_sd[j], "ci_low": np.nan,
                     "ci_high": np.nan, "p": np.nan, "odds_ratio": np.nan})
    print("\nM3 fixed effects:\n", pd.DataFrame({"term": names, "mean": m3.fe_mean, "sd": m3.fe_sd}).round(3))
except Exception as e:                                                   # pragma: no cover
    print("M3 failed:", e)

# M4: feature-count models (clustering-free)
for fam, cols in FAMILY.items():
    df[f"n_{fam}"] = df[[c for c in cols if c in df.columns]].fillna(0).sum(axis=1)
d23 = df[df["target"] >= 2].copy()
m4a = smf.ols("dok_reached ~ n_scaffold + n_epistemic + n_constraint + n_guardrail + high_specificity + finish_line + target_c",
              df).fit(cov_type="cluster", cov_kwds={"groups": df["activity"]})
tidy(m4a, "M4a DOK reached, OLS on feature counts, SE clustered by activity")
m4b = smf.logit("achieved ~ n_scaffold + n_epistemic + n_constraint + n_guardrail + high_specificity + finish_line + target_c",
                d23).fit(disp=0, cov_type="cluster", cov_kwds={"groups": d23["activity"]})
tidy(m4b, "M4b goal achieved, logit on feature counts, SE clustered by activity", exp=True)
print("\nM4a\n", m4a.summary().tables[1], "\nM4b\n", m4b.summary().tables[1])

models = pd.DataFrame(rows)
models.to_csv(outpath("rq1_models.csv"), index=False)

# --------------------------------------------------------------------------
# 5. Minimum detectable effects for pattern contrasts (80% power, two-sided .05)
# --------------------------------------------------------------------------
n_act = act["pattern"].nunique() + 1                    # parameters: patterns + target
df_res = act.shape[0] - n_act
mult = stats.t.ppf(0.975, df_res) + stats.t.ppf(0.80, df_res)
mde_rows = []
for t in m1.bse.index:
    if t.startswith("C(pattern)"):
        mde_rows.append({"model": "M1 DOK reached", "term": t, "se": m1.bse[t],
                         "mde_dok_levels": mult * m1.bse[t]})
for t in m2.bse.index:
    if t.startswith("C(pattern)"):
        mde_rows.append({"model": "M2 goal achieved (log odds)", "term": t, "se": m2.bse[t],
                         "mde_log_odds": mult * m2.bse[t], "mde_odds_ratio": np.exp(mult * m2.bse[t])})
mde = pd.DataFrame(mde_rows)
mde.to_csv(outpath("rq1_mde.csv"), index=False)
print(f"\nMDE multiplier (df = {df_res}): {mult:.3f}\n", mde.round(3).to_string(index=False))

# --------------------------------------------------------------------------
# 5b. Discipline split (ELA versus STEM/CTE)
# --------------------------------------------------------------------------
if args.discipline_split:
    df["discipline"] = np.where(df["subject"] == "ELA", "ELA", "STEM")
    act["discipline"] = np.where(act["subject"] == "ELA", "ELA", "STEM")
    print("\nDiscipline split. Activities by pattern and discipline:\n", pd.crosstab(act["pattern"], act["discipline"]))
    dout = (df.groupby(["discipline", "pattern", "target"])
              .agg(n_conv=("conversationId", "size"), n_act=("discussionId", "nunique"),
                   n_teachers=("teacher_name", "nunique"), mean_dok=("dok_reached", "mean"),
                   ach_rate=("achieved", "mean")).reset_index())
    dall = (df.groupby(["discipline", "target"])
              .agg(n_conv=("conversationId", "size"), n_act=("discussionId", "nunique"),
                   n_teachers=("teacher_name", "nunique"), mean_dok=("dok_reached", "mean"),
                   ach_rate=("achieved", "mean")).reset_index().assign(pattern="all"))
    dout = pd.concat([dall, dout], ignore_index=True)
    dout.to_csv(outpath("rq1_discipline_outcomes.csv"), index=False)
    print("\nOutcomes by discipline, pattern and target:\n", dout.round(3).to_string(index=False))
    drows = []
    d23 = df[df["target"] >= 2].copy()
    # interaction model
    mi = smf.logit("achieved ~ C(pattern) * C(discipline) + target_c", d23).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": d23["activity"]})
    ci = mi.conf_int()
    for t in mi.params.index:
        drows.append({"model": "MD1 goal achieved, pattern x discipline interaction, SE clustered by activity",
                      "term": t, "estimate": mi.params[t], "se": mi.bse[t], "ci_low": ci.loc[t, 0],
                      "ci_high": ci.loc[t, 1], "p": mi.pvalues[t], "odds_ratio": np.exp(mi.params[t])})
    inter = [t for t in mi.params.index if ":" in t]
    w = mi.wald_test(", ".join(f"{t} = 0" for t in inter), scalar=True)
    drows.append({"model": "MD1 Wald test, interaction block", "term": "pattern x discipline",
                  "estimate": float(w.statistic), "se": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                  "p": float(w.pvalue), "odds_ratio": np.nan})
    print(f"\nMD1 interaction Wald: stat = {float(w.statistic):.2f}, p = {float(w.pvalue):.3f}")
    # discipline main effect model
    md = smf.logit("achieved ~ C(pattern) + C(discipline) + target_c", d23).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": d23["activity"]})
    ci = md.conf_int()
    for t in md.params.index:
        drows.append({"model": "MD2 goal achieved, pattern + discipline, SE clustered by activity",
                      "term": t, "estimate": md.params[t], "se": md.bse[t], "ci_low": ci.loc[t, 0],
                      "ci_high": ci.loc[t, 1], "p": md.pvalues[t], "odds_ratio": np.exp(md.params[t])})
    md1 = smf.mixedlm("dok_reached ~ C(pattern) + C(discipline) + target_c", df, groups=df["teacher"],
                      re_formula="1", vc_formula={"activity": "0 + C(activity)"}).fit(reml=True)
    ci = md1.conf_int()
    for t in md1.params.index:
        drows.append({"model": "MD3 DOK reached, LMM, pattern + discipline", "term": t,
                      "estimate": md1.params[t], "se": md1.bse[t], "ci_low": ci.loc[t, 0],
                      "ci_high": ci.loc[t, 1], "p": md1.pvalues[t], "odds_ratio": np.nan})
    print("\nMD2\n", md.summary().tables[1], "\nMD3\n", md1.summary().tables[1])
    # stratified models
    for disc in ["ELA", "STEM"]:
        sub = d23[d23["discipline"] == disc]
        subf = df[df["discipline"] == disc]
        keep = sub["pattern"].value_counts()
        sub = sub[sub["pattern"].isin(keep[keep >= 20].index)]
        try:
            ms = smf.logit("achieved ~ C(pattern) + target_c", sub).fit(
                disp=0, cov_type="cluster", cov_kwds={"groups": sub["activity"]})
            ci = ms.conf_int()
            for t in ms.params.index:
                drows.append({"model": f"MD4 goal achieved, {disc} only, SE clustered by activity (n = {len(sub)})",
                              "term": t, "estimate": ms.params[t], "se": ms.bse[t], "ci_low": ci.loc[t, 0],
                              "ci_high": ci.loc[t, 1], "p": ms.pvalues[t], "odds_ratio": np.exp(ms.params[t])})
            print(f"\nMD4 {disc}\n", ms.summary().tables[1])
        except Exception as e:
            print(f"MD4 {disc} failed: {e}")
        try:
            m4 = smf.logit("achieved ~ n_scaffold + n_epistemic + n_constraint + n_guardrail + high_specificity + finish_line + target_c",
                           sub).fit(disp=0, cov_type="cluster", cov_kwds={"groups": sub["activity"]})
            ci = m4.conf_int()
            for t in m4.params.index:
                drows.append({"model": f"MD5 goal achieved, feature counts, {disc} only (n = {len(sub)})",
                              "term": t, "estimate": m4.params[t], "se": m4.bse[t], "ci_low": ci.loc[t, 0],
                              "ci_high": ci.loc[t, 1], "p": m4.pvalues[t], "odds_ratio": np.exp(m4.params[t])})
            print(f"\nMD5 {disc}\n", m4.summary().tables[1])
        except Exception as e:
            print(f"MD5 {disc} failed: {e}")
    pd.DataFrame(drows).to_csv(outpath("rq1_discipline_models.csv"), index=False)

# --------------------------------------------------------------------------
# 6. Figures
# --------------------------------------------------------------------------
patterns = sorted(act["pattern"].unique())
pcol = dict(zip(patterns, PATTERN_COLORS))

# 6a. Pattern profile heatmap (one hue, light -> dark)
fig, ax = plt.subplots(figsize=(6.2, 5.2))
mat = prof[patterns].values
im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(patterns)))
ax.set_xticklabels([f"{p}\n(n={int((act['pattern']==p).sum())})" for p in patterns], fontsize=9)
ax.set_yticks(range(len(prof.index)))
ax.set_yticklabels(prof.index, fontsize=9)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        v = mat[i, j]
        ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                color="white" if v > 0.6 else PAL["ink"])
ax.set_title("Prompt feature prevalence within each pattern", fontsize=11, loc="left", color=PAL["ink"])
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("Share of activities with feature", fontsize=8, color=PAL["ink2"])
cb.ax.tick_params(labelsize=8, colors=PAL["ink2"])
fig.tight_layout()
fig.savefig(figpath("fig_rq1_pattern_profiles.png"), dpi=300)
plt.close(fig)

# 6b. Achievement rate by pattern within target DOK 2 and DOK 3, with Wilson CIs
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
for ax, tgt in zip(axes, [2, 3]):
    sub = df[df["target"] == tgt]
    xs, ys, lo, hi, ns = [], [], [], [], []
    for i, p in enumerate(patterns):
        s = sub[sub["pattern"] == p]["achieved"]
        n = len(s)
        if n < 20:                       # too few conversations to display a rate
            continue
        k = s.sum()
        ci = sm.stats.proportion_confint(k, n, method="wilson")
        xs.append(i); ys.append(k / n); lo.append(ci[0]); hi.append(ci[1]); ns.append(n)
    ax.bar(xs, ys, width=0.62, color=[pcol[patterns[i]] for i in xs], linewidth=0)
    ax.errorbar(xs, ys, yerr=[np.array(ys) - np.array(lo), np.array(hi) - np.array(ys)],
                fmt="none", ecolor=PAL["ink2"], elinewidth=1, capsize=2)
    for x, y, h, n in zip(xs, ys, hi, ns):
        ax.text(x, h + 0.02, f"{y:.0%}\nn={n}", ha="center", va="bottom", fontsize=7.5, color=PAL["ink2"])
    ax.set_xticks(range(len(patterns)))
    ax.set_xticklabels(patterns, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Target DOK {tgt}", fontsize=10, loc="left", color=PAL["ink"])
    style_axes(ax)
axes[0].set_ylabel("Share of conversations reaching target", fontsize=9, color=PAL["ink2"])
fig.suptitle("Goal achievement by prompt pattern (Wilson 95% CIs; cells with n < 20 omitted)", fontsize=11, x=0.01, ha="left", color=PAL["ink"])
fig.tight_layout()
fig.savefig(figpath("fig_rq1_achievement_by_pattern.png"), dpi=300)
plt.close(fig)

# 6c. Forest plot: pattern contrasts (vs P1) in M1 and M2
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
for ax, (res, title, xlab) in zip(axes, [
        (m1, "DOK reached (M1, difference in levels)", "Difference vs P1 (DOK levels)"),
        (m2, "Goal achieved (M2, log odds)", "Log odds vs P1")]):
    terms = [t for t in res.params.index if t.startswith("C(pattern)")]
    ci = res.conf_int().loc[terms]
    y = np.arange(len(terms))[::-1]
    est = res.params[terms].values
    sig = res.pvalues[terms].values < 0.05
    ax.axvline(0, color=PAL["ink2"], linewidth=1)
    ax.hlines(y, ci[0].values, ci[1].values, color=PAL["ink2"], linewidth=1.5)
    ax.scatter(est, y, s=42, color=[PAL["blue"] if s else "#ffffff" for s in sig],
               edgecolor=PAL["blue"], linewidth=1.5, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([t.replace("C(pattern)[T.", "").rstrip("]") + " vs P1" for t in terms], fontsize=9)
    ax.set_xlabel(xlab, fontsize=9, color=PAL["ink2"])
    ax.set_title(title, fontsize=10, loc="left", color=PAL["ink"])
    style_axes(ax)
    ax.yaxis.grid(False); ax.xaxis.grid(True, color=PAL["grid"], linewidth=0.8)
fig.tight_layout()
fig.savefig(figpath("fig_rq1_forest.png"), dpi=300)
plt.close(fig)

print("\nDone. Outputs written to", OUTPUT_DIR)
