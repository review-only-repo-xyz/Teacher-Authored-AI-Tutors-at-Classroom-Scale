"""
Run the analysis pipeline on the de-identified coded data in data/.

Order
    01_descriptives.py              sample summary
    02_prompt_patterns.py           RQ1 (patterns, mixed models, MDEs)
    03_turn_level_analysis.py       RQ2 (timing, persistence, networks, mining)
    04_revision_analyses.py         reliability, robustness and sensitivity analyses
    04b_glmm_student_re.R           GLMMs with student random intercepts (run if Rscript
                                    and lme4 are available; skipped otherwise)

All outputs land in outputs/. Expect a few minutes end to end. The LLM
coding scripts in coding/ are not part of this run (they need an API key
and operate on the synthetic examples); see README.md.
"""
import shutil
import subprocess
import sys

STEPS = [
    [sys.executable, "01_descriptives.py", "--exclude-subject", "World Language"],
    [sys.executable, "02_prompt_patterns.py",
     "--turn-csv", "data/turn_codes.csv",
     "--exclude-subject", "World Language"],
    [sys.executable, "03_turn_level_analysis.py",
     "data/turn_codes.csv",
     "--exclude-subject", "World Language"],
    [sys.executable, "04_revision_analyses.py"],
]

for step in STEPS:
    print("\n=== " + " ".join(step[1:]) + " ===")
    r = subprocess.run(step)
    if r.returncode != 0:
        sys.exit(f"Step failed: {' '.join(step[1:])}")
if shutil.which("Rscript"):
    print("\n=== 04b_glmm_student_re.R ===")
    r = subprocess.run(["Rscript", "04b_glmm_student_re.R"])
    if r.returncode != 0:
        sys.exit("Step failed: 04b_glmm_student_re.R")
else:
    print("\nRscript not found; skipping 04b_glmm_student_re.R (student random-intercept GLMMs).")
print("\nPipeline complete. See outputs/ and outputs/figures/.")
