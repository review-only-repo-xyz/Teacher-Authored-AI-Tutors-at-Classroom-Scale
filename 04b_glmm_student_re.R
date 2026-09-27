# REPLICATION VERSION.
# Frequentist GLMMs with student random intercepts. Reads the de-identified model frames
# written by 04_revision_analyses.py (section D) and writes outputs/rev_D_glmm_student_re.csv.
# Laplace approximation (lme4::glmer), Wald CIs. Requires R >= 4.1 and lme4.
#
#   Rscript 04b_glmm_student_re.R
suppressPackageStartupMessages(library(lme4))
args <- commandArgs(trailingOnly = FALSE)
here <- dirname(normalizePath(sub("--file=", "", args[grep("--file=", args)])))
out_dir <- file.path(here, "outputs")

cv <- read.csv(file.path(out_dir, "rev_D_modeldata_conversations.csv"), stringsAsFactors = TRUE)
pp <- read.csv(file.path(out_dir, "rev_D_modeldata_person_period.csv"), stringsAsFactors = TRUE)
cv$pattern <- relevel(factor(cv$pattern), ref = "P1")
pp$pattern <- relevel(factor(pp$pattern), ref = "P1")
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

tidy <- function(fit, label, lmm = FALSE) {
  s <- summary(fit)$coefficients
  est <- s[, 1]; se <- s[, 2]
  p <- if (lmm) 2 * pnorm(-abs(est / se)) else s[, 4]
  vc <- as.data.frame(VarCorr(fit))
  data.frame(model = label, term = rownames(s), estimate = est, se = se,
             effect = if (lmm) est else exp(est),
             ci_low = if (lmm) est - 1.96 * se else exp(est - 1.96 * se),
             ci_high = if (lmm) est + 1.96 * se else exp(est + 1.96 * se), p = p,
             n_obs = nobs(fit),
             re_sd = paste(sprintf("%s %.3f", vc$grp, vc$sdcor), collapse = "; "),
             singular = isSingular(fit),
             converged = is.null(fit@optinfo$conv$lme4$messages),
             row.names = NULL)
}

res <- list()
# M1 DOK reached, LMM with teacher, activity and student random intercepts
res[[1]] <- tidy(lmer(dok_reached ~ pattern + target_c + (1 | teacher) + (1 | activity) + (1 | student),
                      data = cv, REML = TRUE), "M1 LMM, teacher + activity + student RI (lme4)", lmm = TRUE)
res[[2]] <- tidy(lmer(dok_reached ~ pattern + target_c + (1 | teacher) + (1 | activity),
                      data = cv, REML = TRUE), "M1 LMM, teacher + activity RI (lme4, original structure)", lmm = TRUE)
# M2 goal achieved, GLMM
d23 <- subset(cv, target >= 2)
res[[3]] <- tidy(glmer(achieved ~ pattern + target_c + (1 | teacher) + (1 | activity) + (1 | student),
                       data = d23, family = binomial, control = ctrl),
                 "M2 GLMM, teacher + activity + student RI")
res[[4]] <- tidy(glmer(achieved ~ pattern + target_c + (1 | teacher) + (1 | activity),
                       data = d23, family = binomial, control = ctrl),
                 "M2 GLMM, teacher + activity RI")
m2full <- glmer(achieved ~ pattern + target_c + (1 | teacher) + (1 | activity) + (1 | student),
                data = d23, family = binomial, control = ctrl)
m2null <- update(m2full, . ~ . - pattern)
lrt <- anova(m2null, m2full)
res[[5]] <- data.frame(model = "M2 GLMM with student RI, likelihood-ratio test of pattern block",
                       term = "pattern", estimate = lrt$Chisq[2], se = NA, effect = NA, ci_low = NA, ci_high = NA,
                       p = lrt$`Pr(>Chisq)`[2], n_obs = nobs(m2full), re_sd = NA, singular = NA, converged = NA)
# Hazard, pattern specification, activity + student random intercepts
moves <- grep("^prev_", names(pp), value = TRUE)
moves <- setdiff(moves, "prev_dok_state")
f_pat <- as.formula(paste("event ~ log_k + log_len + pattern +", paste(moves, collapse = " + "),
                          "+ (1 | activity) + (1 | student)"))
res[[6]] <- tidy(glmer(f_pat, data = pp, family = binomial, control = ctrl),
                 "Hazard, pattern, GLMM with activity + student RI")
# Hazard within the mixed-outcome activities, activity + student random intercepts.
# (A glmer with 34 activity dummies plus a student intercept does not converge; the fixed-effects
#  version with two-way activity x student clustered SEs is in 04_revision_analyses.py, section D.)
ppm <- subset(pp, mixed == 1)
f_m <- as.formula(paste("event ~ log_k + log_len +", paste(moves, collapse = " + "), "+ (1 | activity) + (1 | student)"))
res[[7]] <- tidy(glmer(f_m, data = ppm, family = binomial, control = ctrl),
                 "Hazard, mixed-outcome activities, GLMM with activity + student RI")
f_m2 <- as.formula(paste("event ~ log_k +", paste(moves, collapse = " + "), "+ (1 | activity) + (1 | student)"))
res[[8]] <- tidy(glmer(f_m2, data = ppm, family = binomial, control = ctrl),
                 "Hazard, mixed-outcome activities, GLMM with activity + student RI, without log length")

# Corrected risk set: student turn 1 is the platform's "Start Classroom Discussion" button (log_k = 0)
pp2 <- subset(pp, log_k > 0); ppm2 <- subset(pp2, mixed == 1)
res[[9]] <- tidy(glmer(f_pat, data = pp2, family = binomial, control = ctrl),
                 "Hazard, pattern, GLMM with activity + student RI, turns 2+")
res[[10]] <- tidy(glmer(f_m, data = ppm2, family = binomial, control = ctrl),
                  "Hazard, mixed-outcome activities, GLMM with activity + student RI, turns 2+")
f_ms <- as.formula(paste("event ~ log_k + log_len + prev_dok_state +", paste(moves, collapse = " + "),
                         "+ (1 | activity) + (1 | student)"))
ppm2$prev_dok_state <- relevel(factor(ppm2$prev_dok_state), ref = "D1")
res[[11]] <- tidy(glmer(f_ms, data = ppm2, family = binomial, control = ctrl),
                  "Hazard, mixed-outcome activities, GLMM with activity + student RI, turns 2+, + prior student DOK")

out <- do.call(rbind, res)
write.csv(out, file.path(out_dir, "rev_D_glmm_student_re.csv"), row.names = FALSE)
keep <- out$term %in% c("patternP2", "patternP3", "target_c", "prev_socratic_question", "prev_request_evidence", "pattern")
print(out[keep, c("model", "term", "effect", "ci_low", "ci_high", "p", "converged")], digits = 3, row.names = FALSE)
cat("\nRandom-effect SDs:\n"); print(unique(out[, c("model", "re_sd")]), row.names = FALSE)
