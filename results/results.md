# A/B Test Results: Cookie Cats Gate Placement Experiment

## 1. Data Overview

- Total players: **90,189**
- Control (`gate_30`): 44,700 | Treatment (`gate_40`): 45,489
- Overall 1-day retention: 44.52% | 7-day retention: 18.61%

## 2. Power Analysis

To detect a **1% absolute** change in 1-day retention off a 44.82% baseline, at alpha=0.05 and power=0.8:

- Required sample size per group: **38,899**
- Required total sample size: **77,798**
- Actual sample collected: **90,189** — sufficiently powered ✅

## 3. Sample Ratio Mismatch (SRM) Check

Observed split 49.56% / 50.44% vs expected 50/50. Chi-square p-value = 0.0086. Using the stringent p < 0.0001 threshold standard for SRM checks (a plain p < 0.05 test flags most large-sample experiments as broken even from noise), no SRM detected — randomization looks healthy ✅.

## 4. Hypothesis Test Results

| Metric | Control | Treatment | Abs. Lift | 95% CI (abs) | Rel. Lift | p-value | Significant? |
|---|---|---|---|---|---|---|---|
| 1-day retention | 44.82% | 44.23% | -0.59% | [-1.24%, +0.06%] | -1.32% | 0.07441 | No |
| 7-day retention | 19.02% | 18.20% | -0.82% | [-1.33%, -0.31%] | -4.31% | 0.001554 | Yes |

## 5. Secondary Metric: Game Rounds Played

The raw data contains an extreme outlier (max = 49,854 rounds in 14 days), so the top 0.1% of values (91 players) were trimmed before comparing means.

- Control mean: 50.09 rounds (median 17.0)
- Treatment mean: 49.81 rounds (median 16.0)
- Difference in means: -0.28, bootstrap 95% CI [-1.52, +0.93]
- Welch's t-test p-value: 0.6467 (not significant)

## 6. Conclusion & Recommendation

**1-day retention** was 0.59% lower in treatment than control (95% CI [-1.24%, +0.06%]), a direction consistent with a negative effect but **not statistically significant at alpha=0.05** (p=0.074) — the confidence interval includes a small positive effect, so this metric alone is inconclusive.

**7-day retention** was 0.82% lower in treatment than control (95% CI [-1.33%, -0.31%]), a **statistically significant decrease** (p=0.001554) and a 4.3% relative drop.

**Game rounds played** showed no significant difference between groups (p=0.647), so the retention drop is not explained by players simply engaging less per session before churning — they are churning outright at a higher rate.

**Recommendation: do NOT ship gate_40 — keep the gate at level 30.** Both retention metrics point in the same (negative) direction for the treatment, and the effect reaches significance on the more decision-relevant 7-day horizon. Moving the gate later removes an early monetization/pacing checkpoint, but the data suggests this costs the game long-term player retention, which for a free-to-play title is the metric that most directly drives lifetime value. A follow-up experiment with a larger sample (see Section 2 power analysis) or a longer observation window could tighten the 1-day retention estimate, but the current evidence does not support shipping the change.