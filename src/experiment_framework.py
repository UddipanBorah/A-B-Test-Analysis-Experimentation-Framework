"""
experiment_framework.py
------------------------
A small, reusable toolkit for analyzing A/B tests on binary (conversion /
retention) metrics. Implements the statistics used throughout this project:

  * Power analysis / required sample size for a two-proportion test
  * Sample Ratio Mismatch (SRM) check (chi-square goodness of fit)
  * Two-proportion z-test
  * Wald confidence interval on the absolute and relative lift
  * Cohen's h effect size for two proportions

These are written from first principles (rather than only calling
statsmodels) so the math is transparent, and cross-checked against
statsmodels where an implementation exists.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest, proportion_confint


# ---------------------------------------------------------------------------
# 1. Power analysis / required sample size
# ---------------------------------------------------------------------------

def required_sample_size(baseline_rate: float, mde_abs: float, alpha: float = 0.05,
                          power: float = 0.8, ratio: float = 1.0) -> dict:
    """
    Required sample size PER GROUP to detect an absolute lift of `mde_abs`
    over a `baseline_rate`, for a two-sided two-proportion z-test.

    Uses Cohen's h (variance-stabilizing arcsine transform) so the
    calculation is valid across the full range of baseline rates, not just
    those near 50%.

    Parameters
    ----------
    baseline_rate : control group's expected conversion/retention rate (0-1)
    mde_abs        : minimum detectable absolute effect, e.g. 0.01 = 1pp
    alpha          : significance level (two-sided)
    power          : desired statistical power (1 - beta)
    ratio          : ratio of treatment n to control n (1.0 = equal groups)

    Returns
    -------
    dict with n_per_group (control), n_treatment, n_total, and Cohen's h
    """
    p1 = baseline_rate
    p2 = baseline_rate + mde_abs
    h = 2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(p1))

    analysis = NormalIndPower()
    n_control = analysis.solve_power(effect_size=abs(h), alpha=alpha, power=power,
                                      ratio=ratio, alternative='two-sided')
    n_control = math.ceil(n_control)
    n_treatment = math.ceil(n_control * ratio)

    return {
        "baseline_rate": p1,
        "target_rate": p2,
        "mde_abs": mde_abs,
        "cohens_h": h,
        "alpha": alpha,
        "power": power,
        "n_per_group": n_control,
        "n_treatment": n_treatment,
        "n_total": n_control + n_treatment,
    }


def minimum_detectable_effect(baseline_rate: float, n_per_group: int,
                               alpha: float = 0.05, power: float = 0.8) -> float:
    """Given a fixed sample size per group, back out the smallest absolute
    effect the design could reliably detect (inverse of required_sample_size)."""
    analysis = NormalIndPower()
    h = analysis.solve_power(nobs1=n_per_group, alpha=alpha, power=power,
                              ratio=1.0, alternative='two-sided')
    # invert Cohen's h back to an absolute rate difference around baseline_rate
    phi1 = 2 * math.asin(math.sqrt(baseline_rate))
    phi2 = phi1 + h
    p2 = math.sin(phi2 / 2) ** 2
    return p2 - baseline_rate


# ---------------------------------------------------------------------------
# 2. Sample Ratio Mismatch (SRM) check
# ---------------------------------------------------------------------------

def srm_check(n_control: int, n_treatment: int, expected_ratio: float = 0.5,
              srm_alpha: float = 0.0001) -> dict:
    """
    Chi-square goodness-of-fit test for whether the observed split between
    control and treatment matches the intended randomization ratio.

    The significance threshold for SRM is deliberately much stricter than a
    normal hypothesis test (0.0001 rather than 0.05), following the standard
    used by large-scale experimentation platforms (e.g. Microsoft's ExP
    guidance, Fabijan et al. 2019). Rationale: the chi-square SRM test is
    run on every single experiment an org ships, and with large sample
    sizes even a trivially small, practically meaningless randomization
    imbalance becomes "significant" at p < 0.05 -- that would flag most
    healthy experiments as broken. A p < 0.0001 threshold keeps the false
    positive rate low while still catching real randomization/logging bugs,
    which tend to produce much larger ratio deviations.
    """
    n_total = n_control + n_treatment
    expected_control = n_total * expected_ratio
    expected_treatment = n_total * (1 - expected_ratio)
    chi2, p_value = stats.chisquare(
        f_obs=[n_control, n_treatment],
        f_exp=[expected_control, expected_treatment],
    )
    return {
        "n_control": n_control,
        "n_treatment": n_treatment,
        "observed_ratio": n_control / n_total,
        "expected_ratio": expected_ratio,
        "chi2": chi2,
        "p_value": p_value,
        "srm_alpha": srm_alpha,
        "srm_detected": p_value < srm_alpha,
    }


# ---------------------------------------------------------------------------
# 3. Two-proportion z-test + confidence interval on the lift
# ---------------------------------------------------------------------------

@dataclass
class ProportionTestResult:
    metric: str
    n_control: int
    n_treatment: int
    rate_control: float
    rate_treatment: float
    abs_lift: float
    rel_lift_pct: float
    z_stat: float
    p_value: float
    ci_abs_low: float
    ci_abs_high: float
    ci_rel_low_pct: float
    ci_rel_high_pct: float
    cohens_h: float
    significant: bool
    alpha: float

    def summary(self) -> str:
        sig = "YES" if self.significant else "no"
        return (
            f"[{self.metric}] control={self.rate_control:.4%} (n={self.n_control:,})  "
            f"treatment={self.rate_treatment:.4%} (n={self.n_treatment:,})\n"
            f"  abs lift = {self.abs_lift:+.4%}  "
            f"95% CI [{self.ci_abs_low:+.4%}, {self.ci_abs_high:+.4%}]\n"
            f"  rel lift = {self.rel_lift_pct:+.2f}%  "
            f"95% CI [{self.ci_rel_low_pct:+.2f}%, {self.ci_rel_high_pct:+.2f}%]\n"
            f"  z = {self.z_stat:.3f}   p = {self.p_value:.4g}   "
            f"Cohen's h = {self.cohens_h:.4f}   significant at alpha={self.alpha}: {sig}"
        )


def two_proportion_ztest(successes_control: int, n_control: int,
                          successes_treatment: int, n_treatment: int,
                          metric_name: str = "metric", alpha: float = 0.05) -> ProportionTestResult:
    """
    Two-sided two-proportion z-test comparing a treatment group against a
    control group, with a Wald 95% CI on both the absolute and relative
    lift, and Cohen's h effect size.
    """
    p_c = successes_control / n_control
    p_t = successes_treatment / n_treatment

    # z-test (pooled variance under H0, statsmodels implementation)
    count = np.array([successes_treatment, successes_control])
    nobs = np.array([n_treatment, n_control])
    z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')

    # Wald CI on the absolute difference (unpooled variance, standard for CI)
    abs_lift = p_t - p_c
    se_diff = math.sqrt(p_c * (1 - p_c) / n_control + p_t * (1 - p_t) / n_treatment)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_low = abs_lift - z_crit * se_diff
    ci_high = abs_lift + z_crit * se_diff

    rel_lift_pct = (p_t - p_c) / p_c * 100
    ci_rel_low_pct = ci_low / p_c * 100
    ci_rel_high_pct = ci_high / p_c * 100

    h = 2 * math.asin(math.sqrt(p_t)) - 2 * math.asin(math.sqrt(p_c))

    return ProportionTestResult(
        metric=metric_name,
        n_control=n_control, n_treatment=n_treatment,
        rate_control=p_c, rate_treatment=p_t,
        abs_lift=abs_lift, rel_lift_pct=rel_lift_pct,
        z_stat=z_stat, p_value=p_value,
        ci_abs_low=ci_low, ci_abs_high=ci_high,
        ci_rel_low_pct=ci_rel_low_pct, ci_rel_high_pct=ci_rel_high_pct,
        cohens_h=h, significant=p_value < alpha, alpha=alpha,
    )


# ---------------------------------------------------------------------------
# 4. Continuous-metric test (Welch's t-test) with bootstrap CI on the mean diff
# ---------------------------------------------------------------------------

def welch_ttest_with_bootstrap_ci(control: np.ndarray, treatment: np.ndarray,
                                   metric_name: str = "metric", alpha: float = 0.05,
                                   n_boot: int = 10000, seed: int = 42) -> dict:
    """
    Welch's t-test (does not assume equal variances) for a continuous metric,
    plus a percentile bootstrap CI on the difference in means -- more robust
    than the normal-theory CI when the metric is heavily skewed (as
    engagement / count metrics typically are).
    """
    t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)

    rng = np.random.default_rng(seed)
    boot_diffs = np.empty(n_boot)
    nc, nt = len(control), len(treatment)
    for i in range(n_boot):
        c_sample = control[rng.integers(0, nc, nc)]
        t_sample = treatment[rng.integers(0, nt, nt)]
        boot_diffs[i] = t_sample.mean() - c_sample.mean()

    ci_low, ci_high = np.percentile(boot_diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return {
        "metric": metric_name,
        "mean_control": control.mean(),
        "mean_treatment": treatment.mean(),
        "median_control": np.median(control),
        "median_treatment": np.median(treatment),
        "diff_means": treatment.mean() - control.mean(),
        "t_stat": t_stat,
        "p_value": p_value,
        "bootstrap_ci_low": ci_low,
        "bootstrap_ci_high": ci_high,
        "significant": p_value < alpha,
    }
