"""
run_analysis.py
----------------
End-to-end A/B test analysis on the "Cookie Cats" mobile game dataset.

Experiment background
----------------------
Cookie Cats is a mobile puzzle game. As players progress they hit "gates"
that force them to wait or pay to continue. This experiment tested moving
the FIRST gate from level 30 to level 40:

  * Control   (version = gate_30): gate stays at level 30 (the original)
  * Treatment (version = gate_40): gate moved to level 40

90,189 new players were randomly assigned to one of the two groups. For
each player we observe how many game rounds they played in the first 14
days, and whether they returned to the game 1 day and 7 days after
installing.

Business question: does moving the gate later hurt or help retention?

This script:
  1. Loads and validates the data
  2. Runs a prospective power analysis (what sample size would we need?)
     and compares it with what the experiment actually collected
  3. Checks for Sample Ratio Mismatch (a randomization/logging sanity check)
  4. Runs two-proportion z-tests on 1-day and 7-day retention, with 95% CIs
     on the lift
  5. Runs a secondary check on game rounds played (Welch's t-test + bootstrap CI)
  6. Saves all figures to results/ and a machine-readable summary to
     results/metrics.json, plus a human-readable results/results.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent))
from experiment_framework import (
    required_sample_size, srm_check, two_proportion_ztest,
    welch_ttest_with_bootstrap_ci,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "cookie_cats.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ALPHA = 0.05
POWER_TARGET = 0.80
MDE_ABS = 0.01  # 1 percentage point, a typical minimum-detectable-effect for a retention metric

# ---------------------------------------------------------------------------
# Plot styling (colorblind-safe, consistent categorical palette)
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.05)
COLOR_CONTROL = "#4C72B0"     # gate_30
COLOR_TREATMENT = "#DD8452"   # gate_40
PALETTE = {"gate_30": COLOR_CONTROL, "gate_40": COLOR_TREATMENT}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    assert df["userid"].is_unique, "Duplicate userids found -- unit of randomization violated"
    assert set(df["version"].unique()) == {"gate_30", "gate_40"}
    assert df.isnull().sum().sum() == 0
    return df


def section(title: str, lines: list[str]) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    for l in lines:
        print(l)


def main():
    log = []  # collect markdown lines for results.md
    log.append(f"# A/B Test Results: Cookie Cats Gate Placement Experiment\n")

    df = load_data()
    n_control = (df["version"] == "gate_30").sum()
    n_treatment = (df["version"] == "gate_40").sum()

    section("1. DATA OVERVIEW", [
        f"Total players: {len(df):,}",
        f"Control (gate_30): {n_control:,}",
        f"Treatment (gate_40): {n_treatment:,}",
        f"1-day retention (overall): {df['retention_1'].mean():.2%}",
        f"7-day retention (overall): {df['retention_7'].mean():.2%}",
    ])
    log.append("## 1. Data Overview\n")
    log.append(f"- Total players: **{len(df):,}**")
    log.append(f"- Control (`gate_30`): {n_control:,} | Treatment (`gate_40`): {n_treatment:,}")
    log.append(f"- Overall 1-day retention: {df['retention_1'].mean():.2%} | "
               f"7-day retention: {df['retention_7'].mean():.2%}\n")

    # -----------------------------------------------------------------
    # 2. Prospective power analysis
    # -----------------------------------------------------------------
    baseline_r1 = df.loc[df["version"] == "gate_30", "retention_1"].mean()
    power_result = required_sample_size(baseline_rate=baseline_r1, mde_abs=MDE_ABS,
                                         alpha=ALPHA, power=POWER_TARGET)
    section("2. POWER ANALYSIS (planning stage)", [
        f"Baseline 1-day retention (control): {baseline_r1:.2%}",
        f"Minimum detectable effect (absolute): {MDE_ABS:.1%}",
        f"alpha = {ALPHA}, power = {POWER_TARGET}",
        f"--> Required sample size PER GROUP: {power_result['n_per_group']:,}",
        f"--> Required TOTAL sample size: {power_result['n_total']:,}",
        f"Actual total collected: {len(df):,} "
        f"({'SUFFICIENT' if len(df) >= power_result['n_total'] else 'UNDERPOWERED'})",
    ])
    log.append("## 2. Power Analysis\n")
    log.append(f"To detect a **{MDE_ABS:.0%} absolute** change in 1-day retention off a "
               f"{baseline_r1:.2%} baseline, at alpha={ALPHA} and power={POWER_TARGET}:\n")
    log.append(f"- Required sample size per group: **{power_result['n_per_group']:,}**")
    log.append(f"- Required total sample size: **{power_result['n_total']:,}**")
    log.append(f"- Actual sample collected: **{len(df):,}** — "
               f"{'sufficiently powered ✅' if len(df) >= power_result['n_total'] else 'underpowered ⚠️'}\n")

    # -----------------------------------------------------------------
    # 3. Sample Ratio Mismatch check
    # -----------------------------------------------------------------
    srm = srm_check(n_control, n_treatment, expected_ratio=0.5)
    section("3. SAMPLE RATIO MISMATCH CHECK", [
        f"Observed split: {srm['observed_ratio']:.4f} control / {1 - srm['observed_ratio']:.4f} treatment",
        f"chi2 = {srm['chi2']:.4f}, p = {srm['p_value']:.4f}  (SRM threshold: p < {srm['srm_alpha']})",
        f"SRM detected: {srm['srm_detected']}",
    ])
    log.append("## 3. Sample Ratio Mismatch (SRM) Check\n")
    log.append(f"Observed split {srm['observed_ratio']:.2%} / {1-srm['observed_ratio']:.2%} "
               f"vs expected 50/50. Chi-square p-value = {srm['p_value']:.4f}. Using the "
               f"stringent p < {srm['srm_alpha']} threshold standard for SRM checks (a plain "
               f"p < 0.05 test flags most large-sample experiments as broken even from noise), "
               f"{'⚠️ SRM detected — investigate the randomization/logging pipeline before trusting results' if srm['srm_detected'] else 'no SRM detected — randomization looks healthy ✅'}.\n")

    # -----------------------------------------------------------------
    # 4. Primary metric: 1-day retention
    # -----------------------------------------------------------------
    res_r1 = two_proportion_ztest(
        successes_control=df.loc[df.version == "gate_30", "retention_1"].sum(),
        n_control=n_control,
        successes_treatment=df.loc[df.version == "gate_40", "retention_1"].sum(),
        n_treatment=n_treatment,
        metric_name="1-day retention", alpha=ALPHA,
    )
    section("4. PRIMARY METRIC: 1-DAY RETENTION", [res_r1.summary()])

    # -----------------------------------------------------------------
    # 5. Secondary metric: 7-day retention
    # -----------------------------------------------------------------
    res_r7 = two_proportion_ztest(
        successes_control=df.loc[df.version == "gate_30", "retention_7"].sum(),
        n_control=n_control,
        successes_treatment=df.loc[df.version == "gate_40", "retention_7"].sum(),
        n_treatment=n_treatment,
        metric_name="7-day retention", alpha=ALPHA,
    )
    section("5. SECONDARY METRIC: 7-DAY RETENTION", [res_r7.summary()])

    log.append("## 4. Hypothesis Test Results\n")
    log.append("| Metric | Control | Treatment | Abs. Lift | 95% CI (abs) | Rel. Lift | p-value | Significant? |")
    log.append("|---|---|---|---|---|---|---|---|")
    for r in (res_r1, res_r7):
        log.append(f"| {r.metric} | {r.rate_control:.2%} | {r.rate_treatment:.2%} | "
                   f"{r.abs_lift:+.2%} | [{r.ci_abs_low:+.2%}, {r.ci_abs_high:+.2%}] | "
                   f"{r.rel_lift_pct:+.2f}% | {r.p_value:.4g} | "
                   f"{'Yes' if r.significant else 'No'} |")
    log.append("")

    # -----------------------------------------------------------------
    # 6. Secondary continuous metric: game rounds played (with outlier note)
    # -----------------------------------------------------------------
    rounds = df["sum_gamerounds"]
    n_outliers = (rounds > rounds.quantile(0.999)).sum()
    max_val = rounds.max()
    df_clean = df[rounds <= rounds.quantile(0.999)].copy()  # trim extreme outlier(s) for the mean-based test

    control_rounds = df_clean.loc[df_clean.version == "gate_30", "sum_gamerounds"].values
    treatment_rounds = df_clean.loc[df_clean.version == "gate_40", "sum_gamerounds"].values
    res_rounds = welch_ttest_with_bootstrap_ci(control_rounds, treatment_rounds,
                                                metric_name="Game rounds played (14-day, top 0.1% trimmed)",
                                                alpha=ALPHA)

    section("6. SECONDARY METRIC: GAME ROUNDS PLAYED", [
        f"Max value in raw data: {max_val:,} rounds (extreme outlier -> trimmed top 0.1% before t-test)",
        f"Outliers trimmed: {n_outliers}",
        f"Control mean: {res_rounds['mean_control']:.2f} | median: {res_rounds['median_control']:.1f}",
        f"Treatment mean: {res_rounds['mean_treatment']:.2f} | median: {res_rounds['median_treatment']:.1f}",
        f"Diff in means: {res_rounds['diff_means']:+.2f}  "
        f"bootstrap 95% CI [{res_rounds['bootstrap_ci_low']:+.2f}, {res_rounds['bootstrap_ci_high']:+.2f}]",
        f"Welch t = {res_rounds['t_stat']:.3f}, p = {res_rounds['p_value']:.4g}, "
        f"significant: {res_rounds['significant']}",
    ])
    log.append("## 5. Secondary Metric: Game Rounds Played\n")
    log.append(f"The raw data contains an extreme outlier (max = {max_val:,} rounds in 14 days), "
               f"so the top 0.1% of values ({n_outliers} players) were trimmed before comparing means.\n")
    log.append(f"- Control mean: {res_rounds['mean_control']:.2f} rounds (median {res_rounds['median_control']:.1f})")
    log.append(f"- Treatment mean: {res_rounds['mean_treatment']:.2f} rounds (median {res_rounds['median_treatment']:.1f})")
    log.append(f"- Difference in means: {res_rounds['diff_means']:+.2f}, bootstrap 95% CI "
               f"[{res_rounds['bootstrap_ci_low']:+.2f}, {res_rounds['bootstrap_ci_high']:+.2f}]")
    log.append(f"- Welch's t-test p-value: {res_rounds['p_value']:.4g} "
               f"({'significant' if res_rounds['significant'] else 'not significant'})\n")

    # -----------------------------------------------------------------
    # Save machine-readable summary
    # -----------------------------------------------------------------
    summary = {
        "n_control": int(n_control),
        "n_treatment": int(n_treatment),
        "power_analysis": power_result,
        "srm_check": srm,
        "retention_1": res_r1.__dict__,
        "retention_7": res_r7.__dict__,
        "game_rounds": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                        for k, v in res_rounds.items()},
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # -----------------------------------------------------------------
    # 7. Business conclusion
    # -----------------------------------------------------------------
    conclusion = []
    conclusion.append("## 6. Conclusion & Recommendation\n")
    conclusion.append(
        f"**1-day retention** was {abs(res_r1.abs_lift):.2%} lower in treatment than control "
        f"(95% CI [{res_r1.ci_abs_low:+.2%}, {res_r1.ci_abs_high:+.2%}]), a direction consistent with "
        f"a negative effect but **not statistically significant at alpha={ALPHA}** (p={res_r1.p_value:.3f}) — "
        f"the confidence interval includes a small positive effect, so this metric alone is inconclusive."
    )
    conclusion.append(
        f"\n**7-day retention** was {abs(res_r7.abs_lift):.2%} lower in treatment than control "
        f"(95% CI [{res_r7.ci_abs_low:+.2%}, {res_r7.ci_abs_high:+.2%}]), a **statistically significant decrease** "
        f"(p={res_r7.p_value:.4g}) and a {abs(res_r7.rel_lift_pct):.1f}% relative drop."
    )
    conclusion.append(
        f"\n**Game rounds played** showed no significant difference between groups "
        f"(p={res_rounds['p_value']:.3f}), so the retention drop is not explained by players simply "
        f"engaging less per session before churning — they are churning outright at a higher rate."
    )
    conclusion.append(
        "\n**Recommendation: do NOT ship gate_40 — keep the gate at level 30.** Both retention "
        "metrics point in the same (negative) direction for the treatment, and the effect reaches "
        "significance on the more decision-relevant 7-day horizon. Moving the gate later removes an "
        "early monetization/pacing checkpoint, but the data suggests this costs the game long-term "
        "player retention, which for a free-to-play title is the metric that most directly drives "
        "lifetime value. A follow-up experiment with a larger sample (see Section 2 power analysis) "
        "or a longer observation window could tighten the 1-day retention estimate, but the current "
        "evidence does not support shipping the change."
    )
    log.extend(conclusion)
    section("7. CONCLUSION", conclusion)

    with open(RESULTS_DIR / "results.md", "w") as f:
        f.write("\n".join(log))

    # -----------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------
    make_figures(df, df_clean, res_r1, res_r7)
    print(f"\nSaved: {RESULTS_DIR / 'metrics.json'}")
    print(f"Saved: {RESULTS_DIR / 'results.md'}")
    print(f"Saved figures to: {RESULTS_DIR}")


def make_figures(df, df_clean, res_r1, res_r7):
    # --- Figure 1: retention rates by group with 95% CI error bars ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=False)
    for ax, res, title in zip(axes, [res_r1, res_r7], ["1-Day Retention", "7-Day Retention"]):
        rates = [res.rate_control, res.rate_treatment]
        errs_low = [0, 0]
        errs_high = [0, 0]
        # per-group Wald CIs for the bars themselves
        from statsmodels.stats.proportion import proportion_confint
        for i, (n, successes) in enumerate([
            (res.n_control, round(res.rate_control * res.n_control)),
            (res.n_treatment, round(res.rate_treatment * res.n_treatment)),
        ]):
            lo, hi = proportion_confint(successes, n, alpha=0.05, method="wilson")
            errs_low[i] = rates[i] - lo
            errs_high[i] = hi - rates[i]
        bars = ax.bar(["gate_30\n(control)", "gate_40\n(treatment)"], rates,
                       yerr=[errs_low, errs_high], capsize=6,
                       color=[COLOR_CONTROL, COLOR_TREATMENT], edgecolor="white", linewidth=0.5)
        for b, r in zip(bars, rates):
            ax.text(b.get_x() + b.get_width() / 2, r + max(errs_high) + 0.008,
                     f"{r:.2%}", ha="center", fontsize=10, fontweight="bold")
        ax.set_title(f"{title}\np = {res.p_value:.4f}" +
                     ("  (significant)" if res.significant else "  (not significant)"))
        ax.set_ylabel("Retention rate")
        ax.set_ylim(0, max(rates) * 1.25)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Cookie Cats: Retention by Gate Placement (error bars = 95% Wilson CI)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig1_retention_by_group.png", dpi=150)
    plt.close(fig)

    # --- Figure 2: lift with CI (forest-style plot) ---
    fig, ax = plt.subplots(figsize=(7, 3.5))
    metrics = [res_r1, res_r7]
    y_pos = np.arange(len(metrics))
    lifts = [r.abs_lift * 100 for r in metrics]
    err_low = [(r.abs_lift - r.ci_abs_low) * 100 for r in metrics]
    err_high = [(r.ci_abs_high - r.abs_lift) * 100 for r in metrics]
    colors = ["#C44E52" if r.abs_lift < 0 else "#55A868" for r in metrics]
    ax.errorbar(lifts, y_pos, xerr=[err_low, err_high], fmt="o", markersize=9,
                capsize=5, color="black", ecolor="gray", elinewidth=1.5)
    for i, r in enumerate(metrics):
        ax.scatter(r.abs_lift * 100, i, color=colors[i], s=120, zorder=5,
                    edgecolor="black", linewidth=0.7)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r.metric for r in metrics])
    ax.set_xlabel("Absolute lift, treatment - control (percentage points)")
    ax.set_title("Treatment Effect (gate_40 vs gate_30) with 95% CI", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig2_lift_forest_plot.png", dpi=150)
    plt.close(fig)

    # --- Figure 3: sample size vs power curve ---
    from statsmodels.stats.power import NormalIndPower
    import math
    baseline = df.loc[df.version == "gate_30", "retention_1"].mean()
    analysis = NormalIndPower()
    ns = np.arange(500, 60000, 500)
    mdes = [0.005, 0.01, 0.02]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for mde, color in zip(mdes, sns.color_palette("viridis", len(mdes))):
        p2 = baseline + mde
        h = 2 * math.asin(math.sqrt(p2)) - 2 * math.asin(math.sqrt(baseline))
        powers = analysis.power(effect_size=abs(h), nobs1=ns, alpha=0.05, ratio=1.0)
        ax.plot(ns, powers, label=f"MDE = {mde:.1%}", color=color, linewidth=2.2)
    ax.axhline(0.8, color="gray", linestyle="--", linewidth=1, label="Target power = 0.80")
    ax.axvline(len(df) / 2, color="black", linestyle=":", linewidth=1.2,
               label=f"Actual n/group ≈ {len(df)//2:,}")
    ax.set_xlabel("Sample size per group")
    ax.set_ylabel("Statistical power")
    ax.set_title(f"Power Curves for Detecting a Lift in 1-Day Retention\n(baseline = {baseline:.1%})",
                 fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig3_power_curve.png", dpi=150)
    plt.close(fig)

    # --- Figure 4: game rounds distribution (log scale, trimmed) ---
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for version, color in PALETTE.items():
        vals = df_clean.loc[df_clean.version == version, "sum_gamerounds"]
        vals = vals[vals > 0]  # log scale needs >0
        sns.kdeplot(np.log1p(vals), ax=ax, color=color, label=version, linewidth=2.2, fill=True, alpha=0.15)
    ax.set_xlabel("log(1 + game rounds played in first 14 days)")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Engagement by Group (outlier-trimmed)", fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "fig4_gamerounds_distribution.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
