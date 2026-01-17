"""
Statistical Significance Tests for Publication
Provides rigorous statistical validation of FL performance claims.

For conference papers, these tests prove that FL improvements are
statistically significant, not due to random chance.
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings


@dataclass
class StatisticalTestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float
    effect_size: Optional[float] = None
    effect_interpretation: str = ""
    conclusion: str = ""


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Calculate Cohen's d effect size.

    Interpretation:
    - 0.2: Small effect
    - 0.5: Medium effect
    - 0.8: Large effect
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))

    if pooled_std == 0:
        return 0.0

    return (np.mean(group1) - np.mean(group2)) / pooled_std


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"


def paired_t_test(
    fl_scores: np.ndarray,
    baseline_scores: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "less"  # FL should have LOWER error
) -> StatisticalTestResult:
    """
    Paired t-test for comparing FL vs baseline on same test cases.

    Use when: Same test scenarios are run for both methods.

    Args:
        fl_scores: FL performance scores (e.g., MAE per scenario)
        baseline_scores: Baseline scores
        alpha: Significance level
        alternative: "less" (FL < baseline), "greater", or "two-sided"

    Returns:
        StatisticalTestResult with full analysis
    """
    statistic, p_value = stats.ttest_rel(fl_scores, baseline_scores, alternative=alternative)

    effect = cohens_d(fl_scores, baseline_scores)
    effect_interp = interpret_effect_size(effect)

    is_significant = p_value < alpha

    conclusion = (
        f"FL {'significantly' if is_significant else 'does not significantly'} "
        f"outperform baseline (p={p_value:.4f}, d={effect:.3f} [{effect_interp}])"
    )

    return StatisticalTestResult(
        test_name="Paired t-test",
        statistic=statistic,
        p_value=p_value,
        is_significant=is_significant,
        confidence_level=1 - alpha,
        effect_size=effect,
        effect_interpretation=effect_interp,
        conclusion=conclusion
    )


def wilcoxon_signed_rank_test(
    fl_scores: np.ndarray,
    baseline_scores: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "less"
) -> StatisticalTestResult:
    """
    Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    Use when: Data may not be normally distributed.
    """
    statistic, p_value = stats.wilcoxon(
        fl_scores, baseline_scores,
        alternative=alternative,
        zero_method='zsplit'
    )

    effect = cohens_d(fl_scores, baseline_scores)
    effect_interp = interpret_effect_size(effect)

    is_significant = p_value < alpha

    conclusion = (
        f"FL {'significantly' if is_significant else 'does not significantly'} "
        f"outperform baseline (p={p_value:.4f}, effect={effect_interp})"
    )

    return StatisticalTestResult(
        test_name="Wilcoxon signed-rank test",
        statistic=statistic,
        p_value=p_value,
        is_significant=is_significant,
        confidence_level=1 - alpha,
        effect_size=effect,
        effect_interpretation=effect_interp,
        conclusion=conclusion
    )


def bootstrap_confidence_interval(
    fl_scores: np.ndarray,
    baseline_scores: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95
) -> Dict[str, float]:
    """
    Bootstrap confidence interval for performance difference.

    More robust than parametric methods for small samples.
    """
    differences = fl_scores - baseline_scores
    n = len(differences)

    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(differences, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)

    # Calculate confidence interval
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)

    return {
        "mean_difference": np.mean(differences),
        "ci_lower": lower,
        "ci_upper": upper,
        "confidence": confidence,
        "significant": upper < 0  # Entire CI below zero = FL significantly better
    }


def normality_test(data: np.ndarray) -> Tuple[str, float, bool]:
    """
    Test if data is normally distributed (Shapiro-Wilk test).
    Helps decide between parametric and non-parametric tests.
    """
    if len(data) < 3:
        return "Shapiro-Wilk", 1.0, True

    statistic, p_value = stats.shapiro(data)
    is_normal = p_value > 0.05

    return "Shapiro-Wilk", p_value, is_normal


def run_comprehensive_statistical_analysis(
    fl_results: Dict[str, List[float]],
    baseline_results: Dict[str, List[float]],
    metrics: List[str] = None
) -> Dict[str, Dict]:
    """
    Run comprehensive statistical analysis for publication.

    Args:
        fl_results: FL results for multiple runs {"mae": [1.8, 1.9, ...], ...}
        baseline_results: Baseline results
        metrics: Which metrics to analyze

    Returns:
        Comprehensive analysis results
    """
    if metrics is None:
        metrics = list(fl_results.keys())

    analysis = {}

    for metric in metrics:
        fl_data = np.array(fl_results[metric])
        baseline_data = np.array(baseline_results[metric])

        if len(fl_data) != len(baseline_data):
            warnings.warn(f"Unequal sample sizes for {metric}")
            continue

        # Check normality
        _, fl_normal_p, fl_normal = normality_test(fl_data)
        _, base_normal_p, base_normal = normality_test(baseline_data)

        # Choose appropriate test
        if fl_normal and base_normal and len(fl_data) >= 20:
            # Parametric test
            test_result = paired_t_test(fl_data, baseline_data)
        else:
            # Non-parametric test
            test_result = wilcoxon_signed_rank_test(fl_data, baseline_data)

        # Bootstrap CI
        bootstrap_ci = bootstrap_confidence_interval(fl_data, baseline_data)

        # Compile results
        analysis[metric] = {
            "fl_mean": np.mean(fl_data),
            "fl_std": np.std(fl_data),
            "baseline_mean": np.mean(baseline_data),
            "baseline_std": np.std(baseline_data),
            "improvement_percent": (np.mean(baseline_data) - np.mean(fl_data)) / np.mean(baseline_data) * 100,
            "normality_check": {
                "fl_normal": fl_normal,
                "baseline_normal": base_normal
            },
            "statistical_test": {
                "name": test_result.test_name,
                "statistic": test_result.statistic,
                "p_value": test_result.p_value,
                "significant": test_result.is_significant,
                "effect_size": test_result.effect_size,
                "effect_interpretation": test_result.effect_interpretation
            },
            "bootstrap_ci": bootstrap_ci,
            "conclusion": test_result.conclusion
        }

    return analysis


def generate_publication_table(analysis: Dict[str, Dict]) -> str:
    """
    Generate publication-ready LaTeX table.
    """
    table = []
    table.append("\\begin{table}[htbp]")
    table.append("\\centering")
    table.append("\\caption{Statistical Comparison: Federated Learning vs Local-ML}")
    table.append("\\label{tab:statistical_comparison}")
    table.append("\\begin{tabular}{lccccc}")
    table.append("\\toprule")
    table.append("Metric & FL Mean $\\pm$ SD & Local-ML Mean $\\pm$ SD & Improvement & p-value & Effect \\\\")
    table.append("\\midrule")

    for metric, results in analysis.items():
        fl_str = f"{results['fl_mean']:.3f} $\\pm$ {results['fl_std']:.3f}"
        base_str = f"{results['baseline_mean']:.3f} $\\pm$ {results['baseline_std']:.3f}"
        imp_str = f"{results['improvement_percent']:.1f}\\%"
        p_str = f"{results['statistical_test']['p_value']:.4f}"

        # Mark significance
        if results['statistical_test']['significant']:
            p_str = f"\\textbf{{{p_str}}}*"

        effect_str = results['statistical_test']['effect_interpretation']

        table.append(f"{metric} & {fl_str} & {base_str} & {imp_str} & {p_str} & {effect_str} \\\\")

    table.append("\\bottomrule")
    table.append("\\end{tabular}")
    table.append("\\begin{tablenotes}")
    table.append("\\small")
    table.append("\\item * indicates statistical significance at $\\alpha = 0.05$")
    table.append("\\end{tablenotes}")
    table.append("\\end{table}")

    return "\n".join(table)


def generate_statistical_report(analysis: Dict[str, Dict]) -> str:
    """Generate text report for statistical analysis."""
    report = []
    report.append("=" * 70)
    report.append("STATISTICAL SIGNIFICANCE ANALYSIS")
    report.append("=" * 70)

    for metric, results in analysis.items():
        report.append(f"\n{metric.upper()}")
        report.append("-" * 40)
        report.append(f"  FL:       {results['fl_mean']:.4f} ± {results['fl_std']:.4f}")
        report.append(f"  Baseline: {results['baseline_mean']:.4f} ± {results['baseline_std']:.4f}")
        report.append(f"  Improvement: {results['improvement_percent']:.2f}%")
        report.append(f"")
        report.append(f"  Statistical Test: {results['statistical_test']['name']}")
        report.append(f"  p-value: {results['statistical_test']['p_value']:.6f}")
        report.append(f"  Significant: {'YES' if results['statistical_test']['significant'] else 'NO'}")
        report.append(f"  Effect Size: {results['statistical_test']['effect_size']:.3f} ({results['statistical_test']['effect_interpretation']})")
        report.append(f"")
        report.append(f"  95% CI: [{results['bootstrap_ci']['ci_lower']:.4f}, {results['bootstrap_ci']['ci_upper']:.4f}]")
        report.append(f"")
        report.append(f"  Conclusion: {results['conclusion']}")

    report.append("\n" + "=" * 70)
    report.append("SUMMARY FOR PUBLICATION")
    report.append("=" * 70)

    all_significant = all(r['statistical_test']['significant'] for r in analysis.values())
    if all_significant:
        report.append("\nAll improvements are statistically significant (p < 0.05).")
        report.append("The null hypothesis (no difference) can be rejected.")
        report.append("FL demonstrably outperforms the baseline with high confidence.")
    else:
        report.append("\nSome improvements are not statistically significant.")
        report.append("More experimental runs may be needed to establish significance.")

    return "\n".join(report)


if __name__ == "__main__":
    # Example usage with simulated experimental data
    print("Statistical Significance Testing Demo")
    print("=" * 50)

    # Simulate 10 experimental runs
    np.random.seed(42)

    # FL results (better = lower)
    fl_results = {
        "mae": np.random.normal(1.86, 0.15, 10),
        "waiting_time": np.random.normal(9.49, 0.3, 10),
        "queue_length": np.random.normal(14.51, 0.5, 10)
    }

    # Baseline results (worse = higher)
    baseline_results = {
        "mae": np.random.normal(2.18, 0.20, 10),
        "waiting_time": np.random.normal(9.55, 0.35, 10),
        "queue_length": np.random.normal(14.55, 0.6, 10)
    }

    # Run analysis
    analysis = run_comprehensive_statistical_analysis(fl_results, baseline_results)

    # Print report
    print(generate_statistical_report(analysis))

    # Print LaTeX table
    print("\n" + "=" * 50)
    print("LaTeX TABLE:")
    print("=" * 50)
    print(generate_publication_table(analysis))
