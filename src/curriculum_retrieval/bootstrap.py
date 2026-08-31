"""
Paired grouped bootstrap confidence interval calculation and statistical significance testing.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def compute_paired_grouped_bootstrap(
    baseline_scores: List[float],
    treatment_scores: List[float],
    group_keys: List[str],
    n_replicates: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Perform paired grouped bootstrap over cluster/topic/lecture keys.
    Returns point estimates, 95% CI of the difference, effect size, and group counts.
    """
    assert len(baseline_scores) == len(treatment_scores) == len(group_keys), (
        "Lengths of baseline, treatment, and group_keys must match exactly."
    )

    rng = np.random.default_rng(seed)

    # Group paired records by group key
    groups = defaultdict(lambda: {"b": [], "t": []})
    for b, t, g in zip(baseline_scores, treatment_scores, group_keys):
        groups[g]["b"].append(b)
        groups[g]["t"].append(t)

    unique_groups = list(groups.keys())
    n_groups = len(unique_groups)

    if n_groups == 0:
        return {
            "baseline_mean": 0.0,
            "treatment_mean": 0.0,
            "absolute_diff": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "p_value": 1.0,
            "cohens_d": 0.0,
            "n_groups": 0,
            "n_samples": 0,
            "seed": seed,
            "replicates": n_replicates,
        }

    b_arr = np.array(baseline_scores)
    t_arr = np.array(treatment_scores)
    diff_arr = t_arr - b_arr

    mean_b = float(np.mean(b_arr))
    mean_t = float(np.mean(t_arr))
    obs_diff = mean_t - mean_b

    # Cohen's d for paired samples
    std_diff = float(np.std(diff_arr, ddof=1)) if len(diff_arr) > 1 else 1e-9
    cohens_d = (obs_diff / std_diff) if std_diff > 0 else 0.0

    # Resample groups with replacement
    boot_diffs = []
    for _ in range(n_replicates):
        resampled_groups = rng.choice(unique_groups, size=n_groups, replace=True)
        boot_b = []
        boot_t = []
        for g in resampled_groups:
            boot_b.extend(groups[g]["b"])
            boot_t.extend(groups[g]["t"])
        boot_diffs.append(np.mean(boot_t) - np.mean(boot_b))

    boot_diffs = np.array(boot_diffs)
    alpha = (1.0 - confidence_level) / 2.0
    ci_lower = float(np.percentile(boot_diffs, alpha * 100))
    ci_upper = float(np.percentile(boot_diffs, (1.0 - alpha) * 100))

    # Two-sided empirical p-value for H0: diff == 0
    p_val = float(2.0 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0)))
    p_val = min(1.0, max(1.0 / n_replicates, p_val))

    return {
        "baseline_mean": round(mean_b, 4),
        "treatment_mean": round(mean_t, 4),
        "absolute_diff": round(obs_diff, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "p_value": round(p_val, 4),
        "cohens_d": round(cohens_d, 4),
        "n_groups": n_groups,
        "n_samples": len(baseline_scores),
        "seed": seed,
        "replicates": n_replicates,
    }
