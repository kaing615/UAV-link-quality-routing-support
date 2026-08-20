"""Run-level paired inference for routing strategies."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product

import numpy as np
import pandas as pd


def bootstrap_mean_ci(
    values: np.ndarray,
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a mean, resampling independent runs."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (float("nan"), float("nan"))
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")
    rng = np.random.default_rng(seed)
    samples = rng.choice(x, size=(n_resamples, x.size), replace=True).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (float(np.quantile(samples, alpha)), float(np.quantile(samples, 1.0 - alpha)))


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values in the original order."""
    p = np.asarray(list(p_values), dtype=float)
    if p.size == 0:
        return p
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must be in [0, 1]")
    order = np.argsort(p, kind="stable")
    adjusted_sorted = np.maximum.accumulate(p[order] * (p.size - np.arange(p.size)))
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def _paired_sign_flip_pvalue(delta: np.ndarray) -> float:
    """Two-sided exact sign-flip p-value (run-level paired test)."""
    delta = np.asarray(delta, dtype=float)
    observed = abs(float(delta.mean()))
    n = delta.size
    if n == 0:
        return float("nan")
    if n <= 20:
        means = np.fromiter(
            (abs(float((delta * signs).mean())) for signs in product((-1.0, 1.0), repeat=n)),
            dtype=float,
        )
        return float(np.count_nonzero(means >= observed - 1e-15) / means.size)
    rng = np.random.default_rng(42)
    signs = rng.choice((-1.0, 1.0), size=(100_000, n))
    means = np.abs((signs * delta).mean(axis=1))
    return float((np.count_nonzero(means >= observed - 1e-15) + 1) / (means.size + 1))


def paired_comparisons(
    detailed: pd.DataFrame,
    comparisons: list[tuple[str, str]],
    metrics: dict[str, bool],
    n_resamples: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare comparator minus reference on shared runs.

    ``metrics`` maps metric names to whether larger values are better. The
    returned rows retain metric direction and identify route lifetime as the
    primary routing endpoint.
    """
    rows: list[dict[str, object]] = []
    required = {"run_name", "strategy", *metrics}
    missing = required.difference(detailed.columns)
    if missing:
        raise ValueError(f"detailed results missing columns: {sorted(missing)}")

    for metric, higher_is_better in metrics.items():
        for comparator, reference in comparisons:
            subset = detailed[detailed["strategy"].isin([reference, comparator])]
            pivot = subset.pivot_table(index="run_name", columns="strategy", values=metric, aggfunc="mean")
            if reference not in pivot or comparator not in pivot:
                delta = np.array([], dtype=float)
            else:
                paired = pivot[[reference, comparator]].dropna()
                delta = (paired[comparator] - paired[reference]).to_numpy(dtype=float)
            if delta.size:
                reference_values = paired[reference].to_numpy(dtype=float)
                comparator_values = paired[comparator].to_numpy(dtype=float)
                ci_low, ci_high = bootstrap_mean_ci(delta, n_resamples=n_resamples, seed=seed)
                p_raw = _paired_sign_flip_pvalue(delta)
                values = {
                    "n_pairs": int(delta.size),
                    "reference_mean": float(reference_values.mean()),
                    "comparator_mean": float(comparator_values.mean()),
                    "mean_delta": float(delta.mean()),
                    "median_delta": float(np.median(delta)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "p_raw": p_raw,
                }
            else:
                values = {
                    "n_pairs": 0,
                    "reference_mean": float("nan"),
                    "comparator_mean": float("nan"),
                    "mean_delta": float("nan"),
                    "median_delta": float("nan"),
                    "ci95_low": float("nan"),
                    "ci95_high": float("nan"),
                    "p_raw": float("nan"),
                }
            rows.append(
                {
                    "metric": metric,
                    "reference_strategy": reference,
                    "comparator_strategy": comparator,
                    "higher_is_better": bool(higher_is_better),
                    "primary_endpoint": metric == "mean_route_lifetime",
                    **values,
                }
            )

    result = pd.DataFrame(rows)
    if not result.empty:
        finite = result["p_raw"].notna()
        result["p_holm"] = np.nan
        result.loc[finite, "p_holm"] = holm_adjust(result.loc[finite, "p_raw"].to_numpy())
    else:
        result["p_holm"] = pd.Series(dtype=float)
    return result
