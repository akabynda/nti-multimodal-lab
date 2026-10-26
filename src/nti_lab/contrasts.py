"""Baseline-normalised answer features, truth/lie contrasts and profile-group tests.

Labels: «Правда» (label 0) and «Ложь» (label 1) come from the reviewed reference
comparison; «База» are answers in neutral question blocks (session_summary).
Normalisation uses no labels: centre — participant median of «База», scale — robust
SD (IQR / 1.349) of all the participant's answers; z is clipped to ±Z_LIMIT. The unit of inference is the
participant: one contrast per participant and feature.
"""
from __future__ import annotations

from itertools import combinations, product
import numpy as np
import pandas as pd

from .multimodal_analysis import AUDIO, VIDEO, SEED, BOOTSTRAPS, exact_spearman, holm

CONTRAST_FEATURES = AUDIO + VIDEO + ["bio_ecg_hr_bpm_context120s", "bio_ecg_rmssd_ms_context120s"]
ANOMALY = "audio_anomaly_norm"
MIN_BASE = 2
MIN_SCALE = 4
MIN_GROUP = 3
Z_LIMIT = 5.


def normalize(answers: pd.DataFrame, features=CONTRAST_FEATURES) -> pd.DataFrame:
    """Return z = (x − median «База») / (IQR/1.349) per participant and feature.

    `answers` holds all answers with column `condition` in {truth, lie, base}. A
    participant-feature with fewer than MIN_BASE «База» values, fewer than MIN_SCALE
    answers or zero spread is NaN; z is clipped to [−Z_LIMIT, Z_LIMIT]. `audio_anomaly_norm` is the RMS of the available
    acoustic z-values of an answer.
    """
    result = answers[["trial_id", "participant_id", "condition"]].copy()
    for feature in features:
        result[feature] = np.nan
        for pid, rows in answers.groupby("participant_id"):
            values = rows[feature].astype(float)
            base = values[rows.condition.eq("base")].dropna()
            finite = values.dropna()
            if len(base) < MIN_BASE or len(finite) < MIN_SCALE:
                continue
            spread = np.subtract(*np.quantile(finite, [.75, .25])) / 1.349
            if not spread > 0:
                continue
            result.loc[rows.index, feature] = ((values - base.median()) / spread).clip(-Z_LIMIT, Z_LIMIT)
    acoustic = result[[f for f in AUDIO if f in features]]
    count = acoustic.notna().sum(axis=1)
    result[ANOMALY] = np.where(count > 0, np.sqrt((acoustic ** 2).sum(axis=1) / count.clip(lower=1)), np.nan)
    return result


def participant_contrasts(normalized: pd.DataFrame, features) -> pd.DataFrame:
    """Mean z of «Ложь» minus mean z of «Правда» per participant and feature."""
    rows = []
    for pid, part in normalized.groupby("participant_id"):
        for feature in features:
            truth = part.loc[part.condition.eq("truth"), feature].dropna()
            lie = part.loc[part.condition.eq("lie"), feature].dropna()
            rows.append(dict(participant_id=pid, feature=feature, truth_n=len(truth), lie_n=len(lie),
                             truth_mean_z=truth.mean() if len(truth) else np.nan,
                             lie_mean_z=lie.mean() if len(lie) else np.nan,
                             contrast=lie.mean() - truth.mean() if len(truth) and len(lie) else np.nan))
    return pd.DataFrame(rows)


def sign_flip_test(values, *, bootstrap=BOOTSTRAPS, seed=SEED) -> dict:
    """Exact two-sided sign-flip test of mean = 0 and percentile bootstrap 95% CI."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    result = dict(n=len(x), mean=np.nan, ci_low=np.nan, ci_high=np.nan, p=np.nan, status="not_estimable")
    if len(x) < MIN_GROUP * 2 or np.allclose(x, 0):
        return result
    signs = np.array(list(product((1, -1), repeat=len(x))))
    null = np.abs((signs * x).mean(axis=1))
    boot = x[np.random.default_rng(seed).integers(0, len(x), (bootstrap, len(x)))].mean(axis=1)
    result.update(mean=float(x.mean()), p=float(np.mean(null >= abs(x.mean()) - 1e-12)),
                  ci_low=float(np.quantile(boot, .025)), ci_high=float(np.quantile(boot, .975)), status="estimated")
    return result


def group_difference_test(values, groups) -> dict:
    """Exact two-sided permutation test of the difference in group means (group 1 − group 0)."""
    x, g = np.asarray(values, dtype=float), np.asarray(groups, dtype=bool)
    keep = np.isfinite(x)
    x, g = x[keep], g[keep]
    result = dict(n_group=int(g.sum()), n_other=int((~g).sum()), group_mean=np.nan, other_mean=np.nan,
                  difference=np.nan, p=np.nan, status="not_estimable")
    if g.sum() < MIN_GROUP or (~g).sum() < MIN_GROUP:
        return result
    observed = x[g].mean() - x[~g].mean()
    null = []
    for chosen in combinations(range(len(x)), int(g.sum())):
        mask = np.zeros(len(x), dtype=bool)
        mask[list(chosen)] = True
        null.append(x[mask].mean() - x[~mask].mean())
    result.update(group_mean=float(x[g].mean()), other_mean=float(x[~g].mean()), difference=float(observed),
                  p=float(np.mean(np.abs(null) >= abs(observed) - 1e-12)), status="estimated")
    return result


def hypothesis_tests(contrasts: pd.DataFrame, profiles: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """H1 sign-flip per feature, H2 group difference, H3 Spearman with expression; Holm within each."""
    rows = []
    for feature, part in contrasts.groupby("feature", sort=False):
        part = part.merge(profiles, on="participant_id", how="left")
        h1 = sign_flip_test(part.contrast)
        h2 = group_difference_test(part.contrast, part[group_column].astype(bool))
        h3 = exact_spearman(part.expression, part.contrast)
        rows.append(dict(feature=feature, h1_n=h1["n"], h1_mean=h1["mean"], h1_ci_low=h1["ci_low"],
                         h1_ci_high=h1["ci_high"], h1_p=h1["p"], h2_group_n=h2["n_group"], h2_other_n=h2["n_other"],
                         h2_group_mean=h2["group_mean"], h2_other_mean=h2["other_mean"],
                         h2_difference=h2["difference"], h2_p=h2["p"], h3_n=h3["n"], h3_rho=h3["rho"], h3_p=h3["p"]))
    frame = pd.DataFrame(rows)
    for test in ("h1", "h2", "h3"):
        p = [None if not np.isfinite(v) else v for v in frame[f"{test}_p"].astype(float)]
        frame[f"{test}_p_holm"] = [a if v is not None else np.nan for a, v in zip(holm(p), p)]
    return frame
