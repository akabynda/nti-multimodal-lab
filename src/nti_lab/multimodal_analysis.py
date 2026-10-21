"""Exploratory profile associations and participant-held-out reference consistency.

The target is a reviewed textual fact comparison, never an inferred mental state.
All preprocessing is fitted within a participant-held-out fold. No identifiers,
reference text, question text or comparison scores enter the feature matrix.
"""
from __future__ import annotations

from functools import lru_cache
from itertools import permutations
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 1729
BOOTSTRAPS = 5000
PROFILE_NUMERIC = ["expression", "emotion_positive", "emotion_negative"]
PROFILE_ASSOCIATIONS = PROFILE_NUMERIC + ["emotion_balance"]
PROFILE_LABELS = {"expression": "Выраженность", "emotion_positive": "Эмоц+",
                  "emotion_negative": "Эмоц−", "emotion_balance": "Эмоциональный баланс"}
TYPES = ["содействующий", "контролирующий", "координирующий", "стимулирующий"]
TYPE_CORRECTIONS = {"содейстующий": "содействующий", "стимлурующий": "стимулирующий"}
PROFILE_CATEGORIES = [f"profile_{kind}_{i}" for kind in ("natural", "demonstrated") for i in range(4)]
PROFILE_FEATURES = PROFILE_NUMERIC + PROFILE_CATEGORIES
AUDIO = ["audio_pitch_median_hz", "audio_pitch_iqr_hz", "audio_rms_dbfs",
         "audio_text_rate_wpm", "audio_vad_pause_fraction"]
VIDEO = ["video_head_speed_norm_per_s", "video_torso_motion_deg_per_s",
         "video_eye_opening_ratio", "video_mouth_motion_ratio_per_s"]
BIO_CONTEXT = ["bio_ecg_hr_bpm_context120s", "bio_ecg_rmssd_ms_context120s",
               "bio_rip_rate_bpm_context120s"]
PRIMARY_ASSOCIATION_FEATURES = ["audio_pitch_median_hz", "audio_text_rate_wpm",
    "video_head_speed_norm_per_s", "video_torso_motion_deg_per_s",
    "bio_ecg_hr_bpm_context120s", "bio_ecg_rmssd_ms_context120s"]
FEATURE_LABELS = {
    "audio_pitch_median_hz": "Основной тон речи, Гц", "audio_pitch_iqr_hz": "Разброс F0 (IQR), Гц",
    "audio_rms_dbfs": "Уровень речи, dBFS", "audio_text_rate_wpm": "Темп речи, слов/мин",
    "audio_vad_pause_fraction": "Доля пауз VAD", "audio_word_pause_median_s": "Межсловная пауза, с",
    "video_head_speed_norm_per_s": "Скорость головы, доля кадра/с",
    "video_torso_motion_deg_per_s": "Движение торса, град/с",
    "video_eye_opening_ratio": "Раскрытие глаз, отношение",
    "video_eye_motion_ratio_per_s": "Изменение раскрытия глаз, 1/с",
    "video_mouth_opening_ratio": "Раскрытие рта, отношение",
    "video_mouth_motion_ratio_per_s": "Изменение раскрытия рта, 1/с",
    "video_rppg_hr_bpm": "ЧСС rPPG, уд/мин",
    "video_rppg_hr_bpm_context30s": "ЧСС rPPG, уд/мин (окна 30 с)",
    "bio_ecg_hr_bpm": "ЧСС по RR, уд/мин",
    "bio_ecg_hr_bpm_context120s": "ЧСС ЭКГ, уд/мин (окна 120 с)",
    "bio_ecg_rmssd_ms_context120s": "RMSSD ЭКГ, мс (окна 120 с)",
    "bio_ecg_sdnn_ms_context120s": "SDNN ЭКГ, мс (окна 120 с)",
    "bio_rip_rate_bpm_context120s": "Частота RIP, дых/мин (окна 120 с)",
    "bio_gsr_median_us": "Уровень GSR, мкСм", "bio_gsr_iqr_us": "Разброс GSR, мкСм",
    "bio_ppg_hr_bpm_context120s": "ЧСС контактного ФПГ, уд/мин (окна 120 с)",
    "bio_ppg_rmssd_ms_context120s": "RMSSD контактного ФПГ, мс (окна 120 с)",
}
MODEL_SPECS = [
    ("prior", "Доля классов в обучении", []),
    ("duration", "Только длительность ответа", ["log_duration_s"]),
    ("availability", "Только наличие признаков", [f"available_{c}" for c in AUDIO + VIDEO]),
    ("profiles", "Только анкета", PROFILE_FEATURES),
    ("audio", "Речь", AUDIO), ("video", "Видео", VIDEO),
    ("av", "Видео + речь", AUDIO + VIDEO),
    ("av_profiles", "Видео + речь + анкета", AUDIO + VIDEO + PROFILE_FEATURES),
    ("fusion", "Видео + речь + BioRadio", AUDIO + VIDEO + BIO_CONTEXT),
    ("fusion_profiles", "Видео + речь + BioRadio + анкета", AUDIO + VIDEO + BIO_CONTEXT + PROFILE_FEATURES),
]


class ParticipantWeightedImputer(TransformerMixin, BaseEstimator):
    """Training-only weighted median and training-observed missingness indicators."""

    def fit(self, x, y=None, sample_weight=None):
        x = np.asarray(x, dtype=float)
        weights = np.ones(len(x)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if np.any(weights <= 0) or not np.isfinite(weights).all():
            raise ValueError("Positive finite training weights required")
        self.n_features_in_ = x.shape[1]
        self.indicator_columns_ = np.flatnonzero(np.isnan(x).any(axis=0))
        self.medians_ = np.zeros(x.shape[1])
        for col in range(x.shape[1]):
            keep = np.isfinite(x[:, col])
            if not keep.any():
                continue
            values, mass = x[keep, col], weights[keep]
            # Aggregate identical values so duplicate observations do not affect ties.
            unique, inverse = np.unique(values, return_inverse=True)
            mass = np.bincount(inverse, weights=mass)
            cumulative = np.cumsum(mass) / mass.sum()
            j = int(np.searchsorted(cumulative, .5 - 1e-12))
            self.medians_[col] = ((unique[j] + unique[j+1])/2 if
                j+1 < len(unique) and abs(cumulative[j]-.5) < 1e-12 else unique[j])
        return self

    def transform(self, x):
        x = np.asarray(x, dtype=float)
        if x.shape[1] != self.n_features_in_:
            raise ValueError("Feature schema changed")
        missing = np.isnan(x)
        return np.column_stack((np.where(missing, self.medians_, x), missing[:, self.indicator_columns_].astype(float)))


def encode_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    if profiles.participant_id.duplicated().any() or profiles.participant_id.isna().any():
        raise ValueError("Exactly one profile row per participant is required")
    result = profiles.copy()
    for c in PROFILE_ASSOCIATIONS:
        result[c] = pd.to_numeric(result[c], errors="raise")
    balance = result.emotion_positive - result.emotion_negative
    if not np.allclose(balance, result.emotion_balance, equal_nan=True):
        raise ValueError("Emotion balance differs from positive minus negative")
    for kind in ("natural", "demonstrated"):
        parsed = []
        for raw in result[f"profile_{kind}_raw"]:
            if pd.isna(raw) or not str(raw).strip():
                parsed.append(None)
                continue
            tokens = {TYPE_CORRECTIONS.get(t.strip().lower(), t.strip().lower()) for t in str(raw).split("/")}
            if not tokens <= set(TYPES):
                raise ValueError(f"Unknown profile category: {tokens - set(TYPES)}")
            parsed.append(tokens)
        for i, category in enumerate(TYPES):
            result[f"profile_{kind}_{i}"] = [np.nan if p is None else float(category in p) for p in parsed]
    return result


def holm(p_values):
    """Adjust the entire prespecified family; unestimable hypotheses count as p=1."""
    p = np.asarray([1.0 if v is None or not np.isfinite(v) else v for v in p_values])
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def _correlations(x, y):
    """Rowwise correlation; rank ties are already represented by average ranks."""
    x = x - x.mean(axis=-1, keepdims=True)
    y = y - y.mean(axis=-1, keepdims=True)
    denominator = np.sqrt((x*x).sum(axis=-1) * (y*y).sum(axis=-1))
    return np.divide((x*y).sum(axis=-1), denominator,
                     out=np.full(np.broadcast_shapes(x.shape[:-1], y.shape[:-1]), np.nan), where=denominator > 0)


@lru_cache(maxsize=5)
def _permutations(n):
    if n > 9:
        raise ValueError("Exact permutation implementation is limited to nine participants")
    return np.asarray(list(permutations(range(n))), dtype=np.int16)


def exact_spearman(x, y, *, min_n=6, bootstrap=BOOTSTRAPS, seed=SEED):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("Aligned one-dimensional paired observations required")
    if np.isinf(x).any() or np.isinf(y).any():
        raise ValueError("Infinite feature values are not valid missing data")
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    n = len(x)
    result = dict(n=n, rho=None, p=None, ci_low=None, ci_high=None,
                  status="not_estimable", permutations=0, bootstrap_valid=0,
                  leave_one_out_low=None, leave_one_out_high=None)
    if n < min_n or np.ptp(x) == 0 or np.ptp(y) == 0:
        return result
    rx, ry = rankdata(x), rankdata(y)
    rho = float(_correlations(rx, ry))
    null = _correlations(rx, ry[_permutations(n)])
    p = float(np.mean(np.abs(null) >= abs(rho) - 1e-12))
    indices = np.random.default_rng(seed).integers(0, n, (bootstrap, n))
    boot = _correlations(rankdata(x[indices], axis=1), rankdata(y[indices], axis=1))
    boot = boot[np.isfinite(boot)]
    loo = [float(_correlations(rankdata(np.delete(x, i)), rankdata(np.delete(y, i)))) for i in range(n)]
    loo = [r for r in loo if np.isfinite(r)]
    result.update(status="estimated", rho=rho, p=p, permutations=len(null), bootstrap_valid=len(boot),
        ci_low=float(np.quantile(boot, .025)), ci_high=float(np.quantile(boot, .975)),
        leave_one_out_low=min(loo) if loo else None, leave_one_out_high=max(loo) if loo else None)
    return result


def profile_associations(frame, features, *, bootstrap=BOOTSTRAPS):
    if frame.participant_id.duplicated().any():
        raise ValueError("Associations require one independent row per participant")
    rows = []
    for profile in PROFILE_ASSOCIATIONS:
        for feature in features:
            family = ("primary_12" if profile in ("expression", "emotion_balance") and
                      feature in PRIMARY_ASSOCIATION_FEATURES else "secondary")
            result = exact_spearman(frame[profile], frame[feature], bootstrap=bootstrap)
            rows.append(dict(profile=profile, profile_label=PROFILE_LABELS[profile], feature=feature,
                feature_label=FEATURE_LABELS.get(feature, feature), family=family, **result))
    for family in {r["family"] for r in rows}:
        selected = [r for r in rows if r["family"] == family]
        for row, adjusted in zip(selected, holm([r["p"] for r in selected])):
            row["p_holm"] = float(adjusted) if row["p"] is not None else None
            row["family_size"] = len(selected)
    return rows


def participant_weights(groups):
    groups = np.asarray(groups)
    unique, counts = np.unique(groups, return_counts=True)
    sizes = dict(zip(unique, counts))
    return np.asarray([1.0 / sizes[g] for g in groups])


def model_frame(trials, profiles):
    result = trials.merge(encode_profiles(profiles)[["participant_id"] + PROFILE_FEATURES],
                          on="participant_id", validate="many_to_one", how="left", indicator=True)
    if not result._merge.eq("both").all():
        raise ValueError("Every trial needs a linked profile")
    result = result.drop(columns="_merge")
    if (result.duration_s <= 0).any():
        raise ValueError("Positive response duration required")
    result["log_duration_s"] = np.log(result.duration_s)
    for c in AUDIO + VIDEO:
        result[f"available_{c}"] = result[c].notna().astype(float)
    return result


def held_out_predictions(frame, columns, *, c=.1):
    if frame.trial_id.duplicated().any() or frame.participant_id.isna().any():
        raise ValueError("Unique trials and non-missing groups required")
    groups = frame.participant_id.to_numpy()
    y = frame.label.to_numpy(dtype=int)
    if not set(y) <= {0, 1} or len(set(y)) != 2 or len(set(groups)) < 3:
        raise ValueError("Two target classes and at least three participants required")
    x = frame[columns].to_numpy(dtype=float)
    if np.isinf(x).any():
        raise ValueError("Infinite predictors are not permitted")
    probabilities = np.full(len(frame), np.nan)
    folds = []
    for participant in sorted(set(groups)):
        train, test = groups != participant, groups == participant
        weights = participant_weights(groups[train])
        prior = float(np.average(y[train], weights=weights))
        if not columns or len(set(y[train])) != 2:
            probabilities[test] = prior
            mode = "training_participant_weighted_prevalence"
        else:
            model = make_pipeline(ParticipantWeightedImputer(),
                                  StandardScaler(), LogisticRegression(C=c, solver="lbfgs", max_iter=3000,
                                                                     random_state=SEED))
            model.fit(x[train], y[train], participantweightedimputer__sample_weight=weights,
                      standardscaler__sample_weight=weights, logisticregression__sample_weight=weights)
            probabilities[test] = model.predict_proba(x[test])[:, 1]
            mode = "logistic_l2"
        folds.append(dict(held_out_participant=participant, train_participants=sorted(set(groups[train])),
            train_trials=int(train.sum()), test_trials=int(test.sum()), training_prior=prior,
            all_missing_training_features=[col for j, col in enumerate(columns) if np.isnan(x[train, j]).all()],
            mode=mode))
    return probabilities, folds


def prediction_metrics(y, probabilities, groups, *, bootstrap=BOOTSTRAPS):
    y, probabilities, groups = np.asarray(y), np.asarray(probabilities), np.asarray(groups)
    weights = participant_weights(groups)
    pred = probabilities >= .5
    sensitivity = float(np.average(pred[y == 1], weights=weights[y == 1]))
    specificity = float(np.average(~pred[y == 0], weights=weights[y == 0]))
    loss = (probabilities-y)**2
    unique = sorted(set(groups))
    by_person = [float(loss[groups == g].mean()) for g in unique]
    idx = np.random.default_rng(SEED).integers(0, len(unique), (bootstrap, len(unique)))
    means = np.asarray(by_person)[idx].mean(axis=1)
    return dict(brier=float(np.mean(by_person)), brier_ci_low=float(np.quantile(means, .025)),
        brier_ci_high=float(np.quantile(means, .975)), balanced_accuracy=(sensitivity+specificity)/2,
        sensitivity=sensitivity, specificity=specificity,
        roc_auc=float(roc_auc_score(y, probabilities, sample_weight=weights)),
        accuracy=float(np.average(pred == y, weights=weights)),
        confusion_counts={"tn": int(((y == 0) & ~pred).sum()), "fp": int(((y == 0) & pred).sum()),
                          "fn": int(((y == 1) & ~pred).sum()), "tp": int(((y == 1) & pred).sum())},
        participant_brier=dict(zip(unique, by_person)),
        uncertainty="95% participant bootstrap conditional on fixed OOF predictions; no refitting")


def compare_profile_increment(frame, base, augmented, *, bootstrap=BOOTSTRAPS):
    groups, y = frame.participant_id.to_numpy(), frame.label.to_numpy()
    unique = sorted(set(groups))
    differences = np.array([np.mean((augmented[groups == g]-y[groups == g])**2 -
                                   (base[groups == g]-y[groups == g])**2) for g in unique])
    idx = np.random.default_rng(SEED).integers(0, len(unique), (bootstrap, len(unique)))
    boot = differences[idx].mean(axis=1)
    return dict(delta_brier=float(differences.mean()), ci_low=float(np.quantile(boot, .025)),
        ci_high=float(np.quantile(boot, .975)), participant_deltas=dict(zip(unique, differences.tolist())),
        direction="augmented_minus_base; negative is better", bootstrap=bootstrap,
        uncertainty="participant cluster bootstrap conditional on fixed out-of-fold predictions; no refitting")


def evaluate_models(frame, *, bootstrap=BOOTSTRAPS, specs=None):
    models, predictions, folds = [], [], {}
    outputs = {}
    for identifier, name, columns in (MODEL_SPECS if specs is None else specs):
        probability, audit = held_out_predictions(frame, columns)
        outputs[identifier] = probability
        models.append(dict(id=identifier, name=name, features=columns,
                           metrics=prediction_metrics(frame.label, probability, frame.participant_id, bootstrap=bootstrap)))
        folds[identifier] = audit
        for row, value in zip(frame.itertuples(), probability):
            predictions.append(dict(trial_id=row.trial_id, participant_id=row.participant_id, label=row.label,
                                    model_id=identifier, probability_contradiction=float(value)))
    return models, pd.DataFrame(predictions), folds, outputs
