"""Baseline normalisation, participant contrasts and exact permutation tests."""
import numpy as np
import pandas as pd

from nti_lab.contrasts import (ANOMALY, Z_LIMIT, group_difference_test, hypothesis_tests, normalize,
                               participant_contrasts, sign_flip_test)


def answers(pid="P1", shift=0.):
    rows = [("base", 10), ("base", 12), ("truth", 11), ("truth", 13), ("lie", 15 + shift), ("lie", 17 + shift)]
    return pd.DataFrame([dict(trial_id=f"{pid}-{i}", participant_id=pid, condition=c, feature=v, audio_rms_dbfs=v)
                         for i, (c, v) in enumerate(rows)])


def test_normalize_uses_base_median_and_answer_spread():
    z = normalize(answers(), features=["feature", "audio_rms_dbfs"])
    values = np.array([10, 12, 11, 13, 15, 17.])
    spread = np.subtract(*np.quantile(values, [.75, .25])) / 1.349
    assert np.allclose(z.feature, (values - 11) / spread)
    assert np.allclose(z[ANOMALY], np.abs(z.audio_rms_dbfs))


def test_normalize_requires_base_and_clips_outliers():
    frame = answers()
    frame.loc[frame.condition.eq("base"), "feature"] = [np.nan, 12]
    assert normalize(frame, features=["feature"]).feature.isna().all()
    frame = pd.concat([answers(), answers()], ignore_index=True).assign(trial_id=lambda f: f.index.astype(str))
    frame.loc[0, "feature"] = 1000
    assert normalize(frame, features=["feature"]).feature.max() == Z_LIMIT


def test_contrast_is_lie_minus_truth_mean():
    z = normalize(answers(), features=["feature"])
    row = participant_contrasts(z, ["feature"]).iloc[0]
    assert np.isclose(row.contrast, z.feature[4:].mean() - z.feature[2:4].mean())


def test_sign_flip_test_is_exact():
    result = sign_flip_test([1, 1, 1, 1, 1, 1])
    assert result["p"] == 2 / 64 and result["mean"] == 1
    assert sign_flip_test([1, 2])["status"] == "not_estimable"


def test_group_difference_test_enumerates_assignments():
    result = group_difference_test([5, 5, 5, 0, 0, 0], [1, 1, 1, 0, 0, 0])
    assert result["difference"] == 5 and np.isclose(result["p"], 2 / 20)
    assert group_difference_test([1, 2, 3, 4], [1, 1, 0, 0])["status"] == "not_estimable"


def test_hypothesis_tests_adjust_each_family():
    contrasts = pd.DataFrame([dict(participant_id=f"P{i}", feature=f, contrast=v)
                              for f, values in (("a", range(1, 10)), ("b", [0, 1, -1, 2, -2, 0, 1, -1, 0.5]))
                              for i, v in enumerate(values)])
    profiles = pd.DataFrame(dict(participant_id=[f"P{i}" for i in range(9)], expression=range(9),
                                 group=[1, 1, 1, 1, 1, 1, 0, 0, 0]))
    tests = hypothesis_tests(contrasts, profiles, "group").set_index("feature")
    assert tests.loc["a", "h1_p"] == 2 / 512
    assert tests.loc["a", "h1_p_holm"] == min(1, 2 * tests.loc["a", "h1_p"])
    assert np.isclose(tests.loc["a", "h3_rho"], 1)
