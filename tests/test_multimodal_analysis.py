from itertools import permutations
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from nti_lab.multimodal_analysis import (exact_spearman, holm, held_out_predictions,
    prediction_metrics, compare_profile_increment, encode_profiles, ParticipantWeightedImputer)


def test_exact_tied_ranks_match_independent_enumeration():
    x, y = np.array([1, 2, 2, 4, 5, 6]), np.array([6, 4, 3, 3, 2, 1])
    result = exact_spearman(x, y, bootstrap=100)
    observed = spearmanr(x, y).statistic
    null = [spearmanr(x, p).statistic for p in permutations(y)]
    assert result["rho"] == pytest.approx(observed)
    assert result["p"] == pytest.approx(np.mean(np.abs(null) >= abs(observed)-1e-12))
    assert result["permutations"] == 720


def test_unestimable_hypotheses_keep_their_place_in_family():
    assert exact_spearman([1]*9, range(9))["status"] == "not_estimable"
    assert exact_spearman([1, 2, np.nan], [3, 4, 5])["n"] == 2
    assert holm([.01, .04, None]).tolist() == pytest.approx([.03, .08, 1])


def fixture_trials():
    return pd.DataFrame({"trial_id": list("abcdefgh"), "participant_id": np.repeat(list("ABCD"), 2),
                         "label": [0,1,0,1,1,0,0,1], "x": [1,2,3,np.nan,5,6,7,8]})


def test_held_out_labels_and_extreme_values_cannot_affect_own_model():
    frame = fixture_trials()
    original, folds = held_out_predictions(frame, ["x"])
    changed = frame.copy()
    changed.loc[changed.participant_id == "A", "label"] = [1, 0]
    probabilities, _ = held_out_predictions(changed, ["x"])
    assert probabilities[:2] == pytest.approx(original[:2])
    for fold in folds:
        assert fold["held_out_participant"] not in fold["train_participants"]
    imputer = ParticipantWeightedImputer().fit([[1], [3], [np.nan]], sample_weight=[1,1,1])
    assert imputer.transform([[np.nan], [1e9]]).tolist() == [[2, 1], [1e9, 0]]
    assert imputer.medians_.tolist() == [2]


def test_repeating_one_participants_all_answers_preserves_predictions():
    frame = fixture_trials()
    before, _ = held_out_predictions(frame, ["x"])
    repeated = frame[frame.participant_id == "D"].copy()
    repeated.trial_id += "_copy"
    after, _ = held_out_predictions(pd.concat([frame, repeated], ignore_index=True), ["x"])
    assert after[:len(frame)] == pytest.approx(before, abs=1e-10)


def test_brier_is_participant_macro_and_paired_delta_is_exact():
    y, p, groups = [0, 0, 1], np.array([.2, .2, .7]), ["A", "A", "B"]
    result = prediction_metrics(y, p, groups, bootstrap=100)
    assert result["brier"] == pytest.approx((.04+.09)/2)
    frame = pd.DataFrame({"participant_id": groups, "label": y})
    identical = compare_profile_increment(frame, p, p, bootstrap=100)
    assert identical["delta_brier"] == identical["ci_low"] == identical["ci_high"] == 0


def test_profile_categories_are_unordered_and_balance_not_redefined():
    frame = pd.DataFrame({"participant_id": ["A"], "expression": [2.1], "emotion_positive": [5],
        "emotion_negative": [7], "emotion_balance": [-2],
        "profile_natural_raw": ["Стимлурующий/Содействующий"],
        "profile_demonstrated_raw": ["Контролирующий"]})
    result = encode_profiles(frame)
    assert result.profile_natural_0.iloc[0] == result.profile_natural_3.iloc[0] == 1
    assert result.profile_demonstrated_1.iloc[0] == 1
    assert result.profile_natural_raw.iloc[0] == frame.profile_natural_raw.iloc[0]
    frame.loc[0, "emotion_balance"] = 0
    with pytest.raises(ValueError, match="balance"):
        encode_profiles(frame)
