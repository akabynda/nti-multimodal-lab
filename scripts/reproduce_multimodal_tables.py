"""Verify all analysis tables from the exported feature CSVs, without raw media."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nti_lab.multimodal_analysis import profile_associations, evaluate_models, compare_profile_increment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    root = args.results.resolve()
    saved = json.loads((root/"summary.json").read_text(encoding="utf-8"))
    participants = pd.read_csv(root/"participant_features.csv")
    trials = pd.read_csv(root/"trial_features.csv")
    features = list(dict.fromkeys(r["feature"] for r in saved["associations"]))
    calculated = profile_associations(participants, features)
    checks = 0

    def check(actual, expected):
        nonlocal checks
        if expected is None:
            assert actual is None
        else:
            assert np.isclose(actual, expected, rtol=1e-9, atol=1e-10), (actual, expected)
        checks += 1

    for actual, expected in zip(calculated, saved["associations"]):
        assert (actual["feature"], actual["profile"]) == (expected["feature"], expected["profile"])
        for key in ("n", "rho", "p", "p_holm", "ci_low", "ci_high"):
            check(actual[key], expected[key])
    models, predictions, _, probabilities = evaluate_models(trials)
    for actual, expected in zip(models, saved["models"]):
        assert actual["id"] == expected["id"]
        for key in ("brier", "roc_auc", "balanced_accuracy", "sensitivity", "specificity", "brier_ci_low", "brier_ci_high"):
            check(actual["metrics"][key], expected["metrics"][key])
    original = pd.read_csv(root/"model_predictions.csv")
    assert predictions[["trial_id", "model_id"]].equals(original[["trial_id", "model_id"]])
    assert np.allclose(predictions.probability_contradiction, original.probability_contradiction, atol=1e-10)
    increment = compare_profile_increment(trials, probabilities["av"], probabilities["av_profiles"])
    for key in ("delta_brier", "ci_low", "ci_high"):
        check(increment[key], saved["profile_increment"][key])
    print(json.dumps(dict(status="passed", scalar_checks=checks, predictions_checked=len(predictions),
                          participant_rows=len(participants), trial_rows=len(trials))))


if __name__ == "__main__":
    main()
