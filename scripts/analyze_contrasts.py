"""Normalise answer features to «База», test truth/lie contrasts and profile groups.

Inputs: outputs/research/multimodal/trial_features.csv (labelled answers), saved
audio/video/bio/sync outputs and data/private/participant_profiles.csv.
Output: outputs/research/contrasts/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

from nti_lab import contrasts as lib
from nti_lab import multimodal_analysis as analysis
from nti_lab.multimodal_features import extract_trial_features
from nti_lab.session_summary import session_stages

ROOT = Path(__file__).resolve().parents[1]
GROUP = "profile_natural_leading_cooperative"


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def base_trials(root: Path, labelled: set[str]) -> list[dict]:
    trials = []
    for path in sorted((root/"outputs/audio").glob("*.segments.json")):
        segments = json.loads(path.read_text(encoding="utf-8"))
        rid = segments["recording_id"]
        sync = json.loads((root/"outputs/sync"/f"{rid}.json").read_text(encoding="utf-8"))
        for qa in session_stages(segments, sync.get("baselines", []))["neutral_answers"]:
            tid = f"{rid}:q{qa['question_segment_id']}:a{qa['answer_segment_id']}"
            if tid not in labelled:
                trials.append(dict(trial_id=tid, recording_id=rid, participant_id=segments["participant_id"],
                                   start_s=qa["answer_start_s"], end_s=qa["answer_end_s"]))
    return trials


def leading_cooperative(raw) -> float:
    tokens = [analysis.TYPE_CORRECTIONS.get(t.strip().lower(), t.strip().lower()) for t in str(raw).split("/") if t.strip()]
    return float(bool(tokens) and tokens[0] == "содействующий")


def models(labelled: pd.DataFrame, normalized: pd.DataFrame, profiles: pd.DataFrame) -> tuple[list[dict], dict]:
    z = normalized.set_index("trial_id")[lib.CONTRAST_FEATURES].add_prefix("z_")
    frame = labelled.merge(z, left_on="trial_id", right_index=True, how="left", validate="one_to_one")
    zcols = [f"z_{c}" for c in lib.CONTRAST_FEATURES]
    specs = [("raw", "Исходные признаки", lib.CONTRAST_FEATURES),
             ("normalized", "Нормированные к «Базе»", zcols),
             ("normalized_profiles", "Нормированные к «Базе» + профили", zcols + analysis.PROFILE_FEATURES)]
    results, probabilities = [], {}
    for identifier, name, columns in specs:
        probability, _ = analysis.held_out_predictions(frame, columns)
        probabilities[identifier] = probability
        results.append(dict(id=identifier, name=name, features=columns,
                            metrics=analysis.prediction_metrics(frame.label, probability, frame.participant_id)))
    increments = {"normalized_minus_raw": analysis.compare_profile_increment(frame, probabilities["raw"], probabilities["normalized"]),
                  "profiles_on_normalized": analysis.compare_profile_increment(frame, probabilities["normalized"], probabilities["normalized_profiles"])}
    return results, increments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"outputs/research/contrasts")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    labelled = pd.read_csv(ROOT/"outputs/research/multimodal/trial_features.csv")
    trials = base_trials(ROOT, set(labelled.trial_id))
    base, _ = extract_trial_features(ROOT, trials)
    dialogue = {}
    for path in sorted((ROOT/"outputs/audio").glob("*.segments.json")):
        dialogue.update(lib.dialogue_features(json.loads(path.read_text(encoding="utf-8"))))
    for frame in (labelled, base):
        for column in lib.DIALOGUE:
            frame[column] = [dialogue.get(t, {}).get(column, np.nan) for t in frame.trial_id]
    columns = ["trial_id", "participant_id", "recording_id", "start_s", "end_s", *lib.CONTRAST_FEATURES]
    answers = pd.concat([labelled[columns].assign(condition=np.where(labelled.label.eq(1), "lie", "truth")),
                         base[columns].assign(condition="base")], ignore_index=True)
    normalized = lib.normalize(answers)
    features = lib.CONTRAST_FEATURES + [lib.ANOMALY]
    contrasts = lib.participant_contrasts(normalized, features)
    profiles = analysis.encode_profiles(pd.read_csv(ROOT/"data/private/participant_profiles.csv"))
    profiles[GROUP] = profiles.profile_natural_raw.map(leading_cooperative)
    tests = lib.hypothesis_tests(contrasts, profiles[["participant_id", "expression", GROUP]], GROUP)
    model_results, increments = models(labelled.merge(profiles[["participant_id"]], on="participant_id"), normalized, profiles)
    counts = answers.groupby(["participant_id", "condition"]).size().unstack(fill_value=0).reset_index()
    answers.to_csv(out/"answers.csv", index=False, encoding="utf-8-sig")
    normalized.to_csv(out/"normalized_answers.csv", index=False, encoding="utf-8-sig")
    contrasts.to_csv(out/"participant_contrasts.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(out/"hypothesis_tests.csv", index=False, encoding="utf-8-sig")
    summary = dict(schema_version=2, expected={k: {"direction": d, "source": src} for k, (d, src) in lib.EXPECTED.items()}, labels={"truth": "label 0", "lie": "label 1", "base": "neutral question block answer"},
                   normalization="(x − participant median of base) / (participant IQR of all answers / 1.349)",
                   min_base=lib.MIN_BASE, min_scale_answers=lib.MIN_SCALE, min_group=lib.MIN_GROUP,
                   group={"column": GROUP, "definition": "first natural profile type is содействующий",
                          "participants": profiles.loc[profiles[GROUP].eq(1), "participant_id"].tolist()},
                   counts=counts.to_dict("records"), tests=tests.to_dict("records"),
                   models=model_results, increments=increments)
    (out/"summary.json").write_text(json.dumps(clean(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(tests[["feature", "h1_n", "h1_mean", "h1_p_holm", "h2_difference", "h2_p_holm", "h3_rho", "h3_p_holm"]].to_string())
    print(json.dumps(clean({m["id"]: {k: m["metrics"][k] for k in ("brier", "roc_auc", "balanced_accuracy")} for m in model_results}), indent=1))


if __name__ == "__main__":
    main()
