import json

import numpy as np
import pandas as pd
import pytest

from nti_lab.research import (build_catalog, compare_stories, content_review_report, group_splits,
                              grouped_validation, materialize_content_review, temperament_hypotheses,
                              transcript_windows, validated_trials)


def test_catalog_preserves_text_and_never_makes_topic_into_label():
    text = "Мой обман\nОднажды я опоздал.\nЭкзамен\nЯ сдал."
    doc = {"id": "abc", "text": text, "title": "source", "url": "url",
           "modified_time": "2026-01-01", "extraction_method": "text"}
    spec = {"participant_id": "P1", "source_file_id": "abc",
            "headings": ["Мой обман", "Экзамен"], "participant_mapping_evidence": "filename"}
    catalog = build_catalog([doc], [spec])
    assert [s["text"] for s in catalog] == ["Однажды я опоздал.", "Я сдал."]
    assert all(s["declared_truth_label"] is None for s in catalog)
    truth = build_catalog([doc], [spec], {"label": "truth", "provenance": "explicit-user-statement"})
    assert all(s["declared_truth_label"] == "truth" and
               s["label_provenance"] == "explicit-user-statement" for s in truth)
    with pytest.raises(ValueError):
        build_catalog([doc], [{**spec, "headings": ["unknown"]}])


def test_comparison_finds_paraphrase_but_only_proposes_review():
    stories = [{"participant_id": "P1", "story_id": "forest", "text": "Мы гуляли по лесу с кошкой на поводке.",
                "title": "Лес", "source_url": "url", "text_sha256": "hash"},
               {"participant_id": "P1", "story_id": "exam", "text": "Экзамен по алгебре был очень сложный.",
                "title": "Экзамен", "source_url": "url", "text_sha256": "hash"}]
    transcript = {"model": "test", "segments": [{"start": 10, "end": 20,
                  "text": "Мы гуляли по лесу с кошкой на поводке."}]}
    rows, windows = compare_stories(stories, transcript, "P1")
    assert windows[0]["candidate_story_id"] == "forest"
    assert windows[0]["similarity"] > 0.99
    assert all(r["review_required"] and r["truth_label"] is None for r in rows)
    assert compare_stories(stories, {"segments": []}, "P1") == ([], [])


def test_participants_never_cross_training_and_test():
    participants = np.repeat(["P1", "P2", "P3"], 4)
    folds = group_splits(participants)
    assert len(folds) == 3
    for train, test in folds:
        assert set(participants[train]).isdisjoint(participants[test])
    with pytest.raises(ValueError):
        group_splits(["P1", None])


def test_word_timing_does_not_spread_text_over_long_asr_chunk():
    words = [{"word": f"слово{i}", "start": 120 + i, "end": 120.5 + i} for i in range(10)]
    windows = transcript_windows([{"start": 0, "end": 150, "text": "whole chunk", "words": words}])
    assert windows
    assert all(w["start_s"] >= 120 and w["alignment_unit"] == "word" for w in windows)


def test_evidence_checks_exact_quote_and_preserves_question_context(tmp_path):
    story = {"story_id": "P1_story_01", "participant_id": "P1", "title": "Случай",
             "text": "Я спорил с коллегой.", "source_url": "source", "text_sha256": "hash",
             "declared_truth_label": "truth"}
    doc = {"participant_id": "P1", "model": "test", "segments": [
        {"start": 1, "end": 4, "text": "Вы когда-нибудь спорили?"},
        {"start": 5, "end": 6, "text": "Нет."},
        {"start": 30, "end": 31, "text": "Да."}]}
    (tmp_path / "00001_P1.segments.json").write_text(json.dumps(doc), encoding="utf-8")
    note = {"participant_id": "P1", "story_number": 1, "relation": "contradiction_candidate",
            "source_excerpt": "Я спорил с коллегой.", "spoken_excerpt": "Нет.",
            "start_s": 5, "end_s": 6, "interpretation": "fixture"}
    notes = {"reviewer": "fixture", "notes": [note]}
    rows = materialize_content_review([story], notes, tmp_path)
    assert "Вы когда-нибудь спорили?" in rows[0]["context_excerpt"]
    assert rows[0]["video_truth_label"] is None and rows[0]["audio_verified"] is False
    report = content_review_report([story], rows + [{**rows[0], "comparison_id": "duplicate-evidence"}])
    assert report["stories_reviewed"] == 1 and report["evidence_rows"] == 2
    assert report["stories_with_contradiction_candidates"] == 1
    with pytest.raises(ValueError, match="outside declared interval"):
        materialize_content_review([story], {"reviewer": "fixture", "notes": [{**note, "spoken_excerpt": "Да."}]}, tmp_path)
    with pytest.raises(ValueError, match="Source quote not found"):
        materialize_content_review([story], {"reviewer": "fixture", "notes": [{**note, "source_excerpt": "other"}]}, tmp_path)


def test_label_provenance_and_nonfinite_values_gate_training():
    data = pd.DataFrame({"participant_id": ["A", "B", "C", "D"], "condition": ["truth", "lie", "unknown", "truth"],
                         "label_source": ["expert", "", "expert", "expert"], "reviewed": [True, True, True, False],
                         "feature": [np.inf, 2, 3, 4]})
    clean, audit = validated_trials(data, ["feature"])
    assert clean.empty and audit["excluded_rows"] == 4
    assert grouped_validation(data, ["feature"])["status"] == "blocked"


def test_grouped_model_runs_only_on_verified_fixture_and_rejects_wrong_taxonomy():
    # Deterministic synthetic fixture validates mechanics, never enters study results.
    rows = [{"participant_id": f"P{i}", "condition": condition, "label_source": "synthetic-test-only",
             "reviewed": True, "x": value + i * .01, "temperament": "Содействующий", "temperament_source": "fixture"}
            for i in range(6) for condition, value in [("truth", 0.), ("lie", 1.)]]
    frame = pd.DataFrame(rows)
    result = grouped_validation(frame, ["x"])
    assert result["status"] == "exploratory"
    assert len(result["out_of_fold_probability"]) == len(frame)
    assert all(set(f["train_participants"]).isdisjoint(f["test_participants"]) for f in result["folds"])
    assert grouped_validation(frame, ["x"], True)["status"] == "blocked"
    assert temperament_hypotheses(frame, ["x"])["status"] == "blocked"
