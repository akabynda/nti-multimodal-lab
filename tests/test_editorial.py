import copy

import pytest

from nti_lab.editorial import apply_editorial, join_fragments


def sample():
    source = {"recording_id": "test", "segments": [
        {"id": 3, "start": 1., "end": 3., "text": "Где вы учитесь?", "speaker_id": "A"},
        {"id": 7, "start": 4., "end": 5., "text": "Я учусь", "speaker_id": "B"},
        {"id": 8, "start": 5.1, "end": 6., "text": "в ЭТМО.", "speaker_id": "unknown"},
    ]}
    decision = {"recording_id": "test", "source_sha256": "fixture", "reviewed_segment_ids": [3, 7, 8],
                "roles": [{"segment_id": i, "role": role} for i, role in [(3, "interviewer"), (7, "respondent"), (8, "respondent")]],
                "corrections": [{"segment_id": 8, "before": "в ЭТМО.", "after": "в ИТМО.", "reason": "University spelling"}],
                "merge_groups": [[7, 8]]}
    return source, decision


def test_editorial_preserves_source_and_joins_answer_across_cluster_change():
    source, decision = sample()
    original = copy.deepcopy(source)
    result = apply_editorial(source, decision)
    assert source == original
    assert result["segments"][2]["speaker_id"] == "unknown"
    assert result["segments"][2]["speaker_role"] == "respondent"
    assert result["segments"][2]["pre_editorial_text"] == "в ЭТМО."
    assert result["turns"][1]["text"] == "Я учусь в ИТМО."
    assert result["turns"][1]["segment_ids"] == [7, 8]
    assert result["qa_candidates"][0]["answer"] == "Я учусь в ИТМО."
    assert result["qa_candidates"][0]["latency_s"] == 1
    assert result["editorial_review"]["manual_listening_performed"] is False


@pytest.mark.parametrize("bad", [[3, 7], [3, 7, 7, 8]])
def test_review_rejects_missing_or_duplicate_coverage(bad):
    source, decision = sample()
    decision["reviewed_segment_ids"] = bad
    with pytest.raises(ValueError, match="exactly once"):
        apply_editorial(source, decision)


def test_editorial_rejects_stale_correction():
    source, decision = sample()
    decision["corrections"][0]["before"] = "Not the source"
    with pytest.raises(ValueError, match="text correction"):
        apply_editorial(source, decision)


@pytest.mark.parametrize("group", [[3, 8], [3, 7]])
def test_merge_rejects_nonadjacent_or_different_speakers(group):
    source, decision = sample()
    decision["merge_groups"] = [group]
    with pytest.raises(ValueError):
        apply_editorial(source, decision)


def test_display_split_preserves_source_text_and_timing():
    source, decision = sample()
    source["segments"][0]["text"] = "Вы готовы? Да."
    decision["split_segments"] = [{"segment_id": 3, "parts": [
        {"text": "Вы готовы?", "start": 1., "end": 2., "role": "interviewer"},
        {"text": "Да.", "start": 2., "end": 3., "role": "respondent"},
    ]}]
    result = apply_editorial(source, decision)
    assert len(result["turns"]) == 3
    assert result["segments"][0]["text"] == source["segments"][0]["text"]
    pair = result["qa_candidates"][0]
    assert pair["question_turn_ids"] == ["3.0"]
    assert pair["answer_turn_ids"] == ["3.1", 7]
    assert pair["question_segment_ids"] == [3]
    assert pair["answer_segment_ids"] == [3, 7, 8]
    decision["split_segments"][0]["parts"][1]["text"] = "Иные слова"
    with pytest.raises(ValueError, match="preserve"):
        apply_editorial(source, decision)


def test_join_handles_split_hyphenated_words():
    assert join_fragments(["Кто", "-то из них?"]) == "Кто-то из них?"
    assert join_fragments(["В", "какой-то степени."]) == "В какой-то степени."


def test_timing_correction_reorders_working_copy_without_changing_source():
    source, decision = sample()
    source["duration_s"] = 10
    decision["timing_corrections"] = [{"segment_id": 7, "before_start": 4., "before_end": 5.,
        "start": 7., "end": 8., "words": [{"start": 7., "end": 8., "word": "Я учусь"}],
        "reason": "Repeat ASR timing", "evidence": "test.json"}]
    decision["merge_groups"] = []
    result = apply_editorial(source, decision)
    assert [s["id"] for s in result["segments"]] == [3, 8, 7]
    assert [s["id"] for s in source["segments"]] == [3, 7, 8]
    decision["timing_corrections"][0]["words"][0]["end"] = 9.
    with pytest.raises(ValueError, match="word outside"):
        apply_editorial(source, decision)


def test_recovered_answer_keeps_original_review_coverage():
    source, decision = sample()
    source["duration_s"] = 10
    decision["recovered_segments"] = [{"id": 20, "start": 3.1, "end": 3.8, "text": "Да.", "role": "respondent",
        "words": [{"start": 3.1, "end": 3.8, "word": "Да."}], "reason": "Repeated decoding", "evidence": "clip.json"}]
    result = apply_editorial(source, decision)
    assert result["editorial_review"]["reviewed_segments"] == 3
    assert result["editorial_review"]["recovered_segments"] == 1
    assert [s["id"] for s in result["segments"]] == [3, 20, 7, 8]
    assert result["qa_candidates"][0]["answer"] == "Да. Я учусь в ИТМО."


def test_lexical_comparison_uses_corrected_text_instead_of_stale_words():
    from nti_lab.research import transcript_windows
    source, decision = sample()
    source["segments"][2]["words"] = [{"start": 5.1, "end": 6., "word": "в ЭТМО."}]
    result = apply_editorial(source, decision)
    text = " ".join(w["text"] for w in transcript_windows(result["segments"]))
    assert "ИТМО" in text and "ЭТМО" not in text


def test_repeat_asr_timing_provenance_survives_different_editorial_text():
    from nti_lab.research import transcript_windows
    source, decision = sample()
    source["duration_s"] = 10
    source["segments"][2]["words"] = [{"start": 5.1, "end": 6., "word": "в ЭТМО."}]
    repeated_words = [{"start": 5.2, "end": 6.1, "word": "в ЭТМО."}]
    decision["timing_corrections"] = [{"segment_id": 8, "before_start": 5.1, "before_end": 6.,
        "start": 5.2, "end": 6.1, "words": repeated_words,
        "reason": "Repeat ASR timing", "evidence": "repeat.json"}]
    result = apply_editorial(source, decision)
    segment = next(s for s in result["segments"] if s["id"] == 8)
    assert segment["word_alignment_status"] == "repeat_ASR_word_times_not_realigned_after_editorial_text_change"
    assert segment["words"] == repeated_words
    assert segment["pre_editorial_timing"]["words"] == source["segments"][2]["words"]
    text = " ".join(w["text"] for w in transcript_windows(result["segments"]))
    assert "ИТМО" in text and "ЭТМО" not in text
