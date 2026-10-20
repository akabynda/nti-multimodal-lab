"""Numerical and temporal checks for audio outputs; no downloaded model needed."""
import numpy as np

from nti_lab.audio import SR, acoustic_windows, bound_timestamps, interval_duration, merge_intervals, question_answer_candidates


def test_overlap_never_double_counts_speech_or_pauses():
    speech = merge_intervals([(1, 3), (2, 4), (5, 6)])
    assert interval_duration(speech, 0, 10) == 4
    assert interval_duration(speech, 2, 5.5) == 2.5
    assert interval_duration(speech, 7, 8) == 0


def test_question_gap_and_answer_pauses_have_distinct_clocks():
    segments = [
        {"id": 1, "start": 0, "end": 2, "text": "Как дела?"},
        {"id": 2, "start": 3, "end": 8, "text": "Всё хорошо."},
    ]
    row = question_answer_candidates(segments, [(0, 2), (3, 4), (6, 8)])[0]
    assert row["latency_s"] == 1
    assert row["answer_duration_s"] == 5
    assert row["within_answer_nonspeech_s"] == 2
    assert row["status"] == "candidate_requires_expert_confirmation"


def test_pitch_tracks_known_harmonic_signal_and_silence_is_not_voiced():
    t = np.arange(SR * 6) / SR
    y = (0.2 * np.sin(2 * np.pi * 180 * t) +
         0.1 * np.sin(2 * np.pi * 360 * t)).astype(np.float32)
    rows = acoustic_windows(y, [(0, 6)], step=5, window=5)
    assert abs(rows[0]["pitch_median_hz"] - 180) < 2
    assert rows[0]["clipping_fraction"] == 0
    silent = acoustic_windows(np.zeros(SR * 6, np.float32), [], step=5, window=5)
    assert silent[0]["pitch_median_hz"] is None
    assert silent[0]["speech_fraction"] == 0


def test_asr_recording_edge_is_bounded_without_losing_model_provenance():
    segments = [{"id": 0, "start": 9.8, "end": 10.4, "text": "Да.",
                 "words": [{"start": 9.8, "end": 10.4, "word": "Да."}]}]
    bound_timestamps(segments, 10)
    assert segments[0]["end"] == 10
    assert segments[0]["words"][0]["end"] == 10
    assert segments[0]["raw_model_end"] == 10.4
    assert segments[0]["words"][0]["raw_model_end"] == 10.4
