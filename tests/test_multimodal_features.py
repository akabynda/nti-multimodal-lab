"""Synthetic checks for temporal boundaries and modality-specific quality gates."""
import json

import numpy as np
import pandas as pd
import pytest

from nti_lab.multimodal_features import (
    FEATURE_COLUMNS, extract_session_features, extract_trial_features,
)


PID = "TestParticipant"
RID = "00001_TestParticipant"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def trial(start=0, end=25, trial_id="answer"):
    return {"trial_id": trial_id, "recording_id": RID, "participant_id": PID,
            "start_s": start, "end_s": end}


def segment(start, end, role="respondent"):
    scale = (end - start) / 10
    return {"start": start, "end": end, "speaker_role": role,
            "text": "one two three four five",
            "words": [{"start": start + a * scale, "end": start + b * scale, "word": word}
                      for a, b, word in [(0.5, .8, "one"), (2, 2.4, "two"),
                                         (3, 3.4, "three"), (5, 5.4, "four"),
                                         (7, 7.4, "five")]]}


def metric(center, hr=60, quality="usable_automatic", rmssd=20):
    return {"window_start_s": center - 60, "window_end_s": center + 60,
            "center_s": center, "ecg_quality": quality, "hr_bpm": hr,
            "rmssd_ms": rmssd, "sdnn_ms": rmssd + 10,
            "resp_bpm": 12, "resp_status": "RIP_periodic_candidate"}


@pytest.fixture
def recording(tmp_path):
    out = tmp_path / "outputs"
    segments = [segment(0, 10), segment(10, 15, "interviewer"), segment(15, 25)]
    audio = {"recording_id": RID, "participant_id": PID, "duration_s": 180,
             "segments": segments, "turns": segments,
             "speech_intervals": [[0, 8], [10, 15], [15, 23]]}
    write_json(out / "audio" / f"{RID}.segments.json", audio)
    acoustic = []
    for start, end, pitch in [(0, 5, 100), (5, 10, 100), (7.5, 12.5, 800),
                              (10, 15, 900), (15, 20, 100), (20, 25, 100),
                              (23, 28, 700)]:
        acoustic.append({"start_s": start, "end_s": end, "time_s": (start + end) / 2,
                         "pitch_median_hz": pitch, "pitch_iqr_hz": 5,
                         "rms_dbfs": -20 if pitch == 100 else 0,
                         "speech_fraction": .8, "voiced_fraction": 1})
    pd.DataFrame(acoustic).to_csv(out / "audio" / f"{RID}.features.csv", index=False)
    write_json(out / "video" / f"{RID}.json", {
        "recording_id": RID, "participant_id": PID, "duration_s": 180,
        "source_width": 100, "source_height": 100, "analysis_fps": 12.5,
        "times": [], "kinematics": [], "overlays": [], "windows": [],
    })
    write_json(out / "sync" / f"{RID}.json", {
        "recording_id": RID, "participant_id": PID,
        "bio_to_video_offset_s": 0, "scale": 1, "verified": True,
        "status": "manual_verified",
        "media_timebase": {"audio_first_decoded_pts_s": 1.04,
                           "video_first_decoded_pts_s": 1.04,
                           "sample_aspect_ratio": 1},
    })
    bio = out / "bio" / PID
    write_json(bio / "metadata.json", {
        "subject_id": PID, "analysis_duration_s": 180,
        "gsr_quality": {"status": "usable_automatic"},
    })
    pd.DataFrame([metric(60)]).to_csv(bio / "metrics.csv", index=False)
    pd.DataFrame({"time_s": [2, 4, 6, 8, 16, 18, 20, 22],
                  "rr_ms": [1000] * 8, "accepted": [True] * 8}).to_csv(
        bio / "rr_intervals.csv", index=False)
    np.savez(bio / "signals.npz", gsr_t_s=np.arange(181),
             gsr_us=2 + np.arange(181) / 100)
    return tmp_path


def read_json(root, relative):
    return json.loads((root / "outputs" / relative).read_text(encoding="utf-8"))


def assert_reason(qc, trial_id, feature):
    assert qc["rows"][trial_id]["feature_reasons"][feature]


def test_missing_modalities_remain_nan_with_feature_reasons(tmp_path):
    frame, qc = extract_trial_features(tmp_path, [trial()])
    assert len(frame) == 1
    assert set(FEATURE_COLUMNS).issubset(frame.columns)
    assert frame.loc[0, list(FEATURE_COLUMNS)].isna().all()
    for feature in FEATURE_COLUMNS:
        assert_reason(qc, "answer", feature)


@pytest.mark.parametrize("override, expected_bounds", [(None, (60, 65)), (-10, (55, 60))])
def test_inverse_clock_and_absolute_offset_override(recording, override, expected_bounds):
    sync = read_json(recording, f"sync/{RID}.json")
    sync.update(bio_to_video_offset_s=-20, scale=2, verified=True)
    write_json(recording / "outputs/sync" / f"{RID}.json", sync)
    overrides = None if override is None else {RID: override}
    frame, _ = extract_trial_features(recording, [trial(100, 110)], offset_overrides=overrides)
    row = frame.iloc[0]
    assert (row.bio_start_s, row.bio_end_s) == pytest.approx(expected_bounds)
    assert row.sync_offset_s == (-20 if override is None else override)
    assert row.sync_scale == 2
    if override is not None:
        assert not bool(row.sync_verified)


def test_audio_windows_require_full_respondent_and_trial_containment(recording):
    frame, _ = extract_trial_features(recording, [trial()])
    row = frame.iloc[0]
    assert row.audio_window_count == 4
    assert row.audio_pitch_median_hz == 100
    assert row.audio_pitch_iqr_hz == 5
    assert row.audio_rms_dbfs == -20
    assert row.audio_text_rate_wpm == pytest.approx(30)
    assert row.audio_vad_pause_fraction == pytest.approx(.2)


def test_stale_word_alignment_is_excluded_from_word_pauses(recording):
    audio = read_json(recording, f"audio/{RID}.segments.json")
    audio["segments"][0].update(
        word_alignment_status="edited_text_not_realigned",
        text="one two",
        words=[{"start": 0, "end": .1, "word": "one"},
               {"start": 9, "end": 9.1, "word": "two"}],
    )
    audio["segments"][2]["words"] = [
        {"start": 15, "end": 15.5, "word": "one"},
        {"start": 17, "end": 17.5, "word": "two"},
    ]
    audio["segments"][2]["text"] = "one two"
    audio["turns"] = audio["segments"]
    write_json(recording / "outputs/audio" / f"{RID}.segments.json", audio)
    frame, _ = extract_trial_features(recording, [trial()])
    assert frame.iloc[0].audio_stale_segment_count == 1
    assert frame.iloc[0].audio_word_pause_median_s == pytest.approx(1.5)


def test_nearest_hrv_window_failing_qc_has_no_distant_usable_fallback(recording):
    metrics = [metric(60, hr=60), metric(65, hr=999, quality="review_required"),
               metric(70, hr=120)]
    pd.DataFrame(metrics).to_csv(recording / "outputs/bio" / PID / "metrics.csv", index=False)
    frame, qc = extract_trial_features(recording, [trial(62, 68)])
    row = frame.iloc[0]
    for name in ("bio_ecg_hr_bpm_context120s", "bio_ecg_rmssd_ms_context120s",
                 "bio_ecg_sdnn_ms_context120s"):
        assert np.isnan(row[name])
        assert_reason(qc, "answer", name)
    assert row.bio_context_window_count == 1
    assert row.bio_context_ecg_usable_count == 0


def test_rr_endpoints_and_local_qc_gate_direct_hr(recording):
    bio = recording / "outputs/bio" / PID
    pd.DataFrame([metric(60), metric(65, quality="review_required")]).to_csv(
        bio / "metrics.csv", index=False)
    pd.DataFrame({"time_s": [60, 61, 62, 64, 65, 66],
                  "rr_ms": [500, 1000, 1000, 500, 500, 500],
                  "accepted": [True, True, True, True, False, True]}).to_csv(
        bio / "rr_intervals.csv", index=False)
    frame, _ = extract_trial_features(recording, [trial(60, 66)])
    assert frame.iloc[0].bio_rr_accepted_count == 2
    assert frame.iloc[0].bio_ecg_hr_bpm == pytest.approx(60)


def test_bad_gsr_is_missing_instead_of_a_plausible_number(recording):
    metadata = read_json(recording, f"bio/{PID}/metadata.json")
    metadata["gsr_quality"]["status"] = "flat_or_out_of_range_review_required"
    write_json(recording / "outputs/bio" / PID / "metadata.json", metadata)
    frame, qc = extract_trial_features(recording, [trial()])
    for feature in ("bio_gsr_median_us", "bio_gsr_iqr_us"):
        assert np.isnan(frame.iloc[0][feature])
        assert_reason(qc, "answer", feature)


def test_rppg_hr_requires_quality_and_whole_trial_containment(recording):
    video = read_json(recording, f"video/{RID}.json")
    video["windows"] = [
        {"start_s": start, "end_s": start + 30, "rppg_hr_bpm": hr,
         "face_coverage": face, "spectral_concentration": concentration}
        for start, hr, face, concentration in [
            (10, 60, 1, .35), (15, 80, 1, .8), (20, 200, .99, .9),
            (20, 220, 1, .349), (0, 240, 1, .9), (25, 260, 1, .9),
        ]
    ]
    write_json(recording / "outputs/video" / f"{RID}.json", video)
    frame, _ = extract_trial_features(recording, [trial(10, 50)])
    assert frame.iloc[0].video_rppg_hr_bpm == pytest.approx(70)


def test_session_context_aggregates_only_quality_windows(recording):
    metrics = [metric(60, hr=60, rmssd=20),
               metric(65, hr=999, rmssd=999, quality="review_required"),
               metric(70, hr=100, rmssd=40)]
    pd.DataFrame(metrics).to_csv(recording / "outputs/bio" / PID / "metrics.csv", index=False)
    frame, _ = extract_session_features(recording)
    assert len(frame) == 1
    row = frame.iloc[0]
    assert row.bio_context_window_count == 3
    assert row.bio_context_ecg_usable_count == 2
    assert row.bio_ecg_hr_bpm_context120s == pytest.approx(80)
    assert row.bio_ecg_rmssd_ms_context120s == pytest.approx(30)
    assert row.bio_ecg_sdnn_ms_context120s == pytest.approx(40)


def test_low_session_quality_coverage_does_not_suppress_good_trials(recording):
    metrics = []
    for i in range(11):
        row = metric(60 + i * 5, quality="usable_automatic" if i < 2 else "review_required")
        row["resp_status"] = "RIP_periodic_candidate" if i < 2 else "RIP_low_periodicity"
        metrics.append(row)
    pd.DataFrame(metrics).to_csv(recording / "outputs/bio" / PID / "metrics.csv", index=False)
    video = read_json(recording, f"video/{RID}.json")
    video["windows"] = [
        {"start_s": i * 5, "end_s": i * 5 + 30, "rppg_hr_bpm": 60,
         "face_coverage": 1, "spectral_concentration": .8 if i == 0 else .1}
        for i in range(6)
    ]
    write_json(recording / "outputs/video" / f"{RID}.json", video)
    session, qc = extract_session_features(recording)
    for feature in ("bio_ecg_hr_bpm_context120s", "bio_rip_rate_bpm_context120s",
                    "video_rppg_hr_bpm"):
        assert np.isnan(session.iloc[0][feature])
        assert_reason(qc, f"session:{RID}", feature)
    trials, _ = extract_trial_features(recording, [trial(0, 30)])
    assert trials.iloc[0].bio_ecg_hr_bpm_context120s == 60
    assert trials.iloc[0].bio_rip_rate_bpm_context120s == 12
    assert trials.iloc[0].video_rppg_hr_bpm == 60


def test_video_geometry_respects_pixel_aspect_visibility_and_missing_frames(recording):
    video = read_json(recording, f"video/{RID}.json")
    sync = read_json(recording, f"sync/{RID}.json")
    sync["media_timebase"]["sample_aspect_ratio"] = 2
    write_json(recording / "outputs/sync" / f"{RID}.json", sync)
    video["kinematics"] = [
        {"t": 0, "head_x_velocity": 3, "head_y_velocity": 4, "face_visible": True,
         "torso_tilt_deg_velocity": -2, "shoulders_visible": True, "hips_visible": True},
        {"t": 1, "head_x_velocity": 999, "head_y_velocity": 999, "face_visible": False,
         "torso_tilt_deg_velocity": 999, "shoulders_visible": False, "hips_visible": True},
    ]
    overlays = []
    for t, half_opening in [(0, .1), (1, .2), (2, None), (3, .1)]:
        points = [[.1, .5], [.2, .5 - half_opening], [.3, .5], [.2, .5 + half_opening]] if half_opening else []
        overlays.append({"t": t, "rois": {"eyes": points * 2, "mouth": points}})
    video["overlays"] = overlays
    write_json(recording / "outputs/video" / f"{RID}.json", video)
    frame, _ = extract_trial_features(recording, [trial(0, 4)])
    row = frame.iloc[0]
    assert row.video_head_speed_norm_per_s == 5
    assert row.video_torso_motion_deg_per_s == 2
    assert row.video_face_fraction == .5
    for name in ("eye", "mouth"):
        # Display width is doubled by SAR, so the opening ratio is halved.
        assert row[f"video_{name}_opening_ratio"] == pytest.approx(.5)
        assert row[f"video_{name}_motion_ratio_per_s"] == pytest.approx(.5)
        assert row[f"video_{name}_sample_count"] == 3


def test_text_mismatched_word_times_are_stale_even_without_explicit_flag(recording):
    audio = read_json(recording, f"audio/{RID}.segments.json")
    audio["segments"][0]["text"] = "Corrected wording"
    audio["turns"] = audio["segments"]
    write_json(recording / "outputs/audio" / f"{RID}.segments.json", audio)
    frame, _ = extract_trial_features(recording, [trial()])
    assert frame.iloc[0].audio_stale_segment_count == 1
    assert frame.iloc[0].audio_trusted_pause_count == 4
