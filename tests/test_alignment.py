import numpy as np
import pytest

from nti_lab.alignment import approximate_ear, baseline_windows, estimate_alignment, pulse_template
from nti_lab.biosignals import clean_rr


def synthetic(variable=True, polarity=1):
    rng = np.random.default_rng(20261020)
    peaks, now = [], .5
    while now < 400:
        peaks.append(now)
        now += .75 + (.08 * np.sin(now * .13) + rng.normal(0, .025) if variable else 0)
    t, pulse = pulse_template(peaks, 400)
    video_t = np.arange(0, 380, .08)
    lead = 6.32
    y = polarity * np.interp(video_t + lead, t, pulse)
    y += rng.normal(0, .005, len(y)) if variable else 0
    video = {"recording_id": "sample", "duration_s": 380,
             "times": video_t, "rppg": y, "overlays": []}
    meta = {"analysis_duration_s": 400, "subject_id": "sample"}
    asr = {"segments": [{"start": 5, "end": 10, "text": "Посидите с закрытыми глазами"},
                        {"start": 250, "end": 255, "text": "Снова закройте глаза"}]}
    rr_t, rr, valid = clean_rr(np.asarray(peaks))
    bio = {"r_peaks_s": np.asarray(peaks), "rr_t_s": rr_t, "rr_ms": rr, "rr_valid": valid,
           "ecg_t_s": t, "ecg_mv": pulse * 5, "ecg_filtered_mv": pulse * 5}
    return video, bio, meta, asr, lead


def test_recovers_bounded_candidate_under_polarity_inversion():
    video, bio, metadata, asr, expected = synthetic(polarity=-1)
    result = estimate_alignment(video, bio, metadata, asr)
    assert result["bio_lead_s"] == pytest.approx(expected, abs=.06)
    assert result["bio_to_video_offset_s"] == -result["bio_lead_s"]
    assert result["verified"] is False
    assert result["scale"] == 1
    assert 0 <= result["bio_lead_s"] <= 20
    assert result["status"] == "supported_unverified"


def test_correlated_pulse_with_rejected_rr_is_not_supported():
    video, bio, metadata, asr, _ = synthetic()
    bio["rr_valid"][:] = False
    result = estimate_alignment(video, bio, metadata, asr)
    assert result["status"] == "ambiguous_unverified"
    assert all(not b["ecg_quality"]["usable"] for b in result["baselines"])


def test_periodic_signal_keeps_alias_ambiguity():
    video, bio, metadata, asr, _ = synthetic(variable=False)
    result = estimate_alignment(video, bio, metadata, asr)
    assert result["status"] == "ambiguous_unverified"
    assert result["lead_uncertainty_interval_s"] == [0, 20]
    assert result["joint_support"]["margin"] < .04


def test_missing_rppg_returns_prior_not_success():
    video, bio, metadata, asr, _ = synthetic()
    video["rppg"] = np.full(len(video["times"]), np.nan)
    result = estimate_alignment(video, bio, metadata, asr)
    assert result["status"] == "insufficient_signal"
    assert result["candidate_basis"] == "duration_midpoint_prior_no_signal_evidence"
    assert result["confidence"] == "none"
    assert result["bio_lead_s"] == 10


def test_disagreeing_baseline_offsets_are_not_promoted_to_drift_fit():
    video, bio, metadata, asr, lead = synthetic()
    t, pulse = pulse_template(bio["r_peaks_s"], 400)
    late = np.asarray(video["times"]) > 240
    video["rppg"][late] = np.interp(np.asarray(video["times"])[late] + lead + 1.5, t, pulse)
    result = estimate_alignment(video, bio, metadata, asr)
    assert result["status"] == "ambiguous_unverified"
    assert result["baseline_best_lag_difference_s"] > 1
    assert result["scale"] == 1.
    assert result["scale_status"] == "not_estimated_no_independent_clock_anchors"


def test_negative_duration_bound_is_rejected():
    video, bio, metadata, asr, _ = synthetic()
    metadata["analysis_duration_s"] = 300
    result = estimate_alignment(video, bio, metadata, asr)
    assert result["status"] == "acquisition_order_inconsistent"
    assert result["bio_lead_s"] is None


def test_ear_corrects_anamorphic_pixel_aspect():
    points = [[.1, .1], [.15, .08], [.2, .1], [.15, .12],
              [.3, .1], [.35, .08], [.4, .1], [.35, .12]]
    value = approximate_ear(points, 1440, 1080, 4 / 3)
    assert value == pytest.approx(43.2 / 192)
    assert approximate_ear([], 1440, 1080) is None


def test_baseline_instruction_times_use_decoded_origins():
    video, _, _, asr, _ = synthetic()
    windows = baseline_windows(video, asr, {"audio_first_decoded_pts_s": 1.04,
                                           "video_first_decoded_pts_s": 1.04})
    assert windows[0]["start_s"] == 13
    assert windows[1]["start_s"] == 258
