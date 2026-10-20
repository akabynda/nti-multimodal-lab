"""Bounded, explicitly unverified synchronization candidates from pulse signals."""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal


def approximate_ear(points, width, height, sample_aspect_ratio=1.0):
    """Four-point-per-eye opening ratio, not the conventional six-point EAR."""
    xy = np.asarray(points, dtype=float)
    if xy.shape != (8, 2) or not np.isfinite(xy).all():
        return None
    xy = xy * [width * sample_aspect_ratio, height]
    widths = [np.linalg.norm(xy[0] - xy[2]), np.linalg.norm(xy[4] - xy[6])]
    if min(widths) <= 0:
        return None
    return float(np.mean([np.linalg.norm(xy[1] - xy[3]) / widths[0],
                          np.linalg.norm(xy[5] - xy[7]) / widths[1]]))


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def _filter_finite(values, fs, band=(.65, 3.0)):
    """Filter finite runs; never fill missing video observations."""
    values = np.asarray(values, float)
    output = np.full(len(values), np.nan)
    sos = signal.butter(3, band, btype="bandpass", fs=fs, output="sos")
    for start, end in _runs(np.isfinite(values)):
        if end - start >= 8 * fs:
            output[start:end] = signal.sosfiltfilt(sos, values[start:end])
            margin = int(2 * fs)
            output[start:start + margin] = np.nan
            output[end - margin:end] = np.nan
    return output


def pulse_template(peaks_s, duration_s, fs=25., nominal_transit_s=.2):
    """Gaussian pulses at ECG QRS + nominal transit, then zero-phase bandpass."""
    t = np.arange(0, duration_s, 1 / fs)
    impulses = np.zeros(len(t))
    positions = np.rint((np.asarray(peaks_s) + nominal_transit_s) * fs).astype(int)
    positions = positions[(positions >= 0) & (positions < len(t))]
    np.add.at(impulses, positions, 1)
    smooth = ndimage.gaussian_filter1d(impulses, sigma=.06 * fs)
    return t, _filter_finite(smooth, fs)


def baseline_windows(video, asr, media_info=None):
    """Commanded eye-closure intervals; landmarks can corroborate, not prove rest."""
    media_info = media_info or {}
    duration = float(video["duration_s"])
    audio_to_video = float(media_info.get("audio_first_decoded_pts_s", 0) -
                           media_info.get("video_first_decoded_pts_s", 0))
    instructions = [s for s in asr.get("segments", [])
                    if any(word in s["text"].lower() for word in ("закры", "закро"))
                    and "гла" in s["text"].lower()]
    beginning = [s for s in instructions if s["end"] + audio_to_video < duration * .4]
    ending = [s for s in instructions if s["start"] + audio_to_video > duration * .6]
    windows = []
    for name, chosen, fallback in [
        ("beginning", beginning[0] if beginning else None, (20., min(140., duration * .4))),
        ("ending", ending[-1] if ending else None, (max(0., duration - 125), duration - 5)),
    ]:
        if chosen:
            start = float(chosen["end"] + audio_to_video + 3)
            end = min(start + 115, duration - 2)
            evidence = "ASR_eye_closure_instruction"
        else:
            start, end = fallback
            evidence = "duration_heuristic_no_instruction"
        windows.append({"name": name, "start_s": start, "end_s": end,
                        "evidence": evidence,
                        "instruction": chosen["text"] if chosen else None,
                        "eye_state": "unavailable"})
    width, height = video.get("source_width"), video.get("source_height")
    eye_times, eye_values = [], []
    if width and height:
        for overlay in video.get("overlays", []):
            ear = approximate_ear(overlay.get("rois", {}).get("eyes", []), width, height,
                                  media_info.get("sample_aspect_ratio", 1.0))
            if ear is not None:
                eye_times.append(overlay["t"])
                eye_values.append(ear)
    if len(eye_values) >= 30:
        times = np.asarray(eye_times)
        values = ndimage.median_filter(eye_values, size=5, mode="nearest")
        low, high = np.quantile(values, [.1, .9])
        threshold = float(low + .35 * (high - low))
        for window in windows:
            inside = (times >= window["start_s"]) & (times < window["end_s"])
            if inside.sum() < 20 or high - low < .015:
                window["eye_state"] = "insufficient_landmark_dynamic_range"
                continue
            closed = values[inside] < threshold
            window.update(approx_ear_median=float(np.median(values[inside])),
                          approx_ear_threshold=threshold,
                          low_eye_opening_fraction=float(closed.mean()),
                          eye_state="closure_corroborated" if closed.mean() >= .6 else "closure_not_corroborated")
            # Landmark threshold is not calibrated; keep the commanded interval.
    return windows


def _curve(t, video_pulse, template_t, template, leads, window):
    use = (t >= window["start_s"]) & (t <= window["end_s"])
    times, y = t[use], video_pulse[use]
    curve = np.full(len(leads), np.nan)
    coverage = float(np.isfinite(y).mean()) if len(y) else 0.
    if coverage < .75 or len(y) < 30 or window["end_s"] - window["start_s"] < 40:
        return curve, coverage
    for index, lead in enumerate(leads):
        x = np.interp(times + lead, template_t, template, left=np.nan, right=np.nan)
        good = np.isfinite(x) & np.isfinite(y)
        if good.mean() < .75 or np.std(x[good]) < 1e-12 or np.std(y[good]) < 1e-12:
            continue
        curve[index] = abs(float(np.corrcoef(x[good], y[good])[0, 1]))
    return curve, coverage


def _curve_summary(leads, scores, exclusion_radius=.2):
    good = np.isfinite(scores)
    if not good.any():
        return {"best_lead_s": None, "best_score": None, "alternative_score": None,
                "margin": None}
    best = int(np.nanargmax(scores))
    competing = good & (np.abs(leads - leads[best]) > exclusion_radius)
    second = float(np.max(scores[competing])) if competing.any() else 0.
    return {"best_lead_s": float(leads[best]), "best_score": float(scores[best]),
            "alternative_score": second, "margin": float(scores[best] - second)}


def _ecg_baseline_quality(bio, start_s, end_s):
    """Gate pulse-template support by observed RR and ECG quality."""
    required = ("rr_t_s", "rr_ms", "rr_valid", "ecg_t_s", "ecg_mv", "ecg_filtered_mv")
    if not all(key in bio for key in required):
        return {"status": "not_assessed_missing_arrays", "usable": False}
    from .biosignals import hrv_metrics
    rr = hrv_metrics(bio["rr_t_s"], bio["rr_ms"], bio["rr_valid"], start_s, end_s)
    inside = (bio["ecg_t_s"] >= start_s) & (bio["ecg_t_s"] < end_s)
    raw, filtered = bio["ecg_mv"][inside], bio["ecg_filtered_mv"][inside]
    finite = np.isfinite(filtered)
    amplitude = float(np.quantile(filtered[finite], .995) - np.quantile(filtered[finite], .005)) if finite.any() else None
    nonfinite = float(np.mean(~np.isfinite(raw))) if len(raw) else 1.
    clipping = float(np.mean(np.abs(raw) >= 187.49)) if len(raw) else 1.
    gap = rr["max_interpolation_gap_s"]
    good = (rr["rr_rejected_fraction"] <= .1 and rr["rr_coverage_fraction"] >= .9
            and gap is not None and gap <= 3 and amplitude is not None and amplitude >= .05
            and nonfinite <= .01 and clipping < .001)
    return {"status": "usable_automatic" if good else "review_required", "usable": bool(good),
            "rr_rejected_fraction": rr["rr_rejected_fraction"],
            "rr_coverage_fraction": rr["rr_coverage_fraction"],
            "max_rr_gap_s": gap, "qrs_span_mv": amplitude,
            "nonfinite_fraction": nonfinite, "clipping_fraction": clipping}


def estimate_alignment(video, bio, metadata, asr, media_info=None, grid_step_s=.02):
    """Estimate BioRadio lead; session/video time = bio time - lead.

    All candidates remain unverified. No drift is fitted from periodic pulses.
    """
    media_info = media_info or {}
    bio_duration = float(metadata["analysis_duration_s"])
    video_duration = float(media_info.get("recording_duration_s", video["duration_s"]))
    upper = bio_duration - video_duration
    common = {
        "participant_id": metadata["subject_id"], "recording_id": video["recording_id"],
        "method": "bounded_sign_invariant_QRS_template_rPPG_correlation_in_two_baselines",
        "bio_duration_s": bio_duration, "media_duration_s": video_duration,
        "lead_bounds_s": [0., max(0., upper)],
        "offset_bounds_s": [-max(0., upper), 0.],
        "scale": 1., "scale_status": "not_estimated_no_independent_clock_anchors",
        "verified": False, "bio_to_video_formula": "video_time = bio_time + bio_to_video_offset_s",
        "nominal_pulse_transit_s": .2,
        "transit_sensitivity_assumption_s": [.08, .45],
        "transit_note": "Assumed sensitivity range, not a measured or guaranteed physiological bound.",
        "support_thresholds": {
            "each_baseline_correlation_min": .35, "joint_score_min": .4,
            "joint_competing_peak_margin_min": .04, "each_baseline_peak_margin_min": .02,
            "baseline_lag_difference_max_s": .25, "competing_peak_exclusion_radius_s": .2,
            "ecg_quality_required_in_both_baselines": True,
            "nominal_search_step_s": grid_step_s,
        },
        "media_timebase": media_info,
        "warnings": [
            "Pulse phase and transit time are confounded with clock offset.",
            "Sign-invariant cardiac correlation can have half-cycle and full-cycle aliases.",
            "Eye closure identifies a likely quiet period, not a BioRadio clock event.",
            "No BioRadio-video offset is verified by an independent hardware marker.",
        ],
    }
    if upper < 0:
        common.update(status="acquisition_order_inconsistent", bio_lead_s=None,
                      bio_to_video_offset_s=None, confidence="none",
                      lead_uncertainty_interval_s=None)
        return common
    windows = baseline_windows(video, asr, media_info)
    common["baselines"] = windows
    t = np.asarray(video["times"], float)
    y = np.asarray(video["rppg"], float)
    fs = 1 / np.median(np.diff(t))
    leads = np.linspace(0, upper, max(2, int(np.ceil(upper / grid_step_s)) + 1))
    if fs <= 6 or len(t) < fs * 40:
        y = np.full(len(t), np.nan)
    else:
        y = _filter_finite(y, fs)
    template_t, template = pulse_template(bio["r_peaks_s"], bio_duration)
    curves = []
    for window in windows:
        curve, coverage = _curve(t, y, template_t, template, leads, window)
        support = _curve_summary(leads, curve)
        window.update(support)
        window["finite_pulse_coverage"] = coverage
        curves.append(curve)
    both = np.isfinite(curves[0]) & np.isfinite(curves[1])
    joint = np.full(len(leads), np.nan)
    joint[both] = np.sqrt(curves[0][both] * curves[1][both])
    combined = _curve_summary(leads, joint)
    common.update(joint_support=combined,
                  score_curve={"lead_s": leads.tolist(),
                               "beginning_abs_correlation": _json_scores(curves[0]),
                               "ending_abs_correlation": _json_scores(curves[1]),
                               "joint_score": _json_scores(joint)})
    if not both.any():
        common.update(status="insufficient_signal", confidence="none",
                      bio_lead_s=float(upper / 2), bio_to_video_offset_s=float(-upper / 2),
                      candidate_basis="duration_midpoint_prior_no_signal_evidence",
                      lead_uncertainty_interval_s=[0., upper],
                      baseline_best_lag_difference_s=None)
        return common
    best = int(np.nanargmax(joint))
    lead = float(leads[best])
    for window in windows:
        window["ecg_quality"] = _ecg_baseline_quality(bio, window["start_s"] + lead,
                                                     window["end_s"] + lead)
    difference = abs(windows[0]["best_lead_s"] - windows[1]["best_lead_s"])
    strong = (min(curves[0][best], curves[1][best]) >= .35 and joint[best] >= .4
              and combined["margin"] >= .04 and difference <= .25
              and all(w["margin"] is not None and w["margin"] >= .02
                      and w["ecg_quality"]["usable"] for w in windows))
    near = both & (joint >= joint[best] - .04)
    if strong:
        interval = [max(0., float(leads[near].min()) - .12 - 1 / fs),
                    min(upper, float(leads[near].max()) + .25 + 1 / fs)]
    else:
        interval = [0., upper]
    common.update(status="supported_unverified" if strong else "ambiguous_unverified",
                  confidence="moderate_unverified" if strong else "low",
                  bio_lead_s=lead, bio_to_video_offset_s=-lead,
                  candidate_basis="maximum_joint_pulse_score_under_nominal_transit",
                  lead_uncertainty_interval_s=interval,
                  baseline_best_lag_difference_s=float(difference),
                  beginning_score_at_candidate=float(curves[0][best]),
                  ending_score_at_candidate=float(curves[1][best]))
    local, _ = signal.find_peaks(np.nan_to_num(joint, nan=-1))
    ranked = sorted(local, key=lambda i: joint[i], reverse=True)[:8]
    common["alternative_candidates"] = [
        {"bio_lead_s": float(leads[i]), "score": float(joint[i])} for i in ranked]
    return common


def _json_scores(values):
    return [float(x) if np.isfinite(x) else None for x in values]


def compare_modalities(video, bio, alignment, bio_windows=None):
    """Descriptive matched-window comparisons; offset remains explicitly exploratory."""
    from .biosignals import respiratory_metrics
    lead = alignment.get("bio_lead_s")
    if lead is None or alignment["status"] == "insufficient_signal":
        return []
    rr_t, rr = bio["rr_t_s"], bio["rr_ms"]
    valid = bio["rr_valid"].astype(bool)
    rows = []
    for window in video.get("windows", []):
        start, end = float(window["start_s"]), float(window["end_s"])
        select = (rr_t >= start + lead) & (rr_t < end + lead)
        accepted = select & valid
        count = int(select.sum())
        times = rr_t[accepted]
        ecg_good = count >= 20 and accepted.sum() >= .9 * count and len(times) >= 2
        if ecg_good:
            ecg_good = times[-1] - times[0] >= .85 * (end - start) and np.diff(times).max() <= 3
        if bio_windows is not None and len(bio_windows):
            centers = np.asarray([w["center_s"] for w in bio_windows])
            nearby = bio_windows[int(np.argmin(abs(centers - ((start + end) / 2 + lead))))]
            ecg_good = ecg_good and nearby.get("ecg_quality") == "usable_automatic"
        ecg_hr = float(60000 / rr[accepted].mean()) if ecg_good else None
        remote_hr = window.get("rppg_hr_bpm")
        row = {"participant_id": alignment["participant_id"], "video_start_s": start,
               "video_end_s": end, "bio_start_s": start + lead, "bio_end_s": end + lead,
               "alignment_status": alignment["status"], "comparison_status": "exploratory_candidate_alignment",
               "ecg_hr_bpm": ecg_hr, "rppg_hr_bpm": remote_hr,
               "hr_difference_bpm": float(remote_hr - ecg_hr) if remote_hr is not None and ecg_hr is not None else None,
               "rip_resp_bpm": None, "shoulder_resp_bpm": window.get("respiratory_motion_bpm"),
               "resp_difference_bpm": None}
        if "rip_t_s" in bio:
            use = (bio["rip_t_s"] >= start + lead) & (bio["rip_t_s"] < end + lead)
            segment = bio["rip_mv"][use]
            # Keep the 30 s protocol exactly despite float boundary rounding.
            if len(segment) >= 750:
                response = respiratory_metrics(segment, 25)
                row["rip_resp_bpm"] = response.get("resp_bpm")
                if row["rip_resp_bpm"] is not None and row["shoulder_resp_bpm"] is not None:
                    row["resp_difference_bpm"] = float(row["shoulder_resp_bpm"] - row["rip_resp_bpm"])
        rows.append(row)
    return rows
