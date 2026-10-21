"""Aggregate saved measurements without rerunning ASR, video, or signal analysis.

Trial times are canonical video seconds. ``offset_overrides`` maps recording IDs
to absolute offsets in ``t_video = offset + scale * t_bio``; it never verifies a
clock. Session extraction includes every saved recording, including recordings
with no selected trials. Functions return a DataFrame and JSON-safe QC metadata;
they write no files and perform no inference about answer truth or personality.

Audio windows must fit wholly inside a reviewed respondent block and the trial.
Text rate uses corrected complete respondent turns, independently of old word
alignment. Word-pause measurements exclude explicitly stale or text-mismatched
word timestamps. Acoustic pitch summaries are medians of saved window summaries,
not a reconstructed pooled pitch distribution.

For a trial, context120s/context30s features use the nearest existing window
containing its midpoint, then apply that window's QC. A bad nearest window is
not replaced by a more distant good one. For a session they summarize QC-passing
windows intersecting the video span, with at least 20% usable windows per channel.
Context measures are not short-answer
responses; overlapping source windows are not independent observations.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .alignment import approximate_ear


FEATURE_DEFINITIONS = {
    "audio_pitch_median_hz": "Median of respondent-only 5 s window pitch medians, Hz.",
    "audio_pitch_iqr_hz": "Median of respondent-only 5 s within-window pitch IQRs, Hz.",
    "audio_rms_dbfs": "Median of respondent-only saved-window digital RMS levels, dBFS.",
    "audio_text_rate_wpm": "Corrected text words / duration of complete respondent turns, words/min; not aligned articulation rate.",
    "audio_vad_pause_fraction": "One minus saved VAD speech coverage within respondent blocks; no pause-duration threshold.",
    "audio_word_pause_median_s": "Median positive gap between adjacent current ASR words in the same respondent segment, s.",
    "video_head_speed_norm_per_s": "Median hypot(head_x_velocity, head_y_velocity), normalized frame coordinates/s; visible face only.",
    "video_torso_motion_deg_per_s": "Mean absolute saved torso_tilt_deg_velocity, degrees/s; visible shoulders and hips only. 2-D image proxy.",
    "video_eye_opening_ratio": "Median four-point-per-eye opening ratio, corrected for source pixel aspect ratio; not conventional six-point EAR.",
    "video_eye_motion_ratio_per_s": "Mean absolute opening-ratio derivative of adjacent available overlay samples, 1/s; not blink rate.",
    "video_mouth_opening_ratio": "Median vertical/horizontal lip landmark distance, corrected for source pixel aspect ratio.",
    "video_mouth_motion_ratio_per_s": "Mean absolute mouth-ratio derivative of adjacent available overlay samples, 1/s.",
    "video_rppg_hr_bpm": "Median QC-passing complete 30 s video-pulse windows wholly inside the requested span, beats/min.",
    "video_rppg_hr_bpm_context30s": "Nearest containing 30 s window for trial / median valid complete windows for session, beats/min; context only.",
    "bio_ecg_hr_bpm": "Median instantaneous 60000/RR for accepted complete intervals with nearest containing ECG QC usable, beats/min.",
    "bio_ecg_hr_bpm_context120s": "QC-passing existing 120 s ECG window HR, beats/min; context only.",
    "bio_ecg_rmssd_ms_context120s": "QC-passing existing 120 s ECG RMSSD, ms; context only, never short-answer RMSSD.",
    "bio_ecg_sdnn_ms_context120s": "QC-passing existing 120 s ECG SDNN, ms; context only, never short-answer SDNN.",
    "bio_rip_rate_bpm_context120s": "Existing RIP_periodic_candidate 120 s frequency, breaths/min; automatic periodicity candidate.",
    "bio_gsr_median_us": "Median saved GSR samples, microsiemens, only with explicit usable/validated source QC.",
    "bio_gsr_iqr_us": "IQR saved GSR samples, microsiemens, only with explicit usable/validated source QC.",
}
FEATURE_COLUMNS = tuple(FEATURE_DEFINITIONS)
SESSION_ONLY_FEATURES = ("bio_ppg_hr_bpm_context120s", "bio_ppg_rmssd_ms_context120s")
SESSION_FEATURE_COLUMNS = FEATURE_COLUMNS + SESSION_ONLY_FEATURES
SESSION_MIN_USABLE_FRACTION = .20
_WORDS = re.compile(r"[А-Яа-яЁёA-Za-z]+")
_EPS = 1e-8


def _number(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else np.nan
    except (TypeError, ValueError):
        return np.nan


def _json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _csv(path):
    if not path.exists() or not path.stat().st_size:
        return pd.DataFrame()
    return pd.read_csv(path)


def _merged(intervals):
    result = []
    for a, b in sorted(intervals):
        if not np.isfinite([a, b]).all() or b <= a:
            continue
        if result and a <= result[-1][1] + _EPS:
            result[-1][1] = max(b, result[-1][1])
        else:
            result.append([float(a), float(b)])
    return result


def _clipped(intervals, start, end):
    return _merged((max(start, a), min(end, b)) for a, b in intervals)


def _duration(intervals):
    return sum(b - a for a, b in _merged(intervals))


def _inside(intervals, start, end):
    return any(a <= start + _EPS and b >= end - _EPS for a, b in intervals)


def _overlaps(intervals, start, end):
    return any(min(b, end) - max(a, start) > _EPS for a, b in intervals)


def _subtract(intervals, excluded):
    result = []
    for start, end in intervals:
        cursor = start
        for a, b in _merged(excluded):
            if b <= cursor or a >= end:
                continue
            if a > cursor:
                result.append([cursor, min(a, end)])
            cursor = max(cursor, b)
            if cursor >= end:
                break
        if cursor < end:
            result.append([cursor, end])
    return result


def _reason(qc, feature, reason):
    values = qc["feature_reasons"].setdefault(feature, [])
    if reason not in values:
        values.append(reason)


def _aggregate(row, qc, feature, frame, column, mask=None, how="median"):
    if column not in frame:
        _reason(qc, feature, f"missing_column:{column}")
        return
    values = pd.to_numeric(frame.loc[mask, column] if mask is not None else frame[column], errors="coerce").to_numpy(float)
    values = values[np.isfinite(values)]
    if not len(values):
        _reason(qc, feature, "no_finite_QC_passing_values")
        return
    row[feature] = float(np.mean(values) if how == "mean" else np.median(values))


def _window_times(frame, start_key="start_s", end_key="end_s"):
    if start_key not in frame or end_key not in frame:
        return np.full(len(frame), np.nan), np.full(len(frame), np.nan)
    return (pd.to_numeric(frame[start_key], errors="coerce").to_numpy(float),
            pd.to_numeric(frame[end_key], errors="coerce").to_numpy(float))


def _context_indices(frame, start, end, scope, start_key="start_s", end_key="end_s"):
    a, b = _window_times(frame, start_key, end_key)
    finite = np.isfinite(a) & np.isfinite(b) & (b > a)
    if scope == "session":
        return np.flatnonzero(finite & (a < end) & (b > start))
    middle = (start + end) / 2
    candidates = np.flatnonzero(finite & (a <= middle) & (b >= middle))
    if not len(candidates):
        return candidates
    return candidates[np.argsort(np.abs((a[candidates] + b[candidates]) / 2 - middle), kind="stable")[:1]]


def _load(root, rid, pid):
    outputs = root / "outputs"
    paths = {"audio": outputs / "audio" / f"{rid}.segments.json",
             "acoustic": outputs / "audio" / f"{rid}.features.csv",
             "video": outputs / "video" / f"{rid}.json",
             "sync": outputs / "sync" / f"{rid}.json",
             "bio_metadata": outputs / "bio" / pid / "metadata.json",
             "bio_metrics": outputs / "bio" / pid / "metrics.csv",
             "rr": outputs / "bio" / pid / "rr_intervals.csv",
             "signals": outputs / "bio" / pid / "signals.npz",
             "ppg_metrics": outputs / "bio" / pid / "ppg_metrics.csv"}
    data = {k: _json(paths[k]) for k in ("audio", "video", "sync", "bio_metadata")}
    for key in ("audio", "video", "sync"):
        if data[key].get("participant_id", pid) != pid:
            raise ValueError(f"Participant mismatch in {paths[key]}")
    data.update({k: _csv(paths[k]) for k in ("acoustic", "bio_metrics", "rr", "ppg_metrics")})
    data["kinematics"] = pd.DataFrame(data["video"].get("kinematics", []))
    data["rppg"] = pd.DataFrame(data["video"].get("windows", []))
    data["gsr_t_s"], data["gsr_us"] = np.array([]), np.array([])
    if paths["signals"].exists():
        with np.load(paths["signals"], allow_pickle=False) as saved:
            for key in ("gsr_t_s", "gsr_us"):
                if key in saved:
                    data[key] = saved[key]
    data["sources"] = {k: {"path": str(p.relative_to(root)).replace("\\", "/"),
                           "status": "available" if p.exists() else "missing"} for k, p in paths.items()}
    return data


def _audio_features(row, qc, data, start, end):
    audio, frame = data["audio"], data["acoustic"]
    origin = data["sync"].get("media_timebase", {})
    first_audio, first_video = (_number(origin.get(k)) for k in ("audio_first_decoded_pts_s", "video_first_decoded_pts_s"))
    if not np.isfinite([first_audio, first_video]).all():
        for feature in FEATURE_COLUMNS[:6]:
            _reason(qc, feature, "missing_decoded_audio_video_origins")
        return
    shift = first_audio - first_video
    qc["audio_to_video_offset_s"] = shift
    turns = audio.get("turns", audio.get("segments", []))
    turns = sorted(turns, key=lambda t: (t["start"], t["end"]))
    respondent, blocks, blocked = [], [], []
    previous_role = None
    for turn in turns:
        a, b = _number(turn.get("start")) + shift, _number(turn.get("end")) + shift
        role = turn.get("speaker_role", "unknown")
        if role == "respondent":
            respondent.append((a, b, turn))
            if previous_role == role and blocks:
                blocks[-1][1] = max(blocks[-1][1], b)
            else:
                blocks.append([a, b])
        else:
            blocked.append([a, b])
        previous_role = role
    bounds = _subtract(_clipped(blocks, start, end), blocked)
    row["audio_respondent_seconds"] = _duration(bounds)
    if not bounds:
        for feature in FEATURE_COLUMNS[:6]:
            _reason(qc, feature, "no_reviewed_respondent_turns")
        return
    a, b = _window_times(frame)
    mask = np.array([np.isfinite([x, y]).all() and start <= x + shift + _EPS and y + shift <= end + _EPS
                     and _inside(blocks, x + shift, y + shift) and not _overlaps(blocked, x + shift, y + shift)
                     for x, y in zip(a, b)], dtype=bool)
    row["audio_window_count"] = int(mask.sum())
    for feature, column in [("audio_pitch_median_hz", "pitch_median_hz"),
                            ("audio_pitch_iqr_hz", "pitch_iqr_hz"), ("audio_rms_dbfs", "rms_dbfs")]:
        _aggregate(row, qc, feature, frame, column, mask)
        if not mask.any():
            _reason(qc, feature, "no_complete_respondent_acoustic_window")
    complete = [(a, b, t) for a, b, t in respondent if a >= start - _EPS and b <= end + _EPS
                and not _overlaps(blocked, a, b)]
    duration = _duration((a, b) for a, b, _ in complete)
    count = sum(len(_WORDS.findall(t.get("text", ""))) for _, _, t in complete)
    row["audio_word_count"] = count
    if duration > 0:
        row["audio_text_rate_wpm"] = 60 * count / duration
    else:
        _reason(qc, "audio_text_rate_wpm", "no_complete_respondent_turn")
    if "speech_intervals" in audio:
        vad = _merged((a + shift, b + shift) for a, b in audio["speech_intervals"])
        speech = sum(_duration(_clipped(vad, a, b)) for a, b in bounds)
        row["audio_vad_pause_fraction"] = float(np.clip(1 - speech / _duration(bounds), 0, 1))
    else:
        _reason(qc, "audio_vad_pause_fraction", "missing_saved_VAD_intervals")
    gaps, stale = [], 0
    for segment in audio.get("segments", []):
        sa, sb = _number(segment.get("start")) + shift, _number(segment.get("end")) + shift
        if segment.get("speaker_role") != "respondent" or not _overlaps(bounds, sa, sb):
            continue
        words = segment.get("words", [])
        word_text = " ".join(w.get("word", w.get("text", "")) for w in words)
        if "not_realigned" in segment.get("word_alignment_status", "") or (
                _WORDS.findall(word_text.lower()) != _WORDS.findall(segment.get("text", "").lower())):
            stale += 1
            continue
        for left, right in zip(words, words[1:]):
            a, b = _number(left.get("end")) + shift, _number(right.get("start")) + shift
            if np.isfinite([a, b]).all() and b > a and a >= start and b <= end and _inside(bounds, a, b) and not _overlaps(blocked, a, b):
                gaps.append(b - a)
    row["audio_stale_segment_count"], row["audio_trusted_pause_count"] = stale, len(gaps)
    if gaps:
        row["audio_word_pause_median_s"] = float(np.median(gaps))
    else:
        _reason(qc, "audio_word_pause_median_s", "no_positive_current_word_gaps")
    if stale:
        qc["warnings"].append("Stale/mismatched ASR word times excluded; corrected text rate is independent of word alignment.")


def _video_features(row, qc, data, start, end, scope):
    kin, video = data["kinematics"], data["video"]
    if "t" in kin:
        t = pd.to_numeric(kin.t, errors="coerce")
        kin = kin[(t >= start) & (t < end)].copy()
    else:
        kin = pd.DataFrame()
    row["video_kinematic_count"] = len(kin)
    if "face_visible" in kin and len(kin):
        row["video_face_fraction"] = float(kin.face_visible.eq(True).mean())
    head = "video_head_speed_norm_per_s"
    if {"head_x_velocity", "head_y_velocity", "face_visible"} <= set(kin):
        kin[head] = np.hypot(pd.to_numeric(kin.head_x_velocity, errors="coerce"), pd.to_numeric(kin.head_y_velocity, errors="coerce"))
        _aggregate(row, qc, head, kin, head, kin.face_visible.eq(True))
    else:
        _reason(qc, head, "missing_head_velocity_or_visibility")
    torso = "video_torso_motion_deg_per_s"
    if {"torso_tilt_deg_velocity", "shoulders_visible", "hips_visible"} <= set(kin):
        kin[torso] = pd.to_numeric(kin.torso_tilt_deg_velocity, errors="coerce").abs()
        _aggregate(row, qc, torso, kin, torso, kin.shoulders_visible.eq(True) & kin.hips_visible.eq(True), "mean")
    else:
        _reason(qc, torso, "missing_torso_velocity_or_visibility")
    width, height = _number(video.get("source_width")), _number(video.get("source_height"))
    sar = _number(data["sync"].get("media_timebase", {}).get("sample_aspect_ratio"))
    geometry_valid = np.isfinite([width, height, sar]).all() and min(width, height, sar) > 0
    times, eyes, mouths = [], [], []
    if geometry_valid:
        for overlay in video.get("overlays", []):
            t = _number(overlay.get("t"))
            if not start <= t < end:
                continue
            rois = overlay.get("rois") or {}
            times.append(t)
            eyes.append(_number(approximate_ear(rois.get("eyes", []), width, height, sar)))
            xy = np.asarray(rois.get("mouth", []), dtype=float)
            mouth = np.nan
            if xy.shape == (4, 2) and np.isfinite(xy).all():
                xy *= [width * sar, height]
                denominator = np.linalg.norm(xy[0] - xy[2])
                if denominator > 0:
                    mouth = np.linalg.norm(xy[1] - xy[3]) / denominator
            mouths.append(mouth)
    for name, values in [("eye", eyes), ("mouth", mouths)]:
        values = np.asarray(values, float)
        finite = np.isfinite(values)
        row[f"video_{name}_sample_count"] = int(finite.sum())
        opening, motion = f"video_{name}_opening_ratio", f"video_{name}_motion_ratio_per_s"
        if finite.any():
            row[opening] = float(np.median(values[finite]))
        else:
            _reason(qc, opening, "missing_frame_dimensions_SAR" if not geometry_valid else "no_valid_landmark_ratios")
        dt = np.diff(times)
        pairs = finite[:-1] & finite[1:] & (dt > 0)
        if pairs.any():
            row[motion] = float(np.mean(np.abs(np.diff(values)[pairs]) / dt[pairs]))
        else:
            _reason(qc, motion, "no_adjacent_valid_landmark_samples")
    windows = data["rppg"]
    a, b = _window_times(windows)
    valid = np.zeros(len(windows), dtype=bool)
    if {"face_coverage", "spectral_concentration", "rppg_hr_bpm"} <= set(windows):
        valid = (windows.face_coverage.eq(1) & (pd.to_numeric(windows.spectral_concentration, errors="coerce") >= .35)
                 & np.isfinite(pd.to_numeric(windows.rppg_hr_bpm, errors="coerce"))).to_numpy()
    else:
        missing = sorted({"face_coverage", "spectral_concentration", "rppg_hr_bpm"} - set(windows))
        for feature in ("video_rppg_hr_bpm", "video_rppg_hr_bpm_context30s"):
            _reason(qc, feature, "missing_rPPG_columns:" + ",".join(missing))
    inside = (a >= start - _EPS) & (b <= end + _EPS) & valid
    row["video_rppg_window_count"] = int(inside.sum())
    _aggregate(row, qc, "video_rppg_hr_bpm", windows, "rppg_hr_bpm", inside)
    indices = _context_indices(windows, start, end, scope)
    context = windows.iloc[indices]
    _aggregate(row, qc, "video_rppg_hr_bpm_context30s", context, "rppg_hr_bpm", valid[indices])
    qc["context_windows"]["rppg"] = {"selected_count": len(indices), "usable_count": int(valid[indices].sum()),
        "video_intervals_s": [[float(a[i]), float(b[i])] for i in indices] if scope == "trial" else []}
    row["video_rppg_context_window_count"] = len(indices)
    row["video_rppg_context_usable_count"] = int(valid[indices].sum())
    row["video_rppg_usable_fraction"] = float(valid[indices].mean()) if len(indices) else np.nan
    if scope == "session" and (not len(indices) or row["video_rppg_usable_fraction"] < SESSION_MIN_USABLE_FRACTION):
        for feature in ("video_rppg_hr_bpm", "video_rppg_hr_bpm_context30s"):
            row[feature] = np.nan
            _reason(qc, feature, "session_usable_window_fraction_below_0.20")


def _bio_features(row, qc, data, start, end, scope):
    offset, scale = row["sync_offset_s"], row["sync_scale"]
    bio_features = [f for f in row if f.startswith("bio_") and f in SESSION_FEATURE_COLUMNS]
    if not np.isfinite([offset, scale]).all() or scale <= 0:
        for feature in bio_features:
            _reason(qc, feature, "missing_or_invalid_bio_video_clock")
        return
    start, end = (start - offset) / scale, (end - offset) / scale
    row["bio_start_s"], row["bio_end_s"] = start, end
    frame = data["bio_metrics"]
    a, b = _window_times(frame, "window_start_s", "window_end_s")
    indices = _context_indices(frame, start, end, scope, "window_start_s", "window_end_s")
    context = frame.iloc[indices]
    ecg_good = context.ecg_quality.eq("usable_automatic") if "ecg_quality" in context else np.zeros(len(context), bool)
    rip_good = context.resp_status.eq("RIP_periodic_candidate") if "resp_status" in context else np.zeros(len(context), bool)
    row["bio_context_window_count"] = len(context)
    row["bio_context_ecg_usable_count"] = int(np.sum(ecg_good))
    row["bio_context_rip_usable_count"] = int(np.sum(rip_good))
    row["bio_ecg_usable_fraction"] = float(np.mean(ecg_good)) if len(context) else np.nan
    row["bio_rip_usable_fraction"] = float(np.mean(rip_good)) if len(context) else np.nan
    for feature, column in [("bio_ecg_hr_bpm_context120s", "hr_bpm"), ("bio_ecg_rmssd_ms_context120s", "rmssd_ms"),
                            ("bio_ecg_sdnn_ms_context120s", "sdnn_ms")]:
        _aggregate(row, qc, feature, context, column, ecg_good)
        if len(context) and not np.any(ecg_good):
            _reason(qc, feature, "selected_context_ECG_QC_not_usable")
    _aggregate(row, qc, "bio_rip_rate_bpm_context120s", context, "resp_bpm", rip_good)
    qc["context_windows"]["bio"] = {"selected_count": len(indices), "ecg_usable_count": int(np.sum(ecg_good)),
        "rip_usable_count": int(np.sum(rip_good)), "bio_intervals_s": [[float(a[i]), float(b[i])] for i in indices] if scope == "trial" else []}
    rr = data["rr"]
    accepted = []
    if {"time_s", "rr_ms", "accepted"} <= set(rr):
        t = pd.to_numeric(rr.time_s, errors="coerce").to_numpy(float)
        milliseconds = pd.to_numeric(rr.rr_ms, errors="coerce").to_numpy(float)
        complete = np.isfinite(t) & np.isfinite(milliseconds) & (milliseconds > 0)
        complete &= (t - milliseconds / 2000 >= start - _EPS) & (t + milliseconds / 2000 <= end + _EPS)
        row["bio_rr_total_count"] = int(complete.sum())
        # Gate each RR by its nearest containing existing QC window, independently
        # of whether the whole request spans a long session or one short answer.
        for i in np.flatnonzero(complete & rr.accepted.astype(str).str.lower().isin(["true", "1"]).to_numpy()):
            candidates = np.flatnonzero(np.isfinite(a) & np.isfinite(b) & (a <= t[i]) & (b >= t[i]))
            nearest = candidates[np.argmin(np.abs((a[candidates] + b[candidates]) / 2 - t[i]))] if len(candidates) else None
            if nearest is not None and "ecg_quality" in frame and frame.iloc[nearest].ecg_quality == "usable_automatic":
                accepted.append(milliseconds[i])
    else:
        _reason(qc, "bio_ecg_hr_bpm", "missing_RR_columns")
    row["bio_rr_accepted_count"] = len(accepted)
    if accepted:
        row["bio_ecg_hr_bpm"] = float(np.median(60000 / np.asarray(accepted)))
    else:
        _reason(qc, "bio_ecg_hr_bpm", "no_complete_accepted_RR_with_usable_ECG_QC")
    status = data["bio_metadata"].get("gsr_quality", {}).get("status", "missing")
    qc["gsr_source_status"] = status
    gsr_features = ("bio_gsr_median_us", "bio_gsr_iqr_us")
    if status not in {"usable", "usable_automatic", "validated", "valid"}:
        for feature in gsr_features:
            _reason(qc, feature, f"GSR_source_QC:{status}")
    else:
        t, values = data["gsr_t_s"], data["gsr_us"]
        if len(t) == len(values):
            values = values[(t >= start) & (t < end) & np.isfinite(values) & (values >= 0) & (values <= 24000)]
        else:
            values = np.array([])
        row["bio_gsr_sample_count"] = len(values)
        if len(values):
            row[gsr_features[0]] = float(np.median(values))
            row[gsr_features[1]] = float(np.subtract(*np.percentile(values, [75, 25])))
        else:
            for feature in gsr_features:
                _reason(qc, feature, "no_finite_in_range_GSR_samples")
    if scope == "session":
        if not len(context) or row["bio_ecg_usable_fraction"] < SESSION_MIN_USABLE_FRACTION:
            for feature in ("bio_ecg_hr_bpm", "bio_ecg_hr_bpm_context120s", "bio_ecg_rmssd_ms_context120s", "bio_ecg_sdnn_ms_context120s"):
                row[feature] = np.nan
                _reason(qc, feature, "session_usable_window_fraction_below_0.20")
        if not len(context) or row["bio_rip_usable_fraction"] < SESSION_MIN_USABLE_FRACTION:
            row["bio_rip_rate_bpm_context120s"] = np.nan
            _reason(qc, "bio_rip_rate_bpm_context120s", "session_usable_window_fraction_below_0.20")
        ppg = data["ppg_metrics"]
        ix = _context_indices(ppg, start, end, scope, "window_start_s", "window_end_s")
        selected = ppg.iloc[ix]
        usable = selected.ppg_quality.eq("usable_automatic") if "ppg_quality" in selected else np.zeros(len(selected), bool)
        row["bio_ppg_usable_window_count"] = int(np.sum(usable))
        row["bio_ppg_context_window_count"] = len(selected)
        row["bio_ppg_usable_fraction"] = float(np.mean(usable)) if len(selected) else np.nan
        for feature, column in zip(SESSION_ONLY_FEATURES, ("hr_bpm", "rmssd_ms")):
            _aggregate(row, qc, feature, selected, column, usable)


def _extract(root, trials, offset_overrides, scope):
    root = Path(root)
    overrides = offset_overrides or {}
    records, cache, audits, seen = [], {}, {}, set()
    columns = SESSION_FEATURE_COLUMNS if scope == "session" else FEATURE_COLUMNS
    for trial in trials:
        tid, rid, pid = (str(trial[k]) for k in ("trial_id", "recording_id", "participant_id"))
        if tid in seen:
            raise ValueError(f"Duplicate trial_id: {tid}")
        seen.add(tid)
        if any(not s or s in {".", ".."} or any(c in s for c in "/\\:") for s in (rid, pid)):
            raise ValueError("Recording/participant IDs must be file-name components")
        start, end = _number(trial["start_s"]), _number(trial["end_s"])
        if not np.isfinite([start, end]).all() or start < 0 or end <= start:
            raise ValueError(f"Invalid trial interval: {tid}")
        key = rid, pid
        if key not in cache:
            cache[key] = _load(root, rid, pid)
        data = cache[key]
        sync = data["sync"]
        offset = _number(sync.get("bio_to_video_offset_s", sync.get("suggested_offset_s")))
        if rid in overrides:
            offset = _number(overrides[rid])
            if not np.isfinite(offset):
                raise ValueError(f"Invalid offset override: {rid}")
        uncertainty = sync.get("offset_uncertainty_interval_s")
        if not isinstance(uncertainty, (list, tuple)) or len(uncertainty) != 2:
            uncertainty = [None, None]
        row = {"trial_id": tid, "recording_id": rid, "participant_id": pid, "scope": scope,
               "start_s": start, "end_s": end, "duration_s": end - start,
               "audio_status": "available" if data["audio"] else "missing",
               "video_status": "available" if data["video"] else "missing",
               "bio_status": "available" if len(data["bio_metrics"]) else "missing",
               "sync_status": sync.get("status", "missing"), "sync_verified": sync.get("verified") is True and rid not in overrides,
               "sync_offset_overridden": rid in overrides, "sync_offset_s": offset, "sync_scale": _number(sync.get("scale")),
               "sync_uncertainty_low_s": _number(uncertainty[0]), "sync_uncertainty_high_s": _number(uncertainty[1]),
               "bio_start_s": np.nan, "bio_end_s": np.nan, **{f: np.nan for f in columns}}
        for field in ("audio_window_count", "audio_word_count", "audio_stale_segment_count", "audio_trusted_pause_count",
                      "audio_respondent_seconds", "video_kinematic_count", "video_eye_sample_count", "video_mouth_sample_count",
                      "video_rppg_window_count", "bio_rr_total_count", "bio_rr_accepted_count", "bio_context_window_count",
                      "bio_context_ecg_usable_count", "bio_context_rip_usable_count", "bio_gsr_sample_count"):
            row[field] = 0
        for field in ("video_face_fraction", "video_rppg_usable_fraction", "bio_ecg_usable_fraction", "bio_rip_usable_fraction"):
            row[field] = np.nan
        qc = {"feature_reasons": {}, "warnings": [], "context_windows": {}, "sources": data["sources"]}
        if not row["sync_verified"]:
            qc["warnings"].append("Bio/video alignment remains unverified; no clock confidence was upgraded by extraction.")
        if rid in overrides:
            qc["warnings"].append("Absolute offset override used for sensitivity analysis; stored uncertainty still describes the original candidate.")
        _audio_features(row, qc, data, start, end)
        _video_features(row, qc, data, start, end, scope)
        _bio_features(row, qc, data, start, end, scope)
        for feature in columns:
            if not np.isfinite(row[feature]) and feature not in qc["feature_reasons"]:
                _reason(qc, feature, "no_usable_source_values")
        qc["counts"] = {k: v for k, v in row.items() if k.endswith("_count")}
        qc["usable_fractions"] = {k: float(v) if np.isfinite(v) else None for k, v in row.items() if k.endswith("_usable_fraction")}
        records.append(row)
        audits[tid] = qc
    definitions = dict(FEATURE_DEFINITIONS)
    if scope == "session":
        definitions.update({f: f"Median existing QC-usable complete contact-PPG 120 s windows; {unit}; session description only."
                            for f, unit in zip(SESSION_ONLY_FEATURES, ("beats/min", "RMSSD ms"))})
    frame = pd.DataFrame(records) if records else pd.DataFrame(columns=["trial_id", "recording_id", "participant_id", "scope", "start_s", "end_s", *columns])
    return frame, {"schema_version": 1, "scope": scope, "feature_definitions": definitions,
        "rows": audits, "policies": {"respondent_only_audio": True, "stale_word_times_excluded": True,
        "trial_context_selection": "nearest existing window containing midpoint, QC checked after selection",
        "session_min_usable_window_fraction": SESSION_MIN_USABLE_FRACTION,
        "session_QC_denominator": "all existing windows intersecting video span, per channel",
        "context_is_short_answer_response": False, "source_window_overlap_means_nonindependent": True,
        "clock_formula": "t_video = offset + scale * t_bio", "reprocessing_performed": False}}


def extract_trial_features(root, trials, *, offset_overrides=None):
    """Return trial features/QC for list[dict] or DataFrame with five required IDs/times.

    Required: trial_id, participant_id, recording_id, start_s, end_s. Overrides are
    absolute offsets in seconds keyed by recording_id, keeping the stored scale.
    Missing measurements are NaN with per-feature QC reasons, never imputed.
    """
    if isinstance(trials, pd.DataFrame):
        trials = trials.to_dict("records")
    return _extract(root, trials, offset_overrides, "trial")


def extract_session_features(root, *, offset_overrides=None):
    """Return one row per saved audio/video/sync recording, even without trials.

    The session covers the video clock's available duration. Audio summaries use
    respondent speech only; direct BioRadio samples outside video are excluded.
    Context windows intersecting the video span can include boundary pre/post-roll.
    """
    root = Path(root)
    rids = set(p.name.removesuffix(".segments.json") for p in (root / "outputs/audio").glob("*.segments.json"))
    rids.update(p.stem for p in (root / "outputs/video").glob("*.json") if p.stem != "summary")
    rids.update(p.stem for p in (root / "outputs/sync").glob("*.json") if p.stem != "summary")
    trials = []
    for rid in sorted(rids):
        audio = _json(root / "outputs/audio" / f"{rid}.segments.json")
        video = _json(root / "outputs/video" / f"{rid}.json")
        sync = _json(root / "outputs/sync" / f"{rid}.json")
        pid = audio.get("participant_id", video.get("participant_id", sync.get("participant_id")))
        duration = _number(sync.get("media_duration_s", video.get("duration_s", audio.get("duration_s"))))
        if not pid or not np.isfinite(duration) or duration <= 0:
            raise ValueError(f"Missing participant or session duration: {rid}")
        trials.append(dict(trial_id=f"session:{rid}", participant_id=pid, recording_id=rid, start_s=0., end_s=duration))
    return _extract(root, trials, offset_overrides, "session")
