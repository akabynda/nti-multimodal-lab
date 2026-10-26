"""Per-recording summaries of saved HRV, speech and kinematic outputs and session stages.

Inputs are the processed files in outputs/; nothing is re-detected. Missing or
insufficient data give NaN, never a substitute value.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from scipy import signal

from .audio import words_and_syllables

HRV_COLUMNS = ("hr_bpm", "sdnn_ms", "rmssd_ms", "pnn50_pct", "vlf_ms2", "lf_ms2", "hf_ms2", "lf_hf", "total_ms2",
               "stress_index")
# Question topics; the first matching pattern of the final two sentences wins. A question
# without a match continues the previous block (follow-up questions).
TOPIC_PATTERNS = (
    ("Выбор профессии", r"професс|специализ|аспирант|бакалавр|вуз|учит|учеб"),
    ("Экзамен", r"экзамен|сесси|билет|списыв"),
    ("Достижения", r"достиж|положительн"),
    ("Алкоголь", r"алкогол|пьян|пь[её]т|выпив|употребл"),
    ("Обман", r"обман|лгал|лгать|солга|ложь|лж[её]|честн|соврал|врал"),
    ("Конфликт", r"конфликт|ссор"),
    ("Летний день", r"летн|лет[оау]м?\b|каникул"),
    ("Нерешённая проблема", r"проблем|справ"),
    ("Нейтральные вопросы", r"о себе|увлечен|хобби|зовут|представ|чувству|готов"),
)
NEUTRAL = "Нейтральные вопросы"
NOD_BAND_HZ = (.3, 3.)
HAND_TO_FACE = .8  # wrist-to-nose distance in shoulder widths
POSE_VISIBLE_MIN = 10  # pose samples needed for posture fractions


def _median(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else np.nan


def _iqr(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.subtract(*np.quantile(values, [.75, .25]))) if len(values) else np.nan


def hrv_summary(metrics: pd.DataFrame) -> dict:
    """Medians over 120 s ECG windows with `ecg_quality == usable_automatic`.

    Spectral values use usable windows with `spectrum_status == exploratory_120s`.
    RIP rate uses windows with `resp_status == RIP_periodic_candidate`.
    """
    usable = metrics[metrics.ecg_quality.eq("usable_automatic")]
    spectral = usable[usable.spectrum_status.eq("exploratory_120s")]
    row = {"windows": len(metrics), "usable_windows": len(usable)}
    for column in HRV_COLUMNS:
        source = spectral if column in ("vlf_ms2", "lf_ms2", "hf_ms2", "lf_hf", "total_ms2") else usable
        row[column] = _median(source[column]) if column in source else np.nan
    rip = metrics[metrics.resp_status.eq("RIP_periodic_candidate")] if "resp_status" in metrics else metrics.iloc[:0]
    row["rip_windows"] = len(rip)
    row["rip_rate_bpm"] = _median(rip.resp_bpm) if len(rip) else np.nan
    return row


def _respondent_turns(segments: dict) -> list[dict]:
    return [t for t in segments.get("turns", []) if t.get("speaker_role") == "respondent" and t["end"] > t["start"]]


def speech_summary(segments: dict, acoustic: pd.DataFrame) -> dict:
    """Dialogue timing from reviewed QA pairs and acoustics inside respondent turns.

    Acoustic 5 s windows count when their centre lies inside a respondent turn and
    `voiced_fraction >= 0.2`. Rates use the summed duration of respondent turns.
    """
    qa = segments.get("qa_candidates", [])
    latency = [q["latency_s"] for q in qa if q.get("latency_s") is not None and q["latency_s"] >= 0]
    turns = _respondent_turns(segments)
    seconds = sum(t["end"] - t["start"] for t in turns)
    words = syllables = 0
    for turn in turns:
        w, s = words_and_syllables(turn.get("text", ""))
        words, syllables = words + w, syllables + s
    centre = acoustic.time_s.to_numpy(float) if len(acoustic) else np.array([])
    inside = np.zeros(len(centre), dtype=bool)
    for turn in turns:
        inside |= (centre >= turn["start"]) & (centre < turn["end"])
    voiced = acoustic[inside & (acoustic.voiced_fraction.to_numpy(float) >= .2)] if len(acoustic) else acoustic
    row = {"qa_pairs": len(qa), "latency_median_s": _median(latency), "latency_iqr_s": _iqr(latency),
           "answer_median_s": _median([q.get("answer_duration_s") for q in qa]),
           "respondent_s": seconds, "words_per_min": 60 * words / seconds if seconds else np.nan,
           "syllables_per_s": syllables / seconds if seconds else np.nan, "acoustic_windows": len(voiced)}
    for column in ("pitch_median_hz", "pitch_iqr_hz", "rms_dbfs", "spectral_centroid_hz",
                   "spectral_tilt_db_per_octave", "f1_median_hz", "f2_median_hz", "f3_median_hz"):
        row[column] = _median(voiced[column]) if len(voiced) and column in voiced else np.nan
    return row


def _dominant(values, times, band):
    """Dominant frequency and band RMS of contiguous finite runs of at least 20 s."""
    values, times = np.asarray(values, dtype=float), np.asarray(times, dtype=float)
    fs = 1 / np.median(np.diff(times))
    finite = np.isfinite(values)
    edges = np.flatnonzero(np.diff(np.r_[0, finite.astype(int), 0]))
    spectra, rms = [], []
    for a, b in zip(edges[::2], edges[1::2]):
        if (b - a) / fs < 20:
            continue
        x = signal.detrend(values[a:b])
        sos = signal.butter(2, [band[0], min(band[1], .45 * fs)], "bandpass", fs=fs, output="sos")
        rms.append((np.sqrt(np.mean(signal.sosfiltfilt(sos, x) ** 2)), b - a))
        f, p = signal.welch(x, fs, nperseg=min(b - a, int(20 * fs)))
        spectra.append((f, p * (b - a)))
    if not spectra:
        return np.nan, np.nan
    grid = spectra[0][0]
    power = sum(np.interp(grid, f, p) for f, p in spectra)
    keep = (grid >= band[0]) & (grid <= band[1])
    weights = np.array([n for _, n in rms])
    return float(grid[keep][np.argmax(power[keep])]), float(np.average([r for r, _ in rms], weights=weights))


def _oscillation(k, column):
    """Peak frequency of the velocity spectrum and displacement RMS in NOD_BAND_HZ."""
    if column not in k:
        return np.nan, np.nan
    x, t = k[column].to_numpy(float), k.t.to_numpy(float)
    return _dominant(np.gradient(x, t), t, NOD_BAND_HZ)[0], _dominant(x, t, NOD_BAND_HZ)[1]


def _pose_fractions(overlays: list[dict]) -> dict:
    """Hand-to-face, crossed-arm and crossed-leg fractions from saved pose overlays.

    Uses MediaPipe Pose indices: 0 nose, 11/12 shoulders, 15/16 wrists, 23/24 hips,
    27/28 ankles; a point counts when its saved visibility flag is true.
    """
    face = arms = legs = (0, 0)
    for overlay in overlays:
        pose = overlay.get("pose") or []
        if len(pose) < 29:
            continue
        pt = {i: np.array(pose[i][:2]) for i in range(len(pose)) if pose[i][2]}
        if 11 in pt and 12 in pt:
            width = np.linalg.norm(pt[11] - pt[12])
            middle = (pt[11][0] + pt[12][0]) / 2
            if 0 in pt and (15 in pt or 16 in pt) and width > 0:
                near = any(np.linalg.norm(pt[w] - pt[0]) / width < HAND_TO_FACE for w in (15, 16) if w in pt)
                face = face[0] + near, face[1] + 1
            if 15 in pt and 16 in pt:
                crossed = (np.sign(pt[15][0] - middle) == np.sign(pt[12][0] - middle) and
                           np.sign(pt[16][0] - middle) == np.sign(pt[11][0] - middle))
                arms = arms[0] + crossed, arms[1] + 1
        if all(i in pt for i in (23, 24, 27, 28)):
            middle = (pt[23][0] + pt[24][0]) / 2
            crossed = (np.sign(pt[27][0] - middle) == np.sign(pt[24][0] - middle) and
                       np.sign(pt[28][0] - middle) == np.sign(pt[23][0] - middle))
            legs = legs[0] + crossed, legs[1] + 1
    fraction = lambda pair: pair[0] / pair[1] if pair[1] >= POSE_VISIBLE_MIN else np.nan
    return {"hand_to_face_fraction": fraction(face), "crossed_arms_fraction": fraction(arms),
            "crossed_legs_fraction": fraction(legs)}


def kinematics_summary(kinematics: pd.DataFrame, overlays: list[dict]) -> dict:
    """Visibility, torso axis, body-centre shift, head oscillation, hand speed and postures."""
    k = kinematics
    visible = lambda column: float(k[column].astype(str).str.lower().eq("true").mean()) if column in k else np.nan
    centre = (k.shoulders_x + k.hips_x) / 2 if {"shoulders_x", "hips_x"} <= set(k) else pd.Series(dtype=float)
    speed = np.hypot(k.hands_x_velocity, k.hands_y_velocity) if "hands_x_velocity" in k else pd.Series(dtype=float)
    nod_hz, nod_rms = _oscillation(k, "head_y")
    turn_hz, turn_rms = _oscillation(k, "head_x")
    row = {f"{part}_visible_fraction": visible(f"{part}_visible") for part in ("face", "shoulders", "hips", "hands", "feet")}
    row.update(torso_tilt_median_deg=_median(k.get("torso_tilt_deg", [])), torso_tilt_iqr_deg=_iqr(k.get("torso_tilt_deg", [])),
               centre_shift_iqr=_iqr(centre), head_roll_iqr_deg=_iqr(k.get("head_roll_deg", [])),
               nod_frequency_hz=nod_hz, nod_rms=nod_rms, turn_frequency_hz=turn_hz, turn_rms=turn_rms,
               hand_speed_median=_median(speed), **_pose_fractions(overlays))
    return row


def question_topic(question: str, previous: str | None) -> str | None:
    sentences = re.split(r"(?<=[.?!])\s+", question.strip())
    text = " ".join(sentences[-2:]).lower()
    for name, pattern in TOPIC_PATTERNS:
        if re.search(pattern, text):
            return name
    return previous


def session_stages(segments: dict, baselines: list[dict]) -> dict:
    """Rest baselines, dialogue span, topic blocks and neutral («База») answers.

    The dialogue covers QA answers between the end of the beginning baseline and the
    start of the ending baseline. Neutral answers are answers in neutral blocks.
    """
    by_name = {b["name"]: b for b in baselines}
    start = by_name.get("beginning", {}).get("end_s", 0.)
    end = by_name.get("ending", {}).get("start_s", np.inf)
    blocks, topic = [], None
    neutral = []
    for qa in segments.get("qa_candidates", []):
        topic = question_topic(qa.get("question", ""), topic)
        if not start <= qa["answer_start_s"] < end:
            continue
        if blocks and blocks[-1]["topic"] == topic:
            blocks[-1].update(end_s=qa["answer_end_s"], pairs=blocks[-1]["pairs"] + 1)
        else:
            blocks.append({"topic": topic, "start_s": max(qa["question_start_s"], start), "end_s": qa["answer_end_s"], "pairs": 1})
        if topic == NEUTRAL:
            neutral.append(qa)
    return {"baselines": [{"name": b["name"], "start_s": b["start_s"], "end_s": b["end_s"]} for b in baselines],
            "dialogue_start_s": blocks[0]["start_s"] if blocks else np.nan,
            "dialogue_end_s": blocks[-1]["end_s"] if blocks else np.nan,
            "blocks": blocks, "neutral_answers": neutral}
