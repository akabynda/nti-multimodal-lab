"""Synthetic checks for recording summaries and session stage segmentation."""
import numpy as np
import pandas as pd

from nti_lab.session_summary import (NEUTRAL, hrv_summary, kinematics_summary, question_topic,
                                     session_stages, speech_summary, _pose_fractions)


def test_hrv_summary_uses_only_quality_passing_windows():
    metrics = pd.DataFrame({"ecg_quality": ["usable_automatic", "usable_automatic", "review_required"],
                            "spectrum_status": ["exploratory_120s", "insufficient_quality", "exploratory_120s"],
                            "hr_bpm": [60, 70, 200], "sdnn_ms": [40, 50, 1], "rmssd_ms": [30, 40, 1],
                            "pnn50_pct": [10, 20, 0], "vlf_ms2": [1, 99, 99], "lf_ms2": [2, 99, 99],
                            "hf_ms2": [3, 99, 99], "lf_hf": [.5, 9, 9], "total_ms2": [6, 99, 99],
                            "resp_status": ["RIP_periodic_candidate", "RIP_low_periodicity", "RIP_periodic_candidate"],
                            "resp_bpm": [12, 30, 14]})
    row = hrv_summary(metrics)
    assert row["usable_windows"] == 2 and row["hr_bpm"] == 65
    assert row["lf_ms2"] == 2 and row["lf_hf"] == .5
    assert row["rip_windows"] == 2 and row["rip_rate_bpm"] == 13


def test_speech_summary_counts_respondent_turns_and_voiced_windows():
    segments = {"qa_candidates": [{"latency_s": 1.0, "answer_duration_s": 4}, {"latency_s": -.2, "answer_duration_s": 2},
                                  {"latency_s": 3.0, "answer_duration_s": 6}],
                "turns": [{"start": 10, "end": 20, "speaker_role": "respondent", "text": "мама мыла раму"},
                          {"start": 20, "end": 30, "speaker_role": "interviewer", "text": "вопрос"}]}
    acoustic = pd.DataFrame({"time_s": [12, 15, 25], "voiced_fraction": [.5, .1, .9], "pitch_median_hz": [100, 300, 500]})
    row = speech_summary(segments, acoustic)
    assert row["latency_median_s"] == 2 and row["answer_median_s"] == 4
    assert row["words_per_min"] == 18 and row["syllables_per_s"] == .6
    assert row["acoustic_windows"] == 1 and row["pitch_median_hz"] == 100


def test_topics_continue_into_follow_up_questions():
    assert question_topic("Расскажите немножко о себе.", None) == NEUTRAL
    assert question_topic("Как вы относитесь к алкоголю?", NEUTRAL) == "Алкоголь"
    assert question_topic("А почему?", "Алкоголь") == "Алкоголь"
    assert question_topic("Что чувствует человек, который вынужден лгать?", "Алкоголь") == "Обман"


def test_session_stages_exclude_rest_baselines():
    qa = [dict(question="Как вас зовут?", question_start_s=1, answer_start_s=2, answer_end_s=3),
          dict(question="Расскажите о себе.", question_start_s=120, answer_start_s=131, answer_end_s=140),
          dict(question="Есть ли у вас увлечения?", question_start_s=141, answer_start_s=142, answer_end_s=150),
          dict(question="Вы когда-либо лгали?", question_start_s=151, answer_start_s=152, answer_end_s=160),
          dict(question="Как вы себя чувствуете?", question_start_s=900, answer_start_s=901, answer_end_s=905)]
    stages = session_stages({"qa_candidates": qa}, [{"name": "beginning", "start_s": 10, "end_s": 125},
                                                    {"name": "ending", "start_s": 890, "end_s": 1000}])
    assert [b["topic"] for b in stages["blocks"]] == [NEUTRAL, "Обман"]
    assert stages["blocks"][0]["pairs"] == 2 and stages["dialogue_start_s"] == 125
    assert [q["answer_start_s"] for q in stages["neutral_answers"]] == [131, 142]


def pose(points):
    result = [[0., 0., False] for _ in range(33)]
    for index, (x, y) in points.items():
        result[index] = [x, y, True]
    return {"pose": result}


def test_pose_fractions_detect_hand_at_face_and_crossed_arms():
    body = {0: (.5, .2), 11: (.6, .4), 12: (.4, .4), 23: (.55, .7), 24: (.45, .7), 27: (.55, .95), 28: (.45, .95)}
    open_pose = pose({**body, 15: (.7, .7), 16: (.3, .7)})
    face_pose = pose({**body, 15: (.52, .25), 16: (.3, .7)})
    crossed = pose({**body, 15: (.35, .6), 16: (.65, .6)})
    row = _pose_fractions([open_pose] * 10 + [face_pose] * 5 + [crossed] * 5)
    assert row["hand_to_face_fraction"] == .25
    assert row["crossed_arms_fraction"] == .25
    assert row["crossed_legs_fraction"] == 0
    assert np.isnan(_pose_fractions([open_pose] * 3)["hand_to_face_fraction"])


def test_head_oscillation_frequency_is_recovered():
    t = np.arange(0, 60, .08)
    k = pd.DataFrame({"t": t, "head_y": .5 + .01 * np.sin(2 * np.pi * 1.2 * t), "head_x": .5 + 0 * t,
                      "face_visible": True})
    row = kinematics_summary(k, [])
    assert abs(row["nod_frequency_hz"] - 1.2) < .1
    assert abs(row["nod_rms"] - .01 / np.sqrt(2)) < .001
    assert row["face_visible_fraction"] == 1


def test_icc_and_ppg_agreement():
    from nti_lab.session_summary import icc_a1, ppg_agreement
    x = np.array([60., 70, 80, 90])
    assert np.isclose(icc_a1(x, x), 1)
    assert icc_a1(x, x + 10) < icc_a1(x, x + 1) < 1
    comparison = pd.DataFrame({"paired_usable": [True, True, True, False],
                               **{f"{s}_{m}": x + (2 if s == "ppg" else 0) for s in ("ecg", "ppg")
                                  for m in ("hr_bpm", "sdnn_ms", "rmssd_ms", "lf_ms2", "hf_ms2", "lf_hf")}})
    hr = ppg_agreement(comparison)[0]
    assert hr["windows"] == 3 and hr["bias"] == 2 and np.isclose(hr["r"], 1) and hr["icc"] < 1
