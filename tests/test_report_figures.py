"""Report figures write files and return agreement statistics from saved pairs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nti_lab.palette import decimal_comma
from nti_lab.report_figures import ARCHITECTURE, architecture, bland_altman, sync_curves


def test_bland_altman_statistics(tmp_path):
    pairs = pd.DataFrame({"ecg_hr_bpm": [60, 70, 80, None], "rppg_hr_bpm": [62, 70, 84, 90]})
    result = bland_altman([("P01", pairs)], tmp_path / "ba.png")
    assert result["pairs"] == 3 and result["bias_bpm"] == 2
    assert np.isclose(result["high_bpm"] - result["low_bpm"], 2 * 1.96 * 2)
    assert np.isclose(result["within_5_bpm"], 1)
    assert (tmp_path / "ba.png").exists()


def test_sync_curves_handle_missing_curve(tmp_path):
    curve = {"lead_s": [0, 1, 2], "beginning_abs_correlation": [.1, .5, .2],
             "ending_abs_correlation": [.2, .4, .1], "joint_score": [.14, .45, .14]}
    sync_curves([("P01", {"score_curve": curve, "joint_support": {"best_lead_s": 1}, "status": "supported_unverified"}),
                 ("P02", {"status": "insufficient_signal"})], tmp_path / "sync.png")
    assert (tmp_path / "sync.png").exists()


def test_decimal_comma_tick_labels():
    fig, ax = plt.subplots()
    ax.plot([0, .5], [0, 1.5])
    decimal_comma(fig)
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_xticklabels()]
    assert any("," in label for label in labels) and not any("." in label for label in labels)
    plt.close(fig)


def test_architecture_columns_connect_left_to_right(tmp_path):
    columns = {column for column, *_ in ARCHITECTURE}
    assert columns == set(range(5))
    for column, first, last, *_ in ARCHITECTURE:
        if column < 4:
            rows = set(range(first, last + 1))
            assert any(c == column + 1 and set(range(f, l + 1)) & rows for c, f, l, *_ in ARCHITECTURE)
    architecture(tmp_path / "architecture.png")
    assert (tmp_path / "architecture.png").exists()


def test_contrast_dots_skip_untested_features(tmp_path):
    from nti_lab.report_figures import contrast_dots
    contrasts = pd.DataFrame(dict(participant_id=["A", "B", "A"], feature=["x", "x", "y"], contrast=[1., -1., 2.]))
    tests = [dict(feature="x", h1_mean=0., h1_ci_low=-1., h1_ci_high=1., expected=1), dict(feature="y", h1_mean=None)]
    contrast_dots(contrasts, tests, {"x": "X"}, {"A"}, tmp_path / "dots.png")
    assert (tmp_path / "dots.png").exists()


def test_hr_timeline_writes_figure(tmp_path):
    from nti_lab.report_figures import hr_timeline
    metrics = pd.DataFrame(dict(center_s=[60, 65, 70], hr_bpm=[70, 72, 90], ecg_quality=["usable_automatic"] * 2 + ["review_required"]))
    stages = {"blocks": [{"topic": "Обман", "start_s": 10, "end_s": 50}, {"topic": None, "start_s": 50, "end_s": 60}],
              "baselines": [{"name": "beginning", "start_s": 0, "end_s": 10}]}
    hr_timeline([("P01", metrics, -5., stages)], tmp_path / "hr.png")
    assert (tmp_path / "hr.png").exists()
