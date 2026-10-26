"""Report figures write files and return agreement statistics from saved pairs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nti_lab.palette import decimal_comma
from nti_lab.report_figures import bland_altman, sync_curves


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
