"""Report figures from saved synchronization, video and BioRadio outputs."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .palette import BLUE, CYAN_DARK, INK, MUTED, PLOT_STYLE, RED, VIOLET, decimal_comma

STYLE = {**PLOT_STYLE, "font.family": "DejaVu Sans", "font.size": 8,
         "axes.spines.top": False, "axes.spines.right": False}
STATUS = {"supported_unverified": "однозначный максимум", "ambiguous_unverified": "неоднозначный максимум",
          "insufficient_signal": "мало сигнала"}


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    decimal_comma(fig)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def sync_curves(alignments: list[tuple[str, dict]], path: Path):
    """Joint score and per-baseline |r| against BioRadio lead; accepted lead marked."""
    with plt.rc_context(STYLE):
        rows = int(np.ceil(len(alignments) / 3))
        fig, axes = plt.subplots(rows, 3, figsize=(8.3, 2.1 * rows), layout="constrained", squeeze=False)
        for ax, (code, a) in zip(axes.flat, alignments):
            curve = a.get("score_curve") or {}
            lead = np.asarray(curve.get("lead_s", []), dtype=float)
            if len(lead):
                ax.plot(lead, np.asarray(curve["beginning_abs_correlation"], float), color=CYAN_DARK, lw=.6, alpha=.7, label="|r| начальный фон")
                ax.plot(lead, np.asarray(curve["ending_abs_correlation"], float), color=VIOLET, lw=.6, alpha=.7, label="|r| конечный фон")
                ax.plot(lead, np.asarray(curve["joint_score"], float), color=INK, lw=.9, label="общая оценка")
            best = (a.get("joint_support") or {}).get("best_lead_s")
            if best is not None:
                ax.axvline(best, color=RED, lw=.9, ls=":", label="принятое опережение")
            ax.set_ylim(0, 1)
            ax.set_title(f"{code} · {STATUS.get(a.get('status'), a.get('status'))}", fontsize=8)
            ax.grid(alpha=.4)
        for ax in axes.flat[len(alignments):]:
            ax.axis("off")
        for ax in axes[-1]:
            ax.set_xlabel("Опережение БиоРадио, с")
        for ax in axes[:, 0]:
            ax.set_ylabel("Оценка")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
        _save(fig, path)


def bland_altman(pairs: list[tuple[str, pd.DataFrame]], path: Path) -> dict:
    """rPPG minus ECG heart rate against their mean for all paired 30 s windows."""
    frames = [p.assign(code=c).dropna(subset=["ecg_hr_bpm", "rppg_hr_bpm"]) for c, p in pairs]
    data = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    result = {"pairs": len(data)}
    if not len(data):
        return result
    mean = (data.ecg_hr_bpm + data.rppg_hr_bpm) / 2
    diff = data.rppg_hr_bpm - data.ecg_hr_bpm
    bias, sd = float(diff.mean()), float(diff.std(ddof=1))
    result.update(bias_bpm=bias, sd_bpm=sd, low_bpm=bias - 1.96 * sd, high_bpm=bias + 1.96 * sd,
                  within_5_bpm=float((diff.abs() <= 5).mean()))
    colors = plt.get_cmap("tab10")
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 3.8), layout="constrained")
        for i, (code, frame) in enumerate(data.groupby("code")):
            m = (frame.ecg_hr_bpm + frame.rppg_hr_bpm) / 2
            ax.scatter(m, frame.rppg_hr_bpm - frame.ecg_hr_bpm, s=9, color=colors(i % 10), alpha=.75, label=code, lw=0)
        for value, style in ((bias, "-"), (result["low_bpm"], "--"), (result["high_bpm"], "--")):
            ax.axhline(value, color=MUTED, lw=.9, ls=style)
        ax.set_xlabel("Среднее ЧСС rPPG и ЭКГ, уд/мин")
        ax.set_ylabel("ЧСС rPPG − ЭКГ, уд/мин")
        ax.grid(alpha=.4)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize=7)
        _save(fig, path)
    return result


def synchronized_signals(*, turns: list[dict], video: dict, bio: dict, lead_s: float, start_s: float,
                         end_s: float, path: Path):
    """Speech turns, ECG with R peaks, rPPG and RIP respiration on the video clock.

    Video time t_video = t_bio − lead_s, the formula used for all modalities.
    """
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(4, 1, figsize=(8.3, 5.6), sharex=True, layout="constrained",
                                 gridspec_kw={"height_ratios": [.6, 1.2, 1, 1]})
        for turn in turns:
            if turn["end"] < start_s or turn["start"] > end_s:
                continue
            role = turn.get("speaker_role")
            axes[0].barh(1 if role == "interviewer" else 0, turn["end"] - turn["start"], left=turn["start"],
                         color=MUTED if role == "interviewer" else BLUE, height=.6)
        axes[0].set_yticks([0, 1], ["Респондент", "Интервьюер"])
        axes[0].set_title("Реплики (стенограмма)", loc="left", fontsize=8)
        t = np.asarray(bio["ecg_t_s"]) - lead_s
        keep = (t >= start_s) & (t <= end_s)
        axes[1].plot(t[keep], np.asarray(bio["ecg_filtered_mv"])[keep], color=RED, lw=.5)
        peaks = np.asarray(bio["r_peaks_s"]) - lead_s
        peaks = peaks[(peaks >= start_s) & (peaks <= end_s)]
        axes[1].plot(peaks, np.interp(peaks, t[keep], np.asarray(bio["ecg_filtered_mv"])[keep]), "v", color=INK, ms=3)
        axes[1].set_ylabel("ЭКГ, мВ")
        axes[1].set_title("ЭКГ после фильтрации и R-зубцы", loc="left", fontsize=8)
        times = np.asarray(video["times"], dtype=float)
        rppg = np.asarray([np.nan if v is None else v for v in video["rppg"]], dtype=float)
        keep = (times >= start_s) & (times <= end_s)
        axes[2].plot(times[keep], rppg[keep], color=BLUE, lw=.8)
        axes[2].set_ylabel("rPPG, отн. ед.")
        axes[2].set_title("Видеопульс rPPG (POS)", loc="left", fontsize=8)
        t = np.asarray(bio["rip_t_s"]) - lead_s
        keep = (t >= start_s) & (t <= end_s)
        axes[3].plot(t[keep], np.asarray(bio["respiration_mv"])[keep], color=VIOLET, lw=.8)
        axes[3].set_ylabel("RIP, мВ")
        axes[3].set_title("Дыхание по поясу RIP", loc="left", fontsize=8)
        axes[3].set_xlabel("Время видео, с")
        axes[3].set_xlim(start_s, end_s)
        for ax in axes:
            ax.grid(alpha=.4)
        _save(fig, path)
