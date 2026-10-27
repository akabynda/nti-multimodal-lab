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



ARCHITECTURE = (
    # (column, first row, last row, title, details); data flows left to right between adjacent columns.
    (0, 0, 0, "Видео MTS", "одна камера"), (0, 1, 1, "Звук", "дорожка видео"),
    (0, 2, 2, "БиоРадио BCRX", "ЭКГ, RIP, GSR, ФПГ"), (0, 3, 3, "Истории, анкета", "DOCX, XLSX"),
    (1, 0, 0, "vision.py", "лицо, rPPG, поза"), (1, 1, 1, "audio.py", "ASR, роли, акустика"),
    (1, 2, 2, "biosignals.py", "RR, ВСР, ИН, дыхание"), (1, 3, 3, "research.py", "сверка историй"),
    (2, 0, 3, "alignment.py\nsync.py", "единая шкала\nвремени видео"),
    (3, 0, 0, "session_summary.py", "этапы, сводки"), (3, 1, 1, "multimodal_\nfeatures.py", "признаки ответов"),
    (3, 2, 2, "contrasts.py", "«База», гипотезы"), (3, 3, 3, "multimodal_\nanalysis.py", "модели LOPO"),
    (4, 0, 1, "app.py", "интерфейс\nиз 5 зон"), (4, 2, 3, "build_report.py", "PDF отчёта"),
)


def architecture(path: Path):
    """Processing scheme: inputs, modality modules, common clock, analysis and outputs."""
    width, height, dx, dy = 1.62, .7, 2.1, 1.0
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8.3, 4.4), layout="constrained")
        boxes = []
        for column, first, last, title, details in ARCHITECTURE:
            x, top, bottom = column * dx, -first * dy + height / 2, -last * dy - height / 2
            boxes.append((column, first, last, x))
            color = MUTED if column == 0 else CYAN_DARK if column == 2 else VIOLET if column == 4 else BLUE
            ax.add_patch(plt.Rectangle((x - width / 2, bottom), width, top - bottom, facecolor="white",
                                       edgecolor=color, linewidth=1.2))
            middle = (top + bottom) / 2
            ax.text(x, middle + .12, title, ha="center", va="bottom", fontsize=6.8, color=INK, weight="bold", linespacing=1)
            ax.text(x, middle - .02, details, ha="center", va="top", fontsize=6.2, color=MUTED, linespacing=1.1)
        for column, first, last, x in boxes:
            for row in range(first, last + 1):
                target = next((b for b in boxes if b[0] == column + 1 and b[1] <= row <= b[2]), None)
                if target:
                    ax.annotate("", (target[3] - width / 2, -row * dy), (x + width / 2, -row * dy),
                                arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=.7, shrinkA=0, shrinkB=0))
        for column, label in enumerate(("Исходные данные", "Обработка", "Синхронизация", "Анализ", "Результаты")):
            ax.text(column * dx, height / 2 + .18, label, ha="center", fontsize=7.5, color=INK)
        ax.set_xlim(-width / 2 - .05, 4 * dx + width / 2 + .05)
        ax.set_ylim(-3 * dy - height / 2 - .05, height / 2 + .45)
        ax.axis("off")
        _save(fig, path)


def contrast_dots(contrasts: pd.DataFrame, tests: list[dict], labels: dict, group: set, path: Path):
    """Participant «Ложь − Правда» contrasts per feature with mean and 95% bootstrap CI."""
    rows = [t for t in tests if t.get("h1_mean") is not None]
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.6, .42 * len(rows) + 1.1), layout="constrained")
        for i, test in enumerate(rows):
            part = contrasts[contrasts.feature.eq(test["feature"])].dropna(subset=["contrast"])
            jitter = np.linspace(-.12, .12, len(part)) if len(part) > 1 else np.zeros(len(part))
            colors = [BLUE if pid in group else VIOLET for pid in part.participant_id]
            ax.scatter(part.contrast, i + jitter, s=14, c=colors, alpha=.8, lw=0, zorder=2)
            ax.errorbar(test["h1_mean"], i, xerr=[[test["h1_mean"] - test["h1_ci_low"]], [test["h1_ci_high"] - test["h1_mean"]]],
                        fmt="D", color=INK, markersize=4, capsize=3, lw=1, zorder=3)
        arrow = {1: " (ожид. ↑)", -1: " (ожид. ↓)"}
        ax.set_yticks(range(len(rows)), [labels.get(t["feature"], t["feature"]) + arrow.get(t.get("expected"), "") for t in rows])
        ax.invert_yaxis()
        ax.axvline(0, color=MUTED, lw=.8)
        ax.set_xlabel("Контраст «Ложь − Правда», z")
        ax.grid(axis="x", alpha=.4)
        handles = [plt.Line2D([], [], marker="o", ls="", color=BLUE, label="ведущий тип «Содействующий»"),
                   plt.Line2D([], [], marker="o", ls="", color=VIOLET, label="остальные участники"),
                   plt.Line2D([], [], marker="D", ls="", color=INK, label="среднее и 95 % ДИ")]
        fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
        _save(fig, path)


TOPIC_COLORS = {"Нейтральные вопросы": "#D9DDE6", "Выбор профессии": "#CFE0FF", "Достижения": "#D6F5FF",
                "Алкоголь": "#E9D9FF", "Конфликт": "#FFD9D6", "Обман": "#FFE9B8", "Экзамен": "#E3F7C9",
                "Летний день": "#F3F0C9", "Нерешённая проблема": "#EADFD3"}


def hr_timeline(sessions: list[tuple[str, pd.DataFrame, float, dict]], path: Path):
    """ECG heart rate of QC-passing 120 s windows on the video clock with session stages.

    Each session: code, metrics.csv rows, BioRadio-to-video offset (t_video = offset + t_bio)
    and `session_summary.session_stages` output. Rest baselines are grey hatched.
    """
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(sessions), 1, figsize=(8.3, 1.05 * len(sessions) + .8), sharex=True,
                                 layout="constrained", squeeze=False)
        used = []
        for ax, (code, metrics, offset, stages) in zip(axes[:, 0], sessions):
            for block in stages["blocks"]:
                topic = block["topic"] or "Нейтральные вопросы"
                ax.axvspan(block["start_s"], block["end_s"], color=TOPIC_COLORS.get(topic, "#EEEEEE"), lw=0)
                used.append(topic)
            for baseline in stages["baselines"]:
                ax.axvspan(baseline["start_s"], baseline["end_s"], facecolor="white", edgecolor=MUTED, hatch="///", lw=0, alpha=.6)
            good = metrics[metrics.ecg_quality.eq("usable_automatic")]
            ax.plot(good.center_s + offset, good.hr_bpm, ".", color=RED, ms=2.2)
            ax.set_ylabel(code, rotation=0, ha="right", va="center")
            ax.grid(axis="y", alpha=.4)
        axes[-1, 0].set_xlabel("Время видео, с (точки — центры окон 120 с, прошедших контроль)")
        fig.supylabel("ЧСС ЭКГ, уд/мин", fontsize=8)
        topics = [t for t in TOPIC_COLORS if t in used]
        handles = [plt.Rectangle((0, 0), 1, 1, color=TOPIC_COLORS[t], label=t) for t in topics]
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=MUTED, hatch="///", label="фон с закрытыми глазами"))
        fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=7)
        _save(fig, path)
