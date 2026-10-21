"""Reproduce the local participant-profile analysis from saved measurements."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nti_lab import multimodal_analysis as analysis
from nti_lab.multimodal_features import (extract_session_features, extract_trial_features,
                                        SESSION_FEATURE_COLUMNS, FEATURE_COLUMNS)
from nti_lab.palette import PLOT_STYLE, BLUE, INK, MUTED, LINE

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/research/multimodal"


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def save(name, value):
    (OUT/name).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def csv(name, frame):
    frame.to_csv(OUT/name, index=False, encoding="utf-8-sig")


def number(value, digits=3):
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def inspect_assumptions(frame, helper_path):
    spec = importlib.util.spec_from_file_location("scientific_assumption_checks", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    diagnostics = {}
    for column in analysis.PROFILE_ASSOCIATIONS + list(SESSION_FEATURE_COLUMNS):
        data = frame[column].astype(float)
        item = dict(n=int(data.notna().sum()), missing=int(data.isna().sum()),
                    minimum=data.min(), median=data.median() if data.notna().any() else None, maximum=data.max())
        if data.notna().sum() >= 3 and data.dropna().nunique() > 1:
            item["normality_descriptive_only"] = helper.check_normality(data, name=column, plot=False)
            item["outlier_review"] = helper.detect_outliers(data, name=column, plot=False)
        diagnostics[column] = item
    save("diagnostics.json", {"unit": "participant", "no_rows_removed_by_diagnostics": True,
         "test_not_selected_by_normality_p": True, "variables": diagnostics})


def figures(frame, associations, models, increment):
    style = dict(PLOT_STYLE, **{"font.family": "DejaVu Sans", "font.size": 8,
                "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    selected = [r for r in associations if r["family"] == "primary_12"]
    with plt.rc_context(style):
        fig, axes = plt.subplots(4, 3, figsize=(8.3, 10.0), layout="constrained")
        for ax, row in zip(axes.flat, selected):
            x, y = frame[row["profile"]].to_numpy(float), frame[row["feature"]].to_numpy(float)
            keep = np.isfinite(x) & np.isfinite(y)
            ax.scatter(x[keep], y[keep], s=23, color=BLUE, edgecolor="white", linewidth=.5)
            points = list(zip(x[keep], y[keep], frame.loc[keep, "participant_code"]))
            for j, (px, py, code) in enumerate(points):
                close = [k for k,(qx,qy,_) in enumerate(points) if
                         abs(qx-px) < max(np.ptp(x[keep])*.08, 1e-8) and abs(qy-py) < max(np.ptp(y[keep])*.08, 1e-8)]
                offset = 3 if len(close) == 1 else (-8 if close.index(j) % 2 == 0 else 6)
                ax.annotate(code, (px, py), xytext=(3, offset), textcoords="offset points", fontsize=6.5, color=MUTED)
            ax.margins(.13)
            ax.set_xlabel(row["profile_label"], fontsize=8)
            ax.set_ylabel(row["feature_label"].replace(", ", "\n", 1), fontsize=8)
            ax.set_title(f"ρ={number(row['rho'],2)} · n={row['n']} · p Holm={number(row['p_holm'],2)}", fontsize=8)
            ax.grid(axis="y", alpha=.45)
        fig.suptitle("Анкеты и измеренные признаки · одна точка — один участник", fontsize=12)
        for extension in ("png", "svg"):
            fig.savefig(OUT/f"associations.{extension}", dpi=300, metadata={"Creator": "nti-lab analyze_multimodal.py"})
        plt.close(fig)
        fig, (ax, right) = plt.subplots(1, 2, figsize=(10.5, 5.4), gridspec_kw={"width_ratios": [1.5, 1]}, layout="constrained")
        for i, model in enumerate(models):
            metrics = model["metrics"]
            color = BLUE if model["id"] in ("av", "av_profiles") else INK
            values = list(metrics["participant_brier"].values())
            ax.scatter(values, np.full(len(values), i), s=12, color=MUTED, alpha=.3, zorder=1)
            ax.errorbar(metrics["brier"], i, xerr=[[metrics["brier"]-metrics["brier_ci_low"]],
                [metrics["brier_ci_high"]-metrics["brier"]]], color=color, fmt="D", markersize=4, capsize=3, zorder=2)
        ax.set_yticks(range(len(models)), [m["name"] for m in models], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Brier · меньше — лучше")
        ax.set_title("Проверка на новом участнике", fontsize=11)
        ax.grid(axis="x", alpha=.4)
        codes = dict(zip(frame.participant_id, frame.participant_code))
        items = sorted(increment["participant_deltas"].items(), key=lambda pair: codes[pair[0]])
        right.scatter([v for _, v in items], range(len(items)), color=BLUE, s=25)
        right.set_yticks(range(len(items)), [codes[k] for k, _ in items])
        right.invert_yaxis()
        right.axvline(0, color=MUTED, linewidth=.8)
        right.set_title("Добавление анкеты к видео и речи", fontsize=11)
        right.set_xlabel("Δ Brier = с анкетой − без анкеты\nОтрицательные значения — улучшение")
        right.grid(axis="x", alpha=.4)
        fig.suptitle("Точки слева — участники; ромбы и интервалы — среднее и условный 95% bootstrap ДИ", fontsize=9)
        for extension in ("png", "svg"):
            fig.savefig(OUT/f"model_comparison.{extension}", dpi=300, metadata={"Creator": "nti-lab analyze_multimodal.py"})
        plt.close(fig)
    save("figure_manifest.json", {"associations": {"data": "participant_features.csv", "tests": "associations.csv",
        "unit": "participant", "rows": len(frame), "no_trend_lines": True},
        "model_comparison": {"data": "model_predictions.csv", "estimates": "summary.json",
        "interval": "95% percentile participant bootstrap of fixed OOF predictions, 5000 replicates"},
        "formats": ["300 dpi PNG", "SVG"], "style": "local ITMO palette; DejaVu Sans"})


def write_report(summary, participants):
    d = summary["dataset"]
    text = ["# Совместный анализ анкет, видео, речи и биосигналов", "",
        "\n\n".join(summary["findings_ru"]), "", "## Данные и проверяемая задача", "",
        f"Профили связаны по ИСУ для {d['participants']} участников. Единица классификации — целый ответ респондента: "
        f"соответствуют эталону — {d['agreements']}, содержат несоответствие — {d['contradictions']}. "
        "Повторные ссылки на один ответ объединены. Полная проверка цитат и причины отбора сохранены в label_review.json. "
        "Несоответствие включает отрицание факта, замену события, примера или существенной детали. "
        "При нескольких проверяемых утверждениях в ответе достаточно одного несоответствия для метки 1.", "",
        "Выраженность, Эмоц+ и Эмоц− входят в модель численно. Естественный и демонстрируемый типы представлены "
        "отдельными признаками наличия каждого из четырёх компонентов. Баланс анализируется в корреляциях; в модели он "
        "не дублирует разность Эмоц+ и Эмоц−. ИСУ, имя, даты заполнения и текст эталона в признаки модели не входят.", "",
        "## Связи анкет с измерениями", "",
        "Одна строка — один участник. Основная семья: 12 сочетаний выраженности и эмоционального баланса с шестью "
        "показателями речи, движения и ЭКГ. Остальные сочетания образуют отдельную разведочную семью. "
        "Spearman ρ, двусторонние точные перестановки одного вектора, сохранённые совпадающие ранги; поправка Холма "
        "внутри каждой заранее заданной семьи. Минимум шесть пригодных участников. 95% ДИ получены парным bootstrap "
        "по участникам (5000 повторов). Таблица содержит также диапазон при исключении каждого участника.", "",
        "| Профиль | Признак | n | ρ | 95% ДИ | p Холма |", "|---|---|---:|---:|---|---:|"]
    for row in summary["associations"]:
        if row["family"] == "primary_12":
            text.append(f"| {row['profile_label']} | {row['feature_label']} | {row['n']} | {number(row['rho'])} | "
                        f"{number(row['ci_low'])}…{number(row['ci_high'])} | {number(row['p_holm'])} |")
    text += ["", "![Индивидуальные наблюдения по основным 12 сочетаниям](associations.png)", "",
        "Все дополнительные проверки, включая нерасчётные, находятся в associations.csv. "
        "Категории профилей и исходные значения сохранены в participant_features.csv.", "", "## Проверка моделей", "",
        summary["explanation_ru"], "",
        "| Модель | Brier | 95% ДИ Brier | Сбалансированная точность | ROC AUC |", "|---|---:|---|---:|---:|"]
    for model in summary["models"]:
        m = model["metrics"]
        text.append(f"| {model['name']} | {number(m['brier'])} | {number(m['brier_ci_low'])}…{number(m['brier_ci_high'])} | "
                    f"{number(m['balanced_accuracy'])} | {number(m['roc_auc'])} |")
    inc = summary["profile_increment"]
    text += ["", f"Основное сравнение: Δ Brier (видео + речь + анкета минус видео + речь) = {inc['delta_brier']:.4f}; "
        f"условный 95% ДИ [{inc['ci_low']:.4f}; {inc['ci_high']:.4f}]. Отрицательное значение означает улучшение.", "",
        "Интервалы моделей — bootstrap по участникам при фиксированных прогнозах из перекрёстной проверки; "
        "они не включают повторное обучение на bootstrap-выборках. "
        f"Выборка для моделей: {summary['dataset']['trials']} ответов; "
        f"несоответствий эталону: {summary['dataset']['contradictions']}. "
        "Это состав модельной выборки, а не число случаев лжи во всём эксперименте. "
        "Длительность ответов и пропуски признаков могут влиять на результат; "
        "длительность и наличие измерений проверены отдельными контрольными моделями.", "",
        "![Сравнение моделей и индивидуальные изменения ошибки](model_comparison.png)", "",
        "## Синхронизация и качество", "",
        "Биосигналы переведены в шкалу видео по t_video = offset + scale × t_bio. Для каждого ответа используются "
        "исходные акустические окна внутри речи респондента и видеопризнаки того же интервала. Метрики ЭКГ и RIP "
        "из 120-секундных окон описывают фон вокруг ответа. Ближайшее плохое окно не подменяется удалённым хорошим. "
        "Для межучастниковых показателей ЭКГ, RIP и rPPG задано покрытие не менее 20% пригодных окон. "
        "GSR не прошёл исходный контроль качества; контактный ФПГ имеется у одного участника и включён описательно.", "",
        "Основное сравнение анкет проведено на синхронных аудио и видео. Расширения с BioRadio пересчитаны на нижней "
        "и верхней границах интервалов автоматического смещения. Эти интервалы характеризуют алгоритм синхронизации, "
        "а не все возможные ошибки часов.", "",
        "| Смещение | Модель | Brier | ROC AUC |", "|---|---|---:|---:|"]
    for scenario in summary["sync_sensitivity"]:
        for m in scenario["models"]:
            text.append(f"| {scenario['name']} | {m['name']} | {number(m['metrics']['brier'])} | {number(m['metrics']['roc_auc'])} |")
    text += ["", "## Воспроизведение", "",
        "План и параметры: analysis_plan.json. Происхождение и SHA-256 входов: manifest.json. "
        "Признаки и причины пропусков: participant_features.csv, trial_features.csv и feature_qc.json. "
        "Прогнозы исключённого участника: model_predictions.csv; состав каждого обучения: fold_audit.json. "
        "Исходный код: scripts/analyze_multimodal.py, src/nti_lab/multimodal_analysis.py и multimodal_features.py.", "",
        "## Источники", "",
        "- [SciPy: Spearman и перестановочный расчёт для малых выборок](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html).",
        "- [Scikit-learn: проверка с группами участников](https://scikit-learn.org/stable/modules/cross_validation.html).",
        "- Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). "
        "[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://doi.org/10.48550/arXiv.2609.00065). "
        "Использованы процедуры Statistical Analysis и Scientific Visualization.", ""]
    (OUT/"analysis.md").write_text("\n".join(text), encoding="utf-8")


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-review", type=Path, required=True)
    parser.add_argument("--assumption-helper", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=OUT,
                        help="Separate result directory for a revised label set; existing plans are immutable")
    args = parser.parse_args()
    OUT = args.output_dir.resolve()
    args.label_review = args.label_review.resolve()
    args.assumption_helper = args.assumption_helper.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    review = json.loads(args.label_review.read_text(encoding="utf-8"))
    plan = dict(outcome_definition=review["label_definitions"], created_before_results=datetime.now(timezone.utc).isoformat(), seed=analysis.SEED,
        unit_associations="participant", unit_classification="unique complete respondent answer", minimum_n=6,
        primary_family={"profiles": ["expression", "emotion_balance"], "features": analysis.PRIMARY_ASSOCIATION_FEATURES},
        secondary_family="all remaining numeric profile x session feature pairs, separate Holm family",
        missing_family_members="not_estimable; retained as p=1 for family size", session_minimum_quality_coverage=.2,
        test="exact absolute-rho two-sided permutation of one rank vector", interval="paired participant percentile bootstrap",
        bootstrap=analysis.BOOTSTRAPS, model="L2 logistic; C=.1; no tuning; fixed 0.5 threshold; lbfgs",
        models=[dict(id=i, name=n, features=c) for i,n,c in analysis.MODEL_SPECS],
        primary_comparison="av_profiles minus av Brier", secondary_comparison="fusion_profiles minus fusion Brier",
        training="leave one entire participant out; training-only imputation/scaling; participant-balanced sample weights",
        evaluation="participant-macro Brier; participant-weighted pooled AUC and balanced accuracy",
        model_intervals="conditional participant bootstrap of fixed OOF predictions; no formal incremental-value p-value",
        sensitivity="absolute lower/upper stored offset interval, scale unchanged", label_review_sha256=sha(args.label_review),
        category_corrections=analysis.TYPE_CORRECTIONS)
    if (OUT/"analysis_plan.json").exists():
        previous = json.loads((OUT/"analysis_plan.json").read_text(encoding="utf-8"))
        if {k:v for k,v in previous.items() if k != "created_before_results"} != {k:v for k,v in plan.items() if k != "created_before_results"}:
            raise ValueError("Prespecified plan differs: preserve the prior analysis and explicitly version a changed plan")
    else:
        save("analysis_plan.json", plan)
    for source in review.get("coverage_audits", []):
        path = ROOT/source["path"]
        if sha(path) != source["sha256"]:
            raise ValueError(f"Stale coverage audit: {path}")
        if path.resolve() != (OUT/path.name).resolve():
            shutil.copy2(path, OUT/path.name)
    topic_path = args.label_review.parent/"topic_audit.json"
    topic_audit = json.loads(topic_path.read_text(encoding="utf-8"))
    if topic_audit["sources"]["label_review"]["sha256"] != sha(args.label_review):
        raise ValueError("Topic audit was made for another set of labels")
    save("topic_audit.json", topic_audit)
    # Fail closed if canonical text evidence has changed since the review.
    source_items = list(review["sources"]["transcripts"].values())
    source_items += [review["sources"]["content_comparison"], review["sources"]["reference_stories"]]
    for source in source_items:
        path = ROOT / source.get("path", source.get("file"))
        if sha(path) != source["sha256"]:
            raise ValueError(f"Stale label review source: {path}")
    trials = review["strict_trials"]
    profiles = pd.read_csv(ROOT/"data/private/participant_profiles.csv")
    profiles = analysis.encode_profiles(profiles)
    print("Extracting session and trial measurements", flush=True)
    sessions, session_qc = extract_session_features(ROOT)
    trial_features, trial_qc = extract_trial_features(ROOT, trials)
    print("Measurements extracted", flush=True)
    sessions = sessions.merge(profiles, on="participant_id", validate="one_to_one", how="left")
    sessions["participant_code"] = [f"P{int(r.split('_',1)[0]):02d}" for r in sessions.recording_id]
    labels = pd.DataFrame([{k:t[k] for k in ("trial_id", "label")} for t in trials])
    trial_features = trial_features.drop(columns="label", errors="ignore").merge(labels, on="trial_id", validate="one_to_one")
    frame = analysis.model_frame(trial_features, profiles)
    csv("participant_features.csv", sessions)
    csv("trial_features.csv", frame)
    save("feature_qc.json", dict(sessions=session_qc, trials=trial_qc))
    if args.label_review != OUT/"label_review.json":
        shutil.copy2(args.label_review, OUT/"label_review.json")
    inspect_assumptions(sessions, args.assumption_helper)
    print("Calculating prespecified participant associations", flush=True)
    associations = analysis.profile_associations(sessions, SESSION_FEATURE_COLUMNS)
    csv("associations.csv", pd.DataFrame(associations))
    print("Fitting participant-held-out models", flush=True)
    models, predictions, folds, probabilities = analysis.evaluate_models(frame)
    csv("model_predictions.csv", predictions)
    save("fold_audit.json", folds)
    increment = analysis.compare_profile_increment(frame, probabilities["av"], probabilities["av_profiles"])
    fusion_increment = analysis.compare_profile_increment(frame, probabilities["fusion"], probabilities["fusion_profiles"])
    sensitivity = []
    for side in ("low", "high"):
        overrides = dict(zip(sessions.recording_id, sessions[f"sync_uncertainty_{side}_s"]))
        overrides = {k:float(v) for k,v in overrides.items() if np.isfinite(v)}
        shifted, qc = extract_trial_features(ROOT, trials, offset_overrides=overrides)
        shifted = shifted.drop(columns="label", errors="ignore").merge(labels, on="trial_id", validate="one_to_one")
        shifted = analysis.model_frame(shifted, profiles)
        sm, sp, sf, _ = analysis.evaluate_models(shifted, specs=[s for s in analysis.MODEL_SPECS if s[0] in ("fusion", "fusion_profiles")])
        sensitivity.append(dict(name="Нижняя граница" if side == "low" else "Верхняя граница", offsets=overrides, models=sm))
        csv(f"trial_features_offset_{side}.csv", shifted)
        csv(f"model_predictions_offset_{side}.csv", sp)
        save(f"feature_qc_offset_{side}.json", qc)
        save(f"fold_audit_offset_{side}.json", sf)
    primary = [r for r in associations if r["family"] == "primary_12"]
    secondary = [r for r in associations if r["family"] != "primary_12"]
    significant = [r for r in associations if r["p_holm"] is not None and r["p_holm"] < .05]
    estimated = sum(r["status"] == "estimated" for r in associations)
    dataset = dict(participants=len(sessions), trials=len(frame), agreements=int(frame.label.eq(0).sum()),
                   contradictions=int(frame.label.eq(1).sum()), participants_with_contradictions=int(frame.loc[frame.label.eq(1),"participant_id"].nunique()))
    m = {r["id"]:r["metrics"] for r in models}
    delta = increment["delta_brier"]
    interval_result = ("Интервал включает ноль: устойчивое улучшение не установлено."
                       if increment['ci_low'] <= 0 <= increment['ci_high'] else
                       "Интервал изменения ошибки не включает ноль.")
    findings = [f"Анкеты всех {len(sessions)} участников включены в совместный анализ. Для проверки моделей отобраны "
        f"{len(frame)} отдельных ответов: соответствуют эталону — {dataset['agreements']}, содержат несоответствие — {dataset['contradictions']}.",
        f"Из {len(primary)} основных и {len(secondary)} дополнительных сочетаний анкетных и измеренных показателей "
        f"рассчитаны {estimated}; для остальных не набралось шести пригодных участников. "
        f"После поправки Холма p < 0,05 у {len(significant)} сочетаний.",
        f"При добавлении анкеты к видео и речи средняя ошибка Brier "
        f"{'снизилась' if delta < 0 else 'выросла'} с {m['av']['brier']:.3f} до {m['av_profiles']['brier']:.3f}; "
        f"Δ = {delta:+.3f}, условный 95% ДИ [{increment['ci_low']:+.3f}; {increment['ci_high']:+.3f}]. "
        + interval_result,
        f"Расширение с BioRadio: Brier {m['fusion']['brier']:.3f} без анкеты и {m['fusion_profiles']['brier']:.3f} с анкетой. "
        "Для него рассчитана чувствительность к смещению часов.",
        f"При пороге 0,5 модель «видео + речь + анкета» выделила {m['av_profiles']['confusion_counts']['tp']} "
        f"из {dataset['contradictions']} несоответствий; сбалансированная точность — {100*m['av_profiles']['balanced_accuracy']:.0f}%.",
        topic_audit["conclusion_ru"]]
    explanation = (f"Каждый из {len(sessions)} участников по очереди полностью исключается из обучения. "
        "Пропуски и масштаб признаков обрабатываются только по обучающей части. "
        "L2-логистическая регрессия, C=0,1; параметры и порог 0,5 фиксированы. "
        "Участники имеют равный вес. Brier оценивает ошибку вероятностного прогноза; меньше — лучше. "
        "Метки ответов получены сопоставлением с правдивыми письменными историями. "
        "BioRadio представлен контекстом 120 с вокруг ответа.")
    summary = dict(schema_version=1, status="complete_exploratory_analysis", date=datetime.now().date().isoformat(),
        outcome_definition=review["label_definitions"], dataset=dataset, dataset_scope="selected_reference_comparable_answers",
        experiment_lie_count=None, findings_ru=findings, explanation_ru=explanation, models=models,
        profile_increment=increment, fusion_profile_increment=fusion_increment,
        associations=associations, sync_sensitivity=sensitivity,
        association_families={"primary":len(primary),"secondary":len(secondary),"estimated":estimated,"significant_holm":len(significant)},
        topic_composition=topic_audit["counts"], topic_conclusion_ru=topic_audit["conclusion_ru"],
        profile_encoding={"numeric":analysis.PROFILE_NUMERIC,"categories":analysis.TYPES,
                          "category_features":analysis.PROFILE_CATEGORIES,"corrections":analysis.TYPE_CORRECTIONS},
        label_review=review["summary"], quality_rule="session ECG/RIP/rPPG >=20% QC passing coverage; no GSR inference")
    save("summary.json", summary)
    figures(sessions, associations, models, increment)
    write_report(summary, sessions)
    packages = {p:importlib.metadata.version(p) for p in ("numpy", "scipy", "pandas", "scikit-learn", "matplotlib")}
    (OUT/"requirements_lock.txt").write_text("\n".join(f"{k}=={v}" for k,v in packages.items())+"\n", encoding="utf-8")
    inputs = set()
    for file in (ROOT/"outputs/audio").glob("*.segments.json"): inputs.add(file)
    for file in (ROOT/"outputs/audio").glob("*.features.csv"): inputs.add(file)
    for file in (ROOT/"outputs/video").glob("*.json"): inputs.add(file)
    for file in (ROOT/"outputs/sync").glob("*.json"): inputs.add(file)
    for participant in sessions.participant_id:
        for name in ("metadata.json", "metrics.csv", "rr_intervals.csv", "ppg_metrics.csv", "signals.npz"):
            path = ROOT/"outputs/bio"/participant/name
            if path.exists(): inputs.add(path)
    inputs.update([ROOT/"data/private/participant_profiles.csv", ROOT/"data/private/stories.json",
                   ROOT/"outputs/research/content_comparison.json", args.label_review, topic_path])
    inputs.update(ROOT/item["path"] for item in review.get("coverage_audits", []))
    code_files = [Path(__file__), ROOT/"scripts/reproduce_multimodal_tables.py", ROOT/"src/nti_lab/__init__.py",
                  ROOT/"src/nti_lab/multimodal_analysis.py", ROOT/"src/nti_lab/multimodal_features.py",
                  ROOT/"src/nti_lab/alignment.py", ROOT/"src/nti_lab/palette.py"]
    manifest = dict(run_completed=datetime.now(timezone.utc).isoformat(), python=sys.version, executable=sys.executable, packages=packages,
        input_files=[dict(path=str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(inputs)],
        code=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in code_files],
        assumption_helper=dict(path=str(args.assumption_helper),sha256=sha(args.assumption_helper)),
        command=f'python scripts/analyze_multimodal.py --label-review "{args.label_review.as_posix()}" '
                f'--output-dir "{OUT.as_posix()}" '
                '--assumption-helper <statistical-analysis/scripts/assumption_checks.py>')
    save("manifest.json", manifest)
    (OUT/"REPRODUCE.txt").write_text(
        "Повтор таблиц и прогнозов из сохранённых признаков\n\n"
        "1. Распаковать архив и установить версии из requirements_lock.txt в отдельное окружение Python 3.12.\n"
        "2. Из папки архива выполнить:\n"
        "python code/scripts/reproduce_multimodal_tables.py --results .\n\n"
        "Проверяются все корреляции, поправки, интервалы, метрики моделей и каждый прогноз. "
        "Для этого исходные видео и биосигналы не нужны.\n"
        "Полное повторное извлечение признаков выполняется scripts/analyze_multimodal.py в исходном проекте; "
        "оно требует локальных данных outputs/audio, outputs/video, outputs/bio, outputs/sync и data/private. "
        "Их SHA-256 перечислены в manifest.json.\n", encoding="utf-8")
    with zipfile.ZipFile(OUT/"analysis-results.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(OUT.iterdir()):
            if path.is_file() and path.suffix != ".zip": bundle.write(path, path.name)
        for path in code_files: bundle.write(path, "code/"+str(path.relative_to(ROOT)).replace("\\", "/"))
        bundle.write(args.assumption_helper, "code/assumption_checks.py")
    print(json.dumps(clean({"dataset":dataset,"profile_increment":increment,"models":[{"id":r["id"],"metrics":r["metrics"]} for r in models]}), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
