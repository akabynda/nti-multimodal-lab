"""Process all supplied BioRadio recordings."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from nti_lab.biosignals import analyze_recording, read_session
from nti_lab.palette import BLUE, PLOT_STYLE, RED


def aggregate_plot(output, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lookup = {s["subject_id"]: s for s in summaries}
    previous = output / "aggregate_summary.csv"
    prior = pd.read_csv(previous).to_dict("records") if previous.exists() else []
    labels = {r["subject_id"]: r["participant"] for r in prior}
    order = [r["subject_id"] for r in prior if r["subject_id"] in lookup]
    for path in sorted((output.parent / "video").glob("*.json")):
        try:
            subject = json.loads(path.read_text(encoding="utf-8")).get("participant_id")
        except json.JSONDecodeError:
            continue
        if subject in lookup and subject not in order:
            order.append(subject)
    order.extend(s for s in sorted(lookup) if s not in order)
    next_label = max([int(v[1:]) for v in labels.values() if v.startswith("P") and v[1:].isdigit()] or [0]) + 1
    records = []
    for subject in order:
        if subject in lookup:
            meta = lookup[subject]
            if subject not in labels:
                labels[subject] = f"P{next_label:02}"
                next_label += 1
            records.append({"participant": labels[subject], "subject_id": subject,
                            "median_hr_bpm": meta["median_hr_bpm"],
                            "usable_windows": meta["usable_ecg_windows"],
                            "total_windows": meta["window_count"],
                            "usable_pct": 100 * meta["usable_ecg_windows"] / meta["window_count"] if meta["window_count"] else 0.,
                            "rejected_rr_pct": 100 * meta["rr_rejected_fraction"]})
    data = pd.DataFrame(records)
    data.to_csv(output / "aggregate_summary.csv", index=False)
    with plt.rc_context({**PLOT_STYLE, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(1, 3, figsize=(11, max(4.6, .38 * len(data) + 1.5)), sharey=True, layout="constrained")
        axes[0].scatter(data["median_hr_bpm"], data["participant"], color=BLUE, marker="o")
        axes[0].set(xlabel="Медианная ЧСС в пригодных\nокнах (уд/мин)", ylabel="Участник")
        axes[1].barh(data["participant"], data["usable_pct"], color=BLUE)
        axes[1].set(xlabel="Окна ЭКГ, прошедшие\nконтроль качества (%)", xlim=(0, 100))
        axes[2].barh(data["participant"], data["rejected_rr_pct"], color=RED)
        axes[2].set(xlabel="Исключённые\nRR-интервалы (%)", xlim=(0, max(20, data["rejected_rr_pct"].max() * 1.1)))
        axes[0].invert_yaxis()
        for ax in axes:
            ax.grid(axis="x", alpha=1)
        fig.suptitle(f"БиоРадио: {len(data)} участников · окна 120 с · шаг 5 с", fontsize=11)
        fig.savefig(output / "aggregate_summary.png", dpi=180, facecolor="white")
        fig.savefig(output / "aggregate_summary.svg", facecolor="white")
        plt.close(fig)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "bio")
    parser.add_argument("--input", type=Path, nargs="+", help="Selected BCRX files; existing results remain in the aggregate")
    parser.add_argument("--subject-id", help="Explicit participant ID for one selected source")
    parser.add_argument("--force", action="store_true", help="Recompute even complete unchanged recordings")
    args = parser.parse_args()
    if args.subject_id and (not args.input or len(args.input) != 1):
        parser.error("--subject-id requires exactly one --input file")
    args.output.mkdir(parents=True, exist_ok=True)
    paths = args.input or sorted(p for p in args.data.iterdir() if p.is_file() and p.suffix.lower() == ".bcrx")
    for path in paths:
        subject = args.subject_id or path.stem
        if not subject or subject in (".", "..") or any(c in subject for c in '/\\:'):
            parser.error("subject ID must be a single directory name")
        destination = args.output / subject
        needed = [destination / n for n in ("metadata.json", "metrics.csv", "rr_intervals.csv", "signals.npz", "timeseries.json")]
        if not args.force and all(p.exists() for p in needed):
            previous = json.loads(needed[0].read_text(encoding="utf-8"))
            if previous.get("source_sha256") == hashlib.sha256(path.read_bytes()).hexdigest():
                print(f"Keeping unchanged {subject}", flush=True)
                continue
        print(f"Analyzing {path.name}", flush=True)
        analyze_recording(path, args.output, subject_id=subject)
    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(args.output.glob("*/metadata.json"))
                 if (p.parent / "metrics.csv").exists()]
    if not summaries:
        raise SystemExit(f"No BCRX inputs or existing results in {args.data}")
    summary_path = args.output / "summary.json"
    previous_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    result = {"recordings": summaries, "session": read_session(args.data / "Session1.bcsx") if (args.data / "Session1.bcsx").exists() else previous_summary.get("session")}
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(summary_path)
    pd.concat([pd.read_csv(args.output / s["subject_id"] / "metrics.csv") for s in summaries], ignore_index=True).to_csv(args.output / "windows.csv", index=False)
    aggregate_plot(args.output, summaries)
    print(f"Done: {len(summaries)} recordings, {sum(s['window_count'] for s in summaries)} windows", flush=True)


if __name__ == "__main__":
    main()
